"""
Simple Hamiltonian Implementation for Track Finding
====================================================

This module provides the reference implementation of the Hamiltonian-based
track finding algorithm for the LHCb VELO detector. The approach formulates
track reconstruction as a quadratic optimization problem.

Algorithm Overview
------------------
1. **Segment Construction**: Create all possible segments between hits on
   adjacent detector layers.

2. **Hamiltonian Construction**: Build a matrix A encoding segment compatibility:
   - Diagonal: -(delta + gamma) penalty terms
   - Off-diagonal: +1 for compatible segments (sharing a hit and nearly collinear)

3. **Solve Linear System**: Find x = A^(-1) * b using conjugate gradient.

4. **Track Extraction**: Group active segments (x > threshold) into tracks.

Parameters
----------
epsilon : float
    Angular tolerance (radians) for segment compatibility. Segments with
    angle difference < epsilon are considered compatible.
    
gamma : float
    Self-interaction penalty term in the diagonal.
    
delta : float
    Bias term encouraging segment activation.
    
theta_d : float
    Width of the ERF smoothing function for convolved compatibility.

Mathematical Details
--------------------
The Hamiltonian energy is:

    H(x) = -0.5 * x^T * A * x + b^T * x

For binary x, this is a QUBO (Quadratic Unconstrained Binary Optimization).
For continuous relaxation, we solve the linear system Ax = b.

The compatibility function between segments i and j (sharing a middle hit) is:

    A_ij = 1    if |arccos(v_i · v_j)| < epsilon   (hard threshold)
    
    A_ij = (1 + erf((epsilon - angle) / (theta_d * sqrt(2))))  (convolution)

where v_i, v_j are normalized direction vectors.

Example
-------
>>> from LHCB_Velo_Toy_Models.simple_hamiltonian import SimpleHamiltonian, get_tracks
>>> 
>>> # Create Hamiltonian with parameters
>>> ham = SimpleHamiltonian(epsilon=0.01, gamma=1.0, delta=1.0)
>>> 
>>> # Build from event
>>> A, b = ham.construct_hamiltonian(event, convolution=False)
>>> 
>>> # Solve
>>> solution = ham.solve_classicaly()
>>> 
>>> # Extract tracks
>>> tracks = get_tracks(ham, solution, event)

See Also
--------
simple_hamiltonian_fast : Optimized implementation
simple_hamiltonian_cpp : C++/CUDA accelerated version
"""

from LHCB_Velo_Toy_Models.state_event_generator import StateEventGenerator
from LHCB_Velo_Toy_Models.state_event_model import *
from LHCB_Velo_Toy_Models.hamiltonian import Hamiltonian

from itertools import product, count
from scipy.special import erf 
import scipy as sci
import numpy as np
from itertools import chain


class SimpleHamiltonian(Hamiltonian):
    """
    Reference implementation of the track-finding Hamiltonian.
    
    This class constructs and solves the Hamiltonian system for track finding
    using scipy sparse matrices.
    
    Parameters
    ----------
    epsilon : float
        Angular tolerance for segment compatibility (radians).
    gamma : float
        Self-interaction penalty coefficient.
    delta : float
        Bias term for the linear part of the Hamiltonian.
    theta_d : float, optional
        Width parameter for ERF-smoothed compatibility (default: 1e-4).
    
    Attributes
    ----------
    A : scipy.sparse.csc_matrix
        The Hamiltonian matrix (negated).
    b : numpy.ndarray
        The bias vector.
    segments : list[Segment]
        All constructed segments.
    segments_grouped : list[list[Segment]]
        Segments organized by layer transition.
    n_segments : int
        Total number of segments.
    
    Examples
    --------
    >>> ham = SimpleHamiltonian(epsilon=0.01, gamma=1.0, delta=1.0)
    >>> A, b = ham.construct_hamiltonian(event)
    >>> solution = ham.solve_classicaly()
    >>> energy = ham.evaluate(solution)
    """
    
    def __init__(self, epsilon, gamma, delta, theta_d=1e-4):
        """
        Initialize the Hamiltonian with physics parameters.
        
        Parameters
        ----------
        epsilon : float
            Angular tolerance for segment compatibility (radians).
            Typical values: 0.001 - 0.1 depending on multiple scattering.
        gamma : float
            Self-interaction penalty. Higher values penalize segment activation.
        delta : float
            Bias term. Higher values encourage more segments to be active.
        theta_d : float, optional
            Width of ERF smoothing for convolved compatibility (default: 1e-4).
        """
        self.epsilon                                    = epsilon
        self.gamma                                      = gamma
        self.delta                                      = delta
        self.theta_d                                   = theta_d
        self.Z                                          = None
        self.A                                          = None
        self.b                                          = None
        self.segments                                   = None
        self.segments_grouped                           = None
        self.n_segments                                 = None
    
    def construct_segments(self, event: StateEventGenerator):
        """
        Build all possible segments between adjacent detector layers.
        
        Creates segments for every pair of hits on consecutive modules,
        representing all potential track segment candidates.
        
        Parameters
        ----------
        event : StateEventGenerator
            Event containing detector modules with hits.
        
        Notes
        -----
        Populates self.segments, self.segments_grouped, and self.n_segments.
        segments_grouped[i] contains all segments from layer i to layer i+1.
        """
        segments_grouped = []
        segments = []
        n_segments = 0
        segment_id = count()

        for idx in range(len(event.modules)-1):
            from_hits = event.modules[idx].hits
            to_hits = event.modules[idx+1].hits

            segments_group = []
            for from_hit, to_hit in product(from_hits, to_hits):
                seg = Segment([from_hit, to_hit],next(segment_id))
                segments_group.append(seg)
                segments.append(seg)
                n_segments = n_segments + 1
        
            segments_grouped.append(segments_group)
            
        self.segments_grouped = segments_grouped
        self.segments = segments
        self.n_segments = n_segments
        
    def construct_hamiltonian(self, event: StateEventGenerator, convolution: bool= False):
        """
        Construct the Hamiltonian matrix A and bias vector b.
        
        Parameters
        ----------
        event : StateEventGenerator
            Event containing detector modules and hits.
        convolution : bool, optional
            If True, use ERF-smoothed compatibility function.
            If False (default), use hard threshold.
        
        Returns
        -------
        A : scipy.sparse.csc_matrix
            The negated Hamiltonian matrix.
        b : numpy.ndarray
            The bias vector (all elements = delta).
        
        Notes
        -----
        The matrix A encodes:
        - Diagonal[i,i] = -(delta + gamma) for self-interaction
        - A[i,j] = 1 if segments i,j share a hit and are angularly compatible
        
        With convolution=True, the step function is smoothed with an ERF:
        A[i,j] = 1 + erf((epsilon - angle) / (theta_d * sqrt(2)))
        """
        # Check to see if EFR < thresh then map to 0.
        Segment.id_counter = 0
        if self.segments_grouped is None:
            self.construct_segments(event)
        A = sci.sparse.eye(self.n_segments,format='lil')*(-(self.delta+self.gamma))
        b = np.ones(self.n_segments)*self.delta
        for group_idx in range(len(self.segments_grouped) - 1):
            for seg_i, seg_j in product(self.segments_grouped[group_idx], self.segments_grouped[group_idx+1]):
                if seg_i.hits[1] == seg_j.hits[0]:
                    cosine = (seg_i * seg_j) 
                    if convolution:
                        convolved_step = (1 + erf((self.epsilon - abs(np.arccos(cosine))) / (self.theta_d * np.sqrt(2))))
                        # if convolved_step < 1e-3:
                        #     convolved_step = 0
                        A[seg_i.segment_id, seg_j.segment_id] = A[seg_j.segment_id, seg_i.segment_id] =  convolved_step
                    else: 
                        if abs(cosine - 1) < self.epsilon:
                            A[seg_i.segment_id, seg_j.segment_id] = A[seg_j.segment_id, seg_i.segment_id] =  1
        A = A.tocsc()

        self.A, self.b = -A, b
        return -A, b
    
    def solve_classicaly(self):
        """
        Solve the linear system Ax = b using conjugate gradient.
        
        Returns
        -------
        numpy.ndarray
            Solution vector x giving segment activation levels.
        
        Raises
        ------
        Exception
            If the Hamiltonian has not been constructed yet.
        """
        if self.A is None:
            raise Exception("Not initialised")
        
        solution, _ = sci.sparse.linalg.cg(self.A, self.b, atol=0)
        return solution
    
    def evaluate(self, solution: list):
        """
        Evaluate the Hamiltonian energy for a given solution.
        
        Computes H(x) = -0.5 * x^T * A * x + b^T * x
        
        Parameters
        ----------
        solution : array-like
            Segment activation vector.
        
        Returns
        -------
        float
            Hamiltonian energy value.
        
        Raises
        ------
        Exception
            If the Hamiltonian has not been constructed yet.
        """

        if self.A is None:
            raise Exception("Not initialised")
        
        if isinstance(solution, list):
            sol = np.array([solution, None])
        elif isinstance(solution, np.ndarray):
            if solution.ndim == 1:
                sol = solution[..., None]
            else: sol = solution
            
            
        return -0.5 * sol.T @ self.A @ sol + self.b.dot(sol)
    
from copy import deepcopy
from LHCB_Velo_Toy_Models.state_event_model import Track


def find_segments(s0: Segment, active: Segment):
    """
    Find segments connected to a given segment.
    
    Two segments are connected if they share an endpoint hit
    (the end of one equals the start of another).
    
    Parameters
    ----------
    s0 : Segment
        The reference segment.
    active : list[Segment]
        List of segments to search for connections.
    
    Returns
    -------
    list[Segment]
        Segments connected to s0.
    """
    found_s = []
    for s1 in active:
        if s0.hits[0].hit_id == s1.hits[1].hit_id or \
        s1.hits[0].hit_id == s0.hits[1].hit_id:
            found_s.append(s1)
    return found_s


def get_tracks(ham: SimpleHamiltonian, classical_solution: list[int], event: StateEventGenerator):
    """
    Extract tracks from the Hamiltonian solution.
    
    Identifies active segments (solution > min) and groups connected
    segments into track candidates.
    
    Parameters
    ----------
    ham : SimpleHamiltonian
        The Hamiltonian instance with constructed segments.
    classical_solution : array-like
        Solution vector from solve_classicaly().
    event : StateEventGenerator
        The event containing hits for track construction.
    
    Returns
    -------
    list[Track]
        List of reconstructed tracks.
    
    Algorithm
    ---------
    1. Filter segments where activation > minimum value
    2. Starting from an arbitrary active segment, grow track by
       finding connected segments (depth-first search)
    3. Convert segment chains to Track objects with ordered hits
    """
    active_segments = [segment for segment,pseudo_state in zip(ham.segments,classical_solution) if pseudo_state > np.min(classical_solution)]
    active = deepcopy(active_segments)
    tracks = []
    while len(active):
        s = active.pop()
        nextt = find_segments(s, active)
        track = set([s.hits[0].hit_id, s.hits[1].hit_id])
        while len(nextt):
            s = nextt.pop()
            try:
                active.remove(s)
            except:
                pass
            nextt += find_segments(s, active)
            track = track.union(set([s.hits[0].hit_id, s.hits[1].hit_id]))
        tracks.append(track)

    tracks_processed = []
    for track_ind, track in enumerate(tracks):
        track_hits = []
        track_segs = []
        for hit_id in track:
            matching_hits = list(filter(lambda b: b.hit_id == hit_id, event.hits))
            if matching_hits:  
                track_hits.append(matching_hits[0])
        for idx in range(len(track_hits) - 1):
            track_segs.append(Segment(hits=[track_hits[idx], track_hits[idx + 1]], segment_id=idx))
        if track_hits and track_segs:
            tracks_processed.append(Track(track_ind, track_hits, track_segs))
    return tracks_processed


def construct_event(detector_geometry, tracks, hits, segments, modules):
    """
    Construct an Event object from its components.
    
    Parameters
    ----------
    detector_geometry : Geometry
        The detector geometry specification.
    tracks : list[Track]
        List of tracks.
    hits : list[Hit]
        List of hits.
    segments : list[Segment]
        List of segments.
    modules : list[Module]
        List of detector modules.
    
    Returns
    -------
    Event
        A complete event object.
    """
    return Event(detector_geometry=detector_geometry,
                                   tracks=tracks,
                                   hits=hits,
                                   segments=segments,
                                   modules=modules)
