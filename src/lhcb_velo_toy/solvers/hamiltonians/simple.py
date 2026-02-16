"""
SimpleHamiltonian: Reference implementation for track finding.

This is the reference (non-optimized) implementation of the track-finding
Hamiltonian, prioritizing clarity over performance.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import cg
from scipy.special import erf

from lhcb_velo_toy.solvers.hamiltonians.base import Hamiltonian

if TYPE_CHECKING:
    from lhcb_velo_toy.solvers.reconstruction.segment import Segment
    from lhcb_velo_toy.generation.entities.event import Event
    from lhcb_velo_toy.generation.generators.state_event import StateEventGenerator


class SimpleHamiltonian(Hamiltonian):
    """
    Reference implementation of the track-finding Hamiltonian.
    
    Constructs a Hamiltonian that encodes track finding as an optimization
    problem. Segments are compatible if they share a hit and are angularly
    aligned within tolerance epsilon.
    
    Parameters
    ----------
    epsilon : float
        Angular tolerance for segment compatibility (radians).
        Segments with angle difference < epsilon are considered compatible.
    gamma : float
        Self-interaction penalty coefficient.
    delta : float
        Bias term coefficient.
    theta_d : float, default 1e-4
        Width parameter for ERF-smoothed thresholding.
    
    Attributes
    ----------
    epsilon : float
        Angular tolerance parameter.
    gamma : float
        Self-interaction penalty.
    delta : float
        Bias term.
    theta_d : float
        ERF smoothing width.
    A : csc_matrix
        Hamiltonian matrix after construction.
    b : ndarray
        Bias vector after construction.
    segments : list[Segment]
        All constructed segments.
    n_segments : int
        Number of segments.
    
    Examples
    --------
    >>> ham = SimpleHamiltonian(epsilon=0.01, gamma=1.5, delta=1.0)
    >>> ham.construct_hamiltonian(event)
    >>> solution = ham.solve_classicaly()
    >>> energy = ham.evaluate(solution)
    
    Notes
    -----
    The Hamiltonian matrix encodes:
    
    - Diagonal elements: A[i,i] = -(γ + δ)
    - Off-diagonal elements: A[i,j] = compatibility(seg_i, seg_j)
    
    With hard threshold (convolution=False):
        compatibility = 1 if θ < ε else 0
    
    With ERF smoothing (convolution=True):
        compatibility = 1 + erf((ε - θ) / (θ_d * √2))
    """
    
    def __init__(
        self,
        epsilon: float,
        gamma: float,
        delta: float,
        theta_d: float = 1e-4,
    ) -> None:
        """Initialize the SimpleHamiltonian."""
        self.epsilon = epsilon
        self.gamma = gamma
        self.delta = delta
        self.theta_d = theta_d
        self.A: Optional[csc_matrix] = None
        self.b: Optional[np.ndarray] = None
        self.segments: list["Segment"] = []
        self.segments_grouped: list[list["Segment"]] = []
        self.n_segments: int = 0
    
    def construct_segments(
        self,
        event: "Event | StateEventGenerator",
    ) -> None:
        """
        Construct all segment candidates from the event.
        
        Creates segments between all pairs of hits on adjacent detector
        modules.
        
        Parameters
        ----------
        event : Event or StateEventGenerator
            The event containing hits and modules.
        
        Notes
        -----
        For N_i hits on module i and N_{i+1} hits on module i+1,
        this creates N_i * N_{i+1} segment candidates.
        """
        from itertools import product, count
        from lhcb_velo_toy.solvers.reconstruction.segment import Segment

        # Accept either an Event or a StateEventGenerator
        evt = getattr(event, "true_event", event)

        segments_grouped: list[list[Segment]] = []
        segments: list[Segment] = []
        n_segments = 0
        segment_id = count()

        for idx in range(len(evt.modules) - 1):
            # Resolve hit objects from IDs
            from_hit_ids = evt.modules[idx].hit_ids
            to_hit_ids = evt.modules[idx + 1].hit_ids
            from_hits = evt.get_hits_by_ids(from_hit_ids)
            to_hits = evt.get_hits_by_ids(to_hit_ids)

            group: list[Segment] = []
            for from_hit, to_hit in product(from_hits, to_hits):
                seg = Segment(
                    hit_start=from_hit,
                    hit_end=to_hit,
                    segment_id=next(segment_id),
                )
                group.append(seg)
                segments.append(seg)
                n_segments += 1

            segments_grouped.append(group)

        self.segments_grouped = segments_grouped
        self.segments = segments
        self.n_segments = n_segments
    
    def construct_hamiltonian(
        self,
        event: "Event | StateEventGenerator",
        convolution: bool = False,
    ) -> tuple[csc_matrix, np.ndarray]:
        """
        Construct the Hamiltonian matrix and bias vector.
        
        Parameters
        ----------
        event : Event or StateEventGenerator
            The event containing hits and geometry.
        convolution : bool, default False
            If True, use ERF-smoothed compatibility.
            If False, use hard threshold.
        
        Returns
        -------
        tuple[csc_matrix, ndarray]
            The matrix A and vector b.
        
        Notes
        -----
        Algorithm:
        1. Call construct_segments to build segment candidates
        2. Initialize A as sparse LIL matrix
        3. Set diagonal A[i,i] = -(γ + δ)
        4. For each pair of segments sharing a hit:
           - Compute angle θ between them
           - Set A[i,j] based on compatibility function
        5. Set b[i] = γ + δ for all i
        6. Convert A to CSC format for efficient solving
        """
        from itertools import product as iterproduct
        from scipy.sparse import eye as speye

        if not self.segments_grouped:
            self.construct_segments(event)

        n = self.n_segments

        # Start with diagonal
        A = speye(n, format="lil") * -(self.delta + self.gamma)
        b = np.ones(n) * self.delta

        # Off-diagonal: pairs of consecutive groups sharing a hit
        for group_idx in range(len(self.segments_grouped) - 1):
            for seg_i, seg_j in iterproduct(
                self.segments_grouped[group_idx],
                self.segments_grouped[group_idx + 1],
            ):
                # Segments must share the middle hit
                if seg_i.hit_end.hit_id == seg_j.hit_start.hit_id:
                    value = self._compute_compatibility(
                        seg_i, seg_j, convolution
                    )
                    if value > 0:
                        A[seg_i.segment_id, seg_j.segment_id] = value
                        A[seg_j.segment_id, seg_i.segment_id] = value

        A = A.tocsc()
        self.A, self.b = -A, b
        return -A, b
    
    def _compute_compatibility(
        self,
        seg_i: "Segment",
        seg_j: "Segment",
        convolution: bool,
    ) -> float:
        """
        Compute the compatibility between two segments.
        
        Parameters
        ----------
        seg_i : Segment
            First segment.
        seg_j : Segment
            Second segment.
        convolution : bool
            Whether to use ERF smoothing.
        
        Returns
        -------
        float
            Compatibility value in [0, 2] for ERF, {0, 1} for hard threshold.
        """
        cosine = seg_i * seg_j  # __mul__ returns cos(angle)

        if convolution:
            angle = abs(np.arccos(np.clip(cosine, -1.0, 1.0)))
            return float(
                1 + erf(
                    (self.epsilon - angle)
                    / (self.theta_d * np.sqrt(2))
                )
            )
        else:
            # Hard threshold: compatible if nearly collinear
            if abs(cosine - 1.0) < self.epsilon:
                return 1.0
            return 0.0
    
    def solve_classicaly(self) -> np.ndarray:
        """
        Solve the linear system A x = b using conjugate gradient.
        
        Returns
        -------
        numpy.ndarray
            Solution vector.
        
        Raises
        ------
        ValueError
            If construct_hamiltonian has not been called.
        """
        if self.A is None or self.b is None:
            raise ValueError(
                "Hamiltonian not constructed. "
                "Call construct_hamiltonian() first."
            )
        solution, _ = cg(self.A, self.b, atol=0)
        return solution
    
    def evaluate(self, solution: np.ndarray) -> float:
        """
        Evaluate the Hamiltonian energy for a solution.
        
        Computes H(x) = -0.5 * x^T A x + b^T x
        
        Parameters
        ----------
        solution : numpy.ndarray
            Segment activation vector.
        
        Returns
        -------
        float
            Hamiltonian energy.
        """
        if self.A is None or self.b is None:
            raise ValueError(
                "Hamiltonian not constructed. "
                "Call construct_hamiltonian() first."
            )

        if isinstance(solution, list):
            sol = np.array(solution)
        else:
            sol = solution

        if sol.ndim == 1:
            sol = sol[:, np.newaxis]

        return float(-0.5 * sol.T @ self.A @ sol + self.b.dot(sol.flatten()))
