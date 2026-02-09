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
        raise NotImplementedError
    
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
        raise NotImplementedError
    
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
        raise NotImplementedError
    
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
        raise NotImplementedError
    
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
        raise NotImplementedError
    
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
        raise NotImplementedError
