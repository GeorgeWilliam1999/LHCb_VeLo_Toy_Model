"""
Abstract base class for track-finding Hamiltonians.

The Hamiltonian formulation converts track finding into an optimization
problem that can be solved classically or with quantum algorithms.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.sparse import csc_matrix

if TYPE_CHECKING:
    from lhcb_velo_toy.solvers.reconstruction.segment import Segment
    from lhcb_velo_toy.generation.entities.event import Event
    from lhcb_velo_toy.generation.generators.state_event import StateEventGenerator


class Hamiltonian(ABC):
    """
    Abstract base class for track-finding Hamiltonians.
    
    A Hamiltonian encodes the track-finding problem as a quadratic
    optimization problem:
    
    .. math::
        H(\\mathbf{x}) = -\\frac{1}{2} \\mathbf{x}^T A \\mathbf{x} + \\mathbf{b}^T \\mathbf{x}
    
    where x is the segment activation vector. The minimum of H corresponds
    to the optimal track configuration.
    
    Attributes
    ----------
    A : scipy.sparse.csc_matrix
        The Hamiltonian matrix encoding segment interactions.
    b : numpy.ndarray
        The bias vector.
    segments : list[Segment]
        List of all segment candidates.
    n_segments : int
        Total number of segments.
    
    Notes
    -----
    Subclasses must implement:
    - `construct_hamiltonian`: Build the A matrix and b vector
    - `evaluate`: Compute H(x) for a given solution
    
    The matrix A typically encodes:
    - Diagonal: Self-interaction penalties (discouraging segment activation)
    - Off-diagonal: Compatibility between segments (angular alignment)
    """
    
    A: Optional[csc_matrix] = None
    b: Optional[np.ndarray] = None
    segments: list["Segment"]
    n_segments: int
    
    @abstractmethod
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
            The event containing hits and geometry information.
        convolution : bool, default False
            If True, use ERF-smoothed thresholding for angular compatibility.
            If False, use hard threshold.
        
        Returns
        -------
        tuple[csc_matrix, ndarray]
            The Hamiltonian matrix A and bias vector b.
        
        Notes
        -----
        This method should:
        1. Construct all segment candidates from adjacent module hits
        2. Compute angular compatibility between segments
        3. Build the sparse matrix A with:
           - Diagonal: -(γ + δ)
           - Off-diagonal: 1 if segments share a hit and are compatible
        4. Build the bias vector b with entries (γ + δ)
        """
        ...
    
    @abstractmethod
    def evaluate(self, solution: np.ndarray) -> float:
        """
        Evaluate the Hamiltonian for a given solution.
        
        Computes H(x) = -0.5 * x^T A x + b^T x
        
        Parameters
        ----------
        solution : numpy.ndarray
            Segment activation vector of length n_segments.
        
        Returns
        -------
        float
            The Hamiltonian energy value.
        
        Raises
        ------
        ValueError
            If solution length doesn't match n_segments.
        """
        ...
    
    def solve_classicaly(self) -> np.ndarray:
        """
        Solve the linear system A x = b classically.
        
        Uses iterative conjugate gradient or direct methods depending
        on matrix size and properties.
        
        Returns
        -------
        numpy.ndarray
            Solution vector x.
        
        Raises
        ------
        ValueError
            If construct_hamiltonian has not been called.
        """
        raise NotImplementedError
    
    def get_matrix_dense(self) -> np.ndarray:
        """
        Get the Hamiltonian matrix as a dense array.
        
        Returns
        -------
        numpy.ndarray
            Dense representation of A.
        
        Warnings
        --------
        This can consume significant memory for large matrices.
        """
        raise NotImplementedError
