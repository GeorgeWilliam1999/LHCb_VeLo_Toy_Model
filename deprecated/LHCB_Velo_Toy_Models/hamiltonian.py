"""
Hamiltonian Abstract Base Class for Track Finding
==================================================

This module defines the abstract interface for Hamiltonian-based track finding
in the LHCb VELO detector. The track finding problem is formulated as minimizing
a quadratic objective function (QUBO-style) over segment activation variables.

Mathematical Formulation
------------------------
The track finding Hamiltonian is structured as:

    H(x) = -0.5 * x^T * A * x + b^T * x

where:
    - x is a vector of segment activation variables (continuous or binary)
    - A is the interaction matrix encoding segment compatibility
    - b is the bias vector encouraging segment activation

The matrix A encodes:
    - Diagonal terms: self-interaction penalties (gamma + delta)
    - Off-diagonal terms: segment-to-segment compatibility based on
      angular alignment (segments sharing a hit and pointing in similar
      directions receive positive coupling)

The solution to the linear system Ax = b gives the optimal segment activations,
which can then be thresholded to extract track candidates.

Implementations
---------------
- SimpleHamiltonian: Reference Python implementation
- SimpleHamiltonianFast: Optimized implementation with vectorized operations
- SimpleHamiltonianCPPWrapper: C++/CUDA accelerated version

See Also
--------
simple_hamiltonian : Reference implementation
simple_hamiltonian_fast : Optimized implementation
"""

from LHCB_Velo_Toy_Models.state_event_generator import StateEventGenerator

from abc import ABC, abstractmethod


class Hamiltonian(ABC):
    """
    Abstract base class for track-finding Hamiltonians.
    
    This class defines the interface that all Hamiltonian implementations must
    follow. Subclasses implement the specific construction logic for building
    the interaction matrix A and bias vector b from detector event data.
    
    Attributes
    ----------
    A : scipy.sparse matrix or numpy.ndarray
        The Hamiltonian interaction matrix (set after construct_hamiltonian).
    b : numpy.ndarray
        The bias vector (set after construct_hamiltonian).
    segments : list
        List of Segment objects (set after construct_hamiltonian).
    n_segments : int
        Total number of segments (set after construct_hamiltonian).
    
    Methods
    -------
    construct_hamiltonian(event)
        Build the Hamiltonian matrix A and vector b from event data.
    evaluate(solution)
        Compute the Hamiltonian energy for a given solution vector.
    """
    
    @abstractmethod
    def construct_hamiltonian(self, event: StateEventGenerator):
        """
        Construct the Hamiltonian matrix and bias vector from event data.
        
        This method processes the hits in the event, constructs all possible
        segments between adjacent detector layers, and builds the interaction
        matrix A encoding segment compatibility based on angular constraints.
        
        Parameters
        ----------
        event : StateEventGenerator
            The event containing detector modules and hits.
        
        Returns
        -------
        A : scipy.sparse.csc_matrix or numpy.ndarray
            The Hamiltonian interaction matrix (negated for minimization).
        b : numpy.ndarray
            The bias vector.
        
        Notes
        -----
        After calling this method, the instance attributes `A`, `b`, 
        `segments`, and `n_segments` should be populated.
        """
        pass
    
    @abstractmethod
    def evaluate(self, solution):
        """
        Evaluate the Hamiltonian energy for a given solution.
        
        Computes H(x) = -0.5 * x^T * A * x + b^T * x
        
        Parameters
        ----------
        solution : array-like
            The segment activation vector to evaluate.
        
        Returns
        -------
        float
            The Hamiltonian energy value.
        
        Raises
        ------
        Exception
            If the Hamiltonian has not been constructed yet.
        """
        pass
    
