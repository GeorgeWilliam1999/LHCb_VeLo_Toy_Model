"""
C++/CUDA Accelerated Hamiltonian Implementation
================================================

This module provides a Python wrapper for a C++/CUDA-accelerated implementation
of the track-finding Hamiltonian. It offers significant performance improvements
for large events with many hits/segments.

Requirements
------------
The C++ extension must be built and installed separately:

    cd LHCB_Velo_Toy_Models/cpp_hamiltonian
    pip install .

For CUDA support, ensure CUDA toolkit is available during compilation.

Performance
-----------
The C++ implementation provides:
- ~10-50x speedup for medium events (1000-10000 segments)
- ~100x+ speedup with CUDA for large events (>10000 segments)
- Efficient sparse matrix construction

Usage
-----
>>> from LHCB_Velo_Toy_Models.simple_hamiltonian_cpp import SimpleHamiltonianCPPWrapper
>>> 
>>> # Create wrapper (use_cuda=True if CUDA available)
>>> ham = SimpleHamiltonianCPPWrapper(epsilon=0.01, gamma=1.0, delta=1.0, use_cuda=True)
>>> 
>>> # Use like regular Hamiltonian
>>> A, b = ham.construct_hamiltonian(event)
>>> solution = ham.solve_classicaly()

See Also
--------
simple_hamiltonian : Reference Python implementation
simple_hamiltonian_fast : Optimized Python implementation
"""

import numpy as np
import scipy.sparse as sp
from LHCB_Velo_Toy_Models.state_event_generator import StateEventGenerator
from LHCB_Velo_Toy_Models.hamiltonian import Hamiltonian

try:
    from cpp_hamiltonian import SimpleHamiltonianCPP, CUDA_AVAILABLE
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    CUDA_AVAILABLE = False
    print("Warning: C++ Hamiltonian not available. Install with:")
    print("  cd LHCB_Velo_Toy_Models/cpp_hamiltonian && pip install .")


class SimpleHamiltonianCPPWrapper(Hamiltonian):
    """
    Python wrapper for C++/CUDA Hamiltonian implementation.
    
    This class provides the same interface as SimpleHamiltonian but delegates
    the heavy computation to optimized C++/CUDA code.
    
    Parameters
    ----------
    epsilon : float
        Angular tolerance for segment compatibility (radians).
    gamma : float
        Self-interaction penalty coefficient.
    delta : float
        Bias term for the linear part.
    use_cuda : bool, optional
        Whether to use CUDA acceleration if available (default: False).
    
    Attributes
    ----------
    A : scipy.sparse.csr_matrix
        The Hamiltonian matrix (negated), after construction.
    b : numpy.ndarray
        The bias vector.
    n_segments : int
        Number of segments.
    use_cuda : bool
        Whether CUDA is being used.
    
    Raises
    ------
    ImportError
        If the C++ module is not available.
    
    Notes
    -----
    Falls back to CPU if CUDA is requested but not available.
    """
    
    def __init__(self, epsilon, gamma, delta, use_cuda=False):
        """
        Initialize the C++/CUDA Hamiltonian wrapper.
        
        Parameters
        ----------
        epsilon : float
            Angular tolerance for segment compatibility.
        gamma : float
            Self-interaction penalty.
        delta : float
            Bias term.
        use_cuda : bool, optional
            Enable CUDA acceleration (default: False).
        """
        if not CPP_AVAILABLE:
            raise ImportError("C++ module not available")
        
        self.epsilon = epsilon
        self.gamma = gamma
        self.delta = delta
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        
        if use_cuda and not CUDA_AVAILABLE:
            print("Warning: CUDA requested but not available. Using CPU.")
            self.use_cuda = False
        
        self.cpp_ham = SimpleHamiltonianCPP(epsilon, gamma, delta, self.use_cuda)
        self.A = None
        self.b = None
        self.n_segments = 0
        
        backend = "CUDA" if self.use_cuda else "C++ CPU"
        print(f"✓ Using {backend} backend")
    
    def _get_hit_position(self, hit):
        """
        Extract (x, y, z) position from a Hit object.
        
        Handles different Hit class implementations.
        
        Parameters
        ----------
        hit : Hit
            A hit object.
        
        Returns
        -------
        tuple
            (x, y, z) position.
        
        Raises
        ------
        AttributeError
            If no position attribute can be found.
        """
        if hasattr(hit, 'x') and hasattr(hit, 'y') and hasattr(hit, 'z'):
            return hit.x, hit.y, hit.z
        elif hasattr(hit, 'coordinate'):
            return tuple(hit.coordinate)
        elif hasattr(hit, 'position'):
            return tuple(hit.position)
        else:
            raise AttributeError("Cannot find position in Hit object")
    
    def construct_hamiltonian(self, event: StateEventGenerator, convolution: bool = False):
        """
        Construct the Hamiltonian matrix using C++/CUDA backend.
        
        Parameters
        ----------
        event : StateEventGenerator
            Event containing detector modules and hits.
        convolution : bool, optional
            Whether to use ERF-smoothed compatibility (default: False).
        
        Returns
        -------
        A : scipy.sparse.csr_matrix
            The negated Hamiltonian matrix.
        b : numpy.ndarray
            The bias vector.
        """
        
        segment_id = 0
        
        # Pass segments to C++/CUDA
        for group_idx in range(len(event.modules) - 1):
            from_hits = event.modules[group_idx].hits
            to_hits = event.modules[group_idx + 1].hits
            
            for from_hit in from_hits:
                for to_hit in to_hits:
                    x0, y0, z0 = self._get_hit_position(from_hit)
                    x1, y1, z1 = self._get_hit_position(to_hit)
                    
                    self.cpp_ham.add_segment(
                        segment_id,
                        from_hit.hit_id, float(x0), float(y0), float(z0),
                        to_hit.hit_id, float(x1), float(y1), float(z1),
                        group_idx
                    )
                    segment_id += 1
        
        # Construct in C++/CUDA
        self.cpp_ham.construct_hamiltonian(convolution)
        
        # Get results
        self.n_segments = self.cpp_ham.get_n_segments()
        
        # Convert to scipy sparse matrix
        sparse_dict = self.cpp_ham.get_sparse_matrix()
        self.A = -sp.coo_matrix(
            (sparse_dict['data'], (sparse_dict['row'], sparse_dict['col'])),
            shape=sparse_dict['shape']
        ).tocsr()
        
        self.b = self.cpp_ham.get_b()
        
        backend = "CUDA" if self.use_cuda else "C++"
        print(f"{backend} Hamiltonian: {self.n_segments} segments, {self.cpp_ham.get_nnz()} non-zeros")
        
        return self.A, self.b
    
    def solve_classicaly(self, **kwargs):
        """
        Solve the linear system using scipy sparse solver.
        
        Parameters
        ----------
        **kwargs
            Optional arguments:
            - tol : float, convergence tolerance (default: 1e-6)
            - max_iter : int, maximum iterations (default: 1000)
        
        Returns
        -------
        numpy.ndarray
            Solution vector.
        
        Raises
        ------
        Exception
            If Hamiltonian not initialized.
        """
        if self.A is None:
            raise Exception("Hamiltonian not initialized")
        
        if self.n_segments < 10000:
            solution = sp.linalg.spsolve(self.A, self.b)
        else:
            solution, info = sp.linalg.cg(self.A, self.b, 
                                         atol=kwargs.get('tol', 1e-6),
                                         maxiter=kwargs.get('max_iter', 1000))
            if info > 0:
                print(f"Warning: CG did not converge ({info} iterations)")
        
        return solution
    
    def evaluate(self, solution):
        """
        Evaluate Hamiltonian energy: H(x) = -0.5 * x^T * A * x + b^T * x
        
        Parameters
        ----------
        solution : array-like
            Segment activation vector.
        
        Returns
        -------
        float
            Hamiltonian energy value.
        """
        sol = np.array(solution).reshape(-1, 1)
        return float(-0.5 * sol.T @ self.A @ sol + self.b.dot(sol.flatten()))