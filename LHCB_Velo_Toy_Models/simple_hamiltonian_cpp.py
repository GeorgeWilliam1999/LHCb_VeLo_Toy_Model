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
    """Python wrapper for C++/CUDA Hamiltonian implementation"""
    
    def __init__(self, epsilon, gamma, delta, use_cuda=False):
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
        """Extract position from Hit object"""
        if hasattr(hit, 'x') and hasattr(hit, 'y') and hasattr(hit, 'z'):
            return hit.x, hit.y, hit.z
        elif hasattr(hit, 'coordinate'):
            return tuple(hit.coordinate)
        elif hasattr(hit, 'position'):
            return tuple(hit.position)
        else:
            raise AttributeError("Cannot find position in Hit object")
    
    def construct_hamiltonian(self, event: StateEventGenerator, convolution: bool = False):
        """Construct Hamiltonian from event data"""
        
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
        """Solve using scipy sparse solver"""
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
        """Evaluate Hamiltonian energy"""
        sol = np.array(solution).reshape(-1, 1)
        return float(-0.5 * sol.T @ self.A @ sol + self.b.dot(sol.flatten()))