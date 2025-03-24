
from LHCB_Velo_Toy_Models.state_event_generator import StateEventGenerator
from LHCB_Velo_Toy_Models.state_event_model import Segment
from LHCB_Velo_Toy_Models.hamiltonian import Hamiltonian

from itertools import product, count
import scipy as sci
import numpy as np

class SimpleHamiltonian(Hamiltonian):
    def __init__(self, epsilon, gamma, delta):
        self.epsilon                                    = epsilon
        self.gamma                                      = gamma
        self.delta                                      = delta
        self.Z                                          = None
        self.A                                          = None
        self.b                                          = None
        self.segments                                   = None
        self.segments_grouped                           = None
        self.n_segments                                 = None
    
    def construct_segments(self, event: StateEventGenerator):
        
        segments_grouped = []
        segments = []
        n_segments = 0
        segment_id = count()

        for idx in range(len(event.modules)-1):
            from_hits = event.modules[idx].hits
            to_hits = event.modules[idx+1].hits
            print(len(to_hits))
            print(to_hits, '\n')

            segments_group = []
            for from_hit, to_hit in product(from_hits, to_hits):
                seg = Segment(next(segment_id),[from_hit, to_hit])
                segments_group.append(seg)
                segments.append(seg)
                n_segments = n_segments + 1
        
            segments_grouped.append(segments_group)
            
        self.segments_grouped = segments_grouped
        self.segments = segments
        self.n_segments = n_segments
        
    def construct_hamiltonian(self, event: StateEventGenerator):

        if self.segments_grouped is None:
            self.construct_segments(event)
        A = sci.sparse.eye(self.n_segments,format='lil')*(-(self.delta+self.gamma))
        b = np.ones(self.n_segments)*self.delta
        for group_idx in range(len(self.segments_grouped) - 1):
            for seg_i, seg_j in product(self.segments_grouped[group_idx], self.segments_grouped[group_idx+1]):
                print(seg_i)
                print(seg_i.hits[0])
                print(seg_i.hits[1])
                print(seg_i.hits[0] == seg_j.hits[1])
                print('\n')
                if seg_i.hits[0] == seg_j.hits[1]:
                    cosine = seg_i * seg_j
                    if abs(cosine - 1) < self.epsilon:
                        A[seg_i.segment_id, seg_j.segment_id] = A[seg_j.segment_id, seg_i.segment_id] =  1
        A = A.tocsc()
        
        self.A, self.b = -A, b
        return -A, b
    
    def solve_classicaly(self):
        if self.A is None:
            raise Exception("Not initialised")
        
        solution, _ = sci.sparse.linalg.cg(self.A, self.b, atol=0)
        return solution
    
    def evaluate(self, solution):
        if self.A is None:
            raise Exception("Not initialised")
        
        if isinstance(solution, list):
            sol = np.array([solution, None])
        elif isinstance(solution, np.ndarray):
            if solution.ndim == 1:
                sol = solution[..., None]
            else: sol = solution
            
            
        return -0.5 * sol.T @ self.A @ sol + self.b.dot(sol)