from LHCB_Velo_Toy_Models.state_event_generator import StateEventGenerator

from abc import ABC, abstractmethod

class Hamiltonian(ABC):
    
    @abstractmethod
    def construct_hamiltonian(self, event: StateEventGenerator):
        pass
    
    @abstractmethod
    def evaluate(self, solution):
        pass
    