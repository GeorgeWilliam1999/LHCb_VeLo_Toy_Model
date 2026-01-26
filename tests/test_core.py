"""
Tests for the core module components.
"""

import pytest
import numpy as np


class TestDataModels:
    """Tests for Hit, Track, Segment data models."""

    def test_hit_creation(self):
        """Test that Hit objects can be created."""
        from velo_toy.core import Hit
        
        hit = Hit(hit_id=0, x=0.1, y=0.2, z=100.0, module_id=0, track_id=0)
        
        assert hit.hit_id == 0
        assert hit.x == 0.1
        assert hit.y == 0.2
        assert hit.z == 100.0
        assert hit.module_id == 0
        assert hit.track_id == 0

    def test_hit_indexing(self):
        """Test that Hit supports indexing for coordinates."""
        from velo_toy.core import Hit
        
        hit = Hit(hit_id=0, x=0.1, y=0.2, z=100.0, module_id=0, track_id=0)
        
        assert hit[0] == 0.1  # x
        assert hit[1] == 0.2  # y
        assert hit[2] == 100.0  # z

    def test_segment_vector(self):
        """Test Segment vector calculation."""
        from velo_toy.core import Hit, Segment
        
        hit1 = Hit(hit_id=0, x=0.0, y=0.0, z=0.0, module_id=0, track_id=0)
        hit2 = Hit(hit_id=1, x=1.0, y=0.0, z=1.0, module_id=1, track_id=0)
        
        segment = Segment(hits=[hit1, hit2], segment_id=0)
        vec = segment.to_vect()
        
        assert vec[0] == 1.0  # dx
        assert vec[1] == 0.0  # dy
        assert vec[2] == 1.0  # dz


class TestSimpleHamiltonian:
    """Tests for SimpleHamiltonian class."""

    def test_hamiltonian_creation(self, hamiltonian_params):
        """Test Hamiltonian can be instantiated."""
        from LHCB_Velo_Toy_Models.simple_hamiltonian import SimpleHamiltonian
        
        ham = SimpleHamiltonian(**hamiltonian_params)
        
        assert ham.epsilon == hamiltonian_params["epsilon"]
        assert ham.gamma == hamiltonian_params["gamma"]
        assert ham.delta == hamiltonian_params["delta"]

    def test_hamiltonian_matrix_is_sparse(self, hamiltonian_params):
        """Test that constructed Hamiltonian is sparse."""
        from LHCB_Velo_Toy_Models.simple_hamiltonian import SimpleHamiltonian
        from LHCB_Velo_Toy_Models.state_event_generator import StateEventGenerator
        import scipy.sparse as sp
        
        # Create a simple event
        generator = StateEventGenerator(
            n_modules=5,
            n_tracks=2,
            hit_resolution=0.0001,
            multi_scatter=0.0002,
        )
        event = generator.generate()
        
        ham = SimpleHamiltonian(**hamiltonian_params)
        A, b = ham.construct_hamiltonian(event)
        
        assert sp.issparse(A)
        assert isinstance(b, np.ndarray)


class TestStateEventGenerator:
    """Tests for event generation."""

    def test_generator_creates_events(self, simple_detector_config):
        """Test that generator creates events with correct properties."""
        from LHCB_Velo_Toy_Models.state_event_generator import StateEventGenerator
        
        generator = StateEventGenerator(**simple_detector_config)
        event = generator.generate()
        
        assert event is not None
        assert hasattr(event, 'hits')
        assert hasattr(event, 'modules')

    def test_generator_module_count(self, simple_detector_config):
        """Test that correct number of modules is created."""
        from LHCB_Velo_Toy_Models.state_event_generator import StateEventGenerator
        
        generator = StateEventGenerator(**simple_detector_config)
        event = generator.generate()
        
        assert len(event.modules) == simple_detector_config["n_modules"]

    def test_generator_reproducibility(self, simple_detector_config, random_seed):
        """Test that events are reproducible with same seed."""
        from LHCB_Velo_Toy_Models.state_event_generator import StateEventGenerator
        
        np.random.seed(random_seed)
        generator1 = StateEventGenerator(**simple_detector_config)
        event1 = generator1.generate()
        
        np.random.seed(random_seed)
        generator2 = StateEventGenerator(**simple_detector_config)
        event2 = generator2.generate()
        
        # Check same number of hits
        assert len(event1.hits) == len(event2.hits)
