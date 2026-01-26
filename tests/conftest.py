"""
Pytest configuration and fixtures for VELO Toy Model tests.
"""

import pytest
import numpy as np


@pytest.fixture
def simple_detector_config():
    """Simple detector configuration for testing."""
    return {
        "n_modules": 10,
        "n_tracks": 3,
        "hit_resolution": 0.0001,
        "multi_scatter": 0.0002,
    }


@pytest.fixture
def hamiltonian_params():
    """Default Hamiltonian parameters for testing."""
    return {
        "epsilon": 0.001,
        "gamma": 1.0,
        "delta": 1.0,
    }


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    seed = 42
    np.random.seed(seed)
    return seed


@pytest.fixture
def small_matrix():
    """Small test matrix for HHL algorithm."""
    return np.array([[1, -1/3], [-1/3, 1]])


@pytest.fixture
def small_vector():
    """Small test vector for HHL algorithm."""
    return np.array([1, 0])
