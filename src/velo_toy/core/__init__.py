"""
Core VELO toy model components.

This module contains the fundamental building blocks:
- Event generation and data models
- Hamiltonian construction and solvers
- Track reconstruction algorithms
- Validation and metrics computation
"""

from .state_event_model import (
    Event, Track, Segment, Hit, Module,
    PlaneGeometry, RectangularVoidGeometry
)
from .state_event_generator import StateEventGenerator
from .simple_hamiltonian import SimpleHamiltonian, get_tracks, construct_event
from .toy_validator import EventValidator
from .hamiltonian import Hamiltonian

__all__ = [
    # Data models
    'Event', 'Track', 'Segment', 'Hit', 'Module',
    'PlaneGeometry', 'RectangularVoidGeometry',
    # Generators
    'StateEventGenerator',
    # Hamiltonian
    'Hamiltonian', 'SimpleHamiltonian',
    'get_tracks', 'construct_event',
    # Validation
    'EventValidator',
]
