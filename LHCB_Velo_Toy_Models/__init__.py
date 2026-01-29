"""
LHCb VELO Toy Models Package
============================

A Python package for simulating and analyzing particle tracking in the LHCb VELO
(Vertex Locator) detector. This package provides tools for:

- **Event Generation**: Simulating particle collision events with configurable
  detector geometry, multiple scattering, and measurement errors.

- **Track Finding**: Implementing Hamiltonian-based track reconstruction using
  segment connectivity and angular constraints.

- **Quantum Algorithms**: Exploring quantum computing approaches (HHL algorithm)
  for solving the track finding linear system.

- **Validation & Plotting**: LHCb-style metrics and visualization tools for
  evaluating reconstruction performance.

Modules
-------
state_event_model
    Core data structures for hits, segments, tracks, and detector geometry.

state_event_generator
    Event generation using LHCb state vectors (x, y, tx, ty, p/q).

multi_scattering_generator
    Legacy event generator with multiple scattering simulation.

hamiltonian
    Abstract base class defining the Hamiltonian interface for track finding.

simple_hamiltonian
    Reference implementation of the track-finding Hamiltonian.

simple_hamiltonian_fast
    Optimized Hamiltonian implementation using vectorized operations.

simple_hamiltonian_cpp
    C++/CUDA accelerated Hamiltonian wrapper (requires optional C++ extension).

toy_validator
    LHCb-style track reconstruction validation and metrics.

lhcb_tracking_plots
    Comprehensive plotting utilities for performance visualization.

Example Usage
-------------
>>> from LHCB_Velo_Toy_Models import state_event_generator, simple_hamiltonian
>>> from LHCB_Velo_Toy_Models.state_event_model import PlaneGeometry
>>>
>>> # Define detector geometry
>>> geometry = PlaneGeometry(
...     module_id=list(range(10)),
...     lx=[50.0] * 10,
...     ly=[50.0] * 10,
...     z=[100 + i * 30 for i in range(10)]
... )
>>>
>>> # Create event generator
>>> generator = state_event_generator.StateEventGenerator(
...     detector_geometry=geometry,
...     events=1,
...     n_particles=[5]
... )

Authors
-------
George William, Marcel Kunze, Alain Chancé, and contributors.

License
-------
See LICENSE file in the repository root.
"""

from LHCB_Velo_Toy_Models import (
    state_event_generator,
    state_event_model,
    multi_scattering_generator,
    hamiltonian,
    simple_hamiltonian
)

__version__ = "0.1.0"
__all__ = [
    "state_event_generator",
    "state_event_model",
    "multi_scattering_generator",
    "hamiltonian",
    "simple_hamiltonian",
]