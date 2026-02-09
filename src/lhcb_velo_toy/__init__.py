"""
LHCb VELO Toy Model Package.

A comprehensive framework for simulating particle tracking in the LHCb Vertex
Locator detector and testing track reconstruction algorithms, including quantum
approaches using HHL.

Submodules
----------
generation
    Data generation: event simulation, geometry, and data models
solvers
    Track reconstruction: Hamiltonians, classical/quantum solvers
analysis
    Validation and plotting utilities

Example
-------
>>> from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
>>> from lhcb_velo_toy.solvers import SimpleHamiltonian, get_tracks
>>> from lhcb_velo_toy.analysis import EventValidator
"""

__version__ = "2.0.0"
__author__ = "George William Scriven, Xenofon Chiotopoulos, Alain Chancé"

# Core types
from lhcb_velo_toy.core.types import (
    HitID,
    ModuleID,
    SegmentID,
    TrackID,
    PVID,
    Position,
    StateVector,
)

# Generation submodule
from lhcb_velo_toy.generation import (
    Hit,
    Track,
    Module,
    Event,
    PrimaryVertex,
    Geometry,
    PlaneGeometry,
    RectangularVoidGeometry,
    StateEventGenerator,
)

# Solvers submodule
from lhcb_velo_toy.solvers import (
    Hamiltonian,
    SimpleHamiltonian,
    SimpleHamiltonianFast,
    get_tracks,
    find_segments,
)

# Segment from reconstruction (computed on-demand, not stored in Event)
from lhcb_velo_toy.solvers.reconstruction import (
    Segment,
    get_segments_from_event,
    get_segments_from_track,
    get_all_possible_segments,
    construct_event,
)

# Analysis submodule
from lhcb_velo_toy.analysis import (
    Match,
    EventValidator,
)

__all__ = [
    # Version
    "__version__",
    # Types
    "HitID",
    "ModuleID",
    "SegmentID",
    "TrackID",
    "PVID",
    "Position",
    "StateVector",
    # Generation
    "Hit",
    "Track",
    "Module",
    "Event",
    "PrimaryVertex",
    "Geometry",
    "PlaneGeometry",
    "RectangularVoidGeometry",
    "StateEventGenerator",
    # Solvers
    "Hamiltonian",
    "SimpleHamiltonian",
    "SimpleHamiltonianFast",
    "get_tracks",
    "find_segments",
    # Reconstruction
    "Segment",
    "get_segments_from_event",
    "get_segments_from_track",
    "get_all_possible_segments",
    "construct_event",
    # Analysis
    "Match",
    "EventValidator",
]
