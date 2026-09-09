"""
LHCb VELO Toy Model Package.

A comprehensive framework for simulating particle tracking in the LHCb Vertex
Locator detector and testing track reconstruction algorithms, including the
quantum solver lanes: HHL, the one-bit quantum filter (1BQF) and the QSVT
polynomial spectral filter (the comb).

Submodules
----------
generation
    Data generation: event simulation, geometry, and data models
solvers
    Track reconstruction: Hamiltonians (+ fork / occupancy subclasses),
    classical / quantum (HHL, 1BQF, QSVT) / Hough solvers, post-selection
analysis
    Validation and plotting utilities
persistence
    Save / load utilities for pipeline state and parametric studies

Example
-------
>>> from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
>>> from lhcb_velo_toy.solvers import SimpleHamiltonian, get_tracks
>>> from lhcb_velo_toy.analysis import EventValidator
>>> from lhcb_velo_toy import QSVT            # the polynomial filter (numpy-only)
"""

__version__ = "2.2.0"
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
    ExtendedHamiltonian,
    ForkHamiltonian,
    OccupancyHamiltonian,
    get_tracks,
    get_tracks_layered,
    get_tracks_optimal,
    get_tracks_optimal_iterative,
    find_segments,
    QSVT,
    design_line_comb_inverse,
    solve_one_bit_matrixfree,
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

# Persistence submodule
from lhcb_velo_toy.persistence import (
    PipelineResult,
    save_pipeline,
    load_pipeline,
    save_events_batch,
    load_events_batch,
    StudyResult,
    save_study,
    load_study,
)

# Geometry dispatch
from lhcb_velo_toy.generation.geometry import geometry_from_dict

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
    "ExtendedHamiltonian",
    "ForkHamiltonian",
    "OccupancyHamiltonian",
    "get_tracks",
    "get_tracks_layered",
    "get_tracks_optimal",
    "get_tracks_optimal_iterative",
    "find_segments",
    "QSVT",
    "design_line_comb_inverse",
    "solve_one_bit_matrixfree",
    # Reconstruction
    "Segment",
    "get_segments_from_event",
    "get_segments_from_track",
    "get_all_possible_segments",
    "construct_event",
    # Analysis
    "Match",
    "EventValidator",
    # Persistence
    "PipelineResult",
    "save_pipeline",
    "load_pipeline",
    "save_events_batch",
    "load_events_batch",
    "StudyResult",
    "save_study",
    "load_study",
    # Geometry dispatch
    "geometry_from_dict",
]
