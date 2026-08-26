"""
Solvers submodule for the LHCb VELO Toy Model.

This module provides Hamiltonian formulations, classical/quantum solvers,
and track reconstruction utilities.

Submodules
----------
hamiltonians
    Hamiltonian base class and implementations
classical
    Classical linear system solvers
quantum
    Quantum algorithm implementations (HHL, OneBitHHL)
reconstruction
    Track extraction from solutions
"""

# Hamiltonians
from lhcb_velo_toy.solvers.hamiltonians import (
    Hamiltonian,
    SimpleHamiltonian,
    SimpleHamiltonianFast,
    ExtendedHamiltonian,
    ForkHamiltonian,
    OccupancyHamiltonian,
)

# Reconstruction
from lhcb_velo_toy.solvers.reconstruction import (
    Segment,
    get_tracks,
    get_tracks_fast,
    get_tracks_layered,
    get_tracks_optimal,
    get_tracks_optimal_iterative,
    find_segments,
    construct_event,
    get_segments_from_event,
    get_segments_from_track,
    get_candidate_segments,
    get_all_possible_segments,
)

# Post-selection (solver-agnostic hit-uniqueness filter)
from lhcb_velo_toy.solvers.postselect import (
    hit_uniqueness_filter,
    uniqueness_frontier,
)

__all__ = [
    # Hamiltonians
    "Hamiltonian",
    "SimpleHamiltonian",
    "SimpleHamiltonianFast",
    "ExtendedHamiltonian",
    "ForkHamiltonian",
    "OccupancyHamiltonian",
    # Post-selection
    "hit_uniqueness_filter",
    "uniqueness_frontier",
    # Reconstruction
    "Segment",
    "get_tracks",
    "get_tracks_fast",
    "get_tracks_layered",
    "get_tracks_optimal",
    "get_tracks_optimal_iterative",
    "find_segments",
    "construct_event",
    "get_segments_from_event",
    "get_segments_from_track",
    "get_candidate_segments",
    "get_all_possible_segments",
]
