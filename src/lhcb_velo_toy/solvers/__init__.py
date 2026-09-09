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
    Quantum solver lanes: HHL, the one-bit quantum filter (OneBQF /
    OneBitHHL, + matrix-free engine) and the QSVT polynomial filter
    ("the comb") with its response families and realisations
hough
    Direct Hough track finding (point / locus voting) with an
    EventValidator adapter
postselect
    Solver-agnostic hit-uniqueness post-selection
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

# QSVT polynomial filter (numpy-only; the qiskit circuits stay lazy in
# lhcb_velo_toy.solvers.quantum)
from lhcb_velo_toy.solvers.quantum.QSVT import (
    QSVT,
    design_band_limited_inverse,
    design_line_comb_inverse,
    design_minimax_comb,
)
from lhcb_velo_toy.solvers.quantum.matrixfree import solve_one_bit_matrixfree

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
    # QSVT + 1BQF matrix-free engine
    "QSVT",
    "design_band_limited_inverse",
    "design_line_comb_inverse",
    "design_minimax_comb",
    "solve_one_bit_matrixfree",
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
