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
)

# Reconstruction
from lhcb_velo_toy.solvers.reconstruction import (
    get_tracks,
    find_segments,
)

__all__ = [
    # Hamiltonians
    "Hamiltonian",
    "SimpleHamiltonian",
    "SimpleHamiltonianFast",
    # Reconstruction
    "get_tracks",
    "find_segments",
]
