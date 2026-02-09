"""
Type definitions and protocols for the LHCb VELO Toy Model.

This module defines type aliases and protocols used throughout the package
to ensure type safety and clear interfaces.
"""

from typing import TypeAlias, Protocol, runtime_checkable

# =============================================================================
# Type Aliases
# =============================================================================

HitID: TypeAlias = int
"""Unique identifier for a detector hit."""

ModuleID: TypeAlias = int
"""Unique identifier for a detector module/layer."""

SegmentID: TypeAlias = int
"""Unique identifier for a track segment."""

TrackID: TypeAlias = int
"""Unique identifier for a particle track."""

PVID: TypeAlias = int
"""Unique identifier for a primary vertex."""

Position: TypeAlias = tuple[float, float, float]
"""3D position as (x, y, z) coordinates in mm."""

StateVector: TypeAlias = dict[str, float]
"""
LHCb state vector dictionary.

Keys
----
x : float
    X position (mm)
y : float
    Y position (mm)
z : float
    Z position (mm)
tx : float
    X slope (dx/dz)
ty : float
    Y slope (dy/dz)
p/q : float
    Momentum divided by charge (MeV/c)
"""


# =============================================================================
# Protocols
# =============================================================================

@runtime_checkable
class SupportsPosition(Protocol):
    """Protocol for objects with 3D position coordinates."""
    
    x: float
    y: float
    z: float


@runtime_checkable
class SupportsIteration(Protocol):
    """Protocol for objects supporting len() and indexing."""
    
    def __len__(self) -> int:
        """Return the number of items."""
        ...
    
    def __getitem__(self, index: int) -> tuple:
        """Get item at index."""
        ...


__all__ = [
    "HitID",
    "ModuleID", 
    "SegmentID",
    "TrackID",
    "PVID",
    "Position",
    "StateVector",
    "SupportsPosition",
    "SupportsIteration",
]
