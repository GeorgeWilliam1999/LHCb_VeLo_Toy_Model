"""
Generation submodule for the LHCb VELO Toy Model.

This module provides data structures, detector geometries, and event generators
for simulating particle interactions in the LHCb VELO detector.

Submodules
----------
entities
    Core data structures: Hit, Track, Module, Event, PrimaryVertex
geometry
    Detector geometries: Geometry (ABC), PlaneGeometry, RectangularVoidGeometry
generators
    Event generators: StateEventGenerator

Note
----
Segments are NOT part of this module. They are computed on-demand during
reconstruction using functions from `solvers.reconstruction.segment`.
"""

# Entities
from lhcb_velo_toy.generation.entities import (
    Hit,
    Track,
    Module,
    Event,
    PrimaryVertex,
)

# Geometry
from lhcb_velo_toy.generation.geometry import (
    Geometry,
    PlaneGeometry,
    RectangularVoidGeometry,
)

# Generators
from lhcb_velo_toy.generation.generators import (
    StateEventGenerator,
)

__all__ = [
    # Models
    "Hit",
    "Track",
    "Module",
    "Event",
    "PrimaryVertex",
    # Geometry
    "Geometry",
    "PlaneGeometry",
    "RectangularVoidGeometry",
    # Generators
    "StateEventGenerator",
]
