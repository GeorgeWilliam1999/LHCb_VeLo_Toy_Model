"""
Generation submodule for the LHCb VELO Toy Model.

This module provides data structures, detector geometries, and event generators
for simulating particle interactions in the LHCb VELO detector.

Submodules
----------
models
    Data structures: Hit, Segment, Track, Module, Event
geometry
    Detector geometries: Geometry (ABC), PlaneGeometry, RectangularVoidGeometry
generators
    Event generators: StateEventGenerator
"""

# Models
from lhcb_velo_toy.generation.models import (
    Hit,
    Segment,
    Track,
    Module,
    Event,
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
    "Segment",
    "Track",
    "Module",
    "Event",
    # Geometry
    "Geometry",
    "PlaneGeometry",
    "RectangularVoidGeometry",
    # Generators
    "StateEventGenerator",
]
