"""Detector geometry classes for the LHCb VELO Toy Model."""

from lhcb_velo_toy.generation.geometry.base import Geometry
from lhcb_velo_toy.generation.geometry.plane import PlaneGeometry
from lhcb_velo_toy.generation.geometry.rectangular_void import RectangularVoidGeometry

__all__ = [
    "Geometry",
    "PlaneGeometry",
    "RectangularVoidGeometry",
]
