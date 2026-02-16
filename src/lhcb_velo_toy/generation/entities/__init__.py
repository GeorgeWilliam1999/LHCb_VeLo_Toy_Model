"""
Core entities for the LHCb VELO Toy Model.

This module exports the core data structures used throughout the package:

Hierarchy:
    Event
    ├── PrimaryVertex (collision points)
    ├── Track (particle trajectories, with hit_ids)
    ├── Hit (detector measurements, with track_id)
    └── Module (detector layers)

Note: Segments are NOT part of generation entities. They are computed
on-demand in the reconstruction phase using `get_segments_from_track()`.
"""

from lhcb_velo_toy.generation.entities.hit import Hit
from lhcb_velo_toy.generation.entities.track import Track
from lhcb_velo_toy.generation.entities.module import Module
from lhcb_velo_toy.generation.entities.event import Event
from lhcb_velo_toy.generation.entities.primary_vertex import PrimaryVertex

__all__ = [
    "Hit",
    "Track",
    "Module",
    "Event",
    "PrimaryVertex",
]
