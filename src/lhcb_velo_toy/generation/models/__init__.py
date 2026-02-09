"""
Data models for the LHCb VELO Toy Model.

This module exports the core data structures used throughout the package:

Hierarchy:
    Event
    ├── PrimaryVertex (collision points)
    ├── Track (particle trajectories, with hit_ids)
    ├── Hit (detector measurements, with track_id)
    └── Module (detector layers)

Note: Segments are NOT part of generation models. They are computed
on-demand in the reconstruction phase using `get_segments_from_track()`.
"""

from lhcb_velo_toy.generation.models.hit import Hit
from lhcb_velo_toy.generation.models.track import Track
from lhcb_velo_toy.generation.models.module import Module
from lhcb_velo_toy.generation.models.event import Event
from lhcb_velo_toy.generation.models.primary_vertex import PrimaryVertex

__all__ = [
    "Hit",
    "Track",
    "Module",
    "Event",
    "PrimaryVertex",
]
