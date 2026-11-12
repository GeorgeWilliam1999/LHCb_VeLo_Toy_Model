"""
Track reconstruction utilities.

This module provides tools for reconstructing tracks from detector hits,
including:

- Segment: A track segment connecting two hits (computed on-demand)
- get_segments_from_event: Generate all true segments from an event
- get_segments_from_track: Generate segments for a single track
- get_candidate_segments: Generate all possible candidate segments
- get_all_possible_segments: Generate all hit-to-hit segment candidates
- get_tracks: Extract tracks from Hamiltonian solutions
- get_tracks_fast: Optimized track extraction
- get_tracks_coupled: Tracks along the Hamiltonian couplings, with hit uniqueness
- find_segments: Find connected segments for track building
- construct_event: Build reconstructed event from tracks and hits

Note: Segments are NOT stored in Events. They are computed on-demand
using the segment generation functions in this module.
"""

from lhcb_velo_toy.solvers.reconstruction.segment import (
    Segment,
    get_segments_from_track,
    get_segments_from_event,
    get_candidate_segments,
)
from lhcb_velo_toy.solvers.reconstruction.track_finder import (
    get_tracks,
    get_tracks_fast,
    get_tracks_layered,
    get_tracks_optimal,
    get_tracks_optimal_iterative,
    find_segments,
    construct_event,
    get_all_possible_segments,
)
from lhcb_velo_toy.solvers.reconstruction.coupled_readout import (
    coupled_hit_groups,
    get_tracks_coupled,
)

__all__ = [
    # Segment class and generation
    "Segment",
    "get_segments_from_event",
    "get_segments_from_track",
    "get_candidate_segments",
    "get_all_possible_segments",
    # Track finding
    "get_tracks",
    "get_tracks_fast",
    "get_tracks_layered",
    "get_tracks_optimal",
    "get_tracks_optimal_iterative",
    "get_tracks_coupled",
    "coupled_hit_groups",
    "find_segments",
    "construct_event",
]
