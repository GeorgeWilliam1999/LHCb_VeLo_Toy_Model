"""
Track finding utilities for extracting tracks from Hamiltonian solutions.

Functions for converting segment activation vectors into reconstructed tracks
and computing segments from events on-demand.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from lhcb_velo_toy.generation.models.hit import Hit
    from lhcb_velo_toy.solvers.reconstruction.segment import Segment
    from lhcb_velo_toy.generation.models.track import Track
    from lhcb_velo_toy.generation.models.module import Module
    from lhcb_velo_toy.generation.models.event import Event
    from lhcb_velo_toy.generation.geometry.base import Geometry
    from lhcb_velo_toy.generation.generators.state_event import StateEventGenerator
    from lhcb_velo_toy.solvers.hamiltonians.base import Hamiltonian


def find_segments(
    segment: "Segment",
    active_segments: list["Segment"],
) -> list["Segment"]:
    """
    Find all segments connected to a given segment.
    
    Two segments are connected if they share an endpoint hit.
    
    Parameters
    ----------
    segment : Segment
        The reference segment to find connections for.
    active_segments : list[Segment]
        Pool of candidate segments to search.
    
    Returns
    -------
    list[Segment]
        Segments that share a hit with the reference segment.
    
    Examples
    --------
    >>> # seg1: hit_A -> hit_B
    >>> # seg2: hit_B -> hit_C
    >>> # seg3: hit_D -> hit_E (no connection)
    >>> connected = find_segments(seg1, [seg2, seg3])
    >>> len(connected)
    1
    
    Notes
    -----
    This function is used in the track-building algorithm to group
    connected segments into track candidates.
    """
    return [s for s in active_segments if segment.shares_hit_with(s) and s != segment]


def get_tracks(
    hamiltonian: "Hamiltonian",
    solution: np.ndarray,
    event: Union["Event", "StateEventGenerator"],
    threshold: float = 0.0,
) -> list["Track"]:
    """
    Extract tracks from a Hamiltonian solution.
    
    Converts the continuous segment activation vector into discrete
    tracks by thresholding and grouping connected segments.
    
    Parameters
    ----------
    hamiltonian : Hamiltonian
        The Hamiltonian containing segment information.
    solution : numpy.ndarray
        Segment activation vector from solving A x = b.
    event : Event or StateEventGenerator
        The event containing hit and geometry information.
    threshold : float, default 0.0
        Minimum activation value for a segment to be considered active.
        Segments with solution[i] > threshold are included.
    
    Returns
    -------
    list[Track]
        Reconstructed tracks.
    
    Examples
    --------
    >>> ham = SimpleHamiltonian(epsilon=0.01, gamma=1.5, delta=1.0)
    >>> ham.construct_hamiltonian(event)
    >>> solution = ham.solve_classicaly()
    >>> tracks = get_tracks(ham, solution, event)
    
    Notes
    -----
    Algorithm:
    1. Filter segments where activation > threshold
    2. Build adjacency graph of connected segments
    3. Find connected components via depth-first search
    4. Convert each component to a Track object
    5. Order hits within each track by z coordinate
    """
    from lhcb_velo_toy.generation.models.track import Track
    
    # Get active segments
    active_indices = np.where(solution > threshold)[0]
    active_segments = [hamiltonian.segments[i] for i in active_indices]
    
    if not active_segments:
        return []
    
    # Group segments into tracks
    segment_groups = _group_segments_into_tracks(active_segments)
    
    # Convert groups to Track objects
    tracks = []
    for track_id, group in enumerate(segment_groups):
        track = _segments_to_track(group, track_id)
        tracks.append(track)
    
    return tracks


def get_tracks_fast(
    hamiltonian: "Hamiltonian",
    solution: np.ndarray,
    event: Union["Event", "StateEventGenerator"],
    threshold: float = 0.0,
) -> list["Track"]:
    """
    Optimized track extraction using vectorized operations.
    
    Parameters
    ----------
    hamiltonian : Hamiltonian
        The Hamiltonian containing segment information.
    solution : numpy.ndarray
        Segment activation vector.
    event : Event or StateEventGenerator
        The event containing hit and geometry information.
    threshold : float, default 0.0
        Minimum activation threshold.
    
    Returns
    -------
    list[Track]
        Reconstructed tracks.
    
    Notes
    -----
    Uses pre-computed data structures from SimpleHamiltonianFast
    for improved performance.
    """
    # For now, use the same implementation as get_tracks
    # Future optimization can use pre-computed adjacency matrices
    return get_tracks(hamiltonian, solution, event, threshold)


def construct_event(
    detector_geometry: "Geometry",
    tracks: list["Track"],
    hits: list["Hit"],
    modules: list["Module"],
    primary_vertices: Optional[list] = None,
) -> "Event":
    """
    Construct an Event object from components.
    
    Factory function for creating Event objects with all required
    data structures properly initialized.
    
    Parameters
    ----------
    detector_geometry : Geometry
        The detector geometry configuration.
    tracks : list[Track]
        List of particle tracks.
    hits : list[Hit]
        List of all hits.
    modules : list[Module]
        List of detector modules.
    primary_vertices : list[PrimaryVertex], optional
        List of primary vertices. If None, creates an empty list.
    
    Returns
    -------
    Event
        Constructed event object.
    
    Examples
    --------
    >>> event = construct_event(geometry, tracks, hits, modules)
    
    Notes
    -----
    Segments are NOT passed to this function. They are computed on-demand
    using `get_segments_from_event()` when needed.
    """
    from lhcb_velo_toy.generation.models.event import Event
    
    return Event(
        detector_geometry=detector_geometry,
        primary_vertices=primary_vertices or [],
        tracks=tracks,
        hits=hits,
        modules=modules,
    )


def _group_segments_into_tracks(
    active_segments: list["Segment"],
) -> list[list["Segment"]]:
    """
    Group connected segments into track candidates.
    
    Parameters
    ----------
    active_segments : list[Segment]
        Segments that passed the activation threshold.
    
    Returns
    -------
    list[list[Segment]]
        Groups of connected segments, each forming a track candidate.
    """
    if not active_segments:
        return []
    
    # Track visited segments
    visited: set[int] = set()
    groups: list[list["Segment"]] = []
    
    def dfs(segment: "Segment", group: list["Segment"]) -> None:
        """Depth-first search to find all connected segments."""
        if segment.segment_id in visited:
            return
        visited.add(segment.segment_id)
        group.append(segment)
        
        # Find connected segments
        for other in active_segments:
            if other.segment_id not in visited and segment.shares_hit_with(other):
                dfs(other, group)
    
    # Find all connected components
    for segment in active_segments:
        if segment.segment_id not in visited:
            group: list["Segment"] = []
            dfs(segment, group)
            if group:
                groups.append(group)
    
    return groups


def _segments_to_track(
    segment_group: list["Segment"],
    track_id: int,
) -> "Track":
    """
    Convert a group of segments into a Track object.
    
    Parameters
    ----------
    segment_group : list[Segment]
        Connected segments forming a track.
    track_id : int
        Unique identifier for the track.
    
    Returns
    -------
    Track
        The constructed track with ordered hits.
    """
    from lhcb_velo_toy.generation.models.track import Track
    
    # Collect all unique hits from segments
    hit_set: set[int] = set()
    hits_list: list = []
    
    for segment in segment_group:
        for hit in [segment.hit_start, segment.hit_end]:
            if hit.hit_id not in hit_set:
                hit_set.add(hit.hit_id)
                hits_list.append(hit)
    
    # Sort hits by z coordinate
    hits_list.sort(key=lambda h: h.z)
    
    # Extract hit IDs
    hit_ids = [h.hit_id for h in hits_list]
    
    return Track(track_id=track_id, hit_ids=hit_ids)


def get_segments_from_track(
    track: "Track",
    event: "Event",
) -> list["Segment"]:
    """
    Compute segments for a single track.
    
    Creates segments between consecutive hits (ordered by z) on the track.
    
    Parameters
    ----------
    track : Track
        The track to compute segments for.
    event : Event
        The event containing hit data.
    
    Returns
    -------
    list[Segment]
        List of segments connecting consecutive hits.
    
    Examples
    --------
    >>> segments = get_segments_from_track(track, event)
    >>> print(f"Track has {len(segments)} segments")
    """
    from lhcb_velo_toy.solvers.reconstruction.segment import Segment
    
    # Get hits and sort by z
    hits = event.get_hits_by_ids(track.hit_ids)
    hits_sorted = sorted(hits, key=lambda h: h.z)
    
    segments = []
    for i in range(len(hits_sorted) - 1):
        segment = Segment(
            hit_start=hits_sorted[i],
            hit_end=hits_sorted[i + 1],
            segment_id=len(segments),  # Local ID within this track
            track_id=track.track_id,
            pv_id=track.pv_id,
        )
        segments.append(segment)
    
    return segments


def get_segments_from_event(
    event: "Event",
    include_ghost_tracks: bool = False,
) -> list["Segment"]:
    """
    Compute all segments from an event's tracks.
    
    This function generates segments on-demand from the event's tracks.
    Segments are NOT stored in the Event; use this function when needed.
    
    Parameters
    ----------
    event : Event
        The event containing tracks and hits.
    include_ghost_tracks : bool, default False
        If True, includes segments from ghost hits (track_id == -1).
    
    Returns
    -------
    list[Segment]
        List of all segments with globally unique segment_ids.
    
    Examples
    --------
    >>> segments = get_segments_from_event(event)
    >>> print(f"Event has {len(segments)} segments")
    
    Notes
    -----
    Segment IDs are assigned globally across the entire event to ensure
    uniqueness for Hamiltonian construction.
    """
    from lhcb_velo_toy.solvers.reconstruction.segment import Segment
    
    all_segments: list["Segment"] = []
    segment_id_counter = 0
    
    for track in event.tracks:
        if not include_ghost_tracks and track.track_id == -1:
            continue
        
        # Get hits and sort by z
        hits = event.get_hits_by_ids(track.hit_ids)
        hits_sorted = sorted(hits, key=lambda h: h.z)
        
        # Create segments between consecutive hits
        for i in range(len(hits_sorted) - 1):
            segment = Segment(
                hit_start=hits_sorted[i],
                hit_end=hits_sorted[i + 1],
                segment_id=segment_id_counter,
                track_id=track.track_id,
                pv_id=track.pv_id,
            )
            all_segments.append(segment)
            segment_id_counter += 1
    
    return all_segments


def get_all_possible_segments(
    event: "Event",
    max_z_gap: int = 1,
) -> list["Segment"]:
    """
    Generate all possible segment candidates between hits on adjacent modules.
    
    This is used for Hamiltonian construction where we need to consider
    ALL possible hit-to-hit connections, not just those from known tracks.
    
    Parameters
    ----------
    event : Event
        The event containing hits and modules.
    max_z_gap : int, default 1
        Maximum module gap between hits. 1 = adjacent modules only.
    
    Returns
    -------
    list[Segment]
        List of all possible segment candidates.
    
    Examples
    --------
    >>> candidates = get_all_possible_segments(event)
    >>> print(f"Generated {len(candidates)} segment candidates")
    """
    from lhcb_velo_toy.solvers.reconstruction.segment import Segment
    
    # Group hits by module
    hits_by_module: dict[int, list] = {}
    for hit in event.hits:
        if hit.module_id not in hits_by_module:
            hits_by_module[hit.module_id] = []
        hits_by_module[hit.module_id].append(hit)
    
    # Get sorted module IDs
    module_ids = sorted(hits_by_module.keys())
    
    segments: list["Segment"] = []
    segment_id = 0
    
    # Generate segments between adjacent modules
    for i, mod_id in enumerate(module_ids):
        for j in range(1, max_z_gap + 1):
            if i + j >= len(module_ids):
                break
            next_mod_id = module_ids[i + j]
            
            # Create all hit pairs between these modules
            for hit1 in hits_by_module[mod_id]:
                for hit2 in hits_by_module[next_mod_id]:
                    segment = Segment(
                        hit_start=hit1,
                        hit_end=hit2,
                        segment_id=segment_id,
                        track_id=-1,  # Unknown
                        pv_id=-1,     # Unknown
                    )
                    segments.append(segment)
                    segment_id += 1
    
    return segments
