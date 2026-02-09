"""
Segment dataclass and utilities for on-demand segment generation.

Segments connect two hits on adjacent detector modules and represent
candidate track elements. They are computed on-demand from Tracks/Hits
rather than being stored as part of the Event.

This module is part of the reconstruction phase, NOT the generation phase.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lhcb_velo_toy.core.types import SegmentID

if TYPE_CHECKING:
    from lhcb_velo_toy.generation.models.hit import Hit
    from lhcb_velo_toy.generation.models.track import Track
    from lhcb_velo_toy.generation.models.event import Event


@dataclass
class Segment:
    """
    A track segment connecting two hits on adjacent detector layers.
    
    Segments are the fundamental building blocks for track reconstruction.
    Each segment connects exactly two hits, typically from adjacent detector
    modules, and carries geometric information used in the Hamiltonian
    formulation.
    
    Note
    ----
    Segments are NOT stored in the Event. They are computed on-demand
    using `get_segments_from_event()` or `get_segments_from_track()`.
    
    Attributes
    ----------
    hit_start : Hit
        Starting hit (lower z position).
    hit_end : Hit
        Ending hit (higher z position).
    segment_id : int
        Unique identifier for this segment.
    track_id : int
        ID of the track this segment belongs to (-1 if unknown/candidate).
    pv_id : int
        ID of the primary vertex this segment's track originates from (-1 if unknown).
    
    Examples
    --------
    >>> seg = Segment(hit_start=hit1, hit_end=hit2, segment_id=0, track_id=1, pv_id=0)
    >>> direction = seg.to_vect()  # Get direction vector
    >>> cos_angle = seg1 * seg2    # Compute angle between segments
    >>> seg.track_id  # Which track does this belong to?
    1
    
    Notes
    -----
    The multiplication operator computes the cosine of the angle between
    two segments, which is central to the Hamiltonian track finding.
    """
    
    hit_start: "Hit"
    hit_end: "Hit"
    segment_id: SegmentID
    track_id: int = -1
    pv_id: int = -1
    
    @property
    def hits(self) -> tuple["Hit", "Hit"]:
        """Get both hits as a tuple (start, end)."""
        return (self.hit_start, self.hit_end)
    
    def to_vect(self) -> tuple[float, float, float]:
        """
        Compute the direction vector of this segment.
        
        The direction vector points from the first hit to the second hit.
        
        Returns
        -------
        tuple[float, float, float]
            Direction vector (dx, dy, dz) where:
            - dx = hit_end.x - hit_start.x
            - dy = hit_end.y - hit_start.y
            - dz = hit_end.z - hit_start.z
        
        Examples
        --------
        >>> seg = Segment(hit_at_0_0_0, hit_at_1_1_10, segment_id=0)
        >>> seg.to_vect()
        (1.0, 1.0, 10.0)
        """
        dx = self.hit_end.x - self.hit_start.x
        dy = self.hit_end.y - self.hit_start.y
        dz = self.hit_end.z - self.hit_start.z
        return (dx, dy, dz)
    
    def length(self) -> float:
        """
        Compute the length of this segment.
        
        Returns
        -------
        float
            Euclidean length of the segment.
        """
        dx, dy, dz = self.to_vect()
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def __mul__(self, other: "Segment") -> float:
        """
        Compute the cosine of the angle between this segment and another.
        
        This operation is central to the Hamiltonian track finding algorithm,
        where segments are considered compatible if they are nearly collinear
        (cosine close to 1).
        
        Parameters
        ----------
        other : Segment
            Another segment to compare angles with.
        
        Returns
        -------
        float
            Cosine of the angle between the two segments, in range [-1, 1].
            A value of 1 indicates parallel segments (same direction).
            A value of -1 indicates anti-parallel segments.
            A value of 0 indicates perpendicular segments.
        
        Notes
        -----
        The cosine is computed as:
        
        .. math::
            \\cos(\\theta) = \\frac{\\vec{v}_1 \\cdot \\vec{v}_2}{|\\vec{v}_1| |\\vec{v}_2|}
        
        Examples
        --------
        >>> cos_angle = seg1 * seg2
        >>> if abs(cos_angle - 1.0) < epsilon:
        ...     print("Segments are nearly collinear")
        """
        v1 = self.to_vect()
        v2 = other.to_vect()
        
        # Dot product
        dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        
        # Magnitudes
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot / (mag1 * mag2)
    
    def __eq__(self, other: object) -> bool:
        """
        Check equality by segment_id.
        
        Parameters
        ----------
        other : object
            Object to compare with.
        
        Returns
        -------
        bool
            True if other is a Segment with the same segment_id.
        """
        if not isinstance(other, Segment):
            return False
        return self.segment_id == other.segment_id
    
    def __hash__(self) -> int:
        """
        Hash by segment_id.
        
        Returns
        -------
        int
            Hash value based on segment_id.
        """
        return hash(self.segment_id)
    
    def shares_hit_with(self, other: "Segment") -> bool:
        """
        Check if this segment shares an endpoint hit with another segment.
        
        Two segments that share an endpoint can potentially belong to the
        same track, making this check important for track finding.
        
        Parameters
        ----------
        other : Segment
            Another segment to check for shared hits.
        
        Returns
        -------
        bool
            True if the segments share at least one hit.
        
        Examples
        --------
        >>> # seg1: hit_A -> hit_B
        >>> # seg2: hit_B -> hit_C
        >>> seg1.shares_hit_with(seg2)
        True
        """
        my_hit_ids = {self.hit_start.hit_id, self.hit_end.hit_id}
        other_hit_ids = {other.hit_start.hit_id, other.hit_end.hit_id}
        return len(my_hit_ids & other_hit_ids) > 0
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Returns
        -------
        dict
            Dictionary with segment_id, hit_start_id, hit_end_id, track_id, pv_id.
        """
        return {
            "segment_id": self.segment_id,
            "hit_start_id": self.hit_start.hit_id,
            "hit_end_id": self.hit_end.hit_id,
            "track_id": self.track_id,
            "pv_id": self.pv_id,
        }


# =============================================================================
# Segment Generation Functions
# =============================================================================

def get_segments_from_track(
    track: "Track",
    event: "Event",
    start_segment_id: int = 0,
) -> list[Segment]:
    """
    Generate segments from a single track's hits.
    
    Creates segments between consecutive hits (ordered by z position)
    for a given track. Each segment inherits the track_id and pv_id
    from the parent track.
    
    Parameters
    ----------
    track : Track
        The track to generate segments from.
    event : Event
        The event containing the hit data (for ID lookups).
    start_segment_id : int, default 0
        Starting segment ID counter.
    
    Returns
    -------
    list[Segment]
        List of segments connecting consecutive hits, each with
        track_id and pv_id set from the parent track.
    
    Examples
    --------
    >>> segments = get_segments_from_track(track, event)
    >>> len(segments)  # n_hits - 1 for a single track
    4
    >>> segments[0].track_id == track.track_id
    True
    >>> segments[0].pv_id == track.pv_id
    True
    """
    hits = event.get_hits_by_ids(track.hit_ids)
    
    # Sort by z coordinate
    hits_sorted = sorted(hits, key=lambda h: h.z)
    
    segments = []
    for i, (h1, h2) in enumerate(zip(hits_sorted[:-1], hits_sorted[1:])):
        seg = Segment(
            hit_start=h1,
            hit_end=h2,
            segment_id=start_segment_id + i,
            track_id=track.track_id,
            pv_id=track.pv_id,
        )
        segments.append(seg)
    
    return segments


def get_segments_from_event(
    event: "Event",
    include_ghosts: bool = False,
) -> list[Segment]:
    """
    Generate all segments from all tracks in an event.
    
    Creates segments by connecting consecutive hits on each track.
    Ghost hits (track_id == -1) are excluded by default.
    
    Parameters
    ----------
    event : Event
        The event to process.
    include_ghosts : bool, default False
        If True, also create segments for ghost hits using nearest
        neighbor heuristics. Usually False for ground-truth segments.
    
    Returns
    -------
    list[Segment]
        All segments in the event.
    
    Examples
    --------
    >>> segments = get_segments_from_event(event)
    >>> len(segments)
    125  # Depends on number of tracks and hits per track
    """
    all_segments = []
    segment_id_counter = 0
    
    for track in event.tracks:
        track_segments = get_segments_from_track(
            track, event, start_segment_id=segment_id_counter
        )
        all_segments.extend(track_segments)
        segment_id_counter += len(track_segments)
    
    return all_segments


def get_candidate_segments(
    event: "Event",
    max_delta_z: float | None = None,
) -> list[Segment]:
    """
    Generate all possible candidate segments from hits.
    
    Creates segments between all pairs of hits on adjacent or near-adjacent
    modules. This is used for reconstruction when true track associations
    are not known.
    
    Parameters
    ----------
    event : Event
        The event containing hits.
    max_delta_z : float, optional
        Maximum z-distance between hits to form a segment.
        If None, only adjacent modules are connected.
    
    Returns
    -------
    list[Segment]
        All candidate segments.
    
    Notes
    -----
    The number of candidate segments can be large (O(n_hits^2) worst case),
    so filtering by max_delta_z is recommended for large events.
    """
    # Group hits by module z position
    z_positions = sorted(set(h.z for h in event.hits))
    hits_by_z: dict[float, list["Hit"]] = {z: [] for z in z_positions}
    for hit in event.hits:
        hits_by_z[hit.z].append(hit)
    
    segments = []
    segment_id = 0
    
    # Connect hits on adjacent z layers
    for i in range(len(z_positions) - 1):
        z1, z2 = z_positions[i], z_positions[i + 1]
        
        if max_delta_z is not None and (z2 - z1) > max_delta_z:
            continue
        
        for h1 in hits_by_z[z1]:
            for h2 in hits_by_z[z2]:
                # For candidate segments, track_id and pv_id are unknown (-1)
                seg = Segment(
                    hit_start=h1,
                    hit_end=h2,
                    segment_id=segment_id,
                    track_id=-1,
                    pv_id=-1,
                )
                segments.append(seg)
                segment_id += 1
    
    return segments
