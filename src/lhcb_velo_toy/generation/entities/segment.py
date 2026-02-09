"""
Segment dataclass representing a track segment between two hits.

A Segment connects two hits on adjacent detector modules and represents
a candidate track element in the reconstruction.

Note: Segments are NOT stored in Events. They are computed on-demand
using functions in `solvers/reconstruction/` when needed.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lhcb_velo_toy.core.types import SegmentID, TrackID

if TYPE_CHECKING:
    from lhcb_velo_toy.generation.entities.hit import Hit
    from lhcb_velo_toy.generation.entities.primary_vertex import PVID


@dataclass
class Segment:
    """
    A track segment connecting two hits on adjacent detector layers.
    
    Segments are the fundamental building blocks for track reconstruction.
    Each segment connects exactly two hits, typically from adjacent detector
    modules, and carries geometric information used in the Hamiltonian
    formulation.
    
    Attributes
    ----------
    hit_start : Hit
        Starting hit (lower z coordinate).
    hit_end : Hit
        Ending hit (higher z coordinate).
    segment_id : int
        Unique identifier for this segment.
    track_id : int
        ID of the track this segment belongs to (-1 if unknown).
    pv_id : int
        ID of the primary vertex this segment's track originates from (-1 if unknown).
    
    Examples
    --------
    >>> seg = Segment(hit_start=hit1, hit_end=hit2, segment_id=0)
    >>> direction = seg.to_vect()  # Get direction vector
    >>> cos_angle = seg1 * seg2    # Compute angle between segments
    
    Notes
    -----
    The multiplication operator computes the cosine of the angle between
    two segments, which is central to the Hamiltonian track finding.
    
    Segments store references to Hit objects (not IDs) because they are
    computed on-demand and not serialized with Events.
    """
    
    hit_start: "Hit"
    hit_end: "Hit"
    segment_id: SegmentID
    track_id: TrackID = -1
    pv_id: "PVID" = -1
    
    @property
    def hits(self) -> list["Hit"]:
        """
        Get hits as a list for backward compatibility.
        
        Returns
        -------
        list[Hit]
            [hit_start, hit_end]
        """
        return [self.hit_start, self.hit_end]
    
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
        >>> seg = Segment(hit_start=hit_at_0_0_0, hit_end=hit_at_1_1_10, segment_id=0)
        >>> seg.to_vect()
        (1.0, 1.0, 10.0)
        """
        dx = self.hit_end.x - self.hit_start.x
        dy = self.hit_end.y - self.hit_start.y
        dz = self.hit_end.z - self.hit_start.z
        return (dx, dy, dz)
    
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
        dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        
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
        # Check if any of our hits match any of their hits
        my_hit_ids = {self.hit_start.hit_id, self.hit_end.hit_id}
        other_hit_ids = {other.hit_start.hit_id, other.hit_end.hit_id}
        return bool(my_hit_ids & other_hit_ids)
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Note: This stores hit_ids, not Hit objects.
        
        Returns
        -------
        dict
            Dictionary representation of the segment.
        """
        return {
            "segment_id": self.segment_id,
            "hit_start_id": self.hit_start.hit_id,
            "hit_end_id": self.hit_end.hit_id,
            "track_id": self.track_id,
            "pv_id": self.pv_id,
        }
