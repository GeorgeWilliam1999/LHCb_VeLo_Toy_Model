"""
Segment dataclass representing a track segment between two hits.

A Segment connects two hits on adjacent detector modules and represents
a candidate track element in the reconstruction.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lhcb_velo_toy.core.types import SegmentID

if TYPE_CHECKING:
    from lhcb_velo_toy.generation.models.hit import Hit


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
    hits : list[Hit]
        Exactly two hits: [start_hit, end_hit], ordered by increasing z.
    segment_id : int
        Unique identifier for this segment.
    
    Examples
    --------
    >>> seg = Segment(hits=[hit1, hit2], segment_id=0)
    >>> direction = seg.to_vect()  # Get direction vector
    >>> cos_angle = seg1 * seg2    # Compute angle between segments
    
    Notes
    -----
    The multiplication operator computes the cosine of the angle between
    two segments, which is central to the Hamiltonian track finding.
    """
    
    hits: list["Hit"]
    segment_id: SegmentID
    
    def to_vect(self) -> tuple[float, float, float]:
        """
        Compute the direction vector of this segment.
        
        The direction vector points from the first hit to the second hit.
        
        Returns
        -------
        tuple[float, float, float]
            Direction vector (dx, dy, dz) where:
            - dx = hits[1].x - hits[0].x
            - dy = hits[1].y - hits[0].y
            - dz = hits[1].z - hits[0].z
        
        Examples
        --------
        >>> seg = Segment([hit_at_0_0_0, hit_at_1_1_10], segment_id=0)
        >>> seg.to_vect()
        (1.0, 1.0, 10.0)
        """
        raise NotImplementedError
    
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
        raise NotImplementedError
    
    def __eq__(self, other: object) -> bool:
        """
        Check equality by identity (same object in memory).
        
        Parameters
        ----------
        other : object
            Object to compare with.
        
        Returns
        -------
        bool
            True if other is the same Segment instance.
        """
        raise NotImplementedError
    
    def __hash__(self) -> int:
        """
        Hash by object identity.
        
        Returns
        -------
        int
            Hash value based on object id.
        """
        raise NotImplementedError
    
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
        raise NotImplementedError
