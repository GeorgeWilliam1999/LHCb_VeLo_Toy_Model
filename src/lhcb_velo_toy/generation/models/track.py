"""
Track dataclass representing a particle track through the detector.

A Track is a collection of hits and segments that belong to a single
charged particle traversing the detector.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lhcb_velo_toy.core.types import TrackID

if TYPE_CHECKING:
    from lhcb_velo_toy.generation.models.hit import Hit
    from lhcb_velo_toy.generation.models.segment import Segment


@dataclass
class Track:
    """
    A particle track through the detector.
    
    A track represents the trajectory of a single charged particle as it
    passes through multiple detector layers. It consists of hits (measurement
    points) and segments (connections between adjacent hits).
    
    Attributes
    ----------
    track_id : int
        Unique identifier for this track.
    hits : list[Hit]
        List of hits belonging to this track, ordered by increasing z.
    segments : list[Segment]
        List of segments connecting consecutive hits.
    
    Examples
    --------
    >>> track = Track(
    ...     track_id=0,
    ...     hits=[hit1, hit2, hit3],
    ...     segments=[seg1, seg2]
    ... )
    >>> len(track.hits)
    3
    >>> track.n_hits
    3
    
    Notes
    -----
    For a track with N hits, there should be N-1 segments connecting them.
    The hits are typically ordered by z coordinate (along the beam axis).
    """
    
    track_id: TrackID
    hits: list["Hit"] = field(default_factory=list)
    segments: list["Segment"] = field(default_factory=list)
    
    @property
    def n_hits(self) -> int:
        """
        Get the number of hits in this track.
        
        Returns
        -------
        int
            Number of hits.
        """
        raise NotImplementedError
    
    @property
    def n_segments(self) -> int:
        """
        Get the number of segments in this track.
        
        Returns
        -------
        int
            Number of segments.
        """
        raise NotImplementedError
    
    def get_hit_ids(self) -> list[int]:
        """
        Get the IDs of all hits in this track.
        
        Returns
        -------
        list[int]
            List of hit IDs.
        """
        raise NotImplementedError
    
    def get_module_ids(self) -> list[int]:
        """
        Get the module IDs traversed by this track.
        
        Returns
        -------
        list[int]
            List of module IDs in traversal order.
        """
        raise NotImplementedError
