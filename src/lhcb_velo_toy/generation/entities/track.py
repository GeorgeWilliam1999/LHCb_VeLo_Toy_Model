"""
Track dataclass representing a particle track through the detector.

A Track is a collection of hit references that belong to a single
charged particle traversing the detector.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, ClassVar

from lhcb_velo_toy.core.types import TrackID, HitID, PVID


@dataclass
class Track:
    """
    A particle track through the detector.
    
    A track represents the trajectory of a single charged particle as it
    passes through multiple detector layers. It stores references to hits
    via their IDs (not the hit objects themselves) to enable JSON serialization.
    
    Attributes
    ----------
    track_id : int
        Unique identifier for this track.
    pv_id : int
        ID of the primary vertex this track originates from.
    hit_ids : list[int]
        List of hit IDs belonging to this track, ordered by increasing z.
    extra : dict[str, Any]
        Arbitrary additional data (e.g. MC truth variables such as
        momentum, PID, eta from an external source).  Auto-captured
        from unknown keys in ``from_dict()``.
    
    Examples
    --------
    >>> track = Track(
    ...     track_id=0,
    ...     pv_id=0,
    ...     hit_ids=[0, 5, 12, 18, 25]
    ... )
    >>> track.n_hits
    5
    
    Notes
    -----
    Segments are NOT stored in the Track. They are computed on-demand
    from consecutive hits using `get_segments_from_track()` in the
    reconstruction module when needed for Hamiltonian construction.
    
    The hit_ids should be ordered by z coordinate (along the beam axis).
    To get actual Hit objects, use `event.get_hits_by_ids(track.hit_ids)`.
    """
    
    _KNOWN_KEYS: ClassVar[frozenset[str]] = frozenset({"track_id", "pv_id", "hit_ids", "extra"})
    
    track_id: TrackID
    pv_id: PVID = 0
    hit_ids: list[HitID] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_hits(self) -> int:
        """
        Get the number of hits in this track.
        
        Returns
        -------
        int
            Number of hits.
        """
        return len(self.hit_ids)
    
    def add_hit_id(self, hit_id: HitID) -> None:
        """
        Add a hit ID to this track.
        
        Parameters
        ----------
        hit_id : int
            The hit identifier to add.
        """
        self.hit_ids.append(hit_id)
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Returns
        -------
        dict
            Dictionary representation of the track.
        """
        d: dict[str, Any] = {
            "track_id": self.track_id,
            "pv_id": self.pv_id,
            "hit_ids": self.hit_ids.copy(),
        }
        if self.extra:
            d["extra"] = self.extra.copy()
        return d
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Track":
        """
        Create a Track from a dictionary.
        
        Any keys not in the standard set (track_id, pv_id, hit_ids)
        are automatically captured into ``extra``.
        
        Parameters
        ----------
        data : dict
            Dictionary with track_id, pv_id, and hit_ids keys.
            Additional keys are stored in extra.
        
        Returns
        -------
        Track
            The reconstructed track.
        """
        extra = dict(data.get("extra", {}))
        extra.update({k: v for k, v in data.items() if k not in cls._KNOWN_KEYS})
        return cls(
            track_id=data["track_id"],
            pv_id=data.get("pv_id", 0),
            hit_ids=data.get("hit_ids", []).copy(),
            extra=extra,
        )
