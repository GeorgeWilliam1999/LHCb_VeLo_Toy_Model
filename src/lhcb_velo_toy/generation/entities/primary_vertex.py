"""
PrimaryVertex dataclass representing a collision primary vertex.

A PrimaryVertex (PV) represents the point where protons collided and
particles were produced in the detector.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, ClassVar

from lhcb_velo_toy.core.types import TrackID, PVID


@dataclass
class PrimaryVertex:
    """
    A primary vertex (collision point) in the detector.
    
    A primary vertex represents the location where protons collided
    and produced the particles that form tracks. Each PV can have
    multiple tracks originating from it.
    
    Attributes
    ----------
    pv_id : int
        Unique identifier for this primary vertex.
    x : float
        X coordinate of the vertex position in mm.
    y : float
        Y coordinate of the vertex position in mm.
    z : float
        Z coordinate of the vertex position in mm.
    track_ids : list[int]
        IDs of tracks originating from this vertex.
    extra : dict[str, Any]
        Arbitrary additional data.  Auto-captured from unknown keys
        in ``from_dict()``.
    
    Examples
    --------
    >>> pv = PrimaryVertex(pv_id=0, x=0.01, y=-0.02, z=50.0)
    >>> pv.position
    (0.01, -0.02, 50.0)
    >>> pv.n_tracks
    0
    
    Notes
    -----
    The primary vertex position is typically close to (0, 0, z) where
    z is near the nominal interaction point. Multiple PVs can exist
    in a single event (pile-up).
    """
    
    _KNOWN_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"pv_id", "x", "y", "z", "track_ids", "extra"}
    )
    
    pv_id: PVID
    x: float
    y: float
    z: float
    track_ids: list[TrackID] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    
    @property
    def position(self) -> tuple[float, float, float]:
        """
        Get the vertex position as a tuple.
        
        Returns
        -------
        tuple[float, float, float]
            Position as (x, y, z) in mm.
        """
        return (self.x, self.y, self.z)
    
    @property
    def n_tracks(self) -> int:
        """
        Get the number of tracks from this vertex.
        
        Returns
        -------
        int
            Number of track IDs associated with this PV.
        """
        return len(self.track_ids)
    
    def add_track(self, track_id: TrackID) -> None:
        """
        Associate a track with this vertex.
        
        Parameters
        ----------
        track_id : int
            The track identifier to add.
        """
        if track_id not in self.track_ids:
            self.track_ids.append(track_id)
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Returns
        -------
        dict
            Dictionary representation of the primary vertex.
        """
        d: dict[str, Any] = {
            "pv_id": self.pv_id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "track_ids": self.track_ids.copy(),
        }
        if self.extra:
            d["extra"] = self.extra.copy()
        return d
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrimaryVertex":
        """
        Create a PrimaryVertex from a dictionary.
        
        Any keys not in the standard set are automatically captured
        into ``extra``.
        
        Parameters
        ----------
        data : dict
            Dictionary with pv_id, x, y, z, and track_ids keys.
            Additional keys are stored in extra.
        
        Returns
        -------
        PrimaryVertex
            The reconstructed primary vertex.
        """
        extra = dict(data.get("extra", {}))
        extra.update({k: v for k, v in data.items() if k not in cls._KNOWN_KEYS})
        return cls(
            pv_id=data["pv_id"],
            x=data["x"],
            y=data["y"],
            z=data["z"],
            track_ids=data.get("track_ids", []).copy(),
            extra=extra,
        )
