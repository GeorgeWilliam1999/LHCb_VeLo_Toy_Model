"""
Hit dataclass representing a single detector measurement.

A Hit represents a point where a charged particle traversed a detector module,
leaving an electronic signal that was recorded.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, ClassVar

from lhcb_velo_toy.core.types import HitID, ModuleID, TrackID


@dataclass
class Hit:
    """
    A single detector hit (measurement point).
    
    A hit represents the recorded position where a charged particle crossed
    a detector module. Each hit has a unique identifier and is associated
    with a specific module and (optionally) a truth track.
    
    Attributes
    ----------
    hit_id : int
        Unique identifier for this hit.
    x : float
        X coordinate in mm.
    y : float
        Y coordinate in mm.
    z : float
        Z coordinate in mm (along beam axis).
    module_id : int
        Identifier of the detector module containing this hit.
    track_id : int
        Identifier of the true particle track that created this hit.
        Set to -1 for ghost hits (noise).
    extra : dict[str, Any]
        Arbitrary additional data (e.g. cluster size, charge deposition
        from an external source).  Auto-captured from unknown keys
        in ``from_dict()``.
    
    Examples
    --------
    >>> hit = Hit(hit_id=0, x=1.5, y=-0.3, z=100.0, module_id=1, track_id=0)
    >>> hit.x
    1.5
    >>> hit[2]  # Access z coordinate by index
    100.0
    
    Notes
    -----
    Hits store a back-reference to their track via track_id for easy
    lookup. The Hit and Track cross-reference each other via IDs to
    enable JSON serialization of the entire Event.
    """
    
    _KNOWN_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"hit_id", "x", "y", "z", "module_id", "track_id", "extra"}
    )
    
    hit_id: HitID
    x: float
    y: float
    z: float
    module_id: ModuleID
    track_id: TrackID = -1  # -1 indicates ghost/noise hit
    extra: dict[str, Any] = field(default_factory=dict)
    
    def __getitem__(self, index: int) -> float:
        """
        Access coordinates by index for vector-like operations.
        
        Parameters
        ----------
        index : int
            Coordinate index: 0 for x, 1 for y, 2 for z.
        
        Returns
        -------
        float
            The coordinate value.
        
        Raises
        ------
        IndexError
            If index is not 0, 1, or 2.
        """
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise IndexError(f"Hit index must be 0, 1, or 2, got {index}")
    
    @property
    def position(self) -> tuple[float, float, float]:
        """
        Get the hit position as a tuple.
        
        Returns
        -------
        tuple[float, float, float]
            Position as (x, y, z) in mm.
        """
        return (self.x, self.y, self.z)
    
    @property
    def is_ghost(self) -> bool:
        """
        Check if this is a ghost (noise) hit.
        
        Returns
        -------
        bool
            True if track_id == -1.
        """
        return self.track_id == -1
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Returns
        -------
        dict
            Dictionary representation of the hit.
        """
        d: dict[str, Any] = {
            "hit_id": self.hit_id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "module_id": self.module_id,
            "track_id": self.track_id,
        }
        if self.extra:
            d["extra"] = self.extra.copy()
        return d
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Hit":
        """
        Create a Hit from a dictionary.
        
        Any keys not in the standard set are automatically captured
        into ``extra``.
        
        Parameters
        ----------
        data : dict
            Dictionary with hit_id, x, y, z, module_id, track_id keys.
            Additional keys are stored in extra.
        
        Returns
        -------
        Hit
            The reconstructed hit.
        """
        extra = dict(data.get("extra", {}))
        extra.update({k: v for k, v in data.items() if k not in cls._KNOWN_KEYS})
        return cls(
            hit_id=data["hit_id"],
            x=data["x"],
            y=data["y"],
            z=data["z"],
            module_id=data["module_id"],
            track_id=data.get("track_id", -1),
            extra=extra,
        )
