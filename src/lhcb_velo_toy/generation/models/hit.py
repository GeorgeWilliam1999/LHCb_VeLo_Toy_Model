"""
Hit dataclass representing a single detector measurement.

A Hit represents a point where a charged particle traversed a detector module,
leaving an electronic signal that was recorded.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Union

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
    
    Examples
    --------
    >>> hit = Hit(hit_id=0, x=1.5, y=-0.3, z=100.0, module_id=1, track_id=0)
    >>> hit.x
    1.5
    >>> hit[2]  # Access z coordinate by index
    100.0
    
    Notes
    -----
    Hits are immutable in practice, though not enforced by frozen=True
    to allow flexibility in event generation.
    """
    
    hit_id: HitID
    x: float
    y: float
    z: float
    module_id: ModuleID
    track_id: TrackID
    
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
        
        Examples
        --------
        >>> hit = Hit(0, 1.0, 2.0, 3.0, 0, 0)
        >>> hit[0], hit[1], hit[2]
        (1.0, 2.0, 3.0)
        """
        raise NotImplementedError
    
    def __eq__(self, other: object) -> bool:
        """
        Check equality by identity (same object in memory).
        
        This uses identity comparison rather than value comparison to
        properly handle duplicate coordinates in different hits.
        
        Parameters
        ----------
        other : object
            Object to compare with.
        
        Returns
        -------
        bool
            True if other is the same Hit instance.
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
    
    def to_tuple(self) -> tuple[float, float, float]:
        """
        Convert hit position to a tuple.
        
        Returns
        -------
        tuple[float, float, float]
            Position as (x, y, z).
        """
        raise NotImplementedError
