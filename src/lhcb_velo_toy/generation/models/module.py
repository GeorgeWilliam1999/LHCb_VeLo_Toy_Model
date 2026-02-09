"""
Module dataclass representing a detector layer/sensor.

A Module represents a single detector plane at a specific z position
where particles can leave hits.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lhcb_velo_toy.core.types import ModuleID

if TYPE_CHECKING:
    from lhcb_velo_toy.generation.models.hit import Hit


@dataclass
class Module:
    """
    A detector module (sensor plane) at a specific z position.
    
    A module represents a single layer of the detector where charged
    particles can interact and leave hits. Each module has a defined
    active area and position along the beam axis.
    
    Attributes
    ----------
    module_id : int
        Unique identifier for this module.
    z : float
        Z position of the module in mm (along beam axis).
    lx : float
        Half-width of the active area in x (mm).
    ly : float
        Half-width of the active area in y (mm).
    hit_ids : list[int]
        List of hit IDs recorded on this module.
    
    Examples
    --------
    >>> module = Module(module_id=1, z=100.0, lx=50.0, ly=50.0)
    >>> module.hit_ids.append(hit.hit_id)
    >>> len(module.hit_ids)
    1
    
    Notes
    -----
    The active area of a module spans from (-lx, -ly) to (+lx, +ly)
    in local coordinates, centered at (0, 0, z).
    
    The module stores hit_ids (not hit objects) to enable JSON serialization.
    """
    
    module_id: ModuleID
    z: float
    lx: float
    ly: float
    hit_ids: list[int] = field(default_factory=list)
    
    @property
    def n_hits(self) -> int:
        """
        Get the number of hits on this module.
        
        Returns
        -------
        int
            Number of hits.
        """
        return len(self.hit_ids)
    
    def add_hit_id(self, hit_id: int) -> None:
        """
        Add a hit ID to this module.
        
        Parameters
        ----------
        hit_id : int
            The hit ID to add.
        """
        self.hit_ids.append(hit_id)
    
    def clear_hits(self) -> None:
        """Remove all hit IDs from this module."""
        self.hit_ids.clear()
    
    def contains_point(self, x: float, y: float) -> bool:
        """
        Check if a point (x, y) is within the active area.
        
        Parameters
        ----------
        x : float
            X coordinate in mm.
        y : float
            Y coordinate in mm.
        
        Returns
        -------
        bool
            True if the point is within the module's active area.
        """
        return abs(x) <= self.lx and abs(y) <= self.ly
    
    # =========================================================================
    # JSON Serialization
    # =========================================================================
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert the module to a dictionary for JSON serialization.
        
        Returns
        -------
        dict
            Dictionary representation of the module.
        """
        return {
            "module_id": self.module_id,
            "z": self.z,
            "lx": self.lx,
            "ly": self.ly,
            "hit_ids": self.hit_ids.copy(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Module":
        """
        Create a Module from a dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary with module_id, z, lx, ly, hit_ids keys.
        
        Returns
        -------
        Module
            The reconstructed module.
        """
        return cls(
            module_id=data["module_id"],
            z=data["z"],
            lx=data["lx"],
            ly=data["ly"],
            hit_ids=data.get("hit_ids", []),
        )
