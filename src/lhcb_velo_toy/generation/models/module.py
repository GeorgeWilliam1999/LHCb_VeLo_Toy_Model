"""
Module dataclass representing a detector layer/sensor.

A Module represents a single detector plane at a specific z position
where particles can leave hits.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
    hits : list[Hit]
        List of hits recorded on this module.
    
    Examples
    --------
    >>> module = Module(module_id=1, z=100.0, lx=50.0, ly=50.0)
    >>> module.add_hit(hit)
    >>> len(module.hits)
    1
    
    Notes
    -----
    The active area of a module spans from (-lx, -ly) to (+lx, +ly)
    in local coordinates, centered at (0, 0, z).
    """
    
    module_id: ModuleID
    z: float
    lx: float
    ly: float
    hits: list["Hit"] = field(default_factory=list)
    
    @property
    def n_hits(self) -> int:
        """
        Get the number of hits on this module.
        
        Returns
        -------
        int
            Number of hits.
        """
        raise NotImplementedError
    
    def add_hit(self, hit: "Hit") -> None:
        """
        Add a hit to this module.
        
        Parameters
        ----------
        hit : Hit
            The hit to add.
        """
        raise NotImplementedError
    
    def clear_hits(self) -> None:
        """Remove all hits from this module."""
        raise NotImplementedError
    
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
        raise NotImplementedError
