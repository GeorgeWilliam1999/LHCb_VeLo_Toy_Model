"""
PlaneGeometry: Simple planar detector with rectangular active areas.

This geometry represents a detector with multiple parallel planar modules,
each with a rectangular active area.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator

from lhcb_velo_toy.core.types import ModuleID, StateVector
from lhcb_velo_toy.generation.geometry.base import Geometry


@dataclass(frozen=True)
class PlaneGeometry(Geometry):
    """
    Simple planar detector with rectangular active areas.
    
    A PlaneGeometry consists of multiple parallel detector planes positioned
    along the z-axis. Each plane has a rectangular active area centered at
    (0, 0) in the x-y plane.
    
    Attributes
    ----------
    module_id : list[int]
        List of module identifiers.
    lx : list[float]
        Half-widths in x for each module (mm).
    ly : list[float]
        Half-widths in y for each module (mm).
    z : list[float]
        Z positions of each module (mm).
    
    Examples
    --------
    >>> geometry = PlaneGeometry(
    ...     module_id=[1, 2, 3, 4, 5],
    ...     lx=[50.0, 50.0, 50.0, 50.0, 50.0],
    ...     ly=[50.0, 50.0, 50.0, 50.0, 50.0],
    ...     z=[100.0, 130.0, 160.0, 190.0, 220.0]
    ... )
    >>> geometry.point_on_bulk({'x': 10.0, 'y': 5.0, 'z': 100.0})
    True
    >>> geometry.point_on_bulk({'x': 60.0, 'y': 5.0, 'z': 100.0})
    False
    
    Notes
    -----
    The active region for module i spans:
    - x: [-lx[i], +lx[i]]
    - y: [-ly[i], +ly[i]]
    - z: exactly z[i]
    """
    
    module_id: list[ModuleID]
    lx: list[float]
    ly: list[float]
    z: list[float]
    
    def __getitem__(self, index: int) -> tuple[int, float, float, float]:
        """
        Get geometry data for a specific module.
        
        Parameters
        ----------
        index : int
            Module index (0-based).
        
        Returns
        -------
        tuple[int, float, float, float]
            Tuple of (module_id, lx, ly, z) for the module.
        
        Raises
        ------
        IndexError
            If index is out of range.
        """
        raise NotImplementedError
    
    def point_on_bulk(self, state: StateVector) -> bool:
        """
        Check if a point is within any module's active region.
        
        A point is considered to be on the bulk if its (x, y) coordinates
        fall within the rectangular active area of any module.
        
        Parameters
        ----------
        state : dict[str, float]
            State dictionary with 'x', 'y', 'z' keys.
        
        Returns
        -------
        bool
            True if the point is within an active region.
        
        Notes
        -----
        The check uses the first module's dimensions. For modules with
        varying sizes, override this method.
        """
        raise NotImplementedError
    
    def __len__(self) -> int:
        """Return the number of modules."""
        raise NotImplementedError
    
    def __iter__(self) -> Iterator[tuple[int, float, float, float]]:
        """Iterate over (module_id, lx, ly, z) for each module."""
        raise NotImplementedError
    
    def get_z_positions(self) -> list[float]:
        """Return the z positions of all modules."""
        raise NotImplementedError
