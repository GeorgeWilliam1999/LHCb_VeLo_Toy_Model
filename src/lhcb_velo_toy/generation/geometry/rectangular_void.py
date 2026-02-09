"""
RectangularVoidGeometry: Detector geometry with a central beam pipe void.

This geometry represents a detector where each module has a rectangular
active area with a rectangular void in the center (for the beam pipe).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator

from lhcb_velo_toy.core.types import ModuleID, StateVector
from lhcb_velo_toy.generation.geometry.base import Geometry


@dataclass(frozen=True)
class RectangularVoidGeometry(Geometry):
    """
    Detector geometry with a rectangular beam pipe void in the center.
    
    Each module has an outer rectangular boundary and an inner rectangular
    void region where no detection occurs (typically the beam pipe region).
    
    Attributes
    ----------
    module_id : list[int]
        List of module identifiers.
    z : list[float]
        Z positions of each module (mm).
    void_x_boundary : list[float]
        Half-width of the void region in x for each module (mm).
    void_y_boundary : list[float]
        Half-width of the void region in y for each module (mm).
    lx : list[float]
        Half-width of the outer boundary in x for each module (mm).
    ly : list[float]
        Half-width of the outer boundary in y for each module (mm).
    
    Examples
    --------
    >>> geometry = RectangularVoidGeometry(
    ...     module_id=[1, 2, 3],
    ...     z=[100.0, 150.0, 200.0],
    ...     void_x_boundary=[5.0, 5.0, 5.0],
    ...     void_y_boundary=[5.0, 5.0, 5.0],
    ...     lx=[50.0, 50.0, 50.0],
    ...     ly=[50.0, 50.0, 50.0]
    ... )
    >>> # Point in active region
    >>> geometry.point_on_bulk({'x': 20.0, 'y': 10.0, 'z': 100.0})
    True
    >>> # Point in void region
    >>> geometry.point_on_bulk({'x': 2.0, 'y': 2.0, 'z': 100.0})
    False
    
    Notes
    -----
    A point (x, y) is on the bulk if:
    - It is INSIDE the outer boundary: |x| < lx AND |y| < ly
    - It is OUTSIDE the void: |x| > void_x OR |y| > void_y
    """
    
    module_id: list[ModuleID]
    z: list[float]
    void_x_boundary: list[float]
    void_y_boundary: list[float]
    lx: list[float]
    ly: list[float]
    
    def __getitem__(self, index: int) -> tuple[int, float, float, float, float, float]:
        """
        Get geometry data for a specific module.
        
        Parameters
        ----------
        index : int
            Module index (0-based).
        
        Returns
        -------
        tuple[int, float, float, float, float, float]
            Tuple of (module_id, z, void_x, void_y, lx, ly).
        
        Raises
        ------
        IndexError
            If index is out of range.
        """
        return (
            self.module_id[index],
            self.z[index],
            self.void_x_boundary[index],
            self.void_y_boundary[index],
            self.lx[index],
            self.ly[index],
        )
    
    def point_on_bulk(self, state: StateVector) -> bool:
        """
        Check if a point is within the active (non-void) region.
        
        A point is on the bulk if it is within the outer boundary but
        outside the inner void region.
        
        Parameters
        ----------
        state : dict[str, float]
            State dictionary with 'x', 'y', 'z' keys.
        
        Returns
        -------
        bool
            True if the point is in the active region.
        """
        x, y = abs(state['x']), abs(state['y'])
        # Must be inside outer boundary
        inside_outer = x <= self.lx[0] and y <= self.ly[0]
        # Must be outside void region
        outside_void = x >= self.void_x_boundary[0] or y >= self.void_y_boundary[0]
        return inside_outer and outside_void
    
    def __len__(self) -> int:
        """Return the number of modules."""
        return len(self.module_id)
    
    def __iter__(self) -> Iterator[tuple[int, float, float, float, float, float]]:
        """Iterate over module geometry data."""
        for i in range(len(self)):
            yield self[i]
    
    def get_z_positions(self) -> list[float]:
        """Return the z positions of all modules."""
        return list(self.z)
    
    def is_in_void(self, x: float, y: float, module_index: int = 0) -> bool:
        """
        Check if a point is in the void region.
        
        Parameters
        ----------
        x : float
            X coordinate (mm).
        y : float
            Y coordinate (mm).
        module_index : int, default 0
            Index of the module to check against.
        
        Returns
        -------
        bool
            True if the point is in the void region.
        """
        return (
            abs(x) < self.void_x_boundary[module_index]
            and abs(y) < self.void_y_boundary[module_index]
        )
