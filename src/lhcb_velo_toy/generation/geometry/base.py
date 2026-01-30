"""
Abstract base class for detector geometries.

Geometry defines the interface that all detector geometry classes must
implement, including module layout and active region determination.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

from lhcb_velo_toy.core.types import ModuleID, StateVector


@dataclass(frozen=True)
class Geometry(ABC):
    """
    Abstract base class for detector geometry specifications.
    
    A Geometry defines the layout of detector modules and provides methods
    to determine whether particles are within the active detection region.
    Geometries are immutable (frozen) to ensure consistency.
    
    Attributes
    ----------
    module_id : list[int]
        List of module identifiers, one per detector layer.
    
    Notes
    -----
    Subclasses must implement:
    - `__getitem__`: Return geometry data for a specific module index
    - `point_on_bulk`: Determine if a point is in the active region
    
    Examples
    --------
    >>> geometry = PlaneGeometry(
    ...     module_id=[1, 2, 3],
    ...     lx=[50.0, 50.0, 50.0],
    ...     ly=[50.0, 50.0, 50.0],
    ...     z=[100.0, 150.0, 200.0]
    ... )
    >>> len(geometry)
    3
    >>> for mod_id, lx, ly, z in geometry:
    ...     print(f"Module {mod_id} at z={z}")
    """
    
    module_id: list[ModuleID]
    
    @abstractmethod
    def __getitem__(self, index: int) -> tuple:
        """
        Get geometry data for a specific module.
        
        Parameters
        ----------
        index : int
            Index of the module (0-based).
        
        Returns
        -------
        tuple
            Geometry data for the module. The exact contents depend on
            the concrete geometry class.
        
        Raises
        ------
        IndexError
            If index is out of range.
        """
        ...
    
    @abstractmethod
    def point_on_bulk(self, state: StateVector) -> bool:
        """
        Check if a particle state is within any module's active region.
        
        Parameters
        ----------
        state : dict[str, float]
            Particle state dictionary with keys 'x', 'y', 'z'.
        
        Returns
        -------
        bool
            True if the point (x, y, z) is within the active detection
            region of any module.
        """
        ...
    
    def __len__(self) -> int:
        """
        Get the number of modules in this geometry.
        
        Returns
        -------
        int
            Number of modules.
        """
        raise NotImplementedError
    
    def __iter__(self) -> Iterator[tuple]:
        """
        Iterate over all modules in this geometry.
        
        Yields
        ------
        tuple
            Geometry data for each module.
        """
        raise NotImplementedError
    
    def get_z_positions(self) -> list[float]:
        """
        Get the z positions of all modules.
        
        Returns
        -------
        list[float]
            Z positions in order.
        """
        raise NotImplementedError
