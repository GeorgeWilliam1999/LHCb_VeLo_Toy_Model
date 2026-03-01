"""
PlaneGeometry: Simple planar detector with rectangular active areas.

This geometry represents a detector with multiple parallel planar modules,
each with a rectangular active area.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterator

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
        return (self.module_id[index], self.lx[index], self.ly[index], self.z[index])

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
        x, y = state['x'], state['y']
        return abs(x) <= self.lx[0] and abs(y) <= self.ly[0]

    def __len__(self) -> int:
        """Return the number of modules."""
        return len(self.module_id)

    def __iter__(self) -> Iterator[tuple[int, float, float, float]]:
        """Iterate over (module_id, lx, ly, z) for each module."""
        for i in range(len(self)):
            yield self[i]

    def get_z_positions(self) -> list[float]:
        """Return the z positions of all modules."""
        return list(self.z)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this geometry to a dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary with a ``geometry_class`` discriminator and all
            fields needed to reconstruct the object.
        """
        return {
            "geometry_class": "PlaneGeometry",
            "module_id": list(self.module_id),
            "lx": list(self.lx),
            "ly": list(self.ly),
            "z": list(self.z),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlaneGeometry":
        """
        Construct a PlaneGeometry from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary previously produced by :meth:`to_dict`.

        Returns
        -------
        PlaneGeometry
        """
        return cls(
            module_id=[int(m) for m in data["module_id"]],
            lx=[float(v) for v in data["lx"]],
            ly=[float(v) for v in data["ly"]],
            z=[float(v) for v in data["z"]],
        )
