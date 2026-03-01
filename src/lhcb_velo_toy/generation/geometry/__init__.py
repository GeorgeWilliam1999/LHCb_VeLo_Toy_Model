"""Detector geometry classes for the LHCb VELO Toy Model."""

from __future__ import annotations

from typing import Any

from lhcb_velo_toy.generation.geometry.base import Geometry
from lhcb_velo_toy.generation.geometry.plane import PlaneGeometry
from lhcb_velo_toy.generation.geometry.rectangular_void import RectangularVoidGeometry

# Class registry for deserialization dispatch
_GEOMETRY_REGISTRY: dict[str, type[Geometry]] = {
    "PlaneGeometry": PlaneGeometry,
    "RectangularVoidGeometry": RectangularVoidGeometry,
}


def geometry_from_dict(data: dict[str, Any]) -> Geometry:
    """
    Reconstruct a Geometry subclass from a dictionary.

    The dictionary must contain a ``"geometry_class"`` key whose value
    matches a registered geometry class name.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary previously produced by a geometry's ``to_dict()`` method.

    Returns
    -------
    Geometry
        The reconstructed geometry object.

    Raises
    ------
    ValueError
        If ``geometry_class`` is missing or unrecognised.

    Examples
    --------
    >>> geo = PlaneGeometry(module_id=[0,1], lx=[50,50], ly=[50,50], z=[100,133])
    >>> rebuilt = geometry_from_dict(geo.to_dict())
    >>> rebuilt == geo
    True
    """
    cls_name = data.get("geometry_class")
    if cls_name is None:
        raise ValueError(
            "Dictionary does not contain a 'geometry_class' key. "
            "Cannot determine which geometry to construct."
        )
    geo_cls = _GEOMETRY_REGISTRY.get(cls_name)
    if geo_cls is None:
        raise ValueError(
            f"Unknown geometry class '{cls_name}'. "
            f"Registered classes: {list(_GEOMETRY_REGISTRY.keys())}"
        )
    return geo_cls.from_dict(data)  # type: ignore[attr-defined]


__all__ = [
    "Geometry",
    "PlaneGeometry",
    "RectangularVoidGeometry",
    "geometry_from_dict",
]
