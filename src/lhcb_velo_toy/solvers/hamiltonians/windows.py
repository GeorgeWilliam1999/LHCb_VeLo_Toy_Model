"""Physics windows on segments: drop segments no prompt track can make.

Provenance
----------
New, 2026-11-12 (experiment D of the research plan "cheap and effective VELO
track finding": windowed segment graph). The trigger's triplet search keeps
only segments with a plausible slope that point back to the luminous region;
this module computes the same quantities for the package segments and returns
a keep-mask.

Applying a window
-----------------
For the step-kernel Hamiltonians of the package (``SimpleHamiltonian``,
``SimpleHamiltonianFast``) the diagonal is the constant ``-(delta + gamma)``,
``b`` is constant, and the coupling of two segments depends only on that pair.
So the Hamiltonian built from the kept segments alone is exactly the
sub-matrix ``A[keep][:, keep]`` (and ``b[keep]``): :func:`restrict_system`.
A window can therefore only remove couplings: every connected component of
the windowed system is a subset of a component of the full one.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["segment_geometry", "segment_window_mask", "restrict_system"]


def segment_geometry(endpoints_xyz: np.ndarray, beamline: tuple[float, float] = (0.0, 0.0)) -> dict:
    """Slopes, transverse impact parameter and z of closest approach to the beamline.

    Parameters
    ----------
    endpoints_xyz : (n_seg, 2, 3) array
        Start and end hit coordinates per segment
        (``hamiltonian._segment_endpoints_xyz``).
    beamline : (x, y)
        Beamline position (the beamline is taken parallel to z).

    Returns
    -------
    dict of (n_seg,) arrays: ``tx``, ``ty`` (dx/dz, dy/dz), ``d0`` (distance of
    closest approach of the segment line to the beamline, transverse), ``z0``
    (z at that point; NaN for a line parallel to the beam).
    """
    e = np.asarray(endpoints_xyz, dtype=np.float64)
    p, q = e[:, 0, :], e[:, 1, :]
    dz = q[:, 2] - p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = (q[:, 0] - p[:, 0]) / dz
        ty = (q[:, 1] - p[:, 1]) / dz
    x1 = p[:, 0] - beamline[0]
    y1 = p[:, 1] - beamline[1]
    t2 = tx ** 2 + ty ** 2
    flat = t2 < 1e-24
    with np.errstate(divide="ignore", invalid="ignore"):
        d0 = np.where(flat, np.hypot(x1, y1), np.abs(x1 * ty - y1 * tx) / np.sqrt(t2))
        z0 = np.where(flat, np.nan, p[:, 2] - (x1 * tx + y1 * ty) / t2)
    return dict(tx=tx, ty=ty, d0=d0, z0=z0)


def segment_window_mask(endpoints_xyz: np.ndarray, max_slope: Optional[float] = None,
                        max_d0: Optional[float] = None,
                        z_range: Optional[tuple[float, float]] = None,
                        beamline: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Keep-mask of the segments inside the window.

    Parameters
    ----------
    endpoints_xyz : (n_seg, 2, 3) array
    max_slope : float, optional
        Keep ``|tx| <= max_slope`` and ``|ty| <= max_slope``.
    max_d0 : float, optional
        Keep a transverse impact parameter to the beamline ``d0 <= max_d0``.
    z_range : (z_lo, z_hi), optional
        Keep the z of closest approach inside the range (a line parallel to the
        beam passes).
    beamline : (x, y)

    Returns
    -------
    (n_seg,) bool array. All True when every cut is None.
    """
    g = segment_geometry(endpoints_xyz, beamline)
    keep = np.isfinite(g["tx"]) & np.isfinite(g["ty"])
    if max_slope is not None:
        keep &= (np.abs(g["tx"]) <= max_slope) & (np.abs(g["ty"]) <= max_slope)
    if max_d0 is not None:
        keep &= g["d0"] <= max_d0
    if z_range is not None:
        z0 = g["z0"]
        keep &= np.isnan(z0) | ((z0 >= z_range[0]) & (z0 <= z_range[1]))
    return keep


def restrict_system(A, b: np.ndarray, keep: np.ndarray):
    """The segment system of the kept segments: ``A[keep][:, keep]``, ``b[keep]``.

    Exact for the package's step-kernel Hamiltonians (see the module notes).
    Returns ``(A_w, b_w, index)`` with ``index`` the kept segments' rows in A.
    """
    idx = np.flatnonzero(np.asarray(keep, dtype=bool))
    A = A.tocsr()
    return A[idx][:, idx].tocsr(), np.asarray(b)[idx], idx
