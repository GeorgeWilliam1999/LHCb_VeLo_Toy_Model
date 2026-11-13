"""Primary vertices from first-pass tracks: z clustering plus a 3D vertex fit.

Provenance
----------
New, 2026-11-13 (item J of the research plan "cheap and effective VELO track
finding"). The classical track-based baseline for the direct PV readout of
:mod:`pv_sampler`. The earlier baseline (``Vertexing/Focusing_scan/scripts/
08_pv_baseline.py``) clustered the z of closest approach to the beamline and
took the cluster median. That assumes the vertex is on the beamline; on Run 3
the PVs scatter by about 24 um transversely, which moves a track's z of closest
approach by 24 um / |t|. Here each z cluster seeds a 3D vertex fit (the point
that minimises the summed squared distance to the track lines), with
iterative removal of tracks far from the vertex.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["fit_track_lines", "fit_vertex_point", "vertices_from_tracks"]


def fit_track_lines(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    groups: Sequence[Sequence[int]], min_hits: int = 3):
    """Straight-line fit per track.

    Parameters
    ----------
    x, y, z : hit coordinates (indexed by the group entries).
    groups : hit index lists, one per track.
    min_hits : tracks with fewer hits (or one z only) are skipped.

    Returns
    -------
    (n, 4) array of (ax, ay, tx, ty): x(z) = ax + tx z, y(z) = ay + ty z.
    """
    out = []
    for g in groups:
        g = np.asarray(g, int)
        if len(g) < min_hits or np.ptp(z[g]) == 0:
            continue
        tx, ax = np.polyfit(z[g], x[g], 1)
        ty, ay = np.polyfit(z[g], y[g], 1)
        out.append((ax, ay, tx, ty))
    return np.array(out, float).reshape(-1, 4)


def _dist_to_lines(v, L):
    """Distance of point v to each line (ax, ay, tx, ty)."""
    p = np.c_[L[:, 0], L[:, 1], np.zeros(len(L))]
    d = np.c_[L[:, 2], L[:, 3], np.ones(len(L))]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    w = v[None, :] - p
    return np.linalg.norm(w - (w * d).sum(1, keepdims=True) * d, axis=1)


def fit_vertex_point(L: np.ndarray, z_seed: float, cut: float = 0.3, n_iter: int = 5):
    """3D point closest (least squares) to a set of lines, with outlier removal.

    Parameters
    ----------
    L : (n, 4) track lines.
    z_seed : unused by the linear fit; returned when fewer than 2 lines remain.
    cut : lines farther than ``cut`` [mm] from the fitted point are dropped, then refit.
    n_iter : refit iterations.

    Returns
    -------
    v : (3,) vertex position.
    keep : (n,) bool, lines used in the last fit.
    """
    keep = np.ones(len(L), bool)
    v = np.array([0.0, 0.0, z_seed])
    for _ in range(n_iter):
        if keep.sum() < 2:
            break
        Lk = L[keep]
        p = np.c_[Lk[:, 0], Lk[:, 1], np.zeros(len(Lk))]
        d = np.c_[Lk[:, 2], Lk[:, 3], np.ones(len(Lk))]
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        M = np.zeros((3, 3))
        b = np.zeros(3)
        for pi, di in zip(p, d):
            P = np.eye(3) - np.outer(di, di)
            M += P
            b += P @ pi
        v = np.linalg.solve(M, b)
        new = _dist_to_lines(v, L) <= cut
        if np.array_equal(new, keep):
            break
        keep = new
    return v, keep


def vertices_from_tracks(lines: np.ndarray, gap: float = 1.0, min_trk: int = 3, tmin: float = 0.02,
                         fit: bool = True, cut: float = 0.3, min_sep: float = 0.5):
    """PV list from track lines.

    1. z of closest approach to the beamline per track (tracks with |t| < tmin dropped).
    2. Sorted z split at gaps > ``gap``; clusters with >= ``min_trk`` tracks seed a vertex.
    3. ``fit=False``: the vertex is the cluster median z (the 08_pv_baseline rule).
       ``fit=True``: 3D fit of the cluster's lines (:func:`fit_vertex_point`); the vertex is
       kept if >= ``min_trk`` lines survive the cut. Fitted vertices closer than ``min_sep``
       in z are merged (the one with more tracks is kept).

    Returns
    -------
    list of (z, x, y, n_tracks), sorted by n_tracks descending.
    """
    from .pv_sampler import _poca
    if len(lines) == 0:
        return []
    z0, _, tn = _poca(lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3])
    ok = (tn >= tmin) & np.isfinite(z0)
    L = lines[ok]
    z0 = z0[ok]
    o = np.argsort(z0)
    L, z0 = L[o], z0[o]
    if len(z0) == 0:
        return []
    cuts = np.nonzero(np.diff(z0) > gap)[0] + 1
    out = []
    for idx in np.split(np.arange(len(z0)), cuts):
        if len(idx) < min_trk:
            continue
        if not fit:
            out.append((float(np.median(z0[idx])), 0.0, 0.0, int(len(idx))))
            continue
        v, keep = fit_vertex_point(L[idx], float(np.median(z0[idx])), cut=cut)
        if keep.sum() >= min_trk:
            out.append((float(v[2]), float(v[0]), float(v[1]), int(keep.sum())))
    out.sort(key=lambda r: -r[3])
    kept = []
    for r in out:
        if all(abs(r[0] - k[0]) >= min_sep for k in kept):
            kept.append(r)
    return kept
