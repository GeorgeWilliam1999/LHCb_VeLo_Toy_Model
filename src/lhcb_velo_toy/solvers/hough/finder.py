"""Directional Hough-transform track finding (point and locus voting).

Provenance
----------
Promoted from the Hough deep-dive study (centralise-2.1 Phase 3, 2026-08-26):
``Hough/hough_prototype.py`` (committed prototype: point vote, smoothing,
peak finding, nearest-peak claim) and ``Hough/hough_study_lib.py`` (the
parameterised pipeline and the LOCUS voting that dismantles the vertex
smear — study figures fig14/fig15). Audited study conventions kept as
defaults: tx/ty range (-0.25, 0.25), 3-bin claim radius (``claim_radius`` =
3 w (sigma+1), i.e. r0 ~ 2.5 w merge law), MIN_HITS = 3 distinct planes,
deduplicated locus accumulator, height-priority locus claim.

Physics: with the primary vertex pinned near x = y = 0, each hit's direction
``d = (x/z, y/z)`` ~ (tx, ty), so a track is a concurrent bundle of votes at
one point of a 2-D directional accumulator (point vote). The exact form —
a hit at (x, y, z) is consistent with any beam-line vertex zeta, tracing the
locus ``d(zeta) = (x, y)/(z - zeta)`` — restores exact concurrency at
zeta = z_pv and removes the vertex smear by construction (locus vote).

The accumulator/voting registers are constant in track multiplicity T —
the property that makes this lane quantum-relevant.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "TX_RANGE", "TY_RANGE", "MIN_HITS",
    "hit_directions", "event_arrays", "bin_width",
    "accumulate", "find_peaks", "claim_radius", "assign", "run_point",
    "locus_curves", "accumulate_locus", "n_zeta_for", "assign_locus",
    "run_locus",
]

# Study defaults (audited 2026-06-12): tracks live in [-0.2, 0.2]^2 slopes,
# accumulator padded to +-0.25.
TX_RANGE = (-0.25, 0.25)
TY_RANGE = (-0.25, 0.25)
MIN_HITS = 3          # min hits on distinct planes for a track candidate


def event_arrays(event):
    """Flat hit arrays (xs, ys, zs, planes, truth) from a toy/adapted event."""
    hits = event.hits
    xs = np.array([h.x for h in hits])
    ys = np.array([h.y for h in hits])
    zs = np.array([h.z for h in hits])
    planes = np.array([h.module_id for h in hits])
    truth = np.array([h.track_id for h in hits])
    return xs, ys, zs, planes, truth


def hit_directions(event):
    """Return (D, planes, truth): D[i] = (x/z, y/z) point votes."""
    xs, ys, zs, planes, truth = event_arrays(event)
    return np.column_stack([xs / zs, ys / zs]), planes, truth


def bin_width(n_bins, tx_range=TX_RANGE):
    return (tx_range[1] - tx_range[0]) / n_bins


def accumulate(D, n_bins, weights=None, tx_range=TX_RANGE, ty_range=TY_RANGE):
    """Point-vote accumulator. Returns 2-D counts."""
    w = bin_width(n_bins, tx_range)
    ix = np.clip(((D[:, 0] - tx_range[0]) / w).astype(int), 0, n_bins - 1)
    iy = np.clip(((D[:, 1] - ty_range[0]) / w).astype(int), 0, n_bins - 1)
    flat = ix * n_bins + iy
    counts = np.bincount(flat, weights=weights, minlength=n_bins * n_bins)
    return counts.reshape(n_bins, n_bins).astype(float)


def find_peaks(counts, smooth_sigma=1.0, peak_min=0.0, tx_range=TX_RANGE,
               ty_range=TY_RANGE):
    """Gaussian-smooth + 3x3 local maxima. Returns (centres[N, 2], smoothed).

    Smoothing lets a track's votes, spread over a few adjacent bins by the
    z_pv / scattering smear, reinforce into ONE local maximum.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter

    n_bins = counts.shape[0]
    sm = gaussian_filter(counts, smooth_sigma, mode="constant")
    mx = maximum_filter(sm, size=3, mode="constant")
    pj, pk = np.where((sm == mx) & (sm > peak_min))
    w = bin_width(n_bins, tx_range)
    centres = np.column_stack([tx_range[0] + (pj + 0.5) * w,
                               ty_range[0] + (pk + 0.5) * w])
    return centres, sm


def claim_radius(n_bins, smooth_sigma=1.0, tx_range=TX_RANGE):
    """The prototype's hit-claim radius (smoothing support): 3 w (sigma + 1)."""
    return 3.0 * bin_width(n_bins, tx_range) * (smooth_sigma + 1)


def assign(D, planes, centres, radius, min_hits=MIN_HITS):
    """Nearest-peak assignment, one hit per plane. Returns candidate list.

    Candidates are dicts ``{hit_idx, tx, ty, n_planes}`` with ``hit_idx``
    indexing into the event's hit array.
    """
    from scipy.spatial import cKDTree

    if len(centres) == 0:
        return []
    tree = cKDTree(centres)
    dist, idx = tree.query(D, k=1)
    lab = np.where(dist <= radius, idx, -1)
    cands = []
    for p in np.unique(lab[lab >= 0]):
        members = np.where(lab == p)[0]
        keep = []
        for pl in np.unique(planes[members]):
            cm = members[planes[members] == pl]
            keep.append(cm[np.argmin(np.linalg.norm(D[cm] - centres[p], axis=1))])
        keep = np.array(keep)
        if len(keep) < min_hits:
            continue
        cands.append(dict(hit_idx=keep, tx=D[keep, 0].mean(),
                          ty=D[keep, 1].mean(), n_planes=len(keep)))
    return cands


def run_point(D, planes, n_bins=256, smooth_sigma=1.0):
    """Full point-vote pipeline. Returns (cands, centres, smoothed)."""
    counts = accumulate(D, n_bins)
    centres, sm = find_peaks(counts, smooth_sigma)
    cands = assign(D, planes, centres, claim_radius(n_bins, smooth_sigma))
    return cands, centres, sm


# ---------------------------------------------------------------------------
# Locus voting (the exact hit -> curve Hough)
# ---------------------------------------------------------------------------

def locus_curves(xs, ys, zs, zeta_half=3.5, n_zeta=29):
    """Sampled curves d_i(zeta_m), shape (n_hits, n_zeta, 2), + segment endpoints."""
    zet = np.linspace(-zeta_half, zeta_half, n_zeta)
    denom = zs[:, None] - zet[None, :]
    Dz = np.stack([xs[:, None] / denom, ys[:, None] / denom], axis=-1)
    return Dz, Dz[:, 0, :], Dz[:, -1, :]   # curves, endpoint A, endpoint B


def accumulate_locus(Dz, n_bins, tx_range=TX_RANGE, ty_range=TY_RANGE):
    """Deduplicated locus accumulator: each (hit, cell) pair counts ONCE,
    so a slow curve (small |t|: d(zeta) barely moves) cannot fake a peak."""
    w = bin_width(n_bins, tx_range)
    ix = np.clip(((Dz[..., 0] - tx_range[0]) / w).astype(int), 0, n_bins - 1)
    iy = np.clip(((Dz[..., 1] - ty_range[0]) / w).astype(int), 0, n_bins - 1)
    flat = ix * n_bins + iy                      # (n_hits, n_zeta)
    flat.sort(axis=1)
    first = np.ones_like(flat, dtype=bool)
    first[:, 1:] = flat[:, 1:] != flat[:, :-1]   # first occurrence per hit
    counts = np.bincount(flat[first].ravel(), minlength=n_bins * n_bins)
    return counts.reshape(n_bins, n_bins).astype(float)


def _seg_dist(p, a, b):
    """Distance of points p[N, 2] to segment a-b (each [2])."""
    ab = b - a
    denom = (ab * ab).sum()
    if denom == 0:
        return np.linalg.norm(p - a, axis=-1)
    t = np.clip(((p - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(p - proj, axis=-1)


def assign_locus(D, planes, centres, heights, segA, segB, radius,
                 min_hits=MIN_HITS):
    """Assign each hit to the HIGHEST peak within ``radius`` of its LOCUS SEGMENT.

    Height priority is essential: a hit's curve passes EXACTLY through every
    accidental 2-curve crossing it participates in (segment distance 0 ties),
    so nearest-peak assignment scatters hits across spurious low crossings.
    The true track crossing is the 5-fold (highest) peak -> it wins the claim.
    This is the classic Hough readout (peaks processed by height).
    """
    from scipy.spatial import cKDTree

    if len(centres) == 0:
        return []
    tree = cKDTree(centres)
    mid = 0.5 * (segA + segB)
    halflen = 0.5 * np.linalg.norm(segB - segA, axis=1)
    n = len(D)
    best_peak = np.full(n, -1)
    best_d = np.full(n, np.inf)
    neigh = tree.query_ball_point(mid, halflen + radius)
    for i in range(n):
        cand = neigh[i]
        if not cand:
            continue
        cand = np.asarray(cand)
        dseg = _seg_dist(centres[cand], segA[i], segB[i])
        ok = dseg <= radius
        if not ok.any():
            continue
        j = np.argmax(heights[cand] - 1e-6 * dseg / radius)  # height, dist tiebreak
        if dseg[j] <= radius:
            best_peak[i] = cand[j]
            best_d[i] = dseg[j]
        else:
            jj = np.where(ok)[0][np.argmax(heights[cand[ok]])]
            best_peak[i] = cand[jj]
            best_d[i] = dseg[jj]
    cands = []
    for p in np.unique(best_peak[best_peak >= 0]):
        members = np.where(best_peak == p)[0]
        keep = []
        for pl in np.unique(planes[members]):
            cm = members[planes[members] == pl]
            keep.append(cm[np.argmin(best_d[cm])])
        keep = np.array(keep)
        if len(keep) < min_hits:
            continue
        cands.append(dict(hit_idx=keep, tx=D[keep, 0].mean(),
                          ty=D[keep, 1].mean(), n_planes=len(keep)))
    return cands


def n_zeta_for(n_bins, zeta_half=3.5, d_max=0.25, z_min=33.0):
    """Sample the locus finely enough that consecutive zeta steps move the vote
    by < 1 bin even for the fastest curve (nearest plane, |d| = d_max):
    ``step <= w z_min / d_max``."""
    w = bin_width(n_bins)
    return int(np.ceil(2 * zeta_half * d_max / (w * z_min))) + 3


def run_locus(arrays, n_bins=1024, smooth_sigma=1.0, peak_min=0.35,
              zeta_half=3.5, n_zeta=None, radius_bins=2.0):
    """Full locus-vote pipeline on (xs, ys, zs, planes).

    ``radius_bins`` is the claim radius in BINS (segment distance), default 2.
    Returns (cands, centres, smoothed).
    """
    xs, ys, zs, planes = arrays
    if n_zeta is None:
        n_zeta = n_zeta_for(n_bins, zeta_half)
    Dz, segA, segB = locus_curves(xs, ys, zs, zeta_half, n_zeta)
    counts = accumulate_locus(Dz, n_bins)
    centres, sm = find_peaks(counts, smooth_sigma, peak_min=peak_min)
    n_b = counts.shape[0]
    w = bin_width(n_b)
    pj = np.clip(((centres[:, 0] - TX_RANGE[0]) / w).astype(int), 0, n_b - 1)
    pk = np.clip(((centres[:, 1] - TY_RANGE[0]) / w).astype(int), 0, n_b - 1)
    heights = sm[pj, pk]
    D = np.column_stack([xs / zs, ys / zs])
    cands = assign_locus(D, planes, centres, heights, segA, segB,
                         radius_bins * bin_width(n_bins))
    return cands, centres, sm
