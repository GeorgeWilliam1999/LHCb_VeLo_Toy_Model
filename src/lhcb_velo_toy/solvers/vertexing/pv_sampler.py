"""Direct primary-vertex readout: an amplitude-amplified vertex sampler.

Provenance
----------
New, 2026-11-13 (item I of the research plan "cheap and effective VELO track
finding", reordered by George on 2026-11-13: the goal is a direct readout of the
primary vertices (PVs) by a quantum algorithm, compared by qubits and queries).

The algorithm
-------------
Registers:

* an **index register** over register items. Two item kinds:

  - ``doublet``: every package segment (a hit pair on consecutive modules), so
    the index register is the segment register of paper 1;
  - ``triplet``: a segment and the k-th forward segment that the Hamiltonian
    couples to it (sparse access, as the block encoding of A uses), so the
    index register is ``ceil(log2 n_seg) + ceil(log2 d_max)`` qubits;

* a **vertex register** of ``ceil(log2 n_bins)`` qubits: the z of the point of
  closest approach (POCA) of the item's line to the beamline (x = y = 0), binned;
* one **flag** qubit: the line points to the beamline (distance of closest
  approach <= ``dmax``, slope ``|t| >= tmin``, POCA z inside the window).

One **query** is one call of the oracle
``O |s>|0>|0> = |s>|zeta(s)>|f(s)>``, which loads the item's hits and does the
arithmetic. Amplitude amplification on the flag uses ``r`` rounds (2 queries
each: O and O^dagger inside the reflection), then one more call writes the
vertex register for the measurement. The measurement returns the bin of a
uniformly chosen flagged item. Peaks of the shot histogram are the PVs.

What is exact and what is a model
---------------------------------
* The measurement distribution is exact: it is the normalised histogram of the
  POCA z over the flagged items (the oracle truth table). No circuit is
  simulated. Shots are drawn from it.
* The amplification success probability ``sin^2((2r+1) theta)`` is exact for a
  known flagged fraction (``theta = asin sqrt(P)``, with the register padding
  to ``2^n`` included). Unknown P (exponential search) costs a constant factor
  more; this is not included.
* Qubits of the index, vertex and flag registers are exact. The arithmetic
  workspace of the oracle is a model (:func:`workspace_qubits_model`).

The classical counterparts read the same oracle classically:
exhaustive (every item once, the ``K -> infinity`` limit of the sampler) and
classical rejection sampling (``K / P`` item reads for ``K`` shots).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "line_poca",
    "fit_line_poca",
    "VertexOracleTable",
    "doublet_table",
    "forward_couplings",
    "triplet_table",
    "zeta_bins",
    "output_distribution",
    "amplification",
    "sample_counts",
    "find_vertex_peaks",
    "sampler_resources",
    "workspace_qubits_model",
    "match_vertices",
]


# ── line geometry ────────────────────────────────────────────────────────────
def line_poca(p0: np.ndarray, p1: np.ndarray):
    """POCA of the line through two points to the beamline x = y = 0.

    Parameters
    ----------
    p0, p1 : (n, 3) arrays
        Two points per line (x, y, z), with ``p1.z != p0.z``.

    Returns
    -------
    z0 : (n,) z of closest approach.
    doca : (n,) distance of closest approach.
    tnorm : (n,) transverse slope ``|t| = sqrt(tx^2 + ty^2)``.
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    dz = p1[:, 2] - p0[:, 2]
    tx = (p1[:, 0] - p0[:, 0]) / dz
    ty = (p1[:, 1] - p0[:, 1]) / dz
    # line: x(z) = ax + tx z, y(z) = ay + ty z
    ax = p0[:, 0] - tx * p0[:, 2]
    ay = p0[:, 1] - ty * p0[:, 2]
    return _poca(ax, ay, tx, ty)


def _poca(ax, ay, tx, ty):
    t2 = tx * tx + ty * ty
    tn = np.sqrt(t2)
    with np.errstate(divide="ignore", invalid="ignore"):
        z0 = np.where(t2 > 0, -(ax * tx + ay * ty) / t2, np.nan)
        doca = np.where(tn > 0, np.abs(ax * ty - ay * tx) / tn, np.hypot(ax, ay))
    return z0, doca, tn


def fit_line_poca(points: np.ndarray):
    """Least-squares straight line through k points per line, then its POCA.

    Parameters
    ----------
    points : (n, k, 3) array, k >= 2.

    Returns
    -------
    z0, doca, tnorm as :func:`line_poca`.
    """
    P = np.asarray(points, float)
    z = P[..., 2]
    zm = z.mean(axis=1, keepdims=True)
    dz = z - zm
    szz = (dz * dz).sum(axis=1)
    out = []
    for c in (0, 1):
        v = P[..., c]
        vm = v.mean(axis=1)
        s = (dz * (v - vm[:, None])).sum(axis=1) / szz
        out.append((vm - s * zm[:, 0], s))
    (ax, tx), (ay, ty) = out
    return _poca(ax, ay, tx, ty)


# ── oracle truth table ───────────────────────────────────────────────────────
@dataclass
class VertexOracleTable:
    """The oracle truth table restricted to what the measurement sees.

    Attributes
    ----------
    kind : "doublet" or "triplet".
    n_items : real register items (segments, or forward coupled segment pairs).
    n_index_qubits : index register width.
    z0 : (n_flagged,) POCA z of the flagged items (what the vertex register holds).
    items : (n_flagged, k) hit ids of each flagged item (k = 2 or 3), for truth checks.
    doca : (n_flagged,) distance of closest approach of the flagged items.
    cuts : the flag definition.
    """
    kind: str
    n_items: int
    n_index_qubits: int
    z0: np.ndarray
    items: np.ndarray
    doca: np.ndarray
    cuts: dict = field(default_factory=dict)

    @property
    def n_flagged(self) -> int:
        return int(len(self.z0))

    @property
    def p_flag(self) -> float:
        """Flagged fraction of the real items (no padding)."""
        return self.n_flagged / max(self.n_items, 1)

    @property
    def p_flag_padded(self) -> float:
        """Flagged fraction of the padded ``2^n`` register (what amplification sees)."""
        return self.n_flagged / float(2 ** self.n_index_qubits)


def _flag(z0, doca, tn, dmax, tmin, zwin):
    return (np.isfinite(z0) & (doca <= dmax) & (tn >= tmin)
            & (z0 >= zwin[0]) & (z0 < zwin[1]))


def doublet_table(seg_xyz: np.ndarray, seg_hits: np.ndarray, dmax: float, tmin: float,
                  zwin: tuple) -> VertexOracleTable:
    """Doublet items: one per segment (the paper 1 segment register).

    Parameters
    ----------
    seg_xyz : (n_seg, 2, 3) segment endpoints (``SimpleHamiltonianFast._segment_endpoints_xyz``).
    seg_hits : (n_seg, 2) hit ids (``_segment_to_hit_ids``).
    dmax : flag cut on the distance of closest approach [mm].
    tmin : flag cut on the transverse slope.
    zwin : (lo, hi) z window of the vertex register [mm].
    """
    seg_xyz = np.asarray(seg_xyz, float)
    n = len(seg_xyz)
    z0, doca, tn = line_poca(seg_xyz[:, 0], seg_xyz[:, 1])
    f = _flag(z0, doca, tn, dmax, tmin, zwin)
    return VertexOracleTable("doublet", n, max(1, math.ceil(math.log2(max(n, 2)))),
                             z0[f], np.asarray(seg_hits)[f], doca[f],
                             dict(dmax=dmax, tmin=tmin, zwin=tuple(zwin)))


def forward_couplings(A, seg_hits: np.ndarray):
    """Forward coupled segment pairs (a, b): A couples them and a ends where b starts.

    Parameters
    ----------
    A : sparse (n_seg, n_seg) Hamiltonian matrix; a nonzero off-diagonal is a coupling.
    seg_hits : (n_seg, 2) hit ids.

    Returns
    -------
    a, b : (n_pairs,) segment indices.
    d_max : the largest number of forward neighbours of one segment (the sparse-access
        row width the index register must address).
    """
    import scipy.sparse as sp
    C = sp.coo_matrix(A)
    off = C.row != C.col
    r, c = C.row[off], C.col[off]
    sh = np.asarray(seg_hits)
    fw = sh[r, 1] == sh[c, 0]
    a, b = r[fw], c[fw]
    d_max = int(np.bincount(a, minlength=len(sh)).max()) if len(a) else 1
    return a, b, max(d_max, 1)


def triplet_table(seg_xyz: np.ndarray, seg_hits: np.ndarray, A, dmax: float, tmin: float,
                  zwin: tuple) -> VertexOracleTable:
    """Triplet items: (segment, k-th forward coupled segment), three hits, line fit.

    The index register is ``ceil(log2 n_seg) + ceil(log2 d_max)`` qubits (sparse
    access); slots k >= the segment's neighbour count are unflagged.
    """
    seg_xyz = np.asarray(seg_xyz, float)
    sh = np.asarray(seg_hits)
    n_seg = len(sh)
    a, b, d_max = forward_couplings(A, sh)
    pts = np.stack([seg_xyz[a, 0], seg_xyz[a, 1], seg_xyz[b, 1]], axis=1)
    z0, doca, tn = fit_line_poca(pts) if len(a) else (np.zeros(0),) * 3
    f = _flag(z0, doca, tn, dmax, tmin, zwin)
    nq = max(1, math.ceil(math.log2(max(n_seg, 2)))) + (math.ceil(math.log2(d_max)) if d_max > 1 else 0)
    items = np.stack([sh[a, 0], sh[a, 1], sh[b, 1]], axis=1)[f] if len(a) else np.zeros((0, 3), int)
    # n_items = the real (segment, neighbour) pairs, which a classical loop reads; the
    # padded register (2^nq slots) is what amplification sees (p_flag_padded)
    return VertexOracleTable("triplet", int(len(a)), nq,
                             z0[f], items, doca[f],
                             dict(dmax=dmax, tmin=tmin, zwin=tuple(zwin), d_max=d_max))


# ── vertex register and output distribution ──────────────────────────────────
def zeta_bins(zwin: tuple, width: float):
    """Bin edges of the vertex register and its width in qubits.

    The register has ``2^q >= n_bins`` states; unused states are never written.
    """
    n = int(math.ceil((zwin[1] - zwin[0]) / width))
    edges = zwin[0] + width * np.arange(n + 1)
    return edges, max(1, math.ceil(math.log2(n)))


def output_distribution(table: VertexOracleTable, edges: np.ndarray) -> np.ndarray:
    """Exact measurement distribution of the vertex register (after a flag = 1 shot)."""
    c, _ = np.histogram(table.z0, bins=edges)
    s = c.sum()
    return c / s if s > 0 else c.astype(float)


def amplification(p: float):
    """Rounds and success probability of amplitude amplification at known p.

    Returns
    -------
    rounds : the r in {floor(pi / (4 theta)) - 1, floor(pi / (4 theta))}, theta = asin sqrt(p),
        that needs the fewest queries per accepted shot (0 when amplification does not help).
    p_success : sin^2((2r+1) theta), the flag = 1 probability after r rounds.
    queries_per_shot : (2r + 1) / p_success, the expected oracle calls per accepted shot
        (a failed shot is seen on the flag and repeated).
    """
    if p <= 0:
        return 0, 0.0, math.inf
    th = math.asin(math.sqrt(min(p, 1.0)))
    r = max(0, int(math.floor(math.pi / (4 * th))))
    # floor can overshoot by one round when pi/(4 th) is just above an integer + 1/2
    best = max(range(max(0, r - 1), r + 1), key=lambda k: math.sin((2 * k + 1) * th) ** 2 / (2 * k + 1))
    ps = math.sin((2 * best + 1) * th) ** 2
    return best, ps, (2 * best + 1) / ps


def sample_counts(p: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """Counts per bin of K accepted shots drawn from the exact distribution p."""
    if K <= 0 or p.sum() <= 0:
        return np.zeros_like(p, dtype=np.int64)
    return rng.multinomial(int(K), p / p.sum())


# ── peak finding (the same rule for shots and for the exhaustive histogram) ──
def find_vertex_peaks(counts: np.ndarray, edges: np.ndarray, scale: float, sigma: float,
                      min_content: float, min_sep: float, window: Optional[float] = None,
                      shoulder: float = 0.0):
    """PV list from a vertex-register histogram.

    Parameters
    ----------
    counts : (n_bins,) shot counts (or item counts for the exhaustive histogram).
    edges : bin edges [mm].
    scale : items per count (``n_flagged / K`` for K shots; 1 for item counts), so the
        thresholds are in flagged items whatever K is.
    sigma : Gaussian smoothing [mm].
    min_content : a peak needs at least this many items within +- ``window`` of it.
    min_sep : smallest distance between two peaks [mm].
    window : half-width of the content and centroid window [mm]; default 2 sigma.
    shoulder : drop a peak whose smoothed height is below ``shoulder`` x a higher kept
        peak within 5 mm (0 disables).

    Returns
    -------
    list of (z, content) sorted by content, descending.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    w = float(edges[1] - edges[0])
    window = 2 * sigma if window is None else window
    y = counts.astype(float) * scale
    ys = gaussian_filter1d(y, sigma / w, mode="constant")
    zc = 0.5 * (edges[1:] + edges[:-1])
    pk, _ = find_peaks(ys, distance=max(1, int(round(min_sep / w))))
    hw = max(1, int(round(window / w)))
    cand = []
    for p in pk:
        lo, hi = max(0, p - hw), min(len(y), p + hw + 1)
        content = float(y[lo:hi].sum())
        if content < min_content:
            continue
        z = float(np.average(zc[lo:hi], weights=y[lo:hi])) if y[lo:hi].sum() > 0 else float(zc[p])
        cand.append((z, content, float(ys[p])))
    cand.sort(key=lambda r: -r[1])
    kept = []
    for z, c, h in cand:
        if any(abs(z - k[0]) < min_sep for k in kept):
            continue
        if shoulder > 0 and any(abs(z - k[0]) <= 5.0 and h < shoulder * k[2] for k in kept):
            continue
        kept.append((z, c, h))
    return [(z, c) for z, c, _ in kept]


# ── resources ────────────────────────────────────────────────────────────────
def workspace_qubits_model(kind: str, bits: int = 16) -> int:
    """MODEL of the oracle's arithmetic workspace (not a circuit).

    Loaded coordinates (3 per hit, ``bits`` each, 2 or 3 hits), slopes, the
    intercepts and one division for z0 and the doca comparison, each ``bits``
    wide: (3 k + 6) x bits, with k hits per item.
    """
    k = 2 if kind == "doublet" else 3
    return (3 * k + 6) * bits


def sampler_resources(table: VertexOracleTable, n_zeta_qubits: int, K: int, bits: int = 16) -> dict:
    """Qubits and queries of the sampler for K accepted shots.

    Columns
    -------
    q_index, q_zeta, q_data : exact register widths (data = index + vertex + flag).
    q_work_model : modelled arithmetic workspace (:func:`workspace_qubits_model`).
    P, P_padded : flagged fraction without and with the ``2^n`` padding.
    rounds, p_success, queries_per_shot : :func:`amplification` at P_padded.
    queries : K x queries_per_shot (quantum).
    classical_exhaustive : n_items (every item read once; the K -> infinity limit).
    classical_sampling : K / P (rejection sampling of the real items).
    """
    r, ps, qps = amplification(table.p_flag_padded)
    return dict(kind=table.kind, n_items=table.n_items, n_flagged=table.n_flagged,
                q_index=table.n_index_qubits, q_zeta=n_zeta_qubits,
                q_data=table.n_index_qubits + n_zeta_qubits + 1,
                q_work_model=workspace_qubits_model(table.kind, bits),
                P=table.p_flag, P_padded=table.p_flag_padded, rounds=r, p_success=ps,
                queries_per_shot=qps, K=int(K), queries=K * qps,
                classical_exhaustive=table.n_items,
                classical_sampling=K / table.p_flag if table.p_flag > 0 else math.inf)


# ── scoring ──────────────────────────────────────────────────────────────────
def match_vertices(found_z, true_z, tol: float = 1.0):
    """Greedy nearest matching of found to true vertex z (as the focusing-scan study).

    Returns
    -------
    m : (n_true,) index of the matched found vertex, -1 if none.
    res : residuals found - true of the matched pairs.
    merges : unmatched true vertices within ``tol`` of an already used found vertex.
    fakes : found vertices not matched.
    """
    found_z = np.asarray(found_z, float)
    true_z = np.asarray(true_z, float)
    m = -np.ones(len(true_z), int)
    used = set()
    pairs = sorted((abs(f - t), i, j) for i, t in enumerate(true_z) for j, f in enumerate(found_z))
    for dist, i, j in pairs:
        if dist > tol or m[i] >= 0 or j in used:
            continue
        m[i] = j
        used.add(j)
    res = np.array([found_z[m[i]] - true_z[i] for i in range(len(true_z)) if m[i] >= 0])
    merges = sum(1 for i in range(len(true_z)) if m[i] < 0 and len(found_z)
                 and np.min(np.abs(found_z - true_z[i])) <= tol)
    return m, res, int(merges), int(len(found_z) - len(used))
