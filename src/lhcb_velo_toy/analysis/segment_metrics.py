"""
Segment-level acceptance metrics.

Bakes the helpers previously inlined in
``Toy_Characterisation/Verify_new_results/segment_level_analysis.ipynb``
and ``Toy_Characterisation/condor_pipeline/scripts/helpers.py`` into the
package so all experiments share a single implementation.

Public API
----------
- ``compute_epsilon(sigma_res, sigma_scatt, dz, scale, theta_min)``
- ``pairwise_angle(h_prev, h_mid, h_next)``
- ``collect_segment_pair_angles(event)``
- ``compute_segment_metrics(true_angles, false_angles, epsilon)``
- ``fast_segment_metrics(event, epsilon)``  — Numba-accelerated (optional;
  falls back to an equivalent pure-Python kernel when numba is absent)
- ``segment_truth_mask(ham)``                — truth labels per Hamiltonian segment
- ``solver_segment_metrics(truth, scores, threshold)``  — metrics after thresholding
                                                          a solver's score vector
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

try:                                    # numba is an OPTIONAL accelerator
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:                     # pragma: no cover - depends on the install
    # The kernels below are plain numeric loops over numpy arrays, so they run
    # correctly undecorated — just slower.  Importing the package must never
    # require numba: it is not a runtime dependency, only the ``[fast]`` extra.
    HAS_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        """No-op stand-in for ``numba.njit`` supporting @njit and @njit(...)."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(func):
            return func

        return _decorator


# ---------------------------------------------------------------------------
# Epsilon (calculated) formula
# ---------------------------------------------------------------------------

DEFAULT_DZ = 33.0
DEFAULT_SCALE = 3.0
DEFAULT_THETA_MIN = 1.5e-5


def compute_epsilon(
    sigma_res: float,
    sigma_scatt: float,
    dz: float = DEFAULT_DZ,
    scale: float = DEFAULT_SCALE,
    theta_min: float = DEFAULT_THETA_MIN,
) -> float:
    r"""
    Angular acceptance threshold derived in
    ``docs/reports/segment_analysis/segment_level_report.pdf``:

    .. math::
        \varepsilon
            = \sqrt{2(\text{scale}\cdot\sigma_s)^2
                    + 12\,\arctan^2((\text{scale}\cdot\sigma_r) / \Delta z)
                    + 2\,\theta_{\min}^2}.
    """
    theta_s = scale * sigma_scatt
    theta_r = math.atan((scale * sigma_res) / dz) if dz != 0 else 0.0
    return float(
        math.sqrt(2.0 * theta_s * theta_s
                  + 12.0 * theta_r * theta_r
                  + 2.0 * theta_min * theta_min)
    )


# ---------------------------------------------------------------------------
# Pure-Python reference implementation (kept for validation / small events)
# ---------------------------------------------------------------------------

def pairwise_angle(h_prev, h_mid, h_next) -> float:
    """Angle between segments (prev->mid) and (mid->next)."""
    v1 = np.array([h_mid.x - h_prev.x,
                   h_mid.y - h_prev.y,
                   h_mid.z - h_prev.z])
    v2 = np.array([h_next.x - h_mid.x,
                   h_next.y - h_mid.y,
                   h_next.z - h_mid.z])
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.arccos(cos_a))


def collect_segment_pair_angles(event) -> tuple[list[float], list[float]]:
    """
    Enumerate all (prev, mid, next) hit triplets across consecutive module
    pairs and return ``(true_angles, false_angles)``.

    A triplet is *true* iff both segments are members of the same truth
    track (and all three hits share a non-negative ``track_id``).
    """
    true_seg_set: set[tuple[int, int]] = set()
    for trk in event.tracks:
        hids = trk.hit_ids
        for k in range(len(hids) - 1):
            true_seg_set.add((hids[k], hids[k + 1]))

    hits_by_module: dict[int, list] = {}
    for h in event.hits:
        hits_by_module.setdefault(h.module_id, []).append(h)
    sorted_mod_ids = sorted(hits_by_module.keys())

    true_angles: list[float] = []
    false_angles: list[float] = []

    for mi in range(1, len(sorted_mod_ids) - 1):
        mod_prev = sorted_mod_ids[mi - 1]
        mod_mid = sorted_mod_ids[mi]
        mod_next = sorted_mod_ids[mi + 1]

        for h_mid in hits_by_module[mod_mid]:
            for h_prev in hits_by_module[mod_prev]:
                seg_in_true = (h_prev.hit_id, h_mid.hit_id) in true_seg_set
                for h_next in hits_by_module[mod_next]:
                    seg_out_true = (h_mid.hit_id, h_next.hit_id) in true_seg_set
                    angle = pairwise_angle(h_prev, h_mid, h_next)
                    if (seg_in_true and seg_out_true
                            and h_prev.track_id is not None
                            and h_prev.track_id == h_mid.track_id == h_next.track_id):
                        true_angles.append(angle)
                    else:
                        false_angles.append(angle)
    return true_angles, false_angles


def compute_segment_metrics(
    true_angles: Iterable[float],
    false_angles: Iterable[float],
    epsilon: float,
) -> dict:
    """Acceptance metrics for a given angular threshold ``epsilon``."""
    true_arr = np.asarray(list(true_angles), dtype=np.float64)
    false_arr = np.asarray(list(false_angles), dtype=np.float64)
    n_true = int(true_arr.size)
    n_false = int(false_arr.size)
    true_accepted = int(np.sum(true_arr <= epsilon)) if n_true else 0
    false_accepted = int(np.sum(false_arr <= epsilon)) if n_false else 0
    total_accepted = true_accepted + false_accepted
    return {
        "n_true": n_true,
        "n_false": n_false,
        "true_accepted": true_accepted,
        "false_accepted": false_accepted,
        "segment_efficiency": (true_accepted / n_true) if n_true else 0.0,
        "segment_false_rate": (false_accepted / total_accepted) if total_accepted else 0.0,
    }


# ---------------------------------------------------------------------------
# Numba-accelerated kernel (~100x speedup on large events when numba is
# installed; identical results, without the speedup, when it is not).
# ---------------------------------------------------------------------------

@njit(cache=True)
def _cos_angle_inner(v1x, v1y, v1z, n1, v2x, v2y, v2z):
    n2 = math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
    if n1 == 0.0 or n2 == 0.0:
        return 1.0
    c = (v1x * v2x + v1y * v2y + v1z * v2z) / (n1 * n2)
    if c > 1.0:
        return 1.0
    if c < -1.0:
        return -1.0
    return c


@njit(cache=True)
def _count_one_mid(i, px, py, pz, mx, my, mz, nx_, ny, nz,
                   p_tid, m_tid, n_tid, cos_eps):
    loc_nt = 0
    loc_nf = 0
    loc_ta = 0
    loc_fa = 0
    mid_tid = m_tid[i]
    for j in range(len(px)):
        v1x = mx[i] - px[j]
        v1y = my[i] - py[j]
        v1z = mz[i] - pz[j]
        n1 = math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
        seg_in_true = (p_tid[j] >= 0) and (p_tid[j] == mid_tid)
        for k in range(len(nx_)):
            v2x = nx_[k] - mx[i]
            v2y = ny[k] - my[i]
            v2z = nz[k] - mz[i]
            ca = _cos_angle_inner(v1x, v1y, v1z, n1, v2x, v2y, v2z)
            is_true = seg_in_true and (mid_tid == n_tid[k])
            if is_true:
                loc_nt += 1
                if ca >= cos_eps:
                    loc_ta += 1
            else:
                loc_nf += 1
                if ca >= cos_eps:
                    loc_fa += 1
    return loc_nt, loc_nf, loc_ta, loc_fa


@njit(parallel=True, cache=True)
def _segment_counts_module(px, py, pz, mx, my, mz, nx_, ny, nz,
                           p_tid, m_tid, n_tid, cos_eps):
    n_m = len(mx)
    nt_arr = np.zeros(n_m, dtype=np.int64)
    nf_arr = np.zeros(n_m, dtype=np.int64)
    ta_arr = np.zeros(n_m, dtype=np.int64)
    fa_arr = np.zeros(n_m, dtype=np.int64)
    for i in prange(n_m):
        _nt, _nf, _ta, _fa = _count_one_mid(
            i, px, py, pz, mx, my, mz, nx_, ny, nz,
            p_tid, m_tid, n_tid, cos_eps,
        )
        nt_arr[i] = _nt
        nf_arr[i] = _nf
        ta_arr[i] = _ta
        fa_arr[i] = _fa
    return nt_arr.sum(), nf_arr.sum(), ta_arr.sum(), fa_arr.sum()


def fast_segment_metrics(event, epsilon: float) -> dict:
    """
    Numba-accelerated drop-in replacement for
    ``collect_segment_pair_angles`` + ``compute_segment_metrics``.

    Returns the same dict as :func:`compute_segment_metrics`.
    """
    cos_eps = math.cos(epsilon)

    hits_by_mod: dict[int, list] = {}
    for h in event.hits:
        hits_by_mod.setdefault(h.module_id, []).append(h)
    sorted_mods = sorted(hits_by_mod.keys())

    def _arrays(hlist):
        x = np.fromiter((h.x for h in hlist), dtype=np.float64, count=len(hlist))
        y = np.fromiter((h.y for h in hlist), dtype=np.float64, count=len(hlist))
        z = np.fromiter((h.z for h in hlist), dtype=np.float64, count=len(hlist))
        tid = np.fromiter(
            (h.track_id if h.track_id is not None else -1 for h in hlist),
            dtype=np.int64, count=len(hlist),
        )
        return x, y, z, tid

    mod_arrays = {m: _arrays(hits_by_mod[m]) for m in sorted_mods}

    n_true = n_false = true_accepted = false_accepted = 0
    for mi in range(1, len(sorted_mods) - 1):
        prev_ = mod_arrays[sorted_mods[mi - 1]]
        mid_ = mod_arrays[sorted_mods[mi]]
        nxt_ = mod_arrays[sorted_mods[mi + 1]]
        if mid_[0].size == 0 or prev_[0].size == 0 or nxt_[0].size == 0:
            continue
        nt, nf, ta, fa = _segment_counts_module(
            prev_[0], prev_[1], prev_[2],
            mid_[0], mid_[1], mid_[2],
            nxt_[0], nxt_[1], nxt_[2],
            prev_[3], mid_[3], nxt_[3],
            cos_eps,
        )
        n_true += int(nt)
        n_false += int(nf)
        true_accepted += int(ta)
        false_accepted += int(fa)

    total_accepted = true_accepted + false_accepted
    return {
        "n_true": int(n_true),
        "n_false": int(n_false),
        "true_accepted": int(true_accepted),
        "false_accepted": int(false_accepted),
        "segment_efficiency": (true_accepted / n_true) if n_true else 0.0,
        "segment_false_rate": (false_accepted / total_accepted) if total_accepted else 0.0,
    }


# ---------------------------------------------------------------------------
# Truth labels per Hamiltonian segment + solver metrics after thresholding
# ---------------------------------------------------------------------------

def segment_truth_mask(ham) -> np.ndarray:
    """
    Boolean mask of length ``ham.n_segments``: True iff the segment connects
    two hits sharing the same (non-negative) ``track_id``.
    """
    arr = getattr(ham, "_segment_track_ids", None)
    if arr is not None and len(arr) == ham.n_segments:
        return ((arr[:, 0] == arr[:, 1]) & (arr[:, 0] >= 0)).astype(bool)
    return np.asarray(
        [
            (s.hit_start.track_id == s.hit_end.track_id)
            and (s.hit_start.track_id is not None)
            and (s.hit_start.track_id >= 0)
            for s in ham.segments
        ],
        dtype=bool,
    )


def solver_segment_metrics(
    truth: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    """
    Segment-level metrics after thresholding a solver's continuous score
    vector. Matches the per-Hamiltonian-segment definition used in
    ``run_segment_event.py``.
    """
    active = np.asarray(scores) > threshold
    truth = np.asarray(truth, dtype=bool)
    n_true_all = int(truth.sum())
    n_false_all = int(truth.size - n_true_all)
    n_active = int(active.sum())
    n_true_active = int((active & truth).sum())
    n_false_active = n_active - n_true_active
    return {
        "n_true_all": n_true_all,
        "n_false_all": n_false_all,
        "n_true_active": n_true_active,
        "n_false_active": n_false_active,
        "n_active": n_active,
        "segment_efficiency": n_true_active / max(n_true_all, 1),
        "segment_purity": n_true_active / max(n_active, 1),
        "segment_false_rate": n_false_active / max(n_active, 1),
    }


__all__ = [
    "compute_epsilon",
    "pairwise_angle",
    "collect_segment_pair_angles",
    "compute_segment_metrics",
    "fast_segment_metrics",
    "HAS_NUMBA",
    "segment_truth_mask",
    "solver_segment_metrics",
    "DEFAULT_DZ",
    "DEFAULT_SCALE",
    "DEFAULT_THETA_MIN",
]
