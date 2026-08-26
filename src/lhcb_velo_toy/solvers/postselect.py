"""Hit-uniqueness post-selection — the classical-side occupancy filter.

Provenance
----------
Promoted verbatim from ``Toy_Characterisation/_shared/qtrk_pipeline/
postselect.py`` (centralise-2.1 Phase 2, 2026-08-26). Solver-agnostic: takes
any amplitude vector plus the segment hit-id table.

Denby's per-hit occupancy constraint, applied AFTER the solve as a selection
filter instead of inside the Hamiltonian. Headline result from the
originating study (2026-07-17): heavy T=200 far 0.65 -> 0.04 at eff 0.94.

Relation to the in-matrix occupancy term
(:class:`~lhcb_velo_toy.solvers.hamiltonians.OccupancyHamiltonian`): the
2026-07-07 "occupancy is classical-side only, never a Hamiltonian term"
ruling was REVERSED by the occupancy-refit study — with a refitted filter,
A_occ (alpha = 0.05) is beneficial and also breaks the fragment floor
(verified by George 2026-07-31). The two are complementary tools: this module
filters any solver's output; the Hamiltonian term changes the physics being
solved.

Competition is SAME-ROLE hit sharing only: two segments compete for a hit h
when both use h as their start hit, or both as their end hit (the fork/merge
topology, exactly the pairing of the DP ``B_all``). Consecutive segments of
one track share their interior hit in OPPOSITE roles (end of seg_i, start of
seg_{i+1}) and therefore never compete — a hard per-hit argmax that ignores
roles halves a track's segments (the ``anyrole`` mode exists only as that
negative control).

Modes
-----
strict   one pass: an active segment survives iff it carries the maximum
         amplitude among active segments in BOTH of its (hit, role) slots.
greedy   amplitude-descending slot claiming: walk active segments from the
         largest |x| down; keep a segment iff both its slots are unclaimed,
         then claim them. Resolves the cascade the one-pass misses (a slot
         whose winner was itself dropped elsewhere frees up for the
         runner-up... which greedy correctly still blocks: once a slot is
         claimed by ANY kept segment it stays closed). The default.
anyrole  negative control: per hit (roles pooled) keep only the single
         highest-amplitude active segment. Destroys track interiors.
margin   soft uniqueness: drop a segment only if it LOSES a slot by more
         than the margin — |x_s| < kappa * slotmax. kappa=1 reduces to
         strict; smaller kappa spares near-equal (true-vs-true) contests
         while still killing the false competitors that lose by a wide
         margin (the 0.25-attractor bulk under a ~0.44 true band). The
         efficiency-preserving production mode.
"""
from __future__ import annotations

import numpy as np

__all__ = ["hit_uniqueness_filter", "uniqueness_frontier"]


def _slot_ids(seg_hits: np.ndarray) -> np.ndarray:
    """(n_seg, 2) slot keys: role-tagged hit ids (start keeps id, end offset)."""
    seg_hits = np.asarray(seg_hits, dtype=np.int64)
    off = int(seg_hits.max()) + 1 if seg_hits.size else 1
    slots = seg_hits.copy()
    slots[:, 1] += off                     # end-role slots live in their own id space
    return slots


def hit_uniqueness_filter(sol: np.ndarray, seg_hits: np.ndarray,
                          active: np.ndarray, mode: str = "greedy",
                          kappa: float = 0.8) -> np.ndarray:
    """Prune an active set so each (hit, role) slot is used at most once.

    Parameters
    ----------
    sol      : amplitude vector (any solver; |sol| is used).
    seg_hits : (n_seg, 2) [start_hit_id, end_hit_id] — ``ham._segment_to_hit_ids``.
    active   : boolean mask of the pre-filter active set (e.g. |sol| > tau).
    mode     : "greedy" (default) | "strict" | "margin" | "anyrole" (control).
    kappa    : margin mode only — drop a segment iff |x| < kappa * slot winner.

    Returns the pruned boolean active mask (never adds segments).
    """
    x = np.abs(np.asarray(sol, dtype=np.float64).ravel())
    active = np.asarray(active, dtype=bool).copy()
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return active
    if mode == "anyrole":
        slots = np.asarray(seg_hits, dtype=np.int64)   # roles pooled: raw hit ids
    else:
        slots = _slot_ids(seg_hits)
    s0, s1 = slots[idx, 0], slots[idx, 1]

    if mode in ("strict", "anyrole", "margin"):
        # winner per slot = argmax |x| among active users of that slot
        nslot = int(slots.max()) + 1
        best = np.zeros(nslot)
        np.maximum.at(best, s0, x[idx])
        np.maximum.at(best, s1, x[idx])
        k = float(kappa) if mode == "margin" else 1.0
        keep = (x[idx] >= k * best[s0]) & (x[idx] >= k * best[s1])
        out = np.zeros_like(active)
        out[idx[keep]] = True
        return out

    if mode != "greedy":
        raise ValueError(f"unknown mode {mode!r}")
    order = idx[np.argsort(-x[idx], kind="stable")]
    claimed: set[int] = set()
    out = np.zeros_like(active)
    osl = slots[order]
    for k, seg in enumerate(order):
        a, b = int(osl[k, 0]), int(osl[k, 1])
        if a in claimed or b in claimed:
            continue
        claimed.add(a)
        claimed.add(b)
        out[seg] = True
    return out


def uniqueness_frontier(sol: np.ndarray, seg_hits: np.ndarray, truth: np.ndarray,
                        taus: np.ndarray, mode: str = "greedy",
                        kappa: float = 0.8):
    """(eff, far) of the COMPOSED selector (threshold -> uniqueness) per tau.

    far follows the store convention n_false_active / n_active on the
    post-filter active set. Returns (eff, far) arrays aligned with ``taus``.
    """
    x = np.abs(np.asarray(sol, dtype=np.float64).ravel())
    truth = np.asarray(truth, dtype=bool)
    nt = max(int(truth.sum()), 1)
    eff, far = [], []
    for t in taus:
        act = hit_uniqueness_filter(x, seg_hits, x > t, mode=mode, kappa=kappa)
        na = int(act.sum())
        eff.append(int((act & truth).sum()) / nt)
        far.append(int((act & ~truth).sum()) / max(na, 1))
    return np.asarray(eff), np.asarray(far)
