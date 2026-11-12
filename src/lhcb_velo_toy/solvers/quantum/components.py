"""Split a block-diagonal operator into its connected components, one small circuit each.

Provenance
----------
New, 2026-11-12 (experiment C of the research plan "cheap and effective VELO
track finding": decomposed small circuits). A spectral filter ``f(M)`` of a
block-diagonal ``M`` acts block by block, so every connected component of the
off-diagonal support of ``M`` (segment register: the continuation graph of the
segment Hamiltonian; hit register: the clique graph C) can be run as its own
circuit on its own register of ``ceil(log2 m)`` qubits. This module gives the
split, the per-component Givens price (the gate model of the hit/segment
register studies, ``Vertexing/Focusing_scan/scripts/05_resources.py``), and a
nested multinomial shot sampler with a separate shot budget per component.
No circuit is built or simulated.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from lhcb_velo_toy.solvers.quantum.spectral_filter import continuation_components

__all__ = [
    "split_components",
    "popcount_mean",
    "component_givens_price",
    "shot_allocation",
    "ComponentShotSampler",
]


def split_components(M) -> tuple[list[np.ndarray], np.ndarray]:
    """Connected components of the off-diagonal support of symmetric ``M``.

    Parameters
    ----------
    M : sparse symmetric (n, n)

    Returns
    -------
    comps : list of int arrays
        Row indices of each component, ascending; components ordered by their
        smallest index.
    labels : (n,) int array
        Component label per row, consistent with ``comps``.
    """
    nc, lab = continuation_components(M)
    order = np.argsort(lab, kind="stable")
    cuts = np.flatnonzero(np.diff(lab[order])) + 1
    comps = np.split(order, cuts) if order.size else []
    first = np.array([c[0] for c in comps]) if comps else np.zeros(0, int)
    perm = np.argsort(first, kind="stable")
    comps = [comps[p] for p in perm]
    relabel = np.empty(len(comps), int)
    relabel[perm] = np.arange(len(comps))
    return comps, relabel[lab] if len(comps) else lab


def popcount_mean(i: np.ndarray, j: np.ndarray) -> float:
    """Mean Hamming distance between the binary register indices ``i`` and ``j``.

    1.0 for no pairs (the convention of the 05/07 gate count).
    """
    if len(i) == 0:
        return 1.0
    x = np.bitwise_xor(np.asarray(i, np.int64), np.asarray(j, np.int64))
    cnt = np.zeros(len(x))
    while np.any(x):
        cnt += x & 1
        x = x >> 1
    return float(cnt.mean())


def component_givens_price(M, comps: list[np.ndarray], register_index: np.ndarray,
                           mcrx_cx: Callable[[int], float]) -> dict:
    """Givens price of one product step of each component, on its own register.

    Each coupled pair (i, j) costs ``2 (dH - 1) + CX(MCRX with n_s controls)``
    two-qubit gates (05_resources.py), with ``dH`` the Hamming distance of the
    two register indices and ``n_s = ceil(log2 m)`` the component's register
    width. Inside a component the items keep the order of
    ``register_index`` (e.g. hits by plane then x; segments in package order)
    and are renumbered 0 ... m - 1.

    Parameters
    ----------
    M : sparse symmetric (n, n)
        The operator (its off-diagonal pattern defines the pairs).
    comps : list of int arrays
        From :func:`split_components`.
    register_index : (n,) int array
        Global register order of every item (a permutation of 0 ... n - 1).
    mcrx_cx : callable
        CX count of a multi-controlled RX with ``n`` controls.

    Returns
    -------
    dict of arrays, one entry per component: ``m`` (items), ``n_s`` (register
    qubits, >= 1), ``k`` (coupled pairs), ``dH`` (mean Hamming distance),
    ``cx_step`` (two-qubit gates per product step).
    """
    import scipy.sparse as sp
    U = sp.triu(sp.csr_matrix(M), k=1).tocsr()
    U.eliminate_zeros()
    ri = np.asarray(register_index)
    out = {key: np.zeros(len(comps)) for key in ("m", "n_s", "k", "dH", "cx_step")}
    out["n_s"][:] = 1
    out["dH"][:] = 1.0
    for c, idx in enumerate(comps):
        m = len(idx)
        out["m"][c] = m
        if m == 1:                               # no pair: nothing to price
            continue
        ns = max(1, math.ceil(math.log2(m)))
        local = np.empty(m, int)
        local[np.argsort(ri[idx], kind="stable")] = np.arange(m)
        sub = U[idx][:, idx].tocoo()
        dH = popcount_mean(local[sub.row], local[sub.col])
        k = sub.nnz
        out["n_s"][c], out["k"][c], out["dH"][c] = ns, k, dH
        out["cx_step"][c] = k * (2 * (dH - 1) + mcrx_cx(ns)) if k else 0.0
    return out


def shot_allocation(sizes: np.ndarray, beta: float) -> np.ndarray:
    """Shots per component, ``ceil(beta * m ln m)`` (at least 1).

    The coupon-collector form measured in experiment G (N90 ~ m ln m, m the
    items to collect); ``m`` here is the component size, known before any
    solve.
    """
    m = np.asarray(sizes, dtype=np.float64)
    return np.maximum(1, np.ceil(beta * m * np.log(np.maximum(m, 2.0)))).astype(np.int64)


class ComponentShotSampler:
    """Nested multinomial shots with a separate budget per component.

    Parameters
    ----------
    p : (n,) array
        Per-item measurement probability inside its component: for every
        component ``c``, ``p[comps[c]]`` sums to 1 (or to 0 for a component
        with no probability mass, which then receives no counts).
    comps : list of int arrays
        Components (:func:`split_components`), the items each circuit measures.
    seed : int

    Notes
    -----
    Components are padded to the largest size and drawn in one vectorised
    ``Generator.multinomial`` call per budget step; counts at a larger budget
    contain those at a smaller one (nested), as in
    :class:`~lhcb_velo_toy.solvers.quantum.shots.ShotSampler`.
    """

    def __init__(self, p: np.ndarray, comps: list[np.ndarray], seed: int = 0):
        p = np.asarray(p, dtype=np.float64)
        self.comps = comps
        self.n = p.size
        width = max((len(c) for c in comps), default=1)
        self.P = np.zeros((len(comps), width))
        self.rows = np.repeat(np.arange(len(comps)), [len(c) for c in comps])
        self.cols = np.concatenate([np.arange(len(c)) for c in comps]) if comps else np.zeros(0, int)
        self.items = np.concatenate(comps) if comps else np.zeros(0, int)
        self.P[self.rows, self.cols] = p[self.items]
        s = self.P.sum(1)
        self.live = s > 0
        self.P[self.live] /= s[self.live, None]
        self.rng = np.random.default_rng(seed)
        self.C = np.zeros_like(self.P, dtype=np.int64)
        self.shots = np.zeros(len(comps), dtype=np.int64)

    def counts_at(self, budget: np.ndarray) -> np.ndarray:
        """Per-item counts after ``budget[c]`` shots of component ``c``."""
        budget = np.asarray(budget, dtype=np.int64)
        if np.any(budget < self.shots):
            raise ValueError("budgets must be non-decreasing")
        extra = np.where(self.live, budget - self.shots, 0)
        go = extra > 0
        if go.any():
            self.C[go] += self.rng.multinomial(extra[go], self.P[go])
        self.shots = np.where(self.live, budget, self.shots)
        out = np.zeros(self.n, dtype=np.int64)
        out[self.items] = self.C[self.rows, self.cols]
        return out
