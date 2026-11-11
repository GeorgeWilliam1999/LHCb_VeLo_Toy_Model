"""Denby–Peterson penalty extensions of the segment Hamiltonian.

Provenance
----------
Promoted from the Bifurification study (centralise-2.1 Phase 1, 2026-08-26):
``Toy_Characterisation/Bifurification/{bif.py, dp_terms.py}``. Two penalties
from Denby's original track-finding Hamiltonian, composed onto the base
system A0 x = b built by :class:`SimpleHamiltonianFast` (returned-A
convention: diag = +(gamma+delta), compatible couplings = -1, b = delta*1):

* **Bifurcation (fork)** — penalise co-hit segment pairs within a mutual-angle
  window ``eps_B`` (the epsilon-windowed form; the study's verdict is that the
  dense all-pair fork uniformly rescales the spectrum and breaks the 1BQF, so
  it is deliberately NOT provided as a fork mode)::

      A = A0 + beta * B_fork          (b unchanged; attractor kept)

* **Occupancy (per-hit)** — Denby's hit-uniqueness constraint
  ``E_occ = alpha * sum_h (o_h - 1)^2`` with ``o_h = sum_{s contains h} x_s``::

      A = A0 + 4*alpha*I + 2*alpha*B_all ,   b = (delta + 4*alpha) * 1

  where ``B_all`` couples ALL co-hit pairs regardless of angle. The
  isolated-segment attractor shifts to ``(delta+4a)/(gamma+delta+4a)``;
  :meth:`ExtendedHamiltonian.working_point_tau` tracks it. The verified
  operating point from the occupancy-refit study is ``alpha = 0.05``.

  .. warning::
     ``B_all`` has one entry per co-hit segment pair at ANY angle, which
     scales like O(T^3) with track multiplicity — unlike the base A and
     ``B_fork`` (both ~O(n_seg)). Use ``alpha != 0`` at high T deliberately.

  Denby's GLOBAL count term ``alpha * (sum x - N)^2`` is a dense rank-one
  coupling — classical-only, not implementable as 1BQF pair gates — and is
  intentionally absent here (per the DP study notes).

Both penalties compose (that is :class:`ExtendedHamiltonian`)::

    A = A0 + beta*B_fork + 4*alpha*I + 2*alpha*B_all ,  b = (delta+4*alpha)*1
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from lhcb_velo_toy.solvers.hamiltonians.fast import SimpleHamiltonianFast

__all__ = [
    "ExtendedHamiltonian",
    "ForkHamiltonian",
    "OccupancyHamiltonian",
    "cohit_graph",
    "cohit_graph_eps",
]


def _cohit_groups(seg_hits: np.ndarray):
    """Yield index groups of segments sharing a start hit or an end hit."""
    n = len(seg_hits)
    for col in (0, 1):  # 0 = start hit, 1 = end hit
        order = np.argsort(seg_hits[:, col], kind="stable")
        key = seg_hits[order, col]
        bnd = np.r_[0, np.where(np.diff(key))[0] + 1, n]
        for a, c in zip(bnd[:-1], bnd[1:]):
            grp = order[a:c]
            if len(grp) > 1:
                yield grp


def cohit_graph(seg_hits) -> sp.csr_matrix:
    """B_all: 0/1 adjacency of segments sharing a hit (any mutual angle).

    One entry per co-hit pair — O(T^3) scaling at high multiplicity; see the
    module warning.
    """
    seg_hits = np.asarray(seg_hits)
    n = len(seg_hits)
    rows, cols = [], []
    for grp in _cohit_groups(seg_hits):
        ii, jj = np.meshgrid(grp, grp)
        m = ii < jj
        rows.append(ii[m])
        cols.append(jj[m])
    if not rows:
        return sp.csr_matrix((n, n))
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    B = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    return (B + B.T).tocsr()


def cohit_graph_eps(seg_hits, seg_vecs, eps_B: float) -> sp.csr_matrix:
    """B_fork: 0/1 adjacency of co-hit pairs with mutual angle < eps_B.

    The epsilon-windowed bifurcation adjacency — only segments that share a
    hit AND are genuinely competing continuations (inside the angular window)
    are coupled. Sparse (nnz ~ O(n_seg)), so it preserves the sparse-A
    invariant and keeps the 1BQF tractable. The angle test is the same strict
    ``cos(angle) > cos(eps_B)`` convention as the base Hamiltonian.
    """
    seg_hits = np.asarray(seg_hits)
    seg_vecs = np.asarray(seg_vecs)
    n = len(seg_hits)
    cos_eps = float(np.cos(eps_B))
    rows, cols = [], []
    for grp in _cohit_groups(seg_hits):
        V = seg_vecs[grp]
        cosM = np.clip(V @ V.T, -1.0, 1.0)
        ii, jj = np.triu_indices(len(grp), 1)
        keep = cosM[ii, jj] > cos_eps  # angle < eps_B (strict, canonical)
        if keep.any():
            rows.append(grp[ii[keep]])
            cols.append(grp[jj[keep]])
    if not rows:
        return sp.csr_matrix((n, n))
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    B = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    return (B + B.T).tocsr()


class ExtendedHamiltonian(SimpleHamiltonianFast):
    """Segment Hamiltonian with composable fork and occupancy penalties.

    Parameters
    ----------
    epsilon, gamma, delta, theta_d
        As for :class:`SimpleHamiltonianFast`.
    beta : float, default 0.0
        Fork (bifurcation) penalty strength. 0 disables the term.
    eps_B : float or None, default None
        Angular window for the fork term; ``None`` uses ``epsilon``.
    alpha : float, default 0.0
        Per-hit occupancy penalty strength. 0 disables the term.
        The verified occupancy-refit operating point is 0.05.

    Notes
    -----
    ``construct_hamiltonian`` builds the base system via the parent class
    (identical segments, identical strict-epsilon convention) and then adds
    the enabled penalty terms. ``self.A``/``self.b`` and the returned pair
    are the modified system, so ``solve_classicaly``/``evaluate`` and every
    downstream solver operate on the extended physics transparently.
    ``self.term_info`` records the nnz of each added term.
    """

    def __init__(
        self,
        epsilon: float,
        gamma: float,
        delta: float,
        theta_d: float = 1e-4,
        beta: float = 0.0,
        eps_B: Optional[float] = None,
        alpha: float = 0.0,
    ) -> None:
        super().__init__(epsilon, gamma, delta, theta_d)
        self.beta = float(beta)
        self.eps_B = eps_B
        self.alpha = float(alpha)
        self.term_info: dict = {}

    def construct_hamiltonian(self, event, convolution=False, erf_sigma=None,
                              prune=False):
        """Build the base system, add the enabled penalty terms, then prune.

        ``prune`` is applied to the FINAL operator (after the terms), never to
        the base system: the occupancy term couples every co-hit pair, so a
        segment that is decoupled in ``A0`` is not decoupled in the operator
        actually solved, and pruning it first would change the physics.  With
        ``alpha != 0`` the prune therefore removes (almost) nothing; with the
        fork term alone it removes exactly what the base prune removes.
        """
        A, b = super().construct_hamiltonian(
            event, convolution=convolution, erf_sigma=erf_sigma
        )
        A = A.tocsr()
        n = A.shape[0]
        info = {"A0_nnz": int(A.nnz), "Bfork_nnz": 0, "Ball_nnz": 0}
        seg_hits = np.asarray(self._segment_to_hit_ids)

        if self.beta:
            eps_b = float(self.eps_B) if self.eps_B is not None else self.epsilon
            B_fork = cohit_graph_eps(seg_hits, self._segment_vectors, eps_b)
            A = (A + self.beta * B_fork).tocsr()
            info["Bfork_nnz"] = int(B_fork.nnz)

        if self.alpha:
            B_all = cohit_graph(seg_hits)
            A = (
                A
                + 4.0 * self.alpha * sp.identity(n, format="csr")
                + 2.0 * self.alpha * B_all
            ).tocsr()
            b = b + 4.0 * self.alpha
            info["Ball_nnz"] = int(B_all.nnz)

        info["A_nnz"] = int(A.nnz)
        self.term_info = info
        A = A.tocsc()
        self.A, self.b = A, b
        self._maybe_prune(prune)
        return self.A, self.b

    def working_point_tau(self, margin: float = 0.10) -> float:
        """Attractor-aware absolute threshold: isolated attractor + margin.

        The fork term (off-diagonal mode) leaves the isolated-segment
        attractor unchanged; the occupancy term shifts it to
        ``(delta+4a)/(gamma+delta+4a)``.
        """
        attractor = (self.delta + 4.0 * self.alpha) / (
            self.gamma + self.delta + 4.0 * self.alpha
        )
        return float(attractor + margin)


class ForkHamiltonian(ExtendedHamiltonian):
    """Base Hamiltonian + the epsilon-windowed Denby–Peterson fork penalty."""

    def __init__(
        self,
        epsilon: float,
        gamma: float,
        delta: float,
        beta: float,
        eps_B: Optional[float] = None,
        theta_d: float = 1e-4,
    ) -> None:
        super().__init__(
            epsilon, gamma, delta, theta_d=theta_d, beta=beta, eps_B=eps_B, alpha=0.0
        )


class OccupancyHamiltonian(ExtendedHamiltonian):
    """Base Hamiltonian + the per-hit occupancy (hit-uniqueness) penalty."""

    def __init__(
        self,
        epsilon: float,
        gamma: float,
        delta: float,
        alpha: float = 0.05,
        theta_d: float = 1e-4,
    ) -> None:
        super().__init__(
            epsilon, gamma, delta, theta_d=theta_d, beta=0.0, alpha=alpha
        )
