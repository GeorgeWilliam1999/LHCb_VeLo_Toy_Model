"""Exact spectral-filter application via per-component eigendecomposition.

Provenance
----------
Promoted from the QSVT working folder (centralise-2.1 Phase 2, 2026-08-26):
``QSVT/Codesign/04_fit_comb_to_measured_spectrum.py`` (origin of the pattern)
and ``QSVT/Efficiency_Frontier/01_degree_response_frontier.py`` (the
vectorised ``mode_system``/``amplitudes`` form used for the frontier results),
plus ``Toy_Characterisation/_shared/qtrk_pipeline/solve.spectral_bounds_for``.

The idea: the segment Hamiltonian's off-diagonal support (the continuation
graph) splits A into many small connected components, so any spectral
response f can be applied EXACTLY — ``x = |V f(Lambda) V^T b|`` per component,
isolated segments analytically — at the cost of small dense ``eigh`` calls
instead of a Chebyshev matvec recursion. Diagonalise once, evaluate many
candidate responses for free. This is the official second backend of
:class:`~lhcb_velo_toy.solvers.quantum.QSVT.QSVT` (``solve_eigh``) and the
engine behind the Codesign / Efficiency-Frontier response sweeps.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
from numpy.polynomial import chebyshev as _npcheb
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

__all__ = [
    "continuation_components",
    "mode_system",
    "apply_response",
    "binned_response",
    "chebyshev_realize",
    "spectral_bounds",
]


def continuation_components(M) -> tuple[int, np.ndarray]:
    """Connected components of the off-diagonal support of symmetric M.

    The idiom ``C = sI - A`` / off-diagonal support / ``connected_components``
    repeated across the QSVT studies, factored once. Returns
    ``(n_components, labels)``.
    """
    Off = M.tocsr().copy().tolil()
    Off.setdiag(0)
    Off = Off.tocsr()
    Off.eliminate_zeros()
    return connected_components(Off, directed=False)


def mode_system(M, b, iso_lambda: Optional[float] = None,
                max_comp: int = 20000) -> dict:
    """Per-component eigendecomposition of sparse symmetric M.

    Parameters
    ----------
    M : sparse symmetric (n, n)
    b : (n,) right-hand side
    iso_lambda : float, optional
        Eigenvalue assigned to isolated segments (no off-diagonal coupling).
        ``None`` uses each isolated entry's own diagonal value (a superset of
        the study behaviour, identical when the diagonal is constant).
    max_comp : int
        Guard: refuse a connected component larger than this (dense eigh
        would not be viable).

    Returns
    -------
    dict with U (n x n_modes sparse eigenvector matrix), lam, w = V^T b per
    mode, iso mask, iso_lam per isolated entry, b, and bookkeeping — the
    input to :func:`apply_response`.
    """
    M = M.tocsr()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = M.shape[0]
    ncomp, lab = continuation_components(M)
    sizes = np.bincount(lab)
    if sizes.size and sizes.max() > max_comp:
        raise RuntimeError(
            f"connected component of size {sizes.max()} > max_comp={max_comp}: "
            "dense per-component eigh not viable — use the Chebyshev backend"
        )
    order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], np.arange(ncomp + 1))
    rows, cols, vals = [], [], []
    lams, ws = [], []
    m0 = 0
    for c in np.flatnonzero(sizes > 1):
        idx = order[bounds[c]:bounds[c + 1]]
        lam, V = eigh(M[np.ix_(idx, idx)].toarray())
        wb = V.T @ b[idx]
        k = len(idx)
        rows.append(np.repeat(idx, k))
        cols.append(np.tile(np.arange(m0, m0 + k), k))
        vals.append(V.ravel())
        lams.append(lam)
        ws.append(wb)
        m0 += k
    if m0:
        U = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, m0),
        )
        lam = np.concatenate(lams)
        w = np.concatenate(ws)
    else:
        U = sp.csr_matrix((n, 0))
        lam = np.zeros(0)
        w = np.zeros(0)
    iso = sizes[lab] == 1
    diag = np.asarray(M.diagonal()).ravel()
    iso_lam = (np.full(int(iso.sum()), float(iso_lambda))
               if iso_lambda is not None else diag[iso])
    return dict(
        U=U, lam=lam, w=w, iso=iso, iso_lam=iso_lam, b=b,
        lam_min=float(lam.min()) if lam.size else float(diag.min(initial=0.0)),
        lam_max=float(lam.max()) if lam.size else float(diag.max(initial=0.0)),
        max_comp=int(sizes.max()) if sizes.size else 0,
        n_comp=int(np.count_nonzero(sizes > 1)),
    )


def apply_response(sysd: dict, f: Callable, signed: bool = False) -> np.ndarray:
    """Apply spectral response f to the decomposed system: y = f(M) b, exactly.

    ``signed=False`` (study convention) returns amplitudes ``|y|``;
    ``signed=True`` returns the signed filtered vector (needed for exact
    equivalence with the QSVT circuit semantics).
    """
    iso = sysd["iso"]
    y = np.zeros(sysd["U"].shape[0])
    y[iso] = np.asarray(f(sysd["iso_lam"])).ravel() * sysd["b"][iso]
    y += (sysd["U"] @ (np.asarray(f(sysd["lam"])) * sysd["w"])) * (~iso)
    return y if signed else np.abs(y)


def binned_response(p_bins: np.ndarray, edges: np.ndarray) -> Callable:
    """Piecewise-constant spectral response from bin values + edges."""
    p_bins = np.asarray(p_bins, dtype=np.float64)

    def f(lam):
        bi = np.clip(np.searchsorted(edges, np.asarray(lam)) - 1,
                     0, len(p_bins) - 1)
        return p_bins[bi]

    return f


def chebyshev_realize(f_target: Callable, domain: tuple[float, float],
                      degree: int, grid: int = 4000) -> Callable:
    """Least-squares degree-d Chebyshev realisation of an arbitrary response.

    Fits ``f_target`` on ``grid`` points over ``domain`` and rescales so
    ``max |p| <= 1`` on the grid (implementability as an LCU filter). Returns
    a callable in lambda-space. Port of the Efficiency-Frontier
    ``cheb_realize`` (generalised from binned targets to any callable).
    """
    lo, hi = float(domain[0]), float(domain[1])
    xs = np.linspace(lo, hi, grid)
    targ = np.asarray(f_target(xs), dtype=np.float64)
    u = (2.0 * xs - (lo + hi)) / (hi - lo)
    cf = _npcheb.chebfit(u, targ, degree)
    sc = max(float(np.max(np.abs(_npcheb.chebval(u, cf)))), 1e-12)

    def f(lam, cf=cf, lo=lo, hi=hi, sc=sc):
        uu = (2.0 * np.asarray(lam) - (lo + hi)) / (hi - lo)
        return _npcheb.chebval(uu, cf) / sc

    return f


def spectral_bounds(A, s: Optional[float] = None,
                    pad: float = 0.10) -> tuple[float, float]:
    """Cheap spectral bounds for A = sI - C: rho(C) via one Lanczos 'LA'.

    C is symmetric non-negative, so (Perron–Frobenius) the spectral radius IS
    lambda_max — a single 'LA' eigsh suffices; Gershgorin fallback. ``s``
    defaults to the (constant) diagonal of A. Port of
    ``qtrk_pipeline.solve.spectral_bounds_for`` (itself promoted from
    ``QSVT/qsvt_store_campaign.py``).
    """
    A = A.tocsr() if sp.issparse(A) else sp.csr_matrix(np.asarray(A, float))
    n = A.shape[0]
    if s is None:
        diag = np.asarray(A.diagonal()).ravel()
        if not np.allclose(diag, diag[0]):
            raise ValueError("pass s explicitly for a non-constant diagonal")
        s = float(diag[0])
    C = (s * sp.identity(n, format="csr") - A)
    if n <= 256:
        w = np.linalg.eigvalsh(C.toarray())
        rho = float(max(abs(w[0]), abs(w[-1])))
    else:
        try:
            rho = float(eigsh(C, k=1, which="LA",
                              return_eigenvectors=False, maxiter=3000)[0])
        except Exception:
            d = np.asarray(abs(C).sum(axis=1)).ravel()
            rho = float(d.max())  # Gershgorin on C
    return s - rho - pad, s + rho + pad
