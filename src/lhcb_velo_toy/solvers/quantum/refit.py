"""Refitting the polynomial coefficients to measured events.

Provenance
----------
Promoted from the QSVT working folder (2026-09-07):

* ``chebyshev_moments`` / ``refit_coefficients`` — the ``fitted_moment``
  family of ``QSVT/Efficiency_Frontier/02_hamiltonian_setouts.py`` and
  ``03_highT_frontier.py`` (functions ``cheb_moments``, ``fit_degree``;
  originally ``QSVT/Codesign/06_occupancy_fitted_check.py``).
* ``refit_binned`` — the ``fitted_ridge`` / ``fitted_binned`` family of
  ``QSVT/Efficiency_Frontier/01_degree_response_frontier.py`` (``fit_binned``,
  the ``QSVT/Codesign/04_fit_comb_to_measured_spectrum.py`` recipe on a
  :func:`~lhcb_velo_toy.solvers.quantum.spectral_filter.mode_system`).

What it is
----------
The designed comb places its passes at the closed-form true-track lines. On
noisy events the populated spectrum moves (fragments, tangles, the modified
operators), and the paper's sections 9-10 replace the *target* step of the
design by a measurement: the response's Chebyshev coefficients (or its
binned values) become the unknowns of a ridge-regularised least-squares
problem that pushes the false amplitudes to zero while holding the true
amplitudes to the classical profile,

    minimise  ||M_F c||^2 + mu ||M_T c - x_ref,T||^2 + reg ||c||^2 ,

with ``M = [T_k(X) b]_k`` the Chebyshev moment matrix, ``mu`` selected by the
efficiency at a false-rate target. It is a SUPERVISED fit: it needs the
per-segment truth of the event it is fitted on. Its use is (a) to measure
how much a polynomial of a given degree can buy on a given spectrum, and
(b) the event-to-event transfer test — fit on one event, apply the returned
Chebyshev series through :class:`~lhcb_velo_toy.solvers.quantum.QSVT.QSVT`
(``poly=``) on another.

Status (2026-09-03 QSVT catch-up): the refit results are held back from the
first paper until the refit audit and the transfer check close. This module
is the audited code path for both; it is not a production filter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
from numpy.polynomial import chebyshev as _cheb

__all__ = [
    "DEFAULT_MUS",
    "RefitResult",
    "chebyshev_moments",
    "refit_coefficients",
    "refit_binned",
]

# The study's ridge-strength grid (13 points, 3e-3 .. 30).
DEFAULT_MUS = np.geomspace(3e-3, 30.0, 13)


@dataclass
class RefitResult:
    """Outcome of :func:`refit_coefficients`.

    ``coef`` are Chebyshev coefficients on ``domain`` (raw fit scale);
    ``poly`` is the same series rescaled to ``max|p| <= 0.95`` on the
    domain (QSVT-realisable, usable as ``QSVT(A, b, poly=result.poly)``);
    ``amplitudes`` are ``|M c|`` on the fitted event; ``mu`` the selected
    ridge strength; ``eff_at_far`` the selection score (best efficiency at
    ``far <= far_target``) and ``far_target`` the target it was scored at.
    """
    coef: np.ndarray
    domain: tuple
    mu: float
    amplitudes: np.ndarray
    eff_at_far: float = 0.0
    far_target: float = 0.01
    degree: int = 0

    @property
    def poly(self):
        p = _cheb.Chebyshev(self.coef, domain=list(self.domain))
        x = np.linspace(self.domain[0], self.domain[1], 4000)
        mx = float(np.max(np.abs(p(x))))
        return p / (mx / 0.95) if mx > 0 else p


def chebyshev_moments(A, b, domain, dmax: int, dtype=np.float32) -> np.ndarray:
    """The moment matrix ``M[:, k] = T_k(X) b``, ``k = 0..dmax``.

    ``X = (2A - (lo + hi) I) / (hi - lo)`` for ``domain = (lo, hi)``, which
    must contain the spectrum of ``A`` (pad the measured extremes; the
    studies used a 2 % half-width pad). The recursion runs in float64; the
    stored matrix defaults to float32 (``n x (dmax + 1)`` — 2.5 GB at
    ``T = 1000``, ``dmax = 80`` in float64).
    """
    A = A.tocsr() if sp.issparse(A) else sp.csr_matrix(np.asarray(A, float))
    lo, hi = map(float, domain)
    sc, sh = 2.0 / (hi - lo), (lo + hi) / (hi - lo)
    n = A.shape[0]
    M = np.empty((n, int(dmax) + 1), dtype=dtype)
    t_prev = np.asarray(b, dtype=np.float64).ravel()
    M[:, 0] = t_prev
    if dmax >= 1:
        t_cur = sc * (A @ t_prev) - sh * t_prev
        M[:, 1] = t_cur
        for k in range(2, int(dmax) + 1):
            t_next = 2.0 * (sc * (A @ t_cur) - sh * t_cur) - t_prev
            t_prev, t_cur = t_cur, t_next
            M[:, k] = t_cur
    return M


def _eff_at_far(x: np.ndarray, truth: np.ndarray, far_target: float) -> float:
    """Best segment efficiency reachable at ``far <= far_target`` over all thresholds.

    The selection score of the studies' ``fit_degree`` / ``fit_binned``
    (scale-free; the matched-efficiency false rate floods identically for
    every ``mu`` on the fragment floor, so it cannot rank the fits).
    """
    x = np.abs(np.asarray(x, np.float64).ravel())
    truth = np.asarray(truth, bool)
    n_true = int(truth.sum())
    if n_true == 0:
        return 0.0
    ts = np.sort(x[truth])[::-1]
    xs, xt = np.sort(x), np.sort(x[truth])
    nact = len(x) - np.searchsorted(xs, ts, side="left")
    ntru = n_true - np.searchsorted(xt, ts, side="left")
    far = np.where(nact > 0, (nact - ntru) / np.maximum(nact, 1), np.nan)
    eff = ntru / n_true
    ok = far <= far_target
    return float(eff[ok].max()) if ok.any() else 0.0


def _gram64(X: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    """``X.T @ X`` accumulated in float64 by row chunks (X may be float32)."""
    d = X.shape[1]
    G = np.zeros((d, d))
    for i in range(0, X.shape[0], chunk):
        c = X[i:i + chunk].astype(np.float64)
        G += c.T @ c
    return G


def refit_coefficients(M: np.ndarray, truth: np.ndarray, x_ref: np.ndarray,
                       degree: int, domain, mus=None, far_target: float = 0.01,
                       reg: float = 1e-10) -> RefitResult:
    """Coefficient-space ridge refit, ``mu`` selected by efficiency at a false-rate target.

    Parameters
    ----------
    M : moment matrix from :func:`chebyshev_moments` (columns ``0..dmax``).
    truth : per-segment truth mask of the fitted event.
    x_ref : reference amplitudes the true segments are held to (the
        classical solution, ``|A^{-1} b|``).
    degree : polynomial degree to fit (``<= dmax``).
    domain : the ``(lo, hi)`` the moments were built on (recorded on the result).
    mus : ridge-strength grid (default :data:`DEFAULT_MUS`).
    far_target : the selection axis — the ``mu`` with the best efficiency at
        ``far <= far_target`` wins (the study's ``eff_f010``; the
        matched-efficiency false rate floods identically for every ``mu`` on
        the fragment floor, so it cannot rank the fits there).
    """
    truth = np.asarray(truth, bool)
    x_ref = np.asarray(x_ref, dtype=np.float64)
    mus = DEFAULT_MUS if mus is None else np.asarray(mus, dtype=np.float64)
    Md = M[:, :int(degree) + 1]
    MT, MF = Md[truth], Md[~truth]
    GF, GT = _gram64(MF), _gram64(MT)
    bT = MT.astype(np.float64).T @ x_ref[truth]
    regI = reg * np.trace(GF) / (degree + 1) * np.eye(degree + 1)
    best = None
    for mu in mus:
        c = np.linalg.solve(GF + mu * GT + regI, mu * bT)
        x = np.abs(Md @ c.astype(Md.dtype)).astype(np.float64)
        score = _eff_at_far(x, truth, far_target)
        if best is None or score > best[0]:
            best = (score, float(mu), c, x)
    score, mu, c, x = best
    return RefitResult(coef=np.asarray(c, dtype=np.float64),
                       domain=(float(domain[0]), float(domain[1])),
                       mu=mu, amplitudes=x, eff_at_far=score,
                       far_target=float(far_target), degree=int(degree))


def refit_binned(sysd: dict, truth: np.ndarray, x_ref: np.ndarray,
                 bin_w: float = 0.02, mus=None, far_target: float = 0.01,
                 s: Optional[float] = None, delta: Optional[float] = None,
                 iso_weight_scale: float = 30.0, reg: float = 1e-8
                 ) -> tuple[np.ndarray, np.ndarray, float]:
    """Binned-response ridge refit on a per-component mode system.

    The Codesign/04 recipe: the response is piecewise constant on bins of
    width ``bin_w`` over the populated spectrum; the design matrix
    ``D = U diag(w) onehot`` maps bin values to amplitudes exactly; isolated
    false segments enter as one weighted row at the bin of ``s``; ``mu`` is
    selected as in :func:`refit_coefficients`.

    ``sysd`` is the dict from
    :func:`~lhcb_velo_toy.solvers.quantum.spectral_filter.mode_system` (built
    with the physical ``b`` and ``iso_lambda = s``). Returns
    ``(p_bins, edges, mu)`` with ``max |p_bins| = 1``; realise it with
    :func:`~lhcb_velo_toy.solvers.quantum.spectral_filter.binned_response`
    (exact) or ``chebyshev_realize`` (degree-limited).
    """
    from lhcb_velo_toy.solvers.quantum.spectral_filter import (
        apply_response, binned_response,
    )

    truth = np.asarray(truth, bool)
    x_ref = np.asarray(x_ref, dtype=np.float64)
    mus = DEFAULT_MUS if mus is None else np.asarray(mus, dtype=np.float64)
    iso = sysd["iso"]
    if s is None:
        if not iso.any():
            raise ValueError("pass s explicitly: no isolated segments to infer it from")
        s = float(sysd["iso_lam"][0])
    if delta is None:
        delta = float(sysd["b"][0])
    edges = np.arange(sysd["lam_min"] - bin_w, sysd["lam_max"] + 2 * bin_w, bin_w)
    nb = len(edges) - 1
    bi = np.clip(np.searchsorted(edges, sysd["lam"]) - 1, 0, nb - 1)
    onehot = sp.csr_matrix((sysd["w"], (np.arange(len(bi)), bi)),
                           shape=(len(bi), nb))
    D = np.asarray((sysd["U"] @ onehot).todense())
    incomp = ~iso
    tru = truth & incomp
    fal = (~truth) & incomp
    DT, DF = D[tru], D[fal]
    y_true = x_ref[tru]
    iso_row = np.zeros(nb)
    iso_row[np.clip(np.searchsorted(edges, s) - 1, 0, nb - 1)] = delta
    wiso = np.sqrt(float(np.count_nonzero(iso & ~truth))) / iso_weight_scale
    DF_full = np.vstack([DF, wiso * iso_row])
    GF, GT = DF_full.T @ DF_full, DT.T @ DT
    bT = DT.T @ y_true
    regI = reg * np.eye(nb)
    best = None
    for mu in mus:
        p_b = np.linalg.solve(GF + mu * GT + regI, mu * bT)
        p_b = p_b / max(np.abs(p_b).max(), 1e-12)
        x_o = apply_response(sysd, binned_response(p_b, edges))
        score = _eff_at_far(x_o, truth, far_target)
        if best is None or score > best[0]:
            best = (score, float(mu), p_b)
    _, mu, p_opt = best
    return p_opt, edges, mu
