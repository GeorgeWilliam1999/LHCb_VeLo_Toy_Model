"""Additional QSVT response families: the normalized comb, the one-bit cosine,
and the trigonometric (QETU) comb.

Provenance
----------
Promoted from the QSVT working folder (2026-09-07):

* ``design_normalized_comb`` / ``normalized_coupling`` /
  ``solve_normalized_comb`` — the ``normalized_half_comb`` family of
  ``QSVT/Efficiency_Frontier/01_degree_response_frontier.py`` (functions
  ``design_norm_comb``, ``discriminant``) and of ``02_hamiltonian_setouts.py``
  / ``03_highT_frontier.py`` (Block-Encoding write-up III: the alpha = 1
  normalized +-1/2 comb).
* ``one_bit_response`` / ``design_one_bit_cosine`` — the ``1bqf_cos`` /
  ``onebqf_cos`` family of ``01_degree_response_frontier.py`` and
  ``06b_highT_onebqf.py`` (the exact cosine response of the one-bit filter,
  realised as a degree-60 Chebyshev series so it runs through the same
  matrix-free evaluator as every other response).
* ``design_trig_comb`` — ``trig_comb`` of
  ``QSVT/Paper_planning/section4_qsvt/02_resource_model.py`` (the QETU
  realisation of the comb over the one-bit filter's own primitive
  ``exp(-iAt)``; paper section 4.2, table of realisations).

All three are *responses* f(lambda): weights applied mode by mode to the
segment Hamiltonian. The first two are Chebyshev series and can be handed to
:class:`~lhcb_velo_toy.solvers.quantum.QSVT.QSVT` as ``poly=``; the
trigonometric comb is a cosine series in ``lambda * t`` and is applied
through :func:`~lhcb_velo_toy.solvers.quantum.spectral_filter.apply_response`.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
from numpy.polynomial import chebyshev as _cheb

__all__ = [
    "normalized_coupling",
    "design_normalized_comb",
    "solve_normalized_comb",
    "one_bit_response",
    "design_one_bit_cosine",
    "TrigResponse",
    "design_trig_comb",
    "p4_lines",
]


def p4_lines(s: float = 4.0) -> tuple[float, ...]:
    """The four true-track (P4 chain) eigenvalue lines ``s - 2 cos(k pi / 5)``."""
    s = float(s)
    return tuple(s - 2.0 * np.cos(k * np.pi / 5.0) for k in (1, 2, 3, 4))


# --------------------------------------------------------------------------
# normalized +-1/2 comb on the degree-normalised coupling
# --------------------------------------------------------------------------
def normalized_coupling(A) -> sp.csr_matrix:
    """The degree-normalised coupling ``D^{-1/2} |C| D^{-1/2}`` of ``A``.

    ``C`` is the off-diagonal support of ``A`` (the continuation graph, taken
    in absolute value) and ``D`` its degree matrix. The result is the graph's
    normalised adjacency: spectrum in ``[-1, 1]``, gamma-invariant, with the
    true-track P4 chain at ``+-1/2`` — the operator the normalized comb is
    designed for (Block-Encoding III, the alpha = 1 encoding). Isolated
    segments (degree 0) map to 0.
    """
    A = A.tocsr() if sp.issparse(A) else sp.csr_matrix(np.asarray(A, float))
    Off = A.copy().tolil()
    Off.setdiag(0)
    C = sp.csr_matrix(abs(Off.tocsr()))
    deg = np.asarray(C.sum(1)).ravel()
    inv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-300)), 0.0)
    return sp.csr_matrix(sp.diags(inv) @ C @ sp.diags(inv))


def design_normalized_comb(degree: int = 40, hw: float = 0.10,
                           ngrid: int = 8000):
    """Chebyshev least-squares fit of the +-1/2 comb on ``[-1, 1]``.

    Gaussian passes of half-width ``hw`` at ``x = +1/2`` and ``x = -1/2``
    (the P4 chain's eigenvalues on the normalised adjacency), zero
    elsewhere; rescaled to ``max |p| <= 0.95`` if needed. Apply it to
    :func:`normalized_coupling` with a uniform input, not to ``A``.

    Returns a ``numpy.polynomial.chebyshev.Chebyshev`` with domain ``[-1, 1]``.
    """
    x = np.linspace(-1.0, 1.0, ngrid)
    y = np.zeros_like(x)
    for m in (0.5, -0.5):
        y = np.maximum(y, np.exp(-(((x - m) / hw) ** 2)))
    p = _cheb.Chebyshev.fit(x, y, int(degree), domain=[-1.0, 1.0])
    mx = float(np.max(np.abs(p(x))))
    return p / (mx / 0.95) if mx > 1.0 else p


def _cheb_apply(poly, M, v: np.ndarray) -> np.ndarray:
    """Signed ``poly(M) v`` by the Chebyshev recursion on the poly's domain."""
    lo, hi = map(float, poly.domain)
    sc, sh = 2.0 / (hi - lo), (lo + hi) / (hi - lo)
    c = np.asarray(poly.coef, float)
    t_prev = np.asarray(v, float)
    y = c[0] * t_prev
    if len(c) > 1:
        t_cur = sc * (M @ t_prev) - sh * t_prev
        y = y + c[1] * t_cur
        for k in range(2, len(c)):
            t_next = 2.0 * (sc * (M @ t_cur) - sh * t_cur) - t_prev
            t_prev, t_cur = t_cur, t_next
            y = y + c[k] * t_cur
    return y


def solve_normalized_comb(A, degree: int = 40, hw: float = 0.10,
                          poly=None) -> tuple[np.ndarray, object]:
    """Amplitudes ``|p(D) 1|`` of the normalized comb on the coupling of ``A``.

    The Efficiency-Frontier ``normalized_half_comb`` family verbatim: the
    comb of :func:`design_normalized_comb` (or ``poly``) applied by the
    Chebyshev recursion to :func:`normalized_coupling` with the all-ones
    input. Returns ``(amplitudes, poly)``; amplitudes are NOT normalised
    (isolated segments read ``|p(0)|``), so put them on a scale with
    :func:`~lhcb_velo_toy.analysis.working_points.rescale_to_signal` or use
    the scale-free matched-efficiency metrics.
    """
    Dm = normalized_coupling(A)
    p = design_normalized_comb(degree=degree, hw=hw) if poly is None else poly
    return np.abs(_cheb_apply(p, Dm, np.ones(Dm.shape[0]))), p


# --------------------------------------------------------------------------
# the one-bit filter's cosine as a response
# --------------------------------------------------------------------------
def one_bit_response(s: float = 4.0, t: Optional[float] = None,
                     signed: bool = False) -> Callable:
    """The one-bit filter's spectral response ``cos(lambda t / 2)``.

    With the canonical evolution time ``t = pi / s`` the single zero sits at
    ``lambda = s``, the isolated-segment line. ``signed=False`` (the study
    convention) returns ``|cos|``, the readout of the real-cosine
    idealisation; ``signed=True`` returns the signed cosine.
    """
    s = float(s)
    tt = np.pi / s if t is None else float(t)

    def f(lam, tt=tt, signed=signed):
        v = np.cos(np.asarray(lam, dtype=np.float64) * tt / 2.0)
        return v if signed else np.abs(v)

    f.t = tt
    return f


def design_one_bit_cosine(degree: int = 60, s: float = 4.0, domain=None,
                          t: Optional[float] = None, ngrid: int = 4000,
                          signed: bool = False):
    """Chebyshev realisation of the one-bit cosine response on ``domain``.

    The ``onebqf_cos`` family of ``06b_highT_onebqf.py`` verbatim at the
    defaults (``|cos(pi lambda / 2 s)|``, degree 60, 4000-point grid): the
    exact one-bit response run through the same matrix-free evaluator as
    the comb, so the two can be compared on identical events with identical
    machinery. Not rescaled: ``|cos| <= 1`` already. The ``|cos|`` target has
    a kink at the notch, which limits the degree-60 fit to a cosine of ~0.998
    to the exact response; ``signed=True`` fits the smooth signed cosine to
    machine precision (the readout takes ``|.|`` afterwards anyway).

    Returns a ``numpy.polynomial.chebyshev.Chebyshev`` on ``domain``
    (default ``(s - 3.8, s + 3.8)``), usable as ``QSVT(A, b, poly=...)``.
    """
    s = float(s)
    if domain is None:
        domain = (s - 3.8, s + 3.8)
    lo, hi = map(float, domain)
    f = one_bit_response(s=s, t=t, signed=signed)
    gr = np.linspace(lo, hi, int(ngrid))
    u = (2.0 * gr - (lo + hi)) / (hi - lo)
    cf = _cheb.chebfit(u, f(gr), int(degree))
    return _cheb.Chebyshev(cf, domain=[lo, hi])


# --------------------------------------------------------------------------
# the trigonometric comb over exp(-iAt) (QETU)
# --------------------------------------------------------------------------
class TrigResponse:
    """A cosine series ``f(lambda) = sum_k c_k cos(k lambda t)``.

    Callable on arrays of eigenvalues; ``coef`` (length ``K + 1``), ``t`` and
    ``K`` are attributes. ``l1`` is the coefficient one-norm.
    """

    def __init__(self, coef: np.ndarray, t: float):
        self.coef = np.asarray(coef, dtype=np.float64)
        self.t = float(t)

    @property
    def K(self) -> int:
        return len(self.coef) - 1

    @property
    def l1(self) -> float:
        return float(np.sum(np.abs(self.coef)))

    def __call__(self, lam):
        lam = np.asarray(lam, dtype=np.float64)
        return np.cos(np.outer(lam.ravel() * self.t, np.arange(self.K + 1))) \
            @ self.coef

    def __repr__(self) -> str:
        return f"TrigResponse(K={self.K}, t={self.t:.5f})"


def design_trig_comb(K: int = 40, t: float = np.pi / 7.9, s: float = 4.0,
                     hw: float = 0.18, domain=None, lines=None,
                     invert: bool = True, ngrid: int = 6000) -> TrigResponse:
    """Least-squares cosine series in ``theta = lambda t`` fitted to the comb.

    The comb target of :func:`~lhcb_velo_toy.solvers.quantum.QSVT.design_line_comb_inverse`
    (Gaussian passes at the P4 lines, ``1/lambda``-weighted) fitted by
    ``sum_{k=0}^{K} c_k cos(k lambda t)`` on ``domain`` and scaled to
    ``max |f| = 1``. This is the response the QETU realisation applies with
    ``K`` controlled evolutions ``exp(-+ i A t)`` on one ancilla — the one-bit
    filter's own primitive and width (``n_sys + 2``). ``t = pi / 7.9`` keeps
    ``lambda t`` inside one period over the production domain ``(0.2, 7.8)``.
    The rotation angles of the QETU sequence are not synthesised here.
    """
    s = float(s)
    if domain is None:
        domain = (s - 3.8, s + 3.8)
    if lines is None:
        lines = p4_lines(s)
    x = np.linspace(float(domain[0]), float(domain[1]), int(ngrid))
    y = np.zeros_like(x)
    for m in lines:
        y = np.maximum(y, np.exp(-(((x - m) / hw) ** 2)))
    if invert:
        y = y / np.clip(x, max(min(lines), 1e-6), None)
    B = np.cos(np.outer(x * t, np.arange(int(K) + 1)))
    c, *_ = np.linalg.lstsq(B, y, rcond=None)
    f = TrigResponse(c, t)
    mx = float(np.max(np.abs(f(x))))
    return TrigResponse(c / mx, t)
