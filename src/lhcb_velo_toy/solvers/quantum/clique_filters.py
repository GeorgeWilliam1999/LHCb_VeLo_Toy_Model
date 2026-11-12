"""Spectral filters for the vertex-conditioned hit operator (the hit register).

Provenance
----------
New, 2026-11-12 (experiment B of the research plan "cheap and effective VELO
track finding", Notion 4155d544b9d98186919cd5024e93e8cb). The hit operator is
the per-vertex coupling matrix of ``Quantum_Track_Reconstruction/Vertexing/
Focusing_scan/scripts/07_full_sweep.py`` (``per_vertex_operator``): one basis
state per hit, ``C_ij = 1`` when hits i and j sit on different planes, are
assigned to the same vertex, and their directions from that vertex agree
within the coupling width. The ideal projector onto the lines {3, 4} is the
one used there (lines 3 and 4 = cliques of 4 and 5 hits).

The spectral picture
--------------------
A straight track of m hits from the vertex is a clique K_m of C, with
eigenvalues ``m - 1`` (the uniform state on the clique) and ``-1``
(multiplicity ``m - 1``). An accidental pair sits at ``+1`` and ``-1``, an
isolated hit at ``0``. With the uniform probe ``u = 1 / sqrt(n)`` a perfect
clique loads only its top line, so a filter that passes every line ``>= 2``
keeps every track of ``m >= 3`` hits and rejects pairs and isolated hits.

The responses here are functions of the eigenvalue ``c`` of C:

* :func:`ideal_line_projector` -- 1 on chosen integer lines (the {3, 4}
  baseline);
* :func:`ideal_step` -- 1 on every integer line above a threshold (the
  projector onto all cliques of ``m >= 3`` hits);
* :func:`soft_band` -- 1 on every eigenvalue above a threshold, integer or
  not (also passes broken cliques);
* :func:`design_step_polynomial` -- a bounded Chebyshev polynomial of the
  step on ``[-alpha, alpha]``, the implementable version;
* :func:`one_bit_clique` -- the one-bit filter on ``A = s I - C``, the
  degree-one baseline.

:func:`filter_hit_operator` applies any of them exactly (per connected
component eigendecomposition, through
:func:`~lhcb_velo_toy.solvers.quantum.spectral_filter.mode_system`) with the
uniform probe, and returns the per-hit output and the success probability.
:func:`tracks_from_filtered_hits` reads tracks from the surviving hits.
No circuit is simulated anywhere in this module.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

import numpy as np
import scipy.sparse as sp
from numpy.polynomial import chebyshev as _cheb
from scipy.sparse.csgraph import connected_components
from scipy.special import erf

from lhcb_velo_toy.solvers.quantum.spectral_filter import apply_response, mode_system

__all__ = [
    "ideal_line_projector",
    "ideal_step",
    "soft_band",
    "design_step_polynomial",
    "one_bit_clique",
    "filter_hit_operator",
    "tracks_from_filtered_hits",
]


# --------------------------------------------------------------------------
# responses
# --------------------------------------------------------------------------
def ideal_line_projector(lines: Iterable[float], tol: float = 1e-6) -> Callable:
    """Ideal projector onto chosen eigenvalue lines of the hit operator.

    Parameters
    ----------
    lines : iterable of float
        The eigenvalues to pass, e.g. ``(3, 4)`` for cliques of 4 and 5 hits.
    tol : float
        An eigenvalue counts as on a line when it is within ``tol`` of it.

    Returns
    -------
    callable
        ``f(c)`` = 1 on a line, 0 elsewhere.
    """
    ls = np.asarray(list(lines), dtype=np.float64)

    def f(c, ls=ls, tol=tol):
        c = np.asarray(c, dtype=np.float64)
        return (np.abs(c[..., None] - ls).min(axis=-1) < tol).astype(np.float64)

    f.kind = "ideal_lines"
    return f


def ideal_step(threshold: float = 1.5, tol: float = 1e-6) -> Callable:
    """Ideal projector onto every integer line above ``threshold``.

    With the default threshold 1.5 this passes the lines 2, 3, 4, ...: every
    exact clique of ``m >= 3`` hits. Non-integer eigenvalues (broken cliques)
    are rejected; :func:`soft_band` passes them too.

    Parameters
    ----------
    threshold : float
        Lowest eigenvalue passed (exclusive).
    tol : float
        Distance to the nearest integer below which an eigenvalue is on a line.

    Returns
    -------
    callable
        ``f(c)`` = 1 on an integer line ``> threshold``, 0 elsewhere.
    """

    def f(c, th=threshold, tol=tol):
        c = np.asarray(c, dtype=np.float64)
        r = np.round(c)
        return ((np.abs(c - r) < tol) & (r > th)).astype(np.float64)

    f.kind = "ideal_step"
    return f


def soft_band(lo: float = 1.5) -> Callable:
    """Pass every eigenvalue above ``lo``, on a line or not.

    The tolerant variant of :func:`ideal_step`: a clique with a missing or a
    displaced coupling has a top eigenvalue between the integer lines, and is
    still passed if it lies above ``lo``.

    Parameters
    ----------
    lo : float
        Lowest eigenvalue passed (exclusive).

    Returns
    -------
    callable
        ``f(c)`` = 1 for ``c > lo``, 0 elsewhere.
    """

    def f(c, lo=lo):
        return (np.asarray(c, dtype=np.float64) > lo).astype(np.float64)

    f.kind = "soft_band"
    return f


def design_step_polynomial(degree: int, alpha: float, threshold: float = 1.5,
                           width: Optional[float] = None, lo: Optional[float] = None,
                           grid: int = 4001) -> _cheb.Chebyshev:
    """Bounded degree-d Chebyshev polynomial of the step at ``threshold``.

    The target is the smoothed step ``(1 + erf((c - threshold) / width)) / 2``
    on ``[-alpha, alpha]``, where ``alpha`` bounds the spectrum of the hit
    operator (the block-encoding normalisation: the circuit sees ``C /
    alpha``). It is fitted by least squares in the Chebyshev basis of the
    domain (the :func:`~lhcb_velo_toy.solvers.quantum.spectral_filter.
    chebyshev_realize` recipe) and divided by the exact ``max |p|`` on the
    domain when that exceeds 1, so ``|p| <= 1`` on the whole domain. A sharp
    step would ring (Gibbs); the default ``width`` (domain length over
    ``2 degree``) is the transition a degree-d polynomial can resolve.

    The spectrum of C is nearly one-sided (cliques give ``m - 1`` and
    ``-1``), so ``lo`` can shrink the domain to ``[lo, alpha]``: the circuit
    then encodes ``(C - c0 I) / h`` with ``c0 = (lo + alpha) / 2`` and ``h =
    (alpha - lo) / 2``, an identity shift (one extra LCU term), and the same
    degree resolves a step about twice as sharp.

    Parameters
    ----------
    degree : int
        Polynomial degree d (d calls to the block encoding).
    alpha : float
        Upper end of the domain; must be ``>= lambda_max(C)``.
    threshold : float
        Centre of the step (1.5: between the pair line 1 and the 3-hit line 2).
    width : float, optional
        erf width of the target; default ``(alpha - lo) / (2 degree)``, which
        is ``alpha / degree`` on the symmetric domain.
    lo : float, optional
        Lower end of the domain; default ``-alpha``. Must be ``<=
        lambda_min(C)``.
    grid : int
        Fit grid size.

    Returns
    -------
    numpy.polynomial.chebyshev.Chebyshev
        Callable in eigenvalue space (domain ``[lo, alpha]``), with
        ``|p| <= 1`` on the domain. ``p.degree()`` is ``degree``.
    """
    d = int(degree)
    a = float(alpha)
    b = -a if lo is None else float(lo)
    if d < 1 or a <= b:
        raise ValueError("degree >= 1 and alpha > lo required")
    wid = (a - b) / (2 * d) if width is None else float(width)
    xs = np.linspace(b, a, int(grid))
    target = 0.5 * (1.0 + erf((xs - threshold) / wid))
    p = _cheb.Chebyshev.fit(xs, target, d, domain=[b, a])
    # exact max |p| on the domain: endpoints and the real critical points
    crit = p.deriv().roots()
    crit = crit[np.isreal(crit)].real
    crit = crit[(crit >= b) & (crit <= a)]
    sc = float(np.max(np.abs(p(np.r_[b, a, crit]))))
    if sc > 1.0:
        p = p / sc
    return p


def one_bit_clique(s: float = 4.0, signed: bool = True) -> Callable:
    """The one-bit filter on ``A = s I - C`` as a response in ``c``.

    The one-bit filter with ``t = pi / s`` has amplitude ``cos(lambda t / 2)``
    on the eigenvalue ``lambda = s - c`` of A (``responses.one_bit_response``),
    i.e. ``cos(pi (s - c) / (2 s))``: 1 at ``c = s``, 0 at ``c = 0`` (isolated
    hits), and 0 again at ``c = 2 s``.

    Parameters
    ----------
    s : float
        Diagonal shift (4 in the hit-register studies).
    signed : bool
        Return the signed cosine (the success probability uses its square).

    Returns
    -------
    callable
        ``f(c)``.
    """
    s = float(s)

    def f(c, s=s):
        v = np.cos(np.pi * (s - np.asarray(c, dtype=np.float64)) / (2.0 * s))
        return v if signed else np.abs(v)

    f.kind = "one_bit"
    return f


# --------------------------------------------------------------------------
# exact application
# --------------------------------------------------------------------------
def filter_hit_operator(C, response: Callable, probe: Optional[np.ndarray] = None,
                        sysd: Optional[dict] = None) -> dict:
    """Apply a response to the hit operator exactly, with the uniform probe.

    Parameters
    ----------
    C : sparse symmetric (n, n)
        Hit operator (zero diagonal, non-negative couplings).
    response : callable
        ``f(c)`` on the eigenvalues of C (any function of this module, or any
        callable that accepts an array).
    probe : (n,) array, optional
        Input state; default uniform ``1 / sqrt(n)`` over the real hits.
    sysd : dict, optional
        A precomputed ``mode_system(C, probe)``; pass it to evaluate many
        responses on one decomposition.

    Returns
    -------
    dict
        ``y`` -- signed output ``f(C) probe`` per hit;
        ``weight`` -- ``|y| sqrt(n)``, which is ``|f(m - 1)|`` on every hit of
        a perfect m-hit clique (so a readout threshold is in units of the
        response);
        ``P`` -- success probability over the real register, ``||y||^2``;
        ``P_padded`` -- the same on the ``2^ceil(log2 n)`` register with the
        probe uniform over it: padded rows of C are 0, so they add
        ``f(0)^2`` each, ``P_padded = (n P + (N - n) f(0)^2) / N``;
        ``P_pad_leak`` -- the padded-row part of ``P_padded``;
        ``n``, ``n_sys`` (``ceil(log2 n)``), ``rho`` (largest |eigenvalue|),
        ``max_comp`` (largest connected component), ``sysd``.
    """
    C = sp.csr_matrix(C, dtype=np.float64)
    n = C.shape[0]
    b = np.full(n, 1.0 / math.sqrt(n)) if probe is None else np.asarray(probe, float)
    if sysd is None:
        sysd = mode_system(C, b, iso_lambda=0.0)
    y = apply_response(sysd, response, signed=True)
    P = float(y @ y)
    ns = max(1, math.ceil(math.log2(max(n, 2))))
    N = 2 ** ns
    f0 = float(np.asarray(response(np.array([0.0]))).ravel()[0])
    leak = (N - n) * f0 ** 2 / N if probe is None else 0.0
    lam = sysd["lam"]
    return dict(y=y, weight=np.abs(y) * math.sqrt(n), P=P, P_padded=P * n / N + leak,
                P_pad_leak=leak, n=n, n_sys=ns,
                rho=float(np.max(np.abs(lam))) if lam.size else 0.0,
                max_comp=sysd["max_comp"], sysd=sysd)


def tracks_from_filtered_hits(C, weight: np.ndarray, threshold: float = 0.5,
                              min_hits: int = 3) -> list[np.ndarray]:
    """Track candidates from the hits that survive a filter.

    A hit survives when its output weight (``filter_hit_operator(...)
    ["weight"]``) is at least ``threshold``. Surviving hits are grouped by the
    connected components of C restricted to them; a group of at least
    ``min_hits`` hits is one candidate.

    Parameters
    ----------
    C : sparse symmetric (n, n)
        Hit operator.
    weight : (n,) array
        Per-hit output weight.
    threshold : float
        Minimum weight of a surviving hit.
    min_hits : int
        Minimum hits per candidate.

    Returns
    -------
    list of int arrays
        Hit indices (rows of C) of each candidate.
    """
    keep = np.flatnonzero(np.asarray(weight) >= threshold)
    if keep.size == 0:
        return []
    sub = sp.csr_matrix(C)[keep][:, keep]
    nc, lab = connected_components(sub, directed=False)
    order = np.argsort(lab, kind="stable")
    cut = np.flatnonzero(np.diff(lab[order])) + 1
    return [keep[g] for g in np.split(order, cut) if len(g) >= min_hits]
