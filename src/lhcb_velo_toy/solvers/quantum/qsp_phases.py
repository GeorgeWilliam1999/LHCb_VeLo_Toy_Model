"""The phase-sequence realisation of a polynomial response (QSVT proper).

Provenance
----------
Promoted from ``QSVT/Paper_planning/section4_qsvt/01_phase_factors.py`` and
``01b_convention_check.py`` (paper section 4.2, "the phase sequence";
2026-09-07). The literature's QSVT applies a real polynomial ``q`` of
definite parity with ``|q| <= 1`` on ``[-1, 1]`` by ``d`` applications of a
block encoding of ``X`` interleaved with single-qubit phase rotations on a
signal ancilla,

    U_Phi = e^{i phi_0 Z} prod_{k=1..d} [ U e^{i phi_k Z} ],
    (<0| (x) I) U_Phi (|0> (x) I) = P(X),   Re P = q .

Angle finding is the symmetric-phase-factor formulation of Dong, Meng,
Whaley and Lin, Phys. Rev. A 103, 042419 (2021): minimise the squared
mismatch between ``Re <0|U_Phi(x)|0>`` and ``q(x)`` on Chebyshev nodes with
the phases constrained symmetric (``phi_k = phi_{d-k}``), L-BFGS-B. Degree
40 takes seconds (even part) to minutes (odd part).

Two conventions are involved and the conversion between them matters:

* ``"wx"``: the signal operator is ``W(x) = [[x, i sqrt(1-x^2)],
  [i sqrt(1-x^2), x]]`` — the convention the optimiser works in.
* ``"reflection"``: the dilation ``U = [[X, sqrt(I-X^2)], [sqrt(I-X^2), -X]]``
  (the block encoding the package simulates) acts in every eigenplane as
  the reflection ``R(x) = [[x, sqrt(1-x^2)], [sqrt(1-x^2), -x]]``. Phases
  found in the wx convention convert by (Martyn, Rossi, Tan, Chuang 2021,
  eq. 17)  ``phi_0 += (2d-1) pi/4``, ``phi_k -= pi/2`` (0 < k < d),
  ``phi_d -= pi/4``. Phases returned by :func:`find_phase_factors` are in
  the reflection convention by default, ready for
  :func:`apply_phase_sequence`.

The production comb has no definite parity; :func:`parity_split` divides it
into even and odd parts, each rescaled to peak at one, and
:func:`phase_sequence_solve` runs both sequences and recombines them — the
real, indefinite-parity polynomial then costs one more ancilla (paper
section 4.2: width ``n_sys + 4``, ``4d`` calls per shot, success
``||q(X)u||^2 / (alpha_e + alpha_o)^2`` with ``q = p / max|p|``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.polynomial import chebyshev as _cheb
from scipy import sparse as _sp

__all__ = [
    "parity_split",
    "qsp_signal",
    "wx_to_reflection",
    "find_phase_factors",
    "apply_phase_sequence",
    "phase_sequence_solve",
]


def parity_split(coef, grid: int = 20001) -> dict:
    """Split a Chebyshev coefficient vector into unit-peak even and odd parts.

    Returns a dict with ``c`` (the series rescaled to ``max|p| = 1`` on
    ``[-1, 1]``), ``ce`` / ``co`` (even / odd parts of ``c`` each rescaled to
    peak at one), ``alpha_e`` / ``alpha_o`` (their peaks before rescaling),
    ``scale`` (``max|p|`` of the input) and ``parity_penalty``
    ``(alpha_e + alpha_o)^2`` — the success-probability price of the split.
    """
    c = np.asarray(coef, dtype=np.float64).copy()
    x = np.linspace(-1.0, 1.0, int(grid))
    scale = float(np.max(np.abs(_cheb.chebval(x, c))))
    c = c / scale
    ce = c.copy()
    ce[1::2] = 0.0
    co = c.copy()
    co[0::2] = 0.0
    ae = float(np.max(np.abs(_cheb.chebval(x, ce))))
    ao = float(np.max(np.abs(_cheb.chebval(x, co))))
    return dict(
        c=c,
        ce=ce / ae if ae > 0 else ce,
        co=co / ao if ao > 0 else co,
        alpha_e=ae, alpha_o=ao, scale=scale,
        parity_penalty=(ae + ao) ** 2,
    )


def qsp_signal(phis, x, convention: str = "reflection") -> np.ndarray:
    """``<0|U_Phi(x)|0>`` (complex) for an array of signal values ``x``.

    ``convention="wx"`` uses ``W(x) = [[x, i s], [i s, x]]``;
    ``"reflection"`` uses ``R(x) = [[x, s], [s, -x]]``, ``s = sqrt(1-x^2)``.
    The real part is the polynomial ``q(x)`` when the phases match the
    convention.
    """
    phis = np.asarray(phis, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).ravel()
    sq = np.sqrt(np.clip(1.0 - x * x, 0.0, None))
    U = np.zeros((len(x), 2, 2), dtype=complex)
    U[:, 0, 0] = np.exp(1j * phis[0])
    U[:, 1, 1] = np.exp(-1j * phis[0])
    W = np.zeros_like(U)
    W[:, 0, 0] = x
    if convention == "wx":
        W[:, 1, 1] = x
        W[:, 0, 1] = 1j * sq
        W[:, 1, 0] = 1j * sq
    elif convention == "reflection":
        W[:, 1, 1] = -x
        W[:, 0, 1] = sq
        W[:, 1, 0] = sq
    else:
        raise ValueError("convention must be 'wx' or 'reflection'")
    for ph in phis[1:]:
        E = np.zeros_like(U)
        E[:, 0, 0] = np.exp(1j * ph)
        E[:, 1, 1] = np.exp(-1j * ph)
        U = U @ W @ E
    return U[:, 0, 0]


def wx_to_reflection(phis) -> np.ndarray:
    """Convert wx-convention phases to the reflection (dilation) convention."""
    r = np.asarray(phis, dtype=np.float64).copy()
    d = len(r) - 1
    r[0] += (2 * d - 1) * np.pi / 4.0
    if d > 1:
        r[1:d] -= np.pi / 2.0
    r[d] -= np.pi / 4.0
    return r


def _parity_of(coef, tol: float = 1e-12) -> str:
    c = np.asarray(coef, dtype=np.float64)
    odd_zero = np.all(np.abs(c[1::2]) <= tol)
    even_zero = np.all(np.abs(c[0::2]) <= tol)
    if odd_zero and not even_zero:
        return "even"
    if even_zero and not odd_zero:
        return "odd"
    if odd_zero and even_zero:
        return "zero"
    return "mixed"


def find_phase_factors(coef, degree: Optional[int] = None,
                       convention: str = "reflection", tol: float = 1e-12,
                       n_trials: int = 6, seed: int = 0, grid: int = 4001,
                       maxiter: int = 20000) -> tuple[np.ndarray, float]:
    """Symmetric QSP phase factors for a definite-parity Chebyshev series.

    Parameters
    ----------
    coef : Chebyshev coefficients on ``[-1, 1]`` of a real polynomial ``q``
        with definite parity and ``|q| <= 1`` (use :func:`parity_split`).
    degree : int, optional
        The QSP degree ``d`` (default: the series' own degree, i.e. the last
        non-zero coefficient). Must have the parity of ``q``.
    convention : ``"reflection"`` (default, for :func:`apply_phase_sequence`)
        or ``"wx"``.
    tol, n_trials, seed, maxiter : optimiser controls — the loss must drop
        below ``tol`` or the best of ``n_trials`` perturbed restarts is kept.

    Returns ``(phases, max_err)`` with ``phases`` of length ``d + 1`` and
    ``max_err = max |Re <0|U_Phi|0> - q|`` on a ``grid``-point check.
    """
    from scipy.optimize import minimize

    c = np.asarray(coef, dtype=np.float64)
    par = _parity_of(c)
    if par == "mixed":
        raise ValueError("q must have definite parity — use parity_split first")
    if par == "zero":
        raise ValueError("q is identically zero")
    nz = np.flatnonzero(np.abs(c) > 1e-12)
    d = int(nz[-1]) if degree is None else int(degree)
    if (d % 2 == 0) != (par == "even"):
        raise ValueError(f"degree {d} does not match the {par} parity of q")
    nodes = np.cos(np.pi * (np.arange(2 * d + 2) + 0.5) / (2 * d + 2))
    target = _cheb.chebval(nodes, c)
    nfree = d // 2 + 1

    def expand(theta):
        phis = np.zeros(d + 1)
        for k in range(nfree):
            phis[k] = theta[k]
            phis[d - k] = theta[k]
        return phis

    def loss(theta):
        return float(np.sum((qsp_signal(expand(theta), nodes, "wx").real
                             - target) ** 2))

    theta0 = np.zeros(nfree)
    theta0[0] = np.pi / 4.0
    best = None
    for trial in range(int(n_trials)):
        th = theta0 if trial == 0 else \
            theta0 + np.random.default_rng(seed + trial).normal(0, 0.3, nfree)
        res = minimize(loss, th, method="L-BFGS-B",
                       options=dict(maxiter=int(maxiter), ftol=1e-18,
                                    gtol=1e-14, maxcor=50))
        if best is None or res.fun < best.fun:
            best = res
        if best.fun < tol:
            break
    phis = expand(best.x)
    if convention == "reflection":
        phis = wx_to_reflection(phis)
    elif convention != "wx":
        raise ValueError("convention must be 'wx' or 'reflection'")
    xg = np.linspace(-1.0, 1.0, int(grid))
    err = float(np.max(np.abs(qsp_signal(phis, xg, convention).real
                              - _cheb.chebval(xg, c))))
    return phis, err


def apply_phase_sequence(X, u, phis) -> np.ndarray:
    """Run the phase sequence on the dilation block encoding, densely.

    ``X`` is real symmetric with ``||X|| <= 1``; the block encoding is the
    dilation ``U = [[X, sqrt(I-X^2)], [sqrt(I-X^2), -X]]`` on (dilation,
    system); the signal rotations ``e^{i phi (2 Pi - I)}`` phase the
    dilation-``|0>`` and ``|1>`` halves oppositely; ``U`` and ``U^dagger``
    alternate (identical here: ``U`` is real symmetric). ``phis`` must be in
    the reflection convention. Returns the system state on the
    dilation-``|0>`` branch (complex); its real part is ``q(X) u``.
    Dense: ``O(n^3)`` for the eigendecomposition, small systems only.
    """
    X = X.toarray() if _sp.issparse(X) else np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    w, V = np.linalg.eigh(X)
    w = np.clip(w, -1.0, 1.0)
    R = (V * np.sqrt(1.0 - w * w)) @ V.T
    U = np.block([[X, R], [R, -X]])
    sgn = np.concatenate([np.ones(n), -np.ones(n)])
    psi = np.concatenate([np.asarray(u, dtype=complex), np.zeros(n, dtype=complex)])
    psi = psi * np.exp(1j * phis[0] * sgn)
    for k in range(1, len(phis)):
        psi = U @ psi
        psi = psi * np.exp(1j * phis[k] * sgn)
    return psi[:n]


def phase_sequence_solve(X, u, coef, phases: Optional[dict] = None,
                         **find_kw) -> tuple[np.ndarray, float, dict]:
    """Apply a general (indefinite-parity) Chebyshev series by phase sequences.

    Splits ``coef`` with :func:`parity_split`, finds (or takes from
    ``phases = {"even": ..., "odd": ...}``, reflection convention) the two
    phase sequences, runs both with :func:`apply_phase_sequence` and
    recombines ``q(X) u = alpha_e Re psi_e + alpha_o Re psi_o``.

    Returns ``(amplitudes, success, info)`` in the QSVT readout convention:
    ``amplitudes = |y| / ||y||``, ``success = ||q(X) u||^2 / (alpha_e +
    alpha_o)^2`` (paper eq. for the phase sequence), and ``info`` holding the
    phases, the scalar-identity errors and the parity data.
    """
    parts = parity_split(coef)
    u = np.asarray(u, dtype=np.float64)
    y = np.zeros(len(u))
    info = dict(alpha_e=parts["alpha_e"], alpha_o=parts["alpha_o"],
                scale=parts["scale"], parity_penalty=parts["parity_penalty"],
                phases={}, max_err={})
    for name, cpart, alpha in (("even", parts["ce"], parts["alpha_e"]),
                               ("odd", parts["co"], parts["alpha_o"])):
        if alpha <= 1e-12:
            continue
        if phases is not None and name in phases:
            ph = np.asarray(phases[name], dtype=np.float64)
            xg = np.linspace(-1.0, 1.0, 4001)
            err = float(np.max(np.abs(qsp_signal(ph, xg).real
                                      - _cheb.chebval(xg, cpart))))
        else:
            ph, err = find_phase_factors(cpart, **find_kw)
        psi = apply_phase_sequence(X, u, ph)
        y = y + alpha * psi.real
        info["phases"][name] = ph
        info["max_err"][name] = err
    success = float(y @ y) / (parts["alpha_e"] + parts["alpha_o"]) ** 2
    amp = np.abs(y)
    nrm = np.linalg.norm(amp)
    return (amp / nrm if nrm > 0 else amp), success, info
