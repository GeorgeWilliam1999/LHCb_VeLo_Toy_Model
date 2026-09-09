"""
QSVT-style polynomial filter solver for the segment Hamiltonian Ax = b.

Context
-------
TrackHHL applies full HHL to the segment linear system: multi-qubit QPE plus a
controlled rotation realise the spectral response f(lambda) ~ 1/lambda.  The
OneBQF (1-bit HHL) collapses the QPE register to a single clock qubit, which
realises the one-bit filter f(lambda) = cos(lambda*t/2): a single notch at
lambda = pi/t that erases the isolated false segments but cannot reach the
coupled failure modes (hubs, bridges).

This solver is the middle road: it applies an (almost) arbitrary *polynomial*
spectral response p(A) -- here a band-limited inverse that is ~1/lambda on the
true-track band, ~0 on the failure-mode eigenvalues.  It is implemented as a
linear combination of Chebyshev polynomials of a qubitized walk operator
(Childs-Kothari-Somma / Low-Chuang), which is exactly the QSVT family of
transformations but needs no phase-angle synthesis:

  *  U = [[X, sqrt(1-X^2)], [sqrt(1-X^2), -X]]  block-encodes  X = affine(A)
     (one ancilla; X is A rescaled so its spectrum lies in [-1, 1]),
  *  the walk operator W = (Z (x) I) U satisfies  <0|W^k|0> = T_k(X)  exactly,
  *  an LCU register of m = ceil(log2(d+1)) qubits sums  p(X) = sum_k c_k T_k(X)
     with the Chebyshev coefficients c_k of the designed filter.

Post-selecting the LCU register on |0..0> and the block-encoding ancilla on |0>
leaves  p(A)|b> / ||c||_1  on the system register: the polynomial-filtered
solution, in direct analogy with the OneBQF's post-selected cos-filtered solve.

Resources: ceil(log2 n) system qubits + 1 block-encoding ancilla + m LCU qubits;
d controlled walk-operator calls (the OneBQF is the d = 1 member of the same
family, cos = T_1).  Success probability ||p(X) b||^2 / ||c||_1^2.

Response families (the ``filter_design`` selector, or pass any Chebyshev
series as ``poly=``):

  "comb"     design_line_comb_inverse -- THE QSVT of the papers ("the comb"):
             narrow ~1/lambda passes at the four P4 true-track lines only,
             degree 40, half-width 0.18.  THE DEFAULT.
  "band"     design_band_limited_inverse -- the retired first design (~1/lambda
             on a contiguous band with notches); fails at density: the
             tangle/chain modes between the true lines fill the band.
  "minimax"  design_minimax_comb -- the same comb fitted in the Chebyshev norm
             (LP); reaches the metric floor at about half the degree.
  "cosine"   responses.design_one_bit_cosine -- the one-bit filter's own
             response |cos(pi lambda / 2s)| as a degree-60 series, so the 1BQF
             and the comb run through one evaluator on identical events.
  (responses.design_normalized_comb / solve_normalized_comb: the +-1/2 comb on
  the degree-normalised coupling; responses.design_trig_comb: the QETU cosine
  series over exp(-iAt); refit.refit_coefficients: coefficients refit to a
  measured event -- see those modules.)

Three realisations of the same response (paper section 4.2; `resources()`):

  LCU of walk powers  -- the build above: 1 + m ancillas, 2^m - 1 walk steps
                         (d with an optimal SELECT), success ||p(X)u||^2/||c||_1^2;
                         no phases, every operator real.
  phase sequence      -- QSVT proper (qsp_phases): d applications of the block
                         encoding with phase rotations on one signal qubit;
                         indefinite-parity polynomials split even/odd (+1
                         ancilla, real-part trick +1): n_sys + 4 qubits, 4d
                         calls, success ||q(X)u||^2/(alpha_e + alpha_o)^2 with
                         q = p/max|p|.  `solve_phase_sequence()` runs it densely.
  QETU trig comb      -- the comb as a cosine series over the 1BQF's own
                         primitive exp(-iAt): n_sys + 2 qubits, K calls
                         (responses.design_trig_comb; angles not synthesised).

Backends that compute the LCU response exactly:

  solve_statevector()  -- exact matrix-free emulation of the circuit semantics
                          via the Chebyshev recursion (sparse matvecs only);
                          scales to full events with millions of segments.
  solve_eigh()         -- the same numbers from one per-component
                          eigendecomposition (spectral_filter.mode_system),
                          reusable across many candidate responses.
  build_circuit() /
  run_circuit()        -- the explicit qiskit circuit (StatePreparation + LCU +
                          controlled walk powers), simulated exactly with
                          qiskit.quantum_info.Statevector; for small systems.
                          Validates that the matrix-free backend is what the
                          circuit computes.  (The only path that needs qiskit.)
  solve_phase_sequence() -- the phase-sequence realisation, dense, small
                          systems; validates the QSVT-proper circuit.

The returned solution follows the OneBQF convention: (|amplitudes| normalised,
success probability), truncated to the original dimension.
"""

import math

import numpy as np
from numpy.polynomial import chebyshev as _cheb
from scipy import sparse as _sp
from scipy.sparse import linalg as _spla

# Spectral structure of the segment Hamiltonian A = sI - C with s = gamma +
# delta: the whole design is expressed as offsets from s, and the default
# polynomial domain is s +- DESIGN_HALF_SPAN (= (0.2, 7.8) at s = 4).
DESIGN_HALF_SPAN = 3.8

# The response families selectable by name.  The default is "comb" -- THE QSVT
# of the papers (George's ruling, 2026-09-07; before that the bare constructor
# built the retired "band" design, which the golden baseline now pins only
# under an explicit filter_design="band").
DEFAULT_DESIGN = "comb"
FILTER_DESIGNS = ("comb", "band", "minimax", "cosine")


# --------------------------------------------------------------------------
# polynomial design: band-limited inverse with failure-mode notches
# --------------------------------------------------------------------------
def design_band_limited_inverse(
    degree=40,
    s=4.0,
    domain=None,
    band=None,
    edge=0.12,
    notches=None,
    notch_hw=0.15,
    invert=True,
    iso_weight=40.0,
    ngrid=6000,
):
    """Chebyshev least-squares fit of the band-limited-inverse target.

    Target response: ~1/lambda on the contiguous true-track band ``band``
    (smooth tanh edges of width ``edge``), multiplied by Gaussian notches of
    half-width ``notch_hw`` at the in-band failure-mode eigenvalues
    ``notches``, and ~0 outside the band.  ``iso_weight`` adds fit weight at
    lambda = s so the isolated-false eigenvalue (a machine-exact diagonal
    value, the bulk of the false grass) is strongly nulled, as it is by the
    OneBQF notch.

    ``s = gamma + delta`` is the constant diagonal of the segment Hamiltonian;
    the whole spectral structure (P4 true band ``s - 2cos(k pi/5)``, P3 bridge
    modes ``s -/+ sqrt(2)``, isolated line at ``s``, hub modes ``s -/+ sqrt(m)``)
    shifts rigidly with it, so the default band/notches/domain are expressed
    as offsets from ``s``.  ``s=4`` reproduces the original gamma=3, delta=1
    design exactly.

    Returns a ``numpy.polynomial.chebyshev.Chebyshev`` series whose ``domain``
    must contain the spectrum of any matrix it is applied to.
    """
    s = float(s)
    if band is None:
        band = (s - 1.85, s + 1.85)
    if notches is None:
        notches = (s - 1.414, s, s + 1.414)
    if domain is None:
        domain = (s - 3.8, s + 3.8)
    lo, hi = domain
    x = np.linspace(lo, hi, ngrid)

    def tophat(v, a, b, e):
        return 0.25 * (1 + np.tanh((v - a) / e)) * (1 + np.tanh((b - v) / e))

    y = tophat(x, band[0], band[1], edge)
    for nf in notches:
        y = y * (1 - np.exp(-(((x - nf) / notch_hw) ** 2)))
    if invert:
        # clip the inverse at the band floor: identical to 1/lambda on the band,
        # bounded below it (where the domain may reach lambda <= 0, e.g. the
        # indefinite gamma=1 regime).
        y = y / np.clip(x, max(band[0], 1e-6), None)
    w = np.ones_like(x)
    if iso_weight:
        w = w + iso_weight * np.exp(-(((x - s) / 0.03) ** 2))
    p = _cheb.Chebyshev.fit(x, y, degree, domain=list(domain), w=w)
    mx = float(np.max(np.abs(p(x))))
    if mx > 1.0:
        # global rescale for QSVT realizability (|p|<=1); downstream metrics are
        # scale-invariant, only the success probability shrinks.
        p = p / (mx / 0.95)
    return p


def design_line_comb_inverse(
    degree=40,
    s=4.0,
    lines=None,
    hw=0.18,
    domain=None,
    invert=True,
    ngrid=6000,
):
    """Chebyshev fit of the line-comb inverse: narrow ~1/lambda passes at the
    true-track eigenvalue *lines* only, ~0 everywhere else.

    Detector geometry pins true tracks to fixed length (P4), so their cluster
    eigenvalues are the four lines ``s - 2cos(k pi/5)``.  Passing only these
    lines (Gaussian half-width ``hw``), instead of a contiguous band, also
    rejects the tangle/chain modes that fall *between* the lines — which is
    what breaks the band design at high track density (the in-band false
    promotions grow to tens of percent by T=400, while the comb keeps the
    false rate at ~0 with ~98 % efficiency).

    Returns a ``numpy.polynomial.chebyshev.Chebyshev`` series whose ``domain``
    must contain the spectrum of any matrix it is applied to.
    """
    s = float(s)
    if lines is None:
        lines = tuple(s - 2.0 * np.cos(k * np.pi / 5.0) for k in (1, 2, 3, 4))
    if domain is None:
        domain = (s - 3.8, s + 3.8)
    x = np.linspace(domain[0], domain[1], ngrid)
    y = np.zeros_like(x)
    for m in lines:
        y = np.maximum(y, np.exp(-(((x - m) / hw) ** 2)))
    if invert:
        y = y / np.clip(x, max(min(lines), 1e-6), None)
    p = _cheb.Chebyshev.fit(x, y, degree, domain=list(domain))
    mx = float(np.max(np.abs(p(x))))
    if mx > 1.0:
        p = p / (mx / 0.95)
    return p


def design_minimax_comb(
    degree=8,
    s=4.0,
    lines=None,
    hw=0.18,
    domain=None,
    invert=True,
    ngrid=1500,
):
    """Weighted-minimax (LP) line-comb inverse — the degree-efficient D4 design.

    Same keep/null target as :func:`design_line_comb_inverse` (narrow ~1/lambda
    passes at the four P4 true-track eigenvalue lines), but fit in the **Chebyshev
    (minimax) norm** via a linear program instead of least squares.  The minimax
    fit equiripples the error and so reaches the metric floor at roughly **half the
    degree** of the least-squares comb (measured: $d\\approx6$-10 vs 12-20), i.e.
    half the walk calls.  Falls back to the least-squares comb if SciPy's LP
    solver is unavailable.

    Returns a ``numpy.polynomial.chebyshev.Chebyshev`` series.
    """
    s = float(s)
    if lines is None:
        lines = tuple(s - 2.0 * np.cos(k * np.pi / 5.0) for k in (1, 2, 3, 4))
    if domain is None:
        domain = (s - 3.8, s + 3.8)
    x = np.linspace(domain[0], domain[1], ngrid)
    y = np.zeros_like(x)
    for m in lines:
        y = np.maximum(y, np.exp(-(((x - m) / hw) ** 2)))
    if invert:
        y = y / np.clip(x, max(min(lines), 1e-6), None)
    try:
        from scipy.optimize import linprog
        basis = np.vstack([_cheb.Chebyshev.basis(j, domain=list(domain))(x)
                           for j in range(degree + 1)]).T          # (ngrid, d+1)
        # minimise t s.t.  -t <= basis @ c - y <= t   (Chebyshev-norm fit)
        A_ub = np.vstack([np.hstack([basis, -np.ones((ngrid, 1))]),
                          np.hstack([-basis, -np.ones((ngrid, 1))])])
        b_ub = np.concatenate([y, -y])
        cobj = np.concatenate([np.zeros(degree + 1), [1.0]])
        res = linprog(cobj, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(None, None)] * (degree + 1) + [(0, None)],
                      method="highs")
        if not res.success:
            raise RuntimeError(res.message)
        p = _cheb.Chebyshev(res.x[:degree + 1], domain=list(domain))
    except Exception:
        p = _cheb.Chebyshev.fit(x, y, degree, domain=list(domain))
    mx = float(np.max(np.abs(p(x))))
    if mx > 1.0:
        p = p / (mx / 0.95)
    return p


# --------------------------------------------------------------------------
# the solver
# --------------------------------------------------------------------------
class QSVT:
    """Polynomial (QSVT-family) filter solver for Ax = b.

    Parameters
    ----------
    matrix_A : (n, n) scipy sparse or dense array, Hermitian
        The segment Hamiltonian.  Unlike OneBQF, a constant diagonal is not
        required (the polynomial is applied to the full spectrum), but the
        designed default filter assumes the gamma=3, delta=1 operating point.
    vector_b : (n,) array
    degree : int
        Polynomial degree d (= number of walk-operator calls).
    poly : numpy.polynomial.chebyshev.Chebyshev, optional
        Override the designed filter with an arbitrary Chebyshev series.
        Its domain must contain the spectrum of A.
    filter_design : {"comb", "band", "minimax", "cosine"}, optional
        Which response family to design when ``poly`` is not given
        (module docstring).  Default :data:`DEFAULT_DESIGN` = "comb", **the
        QSVT of the papers** (the bare constructor built the retired band
        design before 2026-09-07).
    s : float, optional
        The constant diagonal ``gamma + delta`` the design is centred on.
        Default: read off ``A`` when its diagonal is constant (so a
        gamma = 1 Hamiltonian gets its own lines), else 4.0.
    design_kwargs : dict, optional
        Extra arguments for the chosen design function (e.g. ``hw``,
        ``lines``, ``band``, ``notches``, ``domain``).
    spectrum_pad : float
        Fractional padding applied to the estimated spectral interval.
    spectral_bounds : (lo, hi), optional
        Known bounds on the spectrum of A; skips the Lanczos estimate
        (:func:`~lhcb_velo_toy.solvers.quantum.spectral_filter.spectral_bounds`
        computes them cheaply from rho(C)).

    Attributes
    ----------
    poly, coeffs, l1, X, s, filter_design, total_qubits, solution_signed
    """

    def __init__(self, matrix_A, vector_b, degree=40, poly=None,
                 design_kwargs=None, spectrum_pad=0.02, spectral_bounds=None,
                 debug=False, filter_design=None, s=None):
        self.original_dim = matrix_A.shape[0]
        self.debug = debug
        self.degree = int(degree)

        A = matrix_A.tocsr() if _sp.issparse(matrix_A) else _sp.csr_matrix(np.asarray(matrix_A, float))
        self.A = A
        n = self.original_dim

        # --- the diagonal the design is centred on
        diag = np.asarray(A.diagonal()).ravel()
        const_diag = bool(diag.size) and bool(np.allclose(diag, diag[0]))
        if s is None:
            s = float(diag[0]) if const_diag else 4.0
        self.s = float(s)
        self.filter_design = None

        # --- spectral interval (needed so the Chebyshev argument is in [-1,1])
        if spectral_bounds is not None:
            lam_min, lam_max = map(float, spectral_bounds)
        else:
            lam_min, lam_max = self._spectral_bounds(A)
        span = max(lam_max - lam_min, 1e-12)
        lam_lo = lam_min - spectrum_pad * span
        lam_hi = lam_max + spectrum_pad * span

        # --- the polynomial
        if poly is None:
            design = DEFAULT_DESIGN if filter_design is None else str(filter_design)
            if design not in FILTER_DESIGNS:
                raise ValueError(f"filter_design must be one of {FILTER_DESIGNS}")
            kw = dict(design_kwargs or {})
            dom = kw.pop("domain", None)
            if dom is None:
                # the design window must cover both the spectrum and the
                # line/band structure around s (= (0.2, 7.8) at s = 4)
                dom = (min(lam_lo, self.s - DESIGN_HALF_SPAN),
                       max(lam_hi, self.s + DESIGN_HALF_SPAN))
            kw.setdefault("s", self.s)
            if design == "band":
                poly = design_band_limited_inverse(degree=self.degree, domain=dom, **kw)
            elif design == "comb":
                poly = design_line_comb_inverse(degree=self.degree, domain=dom, **kw)
            elif design == "minimax":
                poly = design_minimax_comb(degree=self.degree, domain=dom, **kw)
            else:  # "cosine"
                from lhcb_velo_toy.solvers.quantum.responses import design_one_bit_cosine
                poly = design_one_bit_cosine(degree=self.degree, domain=dom, **kw)
            self.filter_design = design
        else:
            if filter_design is not None:
                raise ValueError("pass either poly or filter_design, not both")
            self.filter_design = "custom"
            dlo, dhi = poly.domain
            if lam_lo < dlo - 1e-9 or lam_hi > dhi + 1e-9:
                raise ValueError(
                    f"spectrum [{lam_lo:.3f}, {lam_hi:.3f}] outside polynomial "
                    f"domain [{dlo:.3f}, {dhi:.3f}]"
                )
            self.degree = len(poly.coef) - 1
        self.poly = poly
        self.coeffs = np.asarray(poly.coef, float)
        self.l1 = float(np.sum(np.abs(self.coeffs)))

        # --- the rescaled operator X = (2A - (dlo+dhi) I) / (dhi - dlo)
        dlo, dhi = float(poly.domain[0]), float(poly.domain[1])
        self._x_scale = 2.0 / (dhi - dlo)
        self._x_shift = (dlo + dhi) / (dhi - dlo)
        self.X = (self._x_scale * A - self._x_shift * _sp.identity(n, format="csr")).tocsr()

        # --- right-hand side (normalised; zero-padded in circuit mode)
        b = np.asarray(vector_b, float).ravel()
        if b.shape[0] != n:
            raise ValueError("vector_b dimension mismatch")
        self.b_norm = b / np.linalg.norm(b)

        # --- qubit accounting
        self.num_system_qubits = max(1, math.ceil(math.log2(n)))
        self.num_lcu_qubits = max(1, math.ceil(math.log2(len(self.coeffs))))
        self.num_be_ancilla = 1
        self.total_qubits = self.num_system_qubits + self.num_be_ancilla + self.num_lcu_qubits

        self.circuit = None
        self._W_dense = None

        if self.debug:
            print("--- QSVT setup ---")
            print(f"n = {n}  (system qubits {self.num_system_qubits})")
            print(f"degree d = {self.degree}  (LCU qubits {self.num_lcu_qubits})")
            print(f"total qubits = {self.total_qubits}")
            print(f"spectral interval [{lam_min:.3f}, {lam_max:.3f}]  "
                  f"poly domain [{dlo:.3f}, {dhi:.3f}]  ||c||_1 = {self.l1:.3f}")

    # ------------------------------------------------------------------
    @classmethod
    def comb(cls, matrix_A, vector_b, degree=40, **kw):
        """The QSVT of the papers: the degree-40 line comb on ``A`` (explicit
        shorthand for ``QSVT(A, b, degree, filter_design="comb", **kw)``, the
        default)."""
        return cls(matrix_A, vector_b, degree=degree, filter_design="comb", **kw)

    # ------------------------------------------------------------------
    @staticmethod
    def _spectral_bounds(A):
        """Cheap bounds on the spectrum of Hermitian A."""
        n = A.shape[0]
        if n <= 256:
            w = np.linalg.eigvalsh(A.toarray())
            return float(w[0]), float(w[-1])
        try:
            hi = float(_spla.eigsh(A, k=1, which="LA",
                                   return_eigenvectors=False, maxiter=2000)[0])
            lo = float(_spla.eigsh(A, k=1, which="SA",
                                   return_eigenvectors=False, maxiter=2000)[0])
            return lo, hi
        except Exception:
            # Gershgorin fallback
            d = np.asarray(A.diagonal()).ravel()
            R = np.asarray(abs(A).sum(axis=1)).ravel() - np.abs(d)
            return float((d - R).min()), float((d + R).max())

    # ------------------------------------------------------------------
    # backend 1: exact matrix-free circuit semantics
    # ------------------------------------------------------------------
    def solve_statevector(self):
        """Exact emulation of the post-selected circuit output.

        Computes y = sum_k (c_k / ||c||_1) T_k(X) b via the Chebyshev
        recursion (sparse matvecs only).  Returns ``(solution, success)``
        with ``solution = |y| / ||y||`` (the OneBQF readout convention:
        post-selected measurement probabilities lose the sign) and
        ``success = ||y||^2`` (the probability of the LCU register and
        block-encoding ancilla both post-selecting on 0).

        The signed filtered vector is kept on ``self.solution_signed``.
        """
        c = self.coeffs / self.l1
        b = self.b_norm
        t_prev = b
        y = c[0] * t_prev
        if len(c) > 1:
            t_cur = self.X @ b
            y = y + c[1] * t_cur
            for k in range(2, len(c)):
                t_next = 2.0 * (self.X @ t_cur) - t_prev
                t_prev, t_cur = t_cur, t_next
                y = y + c[k] * t_cur
        success = float(np.vdot(y, y).real)
        self.solution_signed = y / max(np.linalg.norm(y), 1e-300)
        sol = np.abs(y)
        nrm = np.linalg.norm(sol)
        sol = sol / nrm if nrm > 0 else sol
        return sol[: self.original_dim], success

    # ------------------------------------------------------------------
    # backend 2: exact per-component eigendecomposition
    # ------------------------------------------------------------------
    def solve_eigh(self, max_comp=20000):
        """Exact spectral application via per-component eigendecomposition.

        Same output convention (and, up to float round-off, the same values)
        as :meth:`solve_statevector`: ``(|y|/||y||, ||y||^2)`` with
        ``y = p(A) b_norm / ||c||_1``. Instead of the Chebyshev matvec
        recursion it diagonalises each connected component of the
        continuation graph once (dense eigh on small blocks; isolated
        segments analytically), so many candidate responses can be evaluated
        for free by reusing the decomposition — the Codesign /
        Efficiency-Frontier evaluator, promoted as an official backend
        (centralise-2.1 Phase 2).

        The decomposition is cached on ``self._mode_system``.
        """
        from lhcb_velo_toy.solvers.quantum.spectral_filter import (
            apply_response, mode_system,
        )

        if getattr(self, "_mode_system", None) is None:
            self._mode_system = mode_system(self.A, self.b_norm,
                                            max_comp=max_comp)
        scaled = self.coeffs / self.l1
        dom = (float(self.poly.domain[0]), float(self.poly.domain[1]))

        def f(lam, coef=scaled, dom=dom):
            u = (2.0 * np.asarray(lam, dtype=np.float64) - (dom[0] + dom[1])) \
                / (dom[1] - dom[0])
            from numpy.polynomial import chebyshev as _c
            return _c.chebval(u, coef)

        y = apply_response(self._mode_system, f, signed=True)
        success = float(np.vdot(y, y).real)
        self.solution_signed = y / max(np.linalg.norm(y), 1e-300)
        sol = np.abs(y)
        nrm = np.linalg.norm(sol)
        sol = sol / nrm if nrm > 0 else sol
        return sol[: self.original_dim], success

    # ------------------------------------------------------------------
    # backend 3: the explicit circuit
    # ------------------------------------------------------------------
    def _walk_eigh(self):
        """Eigendecomposition of the padded, rescaled X (cached).

        In each eigenspace (x_i = cos(theta_i), eigenvector v_i) the walk
        operator W = (Z (x) I) . U_dilation acts on the (ancilla (x) v_i) plane
        as the rotation R(theta_i), so

            W^p = [[T_p(X),  S_p], [-S_p,  T_p(X)]],
            T_p(X) = V cos(p*theta) V^T,   S_p = V sin(p*theta) V^T.

        This builds any power from one eigh + two N-dim congruences instead of
        repeated (2N)-dim matrix squarings.
        """
        if self._W_dense is None:
            ns = self.num_system_qubits
            N = 1 << ns
            Xd = np.zeros((N, N))
            Xd[: self.original_dim, : self.original_dim] = self.X.toarray()
            # padded dims get x = 0 (in-domain); never populated (b zero-padded)
            w, V = np.linalg.eigh(Xd)
            w = np.clip(w, -1.0, 1.0)
            self._W_dense = (np.arccos(w), V)
        return self._W_dense

    def _walk_power(self, p):
        """Dense W^p on the (ancilla, system) register, ancilla most significant."""
        theta, V = self._walk_eigh()
        C = (V * np.cos(p * theta)) @ V.T
        S = (V * np.sin(p * theta)) @ V.T
        return np.block([[C, S], [-S, C]])

    def build_circuit(self):
        """Explicit LCU-of-Chebyshev circuit.

        Registers (little-endian): system (ns) | block-encoding ancilla (1) |
        LCU (m).  PREPARE loads sqrt(|c_k|/||c||_1) on the LCU register, a
        diagonal gate applies the coefficient signs, SELECT applies
        controlled-W^(2^j) from LCU qubit j (binary powering: W^k for k in
        binary), then PREPARE^dagger.  Post-select LCU = 0..0 and ancilla = 0.
        """
        from qiskit import QuantumCircuit, QuantumRegister
        from qiskit.circuit.library import StatePreparation, DiagonalGate, UnitaryGate

        ns, m = self.num_system_qubits, self.num_lcu_qubits
        N = 1 << ns
        sys_qr = QuantumRegister(ns, "sys")
        anc_qr = QuantumRegister(1, "be")
        lcu_qr = QuantumRegister(m, "lcu")
        qc = QuantumCircuit(sys_qr, anc_qr, lcu_qr)

        # |b> on the system register (zero-padded)
        b_pad = np.zeros(N)
        b_pad[: self.original_dim] = self.b_norm
        qc.append(StatePreparation(b_pad), sys_qr)

        # PREPARE sqrt(|c|/l1) and the sign diagonal
        M = 1 << m
        amps = np.zeros(M)
        amps[: len(self.coeffs)] = np.sqrt(np.abs(self.coeffs) / self.l1)
        amps = amps / np.linalg.norm(amps)
        signs = np.ones(M)
        signs[: len(self.coeffs)] = np.sign(self.coeffs)
        signs[signs == 0] = 1.0
        qc.append(StatePreparation(amps), lcu_qr)
        qc.append(DiagonalGate(list(signs.astype(complex))), lcu_qr)

        # SELECT: controlled-W^(2^j) from LCU qubit j (binary powering)
        dim_ws = 1 << (ns + 1)
        for j in range(m):
            Wp = self._walk_power(1 << j)
            ctrl = np.eye(2 * dim_ws, dtype=complex)
            ctrl[dim_ws:, dim_ws:] = Wp
            del Wp
            try:
                gate = UnitaryGate(ctrl, label=f"c-W^{1 << j}", check_input=False)
            except TypeError:  # older qiskit without check_input
                gate = UnitaryGate(ctrl, label=f"c-W^{1 << j}")
            del ctrl
            qc.append(gate, list(sys_qr) + list(anc_qr) + [lcu_qr[j]])

        # PREPARE^dagger (sign diagonal folds into the post-selected branch)
        qc.append(StatePreparation(amps).inverse(), lcu_qr)
        self.circuit = qc
        return qc

    def run_circuit(self, streaming=False):
        """Simulate the explicit circuit exactly and post-select.

        ``streaming=False`` builds the full qiskit circuit and evolves it with
        ``qiskit.quantum_info.Statevector`` (all SELECT gates held in memory).
        ``streaming=True`` applies the *identical* gate sequence one gate at a
        time with bounded memory (a single walk power at once) — needed for
        large systems where holding all m controlled-W^(2^j) matrices
        (each ``(2^(ns+2))^2`` complex) would exhaust RAM.

        Returns ``(solution, success)`` with the same convention as
        :meth:`solve_statevector`.
        """
        ns, m = self.num_system_qubits, self.num_lcu_qubits
        N, M = 1 << ns, 1 << m

        if not streaming:
            from qiskit.quantum_info import Statevector

            if self.circuit is None:
                self.build_circuit()
            psi = np.asarray(Statevector(self.circuit).data)
            # little-endian: index = sys + (anc << ns) + (lcu << (ns+1))
            psi = psi.reshape(M, 2, N)  # [lcu, anc, sys]
            branch = psi[0, 0, :]
        else:
            # state tensor [lcu, anc, sys]; same gates, applied sequentially
            amps = np.zeros(M)
            amps[: len(self.coeffs)] = np.sqrt(np.abs(self.coeffs) / self.l1)
            amps = amps / np.linalg.norm(amps)
            signs = np.ones(M)
            signs[: len(self.coeffs)] = np.sign(self.coeffs)
            signs[signs == 0] = 1.0
            b_pad = np.zeros(N)
            b_pad[: self.original_dim] = self.b_norm

            state = np.zeros((M, 2 * N))
            state[:, :N] = (amps * signs)[:, None] * b_pad[None, :]
            # SELECT: controlled-W^(2^j) on the lcu-bit-j = 1 branches
            for j in range(m):
                Wp = self._walk_power(1 << j)
                hot = [k for k in range(M) if (k >> j) & 1]
                state[hot] = state[hot] @ Wp.T
                del Wp
            # PREPARE^dagger, keep the lcu = |0..0> component
            out = amps @ state            # (2N,)
            branch = out[:N]              # anc = 0
        success = float(np.vdot(branch, branch).real)
        sol = np.abs(branch)
        nrm = np.linalg.norm(sol)
        sol = sol / nrm if nrm > 0 else sol
        return sol[: self.original_dim], success

    # ------------------------------------------------------------------
    # backend 4: the phase-sequence realisation (QSVT proper), dense
    # ------------------------------------------------------------------
    def solve_phase_sequence(self, phases=None, max_dim=4096, **find_kw):
        """Apply the polynomial by QSP phase sequences on the dilation.

        The QSVT-proper realisation (paper section 4.2): the polynomial is
        split into even and odd parts, phase factors are found for each
        (:func:`~lhcb_velo_toy.solvers.quantum.qsp_phases.find_phase_factors`,
        seconds at degree ~10, minutes at degree 40 — pass ``phases=
        {"even": ..., "odd": ...}`` to reuse them), both sequences are run
        as dense linear algebra on the dilation block encoding of ``X`` and
        recombined.  Returns ``(solution, success)`` in the usual readout
        convention, with ``success = ||q(X) b||^2 / (alpha_e + alpha_o)^2``,
        ``q = p / max|p|`` — the phase sequence's own probability, larger
        than the LCU's by ``(||c||_1 / max|p|)^2``.  The phases and the
        scalar-identity errors are kept on ``self.phase_info``.

        Dense: refuses systems above ``max_dim``.
        """
        from lhcb_velo_toy.solvers.quantum.qsp_phases import phase_sequence_solve

        n = self.original_dim
        if n > max_dim:
            raise ValueError(f"n = {n} > max_dim = {max_dim}: the phase-sequence "
                             "backend is dense; use solve_statevector/solve_eigh")
        sol, success, info = phase_sequence_solve(self.X.toarray(), self.b_norm,
                                                  self.coeffs, phases=phases,
                                                  **find_kw)
        self.phase_info = info
        return sol[:n], success

    # ------------------------------------------------------------------
    def resources(self, realisation="lcu"):
        """Logical resource accounting for this instance.

        ``realisation`` selects the circuit that applies the response
        (paper section 4.2, table of realisations; ancillas are beyond the
        system register):

        * ``"lcu"`` (the build): ``1 + m`` ancillas, ``2^m - 1`` walk steps
          per shot as built (``d`` with an optimal SELECT), success
          ``||p(X)u||^2 / ||c||_1^2``.
        * ``"phase"``: the phase sequence — 4 ancillas (3 for a
          definite-parity polynomial), ``4d`` calls (``2d``), success
          ``||q(X)u||^2 / (alpha_e + alpha_o)^2``.
        * ``"qetu"``: the trigonometric comb over ``exp(-iAt)`` — 2
          ancillas, ``K`` (= ``walk_calls``) evolution calls.

        Keys common to all: ``n, system_qubits, walk_calls (= degree), l1,
        peak (max|p| on the domain), lcu_penalty ((l1/peak)^2), ancillas,
        total_qubits, calls_per_shot, realisation``; the LCU entry keeps the
        legacy ``be_ancilla`` / ``lcu_qubits`` keys.
        """
        d, m = self.degree, self.num_lcu_qubits
        xg = np.linspace(-1.0, 1.0, 20001)
        peak = float(np.max(np.abs(_cheb.chebval(xg, self.coeffs))))
        base = dict(
            n=self.original_dim,
            system_qubits=self.num_system_qubits,
            walk_calls=d,
            l1=self.l1,
            peak=peak,
            lcu_penalty=(self.l1 / peak) ** 2 if peak > 0 else np.inf,
        )
        parity = self._parity()
        if realisation == "lcu":
            base.update(
                be_ancilla=self.num_be_ancilla,
                lcu_qubits=m,
                ancillas=self.num_be_ancilla + m,
                total_qubits=self.total_qubits,
                calls_per_shot=(1 << m) - 1,
                calls_per_shot_optimal_select=d,
                realisation="lcu",
            )
        elif realisation == "phase":
            anc = 3 if parity != "mixed" else 4
            base.update(
                ancillas=anc,
                total_qubits=self.num_system_qubits + anc,
                calls_per_shot=(2 if parity != "mixed" else 4) * d,
                parity=parity,
                realisation="phase",
            )
        elif realisation == "qetu":
            base.update(
                ancillas=2,
                total_qubits=self.num_system_qubits + 2,
                calls_per_shot=d,
                realisation="qetu",
            )
        else:
            raise ValueError("realisation must be 'lcu', 'phase' or 'qetu'")
        return base

    def resource_table(self):
        """``{realisation: resources(realisation)}`` for the three realisations."""
        return {r: self.resources(r) for r in ("lcu", "phase", "qetu")}

    def _parity(self, tol=1e-12):
        c = self.coeffs
        odd_zero = bool(np.all(np.abs(c[1::2]) <= tol))
        even_zero = bool(np.all(np.abs(c[0::2]) <= tol))
        if odd_zero and not even_zero:
            return "even"
        if even_zero and not odd_zero:
            return "odd"
        return "mixed"
