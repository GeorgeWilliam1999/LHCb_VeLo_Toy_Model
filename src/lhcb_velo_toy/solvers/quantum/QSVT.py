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

Two backends:

  solve_statevector()  -- exact matrix-free emulation of the circuit semantics
                          via the Chebyshev recursion (sparse matvecs only);
                          scales to full events with tens of thousands of
                          segments.
  build_circuit() /
  run_circuit()        -- the explicit qiskit circuit (StatePreparation + LCU +
                          controlled walk powers), simulated exactly with
                          qiskit.quantum_info.Statevector; for small systems.
                          Validates that the matrix-free backend is what the
                          circuit computes.

The returned solution follows the OneBQF convention: (|amplitudes| normalised,
success probability), truncated to the original dimension.
"""

import math

import numpy as np
from numpy.polynomial import chebyshev as _cheb
from scipy import sparse as _sp
from scipy.sparse import linalg as _spla


# --------------------------------------------------------------------------
# polynomial design: band-limited inverse with failure-mode notches
# --------------------------------------------------------------------------
def design_band_limited_inverse(
    degree=40,
    domain=(0.2, 7.8),
    band=(2.15, 5.85),
    edge=0.12,
    notches=(2.586, 4.0, 5.414),
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
    lambda = gamma + delta = 4.0 so the isolated-false eigenvalue (a
    machine-exact diagonal value, the bulk of the false grass) is strongly
    nulled, as it is by the OneBQF notch.

    Returns a ``numpy.polynomial.chebyshev.Chebyshev`` series whose ``domain``
    must contain the spectrum of any matrix it is applied to.
    """
    lo, hi = domain
    x = np.linspace(lo, hi, ngrid)

    def tophat(v, a, b, e):
        return 0.25 * (1 + np.tanh((v - a) / e)) * (1 + np.tanh((b - v) / e))

    y = tophat(x, band[0], band[1], edge)
    for nf in notches:
        y = y * (1 - np.exp(-(((x - nf) / notch_hw) ** 2)))
    if invert:
        y = y / x
    w = np.ones_like(x)
    if iso_weight:
        w = w + iso_weight * np.exp(-(((x - 4.0) / 0.03) ** 2))
    return _cheb.Chebyshev.fit(x, y, degree, domain=list(domain), w=w)


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
    design_kwargs : dict, optional
        Extra arguments for :func:`design_band_limited_inverse`.
    spectrum_pad : float
        Fractional padding applied to the estimated spectral interval.
    """

    def __init__(self, matrix_A, vector_b, degree=40, poly=None,
                 design_kwargs=None, spectrum_pad=0.02, spectral_bounds=None,
                 debug=False):
        self.original_dim = matrix_A.shape[0]
        self.debug = debug
        self.degree = int(degree)

        A = matrix_A.tocsr() if _sp.issparse(matrix_A) else _sp.csr_matrix(np.asarray(matrix_A, float))
        self.A = A
        n = self.original_dim

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
            kw = dict(design_kwargs or {})
            dom = kw.pop("domain", None)
            if dom is None:
                # the design window must cover both the spectrum and the
                # default band/notch structure
                dom = (min(lam_lo, 0.2), max(lam_hi, 7.8))
            poly = design_band_limited_inverse(degree=self.degree, domain=dom, **kw)
        else:
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
    # backend 2: the explicit circuit
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
    def resources(self):
        """Logical resource accounting for this instance."""
        return dict(
            n=self.original_dim,
            system_qubits=self.num_system_qubits,
            be_ancilla=self.num_be_ancilla,
            lcu_qubits=self.num_lcu_qubits,
            total_qubits=self.total_qubits,
            walk_calls=self.degree,
            l1=self.l1,
        )
