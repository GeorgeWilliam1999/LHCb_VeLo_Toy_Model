"""
HHLAlgorithm: Full Harrow-Hassidim-Lloyd algorithm implementation.

Implements the quantum algorithm for solving linear systems A x = b.
"""

from __future__ import annotations
import math
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT, RYGate, UnitaryGate
from scipy.linalg import expm


class HHLAlgorithm:
    """
    Full HHL (Harrow-Hassidim-Lloyd) algorithm for solving linear systems.

    The HHL algorithm solves A x = b on a quantum computer with exponential
    speedup for certain classes of sparse matrices. This implementation
    provides a complete HHL circuit with phase estimation, controlled
    rotation, and uncomputation.

    Parameters
    ----------
    matrix_A : numpy.ndarray
        The system matrix. Will be padded to power of 2 if necessary.
        Should be Hermitian (or will be symmetrized).
    vector_b : numpy.ndarray
        The right-hand side vector. Will be normalized.
    num_time_qubits : int, default 5
        Number of qubits for phase estimation precision.
    shots : int, default 10240
        Number of measurement shots for sampling.
    debug : bool, default False
        If True, print debug information during execution.

    Attributes
    ----------
    A : ndarray
        The normalised (possibly padded) system matrix.
    vector_b : ndarray
        The normalized RHS vector.
    num_time_qubits : int
        Phase estimation precision.
    shots : int
        Measurement shots.
    circuit : QuantumCircuit | None
        The built HHL circuit (after calling build_circuit).

    Examples
    --------
    >>> A = np.array([[2, -1], [-1, 2]])
    >>> b = np.array([1, 0])
    >>> hhl = HHLAlgorithm(A, b, num_time_qubits=4)
    >>> hhl.build_circuit()
    >>> counts = hhl.run()
    >>> solution = hhl.get_solution()

    Notes
    -----
    The algorithm works by:
    1. Encoding |b⟩ in the state register
    2. Phase estimation to extract eigenvalues of A
    3. Controlled rotation conditioned on eigenvalues
    4. Uncomputation of phase estimation
    5. Measurement and post-selection on ancilla

    References
    ----------
    Harrow, Hassidim, Lloyd. "Quantum Algorithm for Linear Systems of
    Equations." Physical Review Letters 103, 150502 (2009).
    """

    def __init__(
        self,
        matrix_A: np.ndarray,
        vector_b: np.ndarray,
        num_time_qubits: int = 5,
        shots: int = 10240,
        debug: bool = False,
    ) -> None:
        """Initialize the HHL algorithm."""
        A = np.array(matrix_A, dtype=complex)
        self.original_dim = A.shape[0]
        self.debug = debug

        # ---- pad to power-of-2 ----
        d = self.original_dim
        n_needed = math.ceil(np.log2(d)) if d > 1 else 1
        padded_dim = 2 ** n_needed

        if padded_dim != d:
            A_padded = np.zeros((padded_dim, padded_dim), dtype=complex)
            A_padded[:d, :d] = A
            A = (A_padded + A_padded.conj().T) / 2

            b_padded = np.zeros(padded_dim, dtype=complex)
            b_padded[:d] = vector_b
            vector_b = b_padded

        # ---- normalise b ----
        b_normalized = vector_b / np.linalg.norm(vector_b)

        # ---- normalise A ----
        self.A_orig = A.copy()
        self.A_norm = np.linalg.norm(A)
        A = A / self.A_norm

        self.A = A
        self.vector_b = b_normalized
        self.num_time_qubits = num_time_qubits
        self.shots = shots

        self.system_dim = A.shape[0]
        self.num_system_qubits = int(np.log2(self.system_dim))

        # quantum registers
        self.time_qr = QuantumRegister(self.num_time_qubits, "time")
        self.b_qr = QuantumRegister(self.num_system_qubits, "b")
        self.ancilla_qr = QuantumRegister(1, "ancilla")
        self.classical_reg = ClassicalRegister(
            1 + self.num_system_qubits, "c"
        )
        self.circuit: Optional[QuantumCircuit] = None
        self.counts: Optional[dict[str, int]] = None

        # time parameter  t = π / λ_max
        self.t = float(
            np.pi / np.max(np.abs(np.linalg.eigvals(A)))
        )
        self.eigenvalues = np.linalg.eigvals(self.A_orig)
        self.eigenvalues_scaled = np.linalg.eigvals(self.A)

    # ------------------------------------------------------------------
    # circuit helpers
    # ------------------------------------------------------------------

    def _create_input_state(self) -> QuantumCircuit:
        """
        Create the state preparation circuit for |b⟩.

        Returns
        -------
        QuantumCircuit
            Circuit that prepares |b⟩ from |0⟩.
        """
        qc_b = QuantumCircuit(self.num_system_qubits)
        qc_b.initialize(self.vector_b, list(range(self.num_system_qubits)))
        return qc_b

    def _apply_controlled_u(
        self,
        qc: QuantumCircuit,
        matrix: np.ndarray,
        control: "QuantumRegister",
        target: list,
        power: int,
    ) -> QuantumCircuit:
        """Apply controlled-U^power to the circuit."""
        U = expm(1j * matrix * self.t * power)
        controlled_U = UnitaryGate(U).control(1)
        qc.append(controlled_U, [control] + target)
        return qc

    def _inverse_qft(self, n_qubits: int) -> QFT:
        """Return the inverse QFT gate."""
        return QFT(n_qubits, do_swaps=True).inverse()

    def _phase_estimation(self, qc: QuantumCircuit) -> None:
        """Apply QPE to the circuit (in-place)."""
        for qubit in self.time_qr:
            qc.h(qubit)

        for i in range(self.num_time_qubits):
            power = 2 ** i
            self._apply_controlled_u(
                qc,
                self.A,
                self.time_qr[self.num_time_qubits - i - 1],
                list(self.b_qr),
                power,
            )

        iqft = self._inverse_qft(self.num_time_qubits).to_gate(label="IQFT")
        qc.append(iqft, self.time_qr[:])

    def _uncompute_phase_estimation(self, qc: QuantumCircuit) -> None:
        """Un-compute QPE (in-place)."""
        qft = QFT(self.num_time_qubits, do_swaps=True).to_gate(label="QFT")
        qc.append(qft, self.time_qr[:])

        for i in reversed(range(self.num_time_qubits)):
            power = 2 ** i
            self._apply_controlled_u(
                qc,
                -self.A,
                self.time_qr[self.num_time_qubits - i - 1],
                list(self.b_qr),
                power,
            )

        for qubit in self.time_qr:
            qc.h(qubit)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def build_circuit(self) -> QuantumCircuit:
        """
        Build the complete HHL quantum circuit.

        Returns
        -------
        QuantumCircuit
            The constructed HHL circuit with all components.

        Notes
        -----
        Circuit structure:
        1. State preparation for |b⟩
        2. Phase estimation (H + controlled-U + IQFT)
        3. Controlled R_y rotations conditioned on eigenvalue register
        4. Uncomputation of phase estimation
        5. Measurements on ancilla + system register
        """
        qc = QuantumCircuit(
            self.time_qr,
            self.b_qr,
            self.ancilla_qr,
            self.classical_reg,
        )

        # 1 – state preparation
        qc.compose(
            self._create_input_state(),
            qubits=list(self.b_qr),
            inplace=True,
        )

        # 2 – phase estimation
        self._phase_estimation(qc)

        # 3 – controlled rotations
        gain = 0.3
        for i in range(2 ** self.num_time_qubits):
            phase = i / (2 ** self.num_time_qubits)
            lam = 2 * np.pi * phase / self.t
            if abs(lam) < 1e-9:
                continue

            inv_lam = 1.0 / lam
            angle = 2 * np.arcsin(min(1.0, gain * inv_lam / 2))
            controls = list(self.time_qr)

            if self.debug:
                bits = format(i, f"0{self.num_time_qubits}b")
                print(
                    f"Time state |{bits}>: phase = {phase:.4f}, "
                    f"\u03bb_scaled = {lam:.4f}, "
                    f"\u03bb_true = {lam * self.A_norm:.4f}, "
                    f"1/\u03bb = {inv_lam:.2f}, Ry angle = {angle:.4f}"
                )

            bits = format(i, f"0{self.num_time_qubits}b")
            for j, bit in enumerate(bits):
                if bit == "0":
                    qc.x(self.time_qr[j])

            cry = RYGate(angle).control(
                num_ctrl_qubits=self.num_time_qubits
            )
            qc.append(cry, [*controls, self.ancilla_qr[0]])

            for j, bit in enumerate(bits):
                if bit == "0":
                    qc.x(self.time_qr[j])

        # 4 – uncompute phase estimation
        self._uncompute_phase_estimation(qc)

        # 5 – measurements
        qc.measure(self.ancilla_qr[0], self.classical_reg[0])
        qc.measure(self.b_qr, self.classical_reg[1:])

        self.circuit = qc
        return qc

    def run(self) -> dict[str, int]:
        """
        Execute the HHL circuit on a simulator.

        Returns
        -------
        dict[str, int]
            Measurement counts from the circuit execution.

        Raises
        ------
        RuntimeError
            If build_circuit has not been called.
        """
        if self.circuit is None:
            raise RuntimeError(
                "Circuit not built. Call build_circuit() first."
            )
        simulator = AerSimulator()
        transpiled_circuit = transpile(self.circuit, simulator)
        job = simulator.run(transpiled_circuit, shots=self.shots)
        result = job.result()
        self.counts = result.get_counts()
        return self.counts

    def get_solution(self, counts: Optional[dict] = None) -> np.ndarray:
        """
        Extract the solution vector from measurement counts.

        Parameters
        ----------
        counts : dict, optional
            Measurement counts. If None, uses stored counts from run().

        Returns
        -------
        ndarray
            The solution vector x (normalized).

        Notes
        -----
        Post-selects on ancilla qubit = 1 and extracts amplitudes
        from the system register measurements.
        """
        if counts is not None:
            self.counts = counts
        if self.counts is None:
            raise ValueError(
                "No measurement results available. Run run() first."
            )

        total_success = 0
        padded_dim = 2 ** self.num_system_qubits
        prob_dist = np.zeros(padded_dim)

        for outcome, count in self.counts.items():
            if outcome[-1] == "1":
                system_bits = outcome[:-1]
                index = int(system_bits, 2)
                prob_dist[index] += count
                total_success += count

        if total_success == 0:
            return np.zeros(self.original_dim)

        prob_dist = prob_dist / total_success
        solution_padded = np.sqrt(prob_dist)
        solution_padded = solution_padded / np.linalg.norm(solution_padded)

        solution_vector = solution_padded[: self.original_dim]
        solution_vector = solution_vector / np.linalg.norm(solution_vector)
        return solution_vector

    def get_quantum_only_circuit(self) -> QuantumCircuit:
        """
        Get the HHL circuit without measurements.

        Returns
        -------
        QuantumCircuit
            Circuit for statevector simulation.
        """
        if self.circuit is None:
            raise ValueError("Circuit has not been built yet.")

        qc_clean = QuantumCircuit(
            self.time_qr, self.b_qr, self.ancilla_qr
        )
        for instr, qargs, cargs in self.circuit.data:
            if instr.name == "measure":
                continue
            qc_clean.append(instr, qargs, cargs)
        return qc_clean

    def simulate_statevector(self):
        """
        Simulate the circuit and return the final statevector.

        Returns
        -------
        Statevector
            The final quantum state.
        """
        from qiskit_aer import Aer

        backend = Aer.get_backend("statevector_simulator")
        job = backend.run(transpile(self.circuit, backend))
        statevector = job.result().get_statevector()
        return statevector

    def extract_postselected_solution(
        self,
        statevector,
    ) -> np.ndarray:
        """
        Extract solution from statevector with post-selection.

        Parameters
        ----------
        statevector : Statevector
            The final quantum state from simulation.

        Returns
        -------
        ndarray
            The solution vector (normalized).
        """
        system_qubits = self.num_system_qubits
        ancilla_index = self.num_time_qubits + self.num_system_qubits

        sol = np.zeros(2 ** system_qubits)
        for i, amp in enumerate(statevector.data):
            if ((i >> ancilla_index) & 1) == 1:
                system_state = i & ((1 << system_qubits) - 1)
                sol[system_state] += np.abs(amp) ** 2

        norm = np.linalg.norm(sol)
        if norm > 0:
            sol = np.sqrt(sol)
            sol = sol / np.linalg.norm(sol)
        return sol
