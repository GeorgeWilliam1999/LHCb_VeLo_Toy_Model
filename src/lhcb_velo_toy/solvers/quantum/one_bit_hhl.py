"""
OneBitHHL: 1-Bit Quantum Filter using Suzuki-Trotter decomposition.

A simplified HHL variant using single-qubit phase estimation, designed
for track reconstruction at LHCb.
"""

from __future__ import annotations
import math
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT, RXGate


class OneBitHHL:
    """
    1-Bit Quantum Filter (OneBQF) for solving linear systems.

    This implementation uses a single time qubit for phase estimation
    and first-order Suzuki-Trotter decomposition for time evolution.
    It provides a more practical circuit depth than full HHL while
    maintaining quantum advantage for specific problem classes.

    Parameters
    ----------
    matrix_A : numpy.ndarray
        The system matrix.
    vector_b : numpy.ndarray
        The right-hand side vector.
    num_time_qubits : int, default 1
        Number of time qubits (typically 1 for this algorithm).
    shots : int, default 1024
        Number of measurement shots.
    debug : bool, default False
        If True, print debug information.

    Attributes
    ----------
    A : ndarray
        The (possibly padded) system matrix.
    vector_b : ndarray
        The normalized RHS vector.
    num_time_qubits : int
        Number of time qubits.
    shots : int
        Measurement shots.
    circuit : QuantumCircuit | None
        The built circuit (after calling build_circuit).

    Examples
    --------
    >>> A = np.array([[2, -1], [-1, 2]])
    >>> b = np.array([1, 0])
    >>> hhl = OneBitHHL(A, b, shots=10000)
    >>> hhl.build_circuit()
    >>> counts = hhl.run(use_noise_model=False)
    >>> solution, success = hhl.get_solution(counts)

    Notes
    -----
    Key differences from full HHL:
    - Single time qubit reduces circuit depth
    - Exact two-level unitary decomposition (Givens rotations)
    - Designed for specific constant-diagonal matrix structures

    The algorithm trades precision for practicality, making it more
    suitable for near-term quantum devices.

    References
    ----------
    Based on the OneBQF algorithm developed for LHCb track reconstruction.
    """

    def __init__(
        self,
        matrix_A: np.ndarray,
        vector_b: np.ndarray,
        num_time_qubits: int = 1,
        shots: int = 1024,
        debug: bool = False,
    ) -> None:
        """Initialize the 1-Bit HHL algorithm."""
        A = np.array(matrix_A, dtype=float)
        self.original_dim = A.shape[0]
        self.debug = debug

        # ---- pad to power-of-2 ----
        d = self.original_dim
        n_needed = math.ceil(np.log2(d)) if d > 1 else 1
        padded_dim = 2 ** n_needed

        if padded_dim != d:
            A_padded = np.zeros((padded_dim, padded_dim))
            A_padded[:d, :d] = A
            # fill extra diagonal with first diagonal value
            for i in range(d, padded_dim):
                diagonal_value = np.diag(A)[0]
                A_padded[i, i] = diagonal_value
            A = (A_padded + A_padded.conj().T) / 2

            b_padded = np.ones(padded_dim)
            b_padded[:d] = vector_b
            vector_b = b_padded

        # ---- normalise b ----
        b_normalized = vector_b / np.linalg.norm(vector_b)

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

        # time scaling  t = π / a_{00}
        self.t = float(np.pi / A[0, 0])

        # validate constant diagonal
        if not np.allclose(np.diag(A), np.diag(A)[0]):
            raise ValueError(
                "Matrix A must have a constant diagonal for this scheme."
            )

        self.diagonal_val = float(np.diag(A)[0])

        # extract interaction structure: B = c·I - A
        B = self.diagonal_val * np.identity(self.system_dim) - self.A
        rows, cols = np.where(np.triu(B) != 0)
        self.interaction_pairs = list(zip(rows.tolist(), cols.tolist()))

        if self.debug:
            print("--- Automated Matrix Analysis ---")
            print(f"Diagonal Value (c): {self.diagonal_val}")
            print(
                f"Found {len(self.interaction_pairs)} interaction pair(s): "
                f"{self.interaction_pairs}"
            )
            print("---------------------------------")

    # ------------------------------------------------------------------
    # controlled-U via exact two-level unitary (Givens rotation)
    # ------------------------------------------------------------------

    def _apply_direct_controlled_u(
        self,
        qc: QuantumCircuit,
        control_qubit,
        target_qubits: list,
        power: int,
        inverse: bool = False,
    ) -> None:
        """
        Implement controlled e^{-iHt} exactly using Givens rotations.

        Works for any Hamming distance and prevents spectral leakage.
        """
        evolution_time = self.t * power
        theta = 2 * evolution_time
        if inverse:
            theta = -theta

        for i, j in self.interaction_pairs:
            xor_val = i ^ j
            differing_indices = [
                k
                for k in range(self.num_system_qubits)
                if (xor_val >> k) & 1
            ]
            if not differing_indices:
                continue

            pivot = differing_indices[0]
            rest_diff = differing_indices[1:]

            # CNOT ladder
            for k in rest_diff:
                qc.cx(target_qubits[pivot], target_qubits[k])

            # transform i to new basis
            i_transformed = i
            for k in rest_diff:
                if (i_transformed >> pivot) & 1:
                    i_transformed ^= 1 << k

            qubits_to_flip: list = []
            full_control_list = [control_qubit]

            for k in range(self.num_system_qubits):
                if k == pivot:
                    continue
                bit_val = (i_transformed >> k) & 1
                full_control_list.append(target_qubits[k])
                if bit_val == 0:
                    qubits_to_flip.append(target_qubits[k])

            if qubits_to_flip:
                qc.x(qubits_to_flip)

            mcrx = RXGate(theta).control(len(full_control_list))
            qc.append(mcrx, full_control_list + [target_qubits[pivot]])

            if qubits_to_flip:
                qc.x(qubits_to_flip)

            # undo CNOT ladder
            for k in reversed(rest_diff):
                qc.cx(target_qubits[pivot], target_qubits[k])

        # global phase from diagonal
        phase = -self.diagonal_val * evolution_time
        if inverse:
            phase = -phase
        qc.p(phase, control_qubit)

    def _apply_controlled_u(
        self,
        qc: QuantumCircuit,
        control_qubit,
        target_qubits: list,
        power: int,
        inverse: bool = False,
    ) -> None:
        """Convenience wrapper for controlled evolution."""
        self._apply_direct_controlled_u(
            qc, control_qubit, target_qubits, power, inverse=inverse
        )

    # ------------------------------------------------------------------
    # QPE sub-circuits
    # ------------------------------------------------------------------

    def _phase_estimation(self, qc: QuantumCircuit) -> None:
        """Apply single-qubit phase estimation."""
        qc.h(self.time_qr)
        for i in range(self.num_time_qubits):
            power = 2 ** i
            self._apply_controlled_u(
                self.circuit,
                self.time_qr[self.num_time_qubits - 1 - i],
                list(self.b_qr),
                power,
            )
        iqft = QFT(self.num_time_qubits, do_swaps=True).inverse()
        qc.append(iqft.to_gate(label="IQFT"), self.time_qr)

    def _uncompute_phase_estimation(self, qc: QuantumCircuit) -> None:
        """Un-compute phase estimation."""
        qft = QFT(self.num_time_qubits, do_swaps=True).to_gate(label="QFT")
        qc.append(qft, self.time_qr)
        for i in reversed(range(self.num_time_qubits)):
            power = 2 ** i
            self._apply_controlled_u(
                self.circuit,
                self.time_qr[self.num_time_qubits - 1 - i],
                list(self.b_qr),
                power,
                inverse=True,
            )
        qc.h(self.time_qr)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def build_circuit(self) -> QuantumCircuit:
        """
        Build the 1-Bit HHL quantum circuit.

        Returns
        -------
        QuantumCircuit
            The constructed circuit.

        Notes
        -----
        Circuit structure:
        1. Hadamard on system register (uniform superposition as |b⟩)
        2. Phase estimation with single time qubit
        3. Conditional flip  +  CNOT ancilla (eigenvalue inversion)
        4. Uncompute phase estimation
        5. Measurements on ancilla + system register
        """
        self.circuit = QuantumCircuit(
            self.time_qr,
            self.b_qr,
            self.ancilla_qr,
            self.classical_reg,
        )

        # state preparation: uniform superposition
        self.circuit.h(self.b_qr)

        # phase estimation
        self._phase_estimation(self.circuit)

        # eigenvalue inversion via single-qubit trick
        self.circuit.x(self.time_qr[0])
        self.circuit.cx(self.time_qr[0], self.ancilla_qr[0])
        self.circuit.x(self.time_qr[0])

        # uncompute
        self._uncompute_phase_estimation(self.circuit)

        # measurements
        self.circuit.measure(self.ancilla_qr[0], self.classical_reg[0])
        self.circuit.measure(self.b_qr, self.classical_reg[1:])

        return self.circuit

    def run(
        self,
        use_noise_model: bool = False,
        backend_name: str = "ibm_torino",
    ) -> dict[str, int]:
        """
        Execute the circuit on a simulator.

        Parameters
        ----------
        use_noise_model : bool, default False
            If True, simulate with a noise model from the specified backend.
        backend_name : str, default "ibm_torino"
            IBM backend name for noise model (only used if use_noise_model=True).

        Returns
        -------
        dict[str, int]
            Measurement counts.

        Raises
        ------
        RuntimeError
            If build_circuit has not been called.
        ImportError
            If use_noise_model=True but qiskit-ibm-runtime not installed.
        """
        if self.circuit is None:
            raise RuntimeError(
                "Circuit not built. Call build_circuit() first."
            )

        simulator = AerSimulator()

        if use_noise_model:
            from qiskit_ibm_runtime import QiskitRuntimeService
            from qiskit_aer.noise import NoiseModel
            from qiskit.transpiler.preset_passmanagers import (
                generate_preset_pass_manager,
            )

            service = QiskitRuntimeService()
            backend = service.backend(backend_name)
            noise_model = NoiseModel.from_backend(backend)

            if self.debug:
                print(f"\n--- Using {backend_name} Noise Model ---")
                print(f"Basis gates: {noise_model.basis_gates}")
                print(f"Number of qubits: {backend.num_qubits}")

            pm = generate_preset_pass_manager(
                optimization_level=3, backend=backend
            )
            transpiled_circuit = pm.run(self.circuit)
            simulator = AerSimulator(noise_model=noise_model)
            job = simulator.run(transpiled_circuit, shots=self.shots)
        else:
            transpiled_circuit = transpile(
                self.circuit, simulator, optimization_level=3
            )
            job = simulator.run(transpiled_circuit, shots=self.shots)

        result = job.result()
        self.counts = result.get_counts()
        return self.counts

    def get_solution(
        self,
        counts: Optional[dict] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Extract the solution vector from measurement counts.

        Parameters
        ----------
        counts : dict, optional
            Measurement counts. If None, uses stored counts.

        Returns
        -------
        tuple[ndarray, int]
            The solution vector (normalized) and the number of
            successful post-selections (ancilla = 1).

        Notes
        -----
        Post-selects on the ancilla qubit being |1⟩ and extracts
        the solution from the system register amplitudes.
        """
        if counts is not None:
            self.counts = counts
        if not self.counts:
            raise ValueError("No measurement results available.")

        total_success = 0
        prob_dist = np.zeros(2 ** self.num_system_qubits)

        for outcome, count in self.counts.items():
            if outcome[-1] == "1":
                system_bits = outcome[:-1]
                index = int(system_bits, 2)
                prob_dist[index] += count
                total_success += count

        if total_success == 0:
            return np.zeros(self.original_dim), 0

        prob_dist /= np.sum(prob_dist)
        solution_padded = np.sqrt(prob_dist)
        norm = np.linalg.norm(solution_padded)
        if norm > 0:
            solution_padded /= norm

        return solution_padded[: self.original_dim], total_success

    def get_success_probability(self) -> float:
        """
        Get the probability of successful post-selection.

        Returns
        -------
        float
            Probability that the ancilla measures |1⟩.
        """
        if not self.counts:
            raise ValueError(
                "No measurement results available. Run run() first."
            )
        total = sum(self.counts.values())
        success = sum(
            c for outcome, c in self.counts.items() if outcome[-1] == "1"
        )
        return success / total if total > 0 else 0.0
