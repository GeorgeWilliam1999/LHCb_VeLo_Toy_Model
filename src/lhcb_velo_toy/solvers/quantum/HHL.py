"""
HHL.py: Standalone HHL implementation with plotting and statevector helpers.

This module provides an alternative ``HHLAlgorithm`` implementation kept
alongside :mod:`lhcb_velo_toy.solvers.quantum.hhl`. It is functionally
similar but ships with extra utilities (matplotlib histogram plotting,
direct statevector extraction) that are convenient for interactive
exploration and debugging of the Harrow-Hassidim-Lloyd algorithm on
toy track-finding linear systems.

Notes
-----
The canonical, package-exposed implementation is
:class:`lhcb_velo_toy.solvers.quantum.hhl.HHLAlgorithm`. This module is
retained as a reference / development variant and is not re-exported
through :mod:`lhcb_velo_toy.solvers.quantum.__init__`.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT, RYGate, UnitaryGate
from scipy.linalg import expm


class HHLAlgorithm:
    """
    Reference HHL (Harrow-Hassidim-Lloyd) implementation with debug helpers.

    Solves the linear system ``A x = b`` on a quantum simulator using
    quantum phase estimation, controlled rotation, and uncomputation.
    Compared with :class:`lhcb_velo_toy.solvers.quantum.hhl.HHLAlgorithm`,
    this class additionally exposes :meth:`plot_results`,
    :meth:`simulate_statevector` and :meth:`extract_postselected_solution`
    for diagnostic work.

    Parameters
    ----------
    matrix_A : numpy.ndarray
        The system matrix. Padded to a power-of-2 dimension if needed and
        symmetrised to ensure Hermiticity.
    vector_b : numpy.ndarray
        The right-hand side vector. Will be normalised internally.
    num_time_qubits : int, default 5
        Number of qubits used for phase estimation precision.
    shots : int, default 10240
        Number of measurement shots when running on the simulator.
    debug : bool, default False
        If True, print eigenvalue / rotation diagnostics during circuit
        construction.

    Attributes
    ----------
    A : ndarray
        The normalised (and possibly padded) system matrix.
    A_orig : ndarray
        Copy of the padded matrix prior to spectral normalisation.
    A_norm : float
        Frobenius norm used to rescale ``A``.
    vector_b : ndarray
        The normalised right-hand side vector.
    num_time_qubits, num_system_qubits : int
        Sizes of the time and system registers.
    t : float
        QPE evolution time, ``π / max|λ(A)|``.
    eigenvalues, eigenvalues_scaled : ndarray
        Eigenvalues of ``A_orig`` and ``A`` respectively.
    circuit : qiskit.QuantumCircuit | None
        The built HHL circuit (after :meth:`build_circuit`).
    counts : dict[str, int] | None
        Measurement counts after :meth:`run`.

    References
    ----------
    Harrow, Hassidim, Lloyd. "Quantum Algorithm for Linear Systems of
    Equations." Physical Review Letters 103, 150502 (2009).
    """

    def __init__(self, matrix_A, vector_b, num_time_qubits=5, shots=10240, debug=False):
        """Initialise registers, normalise inputs, and pre-compute spectra."""
        A = matrix_A
        self.original_dim = A.shape[0]
        self.debug = debug

        d = self.original_dim
        n_needed = math.ceil(np.log2(d))
        padded_dim = 2 ** n_needed

        if padded_dim != d:
            A_padded = np.zeros((padded_dim, padded_dim), dtype=complex)
            A_padded[:d, :d] = A
            A = (A_padded + A_padded.conj().T) / 2

            b_padded = np.zeros(padded_dim, dtype=complex)
            b_padded[:d] = vector_b
            vector_b = b_padded

        b_normalized = vector_b / np.linalg.norm(vector_b)

        self.A_orig = A.copy()
        self.A_norm = np.linalg.norm(A)
        A = A / self.A_norm

        self.A = A
        self.vector_b = b_normalized
        self.num_time_qubits = num_time_qubits
        self.shots = shots

        self.system_dim = A.shape[0]
        self.num_system_qubits = int(np.log2(self.system_dim))

        self.time_qr = QuantumRegister(self.num_time_qubits, "time")
        self.b_qr = QuantumRegister(self.num_system_qubits, "b")
        self.ancilla_qr = QuantumRegister(1, "ancilla")
        self.classical_reg = ClassicalRegister(1 + self.num_system_qubits, "c")
        self.circuit = None
        self.counts = None

        self.t = np.pi / np.max(np.abs(np.linalg.eigvals(A)))
        self.eigenvalues = np.linalg.eigvals(self.A_orig)
        self.eigenvalues_scaled = np.linalg.eigvals(self.A)

    def get_quantum_only_circuit(self):
        """
        Return a copy of :attr:`circuit` with measurements stripped.

        Useful for statevector simulation where classical operations
        must be removed.

        Returns
        -------
        qiskit.QuantumCircuit
            Circuit containing only the quantum instructions.

        Raises
        ------
        ValueError
            If :meth:`build_circuit` has not been called yet.
        """
        if self.circuit is None:
            raise ValueError("Circuit has not been built yet.")
        
        # Create a new QuantumCircuit with only quantum registers
        qc_clean = QuantumCircuit(self.time_qr, self.b_qr, self.ancilla_qr)

        # Copy only the quantum instructions
        for instr, qargs, cargs in self.circuit.data:
            if instr.name == 'measure':
                continue  # Skip measurement
            qc_clean.append(instr, qargs, cargs)

        return qc_clean

    def create_input_state(self):
        """
        Build the state-preparation circuit for ``|b⟩``.

        Returns
        -------
        qiskit.QuantumCircuit
            Sub-circuit that prepares the normalised RHS in the system
            register.
        """
        qc_b = QuantumCircuit(self.num_system_qubits)
        qc_b.initialize(self.vector_b, list(range(self.num_system_qubits)))
        return qc_b

    def apply_controlled_u(self, qc, matrix, control, target, power):
        """
        Append a controlled ``e^{i·matrix·t·power}`` to ``qc``.

        Parameters
        ----------
        qc : qiskit.QuantumCircuit
            The circuit to append to.
        matrix : ndarray
            Hamiltonian to exponentiate (typically ``±A``).
        control : Qubit
            Control qubit from the time register.
        target : list[Qubit]
            Target qubits (the system register).
        power : int
            Power of the unitary, ``2**i`` in standard QPE.
        """
        U = expm(1j * matrix * self.t * power)
        controlled_U = UnitaryGate(U).control(1)
        qc.append(controlled_U, [control] + target)
        return qc

    def inverse_qft(self, n_qubits):
        """Return an inverse QFT gate over ``n_qubits`` (with swaps)."""
        return QFT(n_qubits, do_swaps=True).inverse()

    def phase_estimation(self, qc):
        """
        Apply quantum phase estimation in-place on ``qc``.

        Hadamards the time register, applies controlled-``U^{2**i}``
        operators, then an inverse QFT.
        """
        for qubit in self.time_qr:
            qc.h(qubit)

        for i in range(self.num_time_qubits):
            power = 2 ** i
            self.apply_controlled_u(qc, self.A, self.time_qr[self.num_time_qubits - i - 1], list(self.b_qr), power)

        iqft = self.inverse_qft(self.num_time_qubits).to_gate(label="IQFT")
        qc.append(iqft, self.time_qr[:])

    def uncompute_phase_estimation(self, qc):
        """Inverse of :meth:`phase_estimation` applied in-place to ``qc``."""
        qft = QFT(self.num_time_qubits, do_swaps=True).to_gate(label="QFT")
        qc.append(qft, self.time_qr[:])

        for i in reversed(range(self.num_time_qubits)):
            power = 2 ** i
            self.apply_controlled_u(qc, -self.A, self.time_qr[self.num_time_qubits - i - 1], list(self.b_qr), power)

        for qubit in self.time_qr:
            qc.h(qubit)
    
    def R_rotation(self, qc, target_qubit):
        """Grover rotation operator I-2*|1><1| = Z on ancilla qubit"""
        #apply z to ancilla register 
        qc.z(target_qubit)

    def WRW_operator(self,qc,ancilla_qubit, target_qubit):
        """Placeholder for the W·R·W amplitude-amplification operator."""
        pass

    def build_circuit(self):
        """
        Construct the full HHL circuit.

        Returns
        -------
        qiskit.QuantumCircuit
            The complete circuit, also stored on :attr:`circuit`.

        Notes
        -----
        Steps:

        1. Prepare ``|b⟩`` in the system register.
        2. Apply phase estimation.
        3. Apply controlled R_y rotations conditioned on each eigenvalue
           bitstring (``gain = 0.3``, dropping near-zero eigenvalues).
        4. Uncompute phase estimation.
        5. Measure ancilla and system registers.
        """
        qc = QuantumCircuit(self.time_qr, self.b_qr, self.ancilla_qr, self.classical_reg)

        qc.compose(self.create_input_state(), qubits=list(self.b_qr), inplace=True)

        self.phase_estimation(qc)

        gain = 0.3
        #n_time = self.num_time_qubits
        for i in range(2 ** self.num_time_qubits):
            #phase = i / (2 ** n_time)
            #if phase >= 0.5:
            #    phase = phase - 1.0

            phase = i / (2 ** self.num_time_qubits)
            lam = 2 * np.pi * phase / self.t
            if abs(lam) < 1e-9:# or abs(lam) > 10.0:
                continue

            inv_lam = 1.0 / lam
            angle = 2 * np.arcsin(min(1.0, gain * inv_lam / 2))
            controls = list(self.time_qr)

            if self.debug:
                bits = format(i, f"0{self.num_time_qubits}b")
                print(f"Time state |{bits}>: phase = {phase:.4f}, ",
                      f"\u03bb_scaled = {lam:.4f}, \u03bb_true = {lam * self.A_norm:.4f}, ",
                      f"1/\u03bb = {inv_lam:.2f}, Ry angle = {angle:.4f}")

            bits = format(i, f"0{self.num_time_qubits}b")
            for j, bit in enumerate(bits):
                if bit == '0':
                    qc.x(self.time_qr[j])

            cry = RYGate(angle).control(num_ctrl_qubits=self.num_time_qubits)
            qc.append(cry, [*controls, self.ancilla_qr[0]])

            for j, bit in enumerate(bits):
                if bit == '0':
                    qc.x(self.time_qr[j])

        self.uncompute_phase_estimation(qc)

        qc.measure(self.ancilla_qr[0], self.classical_reg[0])
        qc.measure(self.b_qr, self.classical_reg[1:])

        self.circuit = qc
        return qc

    def run(self):
        """
        Execute :attr:`circuit` on :class:`AerSimulator`.

        Returns
        -------
        dict[str, int]
            Measurement counts, also stored on :attr:`counts`.
        """
        simulator = AerSimulator()
        transpiled_circuit = transpile(self.circuit, simulator)
        job = simulator.run(transpiled_circuit, shots=self.shots)
        result = job.result()
        self.counts = result.get_counts()
        return self.counts

    def get_solution(self):
        """
        Decode the solution vector ``x`` from :attr:`counts`.

        Post-selects on the ancilla being ``|1⟩`` and converts the
        resulting probability distribution into amplitudes, truncated
        back to the original (un-padded) dimension.

        Returns
        -------
        ndarray | None
            Unit-normalised solution vector, or ``None`` if no successful
            ancilla measurement was observed.

        Raises
        ------
        ValueError
            If :meth:`run` has not been called.
        """
        if self.counts is None:
            raise ValueError("No measurement results available. Run run() first.")

        total_success = 0
        padded_dim = 2 ** self.num_system_qubits
        prob_dist = np.zeros(padded_dim)

        for outcome, count in self.counts.items():
            if outcome[-1] == '1':
                print(f"Outcome: {outcome}, Count: {count}")
                #print(outcome[:-1])
                system_bits = outcome[:-1]#[::-1]
                #print(f"System bits: {system_bits}")
                index = int(system_bits, 2)
                prob_dist[index] += count
                total_success += count

        if total_success == 0:
            print("No valid solution: ancilla was never measured as |1⟩.")
            return None

        prob_dist = prob_dist / total_success
        solution_padded = np.sqrt(prob_dist)
        solution_padded = solution_padded / np.linalg.norm(solution_padded)

        solution_vector = solution_padded[:self.original_dim]
        solution_vector = solution_vector / np.linalg.norm(solution_vector)
        #solution_vector = solution_vector / self.A_norm
        return solution_vector

    def plot_results(self, filename="hhl_results.png"):
        """
        Save a histogram of :attr:`counts` to ``filename``.

        Parameters
        ----------
        filename : str, default "hhl_results.png"
            Output PNG path.

        Raises
        ------
        ValueError
            If :meth:`run` has not been called.
        """
        if self.counts is None:
            raise ValueError("No measurement results available. Run run() first.")
        plot_histogram(self.counts)
        plt.title("HHL Algorithm Results")
        plt.savefig(filename)
        print(f"Results histogram saved as '{filename}'.")

    def simulate_statevector(self):
        """
        Run :attr:`circuit` on the statevector simulator.

        Returns
        -------
        qiskit.quantum_info.Statevector
            Final statevector of the circuit.
        """
        from qiskit_aer import Aer
        backend = Aer.get_backend("statevector_simulator")
        job = backend.run(transpile(self.circuit, backend))
        statevector = job.result().get_statevector()
        return statevector

    def extract_postselected_solution(self, statevector):
        """
        Extract ``x`` directly from a final statevector.

        Parameters
        ----------
        statevector : qiskit.quantum_info.Statevector
            Output of :meth:`simulate_statevector`.

        Returns
        -------
        ndarray
            Unit-normalised solution vector in the system register basis,
            obtained by post-selecting on ancilla = ``|1⟩``.
        """
        dim = 2 ** (self.num_time_qubits + self.num_system_qubits + 1)
        total_qubits = self.num_time_qubits + self.num_system_qubits + 1
        system_qubits = self.num_system_qubits
        ancilla_index = self.num_time_qubits + self.num_system_qubits

        probs = np.abs(statevector.data) ** 2
        sol = np.zeros(2 ** system_qubits)

        for i, amp in enumerate(statevector.data):
            if ((i >> ancilla_index) & 1) == 1:
                system_state = (i >> 0) & ((1 << system_qubits) - 1)
                sol[system_state] += np.abs(amp) ** 2

        sol = np.sqrt(sol)
        sol = sol / np.linalg.norm(sol)
        return sol
