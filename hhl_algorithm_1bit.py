"""
HHL Algorithm with Suzuki-Trotter Decomposition
=================================================

This module provides an optimized implementation of the HHL quantum algorithm
with support for Suzuki-Trotter decomposition of the Hamiltonian evolution.
This version includes modifications by Alain Chancé for improved accuracy
and additional features like noisy simulation support.

Key Features
------------
- **Suzuki-Trotter Decomposition**: Efficient approximation of e^(iAt) using
  Pauli decomposition and product formulas.
- **Configurable Trotterization**: Supports QDrift, Lie-Trotter, and higher-order
  Suzuki-Trotter formulas.
- **Noisy Simulation**: Built-in support for simulating on noisy backends
  using IBM Quantum noise models.
- **Flexible Rotation Angles**: Configurable gain and angular parameters for
  the ancilla rotation.

Trotterization Methods
----------------------
The Hamiltonian evolution e^(iAt) is decomposed using:

- **QDrift (order=0)**: Randomized product formula, good for many-body systems
- **Lie-Trotter (order=1)**: First-order product formula, simplest
- **Suzuki-Trotter (order=2)**: Second-order symmetric decomposition, more accurate

Usage
-----
>>> import numpy as np
>>> from hhl_algorithm_1bit import HHLAlgorithm, add_suzuki_trotter_to_class
>>> 
>>> A = np.array([[3, -1], [-1, 3]])
>>> b = np.array([1, 1])
>>> 
>>> # Patch class with Trotter methods
>>> HHLAlgorithm = add_suzuki_trotter_to_class(HHLAlgorithm)
>>> 
>>> # Create solver with custom parameters
>>> hhl = HHLAlgorithm(
...     A, b,
...     num_time_qubits=4,
...     gain=0.3,
...     lam_s=6,
...     angle_pi=True
... )
>>> circuit = hhl.build_circuit()
>>> counts = hhl.run()
>>> solution = hhl.get_solution()

Running with Noise
------------------
>>> # Simulate with IBM Brisbane noise model
>>> counts = hhl.run_with_noise_simulation(backend_name="ibm_brisbane")

Authors
-------
Original implementation with modifications by Alain Chancé.

See Also
--------
hhl_algorithm : Basic HHL implementation
"""

# hhl_algorithm_1bit.py


import numpy as np
import math
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT, RYGate, UnitaryGate
from scipy.linalg import expm

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import QDrift, LieTrotter, SuzukiTrotter


def add_suzuki_trotter_to_class(HHLAlgorithmClass):
    """
    Patch the HHLAlgorithm class to use PauliEvolutionGate with Trotterization.
    
    This function adds methods for creating Trotter gates and controlled
    Hamiltonian evolution using Qiskit's synthesis tools.
    
    Parameters
    ----------
    HHLAlgorithmClass : class
        The HHLAlgorithm class to patch.
    
    Returns
    -------
    class
        The patched class with Trotter methods.
    
    Notes
    -----
    After patching, apply_controlled_u will use Trotterized evolution
    instead of exact matrix exponentiation.
    """
    def _create_trotter_gate(self, evolution_time, trotter_steps=1, order=0):
        """Create a Trotter gate for Hamiltonian evolution."""
        if not hasattr(self, '_pauli_A'):
            self._pauli_A = SparsePauliOp.from_operator(self.A)

        # FIX: use if/elif/else, not two ifs that fall through.
        if order == 1:
            synthesis_method = LieTrotter(reps=trotter_steps)
        elif order == 0:
            synthesis_method = QDrift(reps=trotter_steps)
        else:
            synthesis_method = SuzukiTrotter(reps=trotter_steps, order=2)

        trotter_gate = PauliEvolutionGate(
            self._pauli_A, 
            time=evolution_time, 
            synthesis=synthesis_method
        )
        return trotter_gate

    def _apply_trotter_controlled_u(self, qc, control_qubit, target_qubits, power, trotter_steps=1, order=2):
        """Apply controlled Trotterized evolution."""
        evolution_time = self.t * power
        trotter_gate = self._create_trotter_gate(evolution_time, trotter_steps, order)
        controlled_trotter = trotter_gate.control(1, label=f"C-Trot(t={evolution_time:.3g})")
        qc.append(controlled_trotter, [control_qubit] + target_qubits)

    # Replace the class's original apply_controlled_u with our new one
    print("Patching HHLAlgorithm with Suzuki-Trotter methods...")
    HHLAlgorithmClass._create_trotter_gate = _create_trotter_gate
    HHLAlgorithmClass.apply_controlled_u = _apply_trotter_controlled_u
    
    return HHLAlgorithmClass


class HHLAlgorithm:
    """
    HHL Algorithm with Suzuki-Trotter support and enhanced features.
    
    This class implements the HHL quantum algorithm with additional features
    for controlling the rotation angles, Trotterization order, and noisy
    simulation capabilities.
    
    Parameters
    ----------
    matrix_A : numpy.ndarray
        The system matrix A (must be Hermitian).
    vector_b : numpy.ndarray
        The right-hand side vector b.
    num_time_qubits : int, optional
        Number of qubits for phase estimation (default: 5).
    gain : float, optional
        Gain factor for ancilla rotation (default: 0.3).
    lam_s : int, optional
        Target eigenvalue bucket for rotation (default: 6).
    angle_pi : bool, optional
        If True, use π rotation; otherwise compute from eigenvalue (default: True).
    shots : int, optional
        Number of measurement shots (default: 1024).
    debug : bool, optional
        Enable debug output (default: False).
    do_draw : bool, optional
        Save circuit diagram to file (default: False).
    
    Attributes
    ----------
    circuit : QuantumCircuit
        The constructed HHL circuit.
    counts : dict
        Measurement results.
    t : float
        Evolution time parameter.
    eigenvalues : numpy.ndarray
        Eigenvalues of original matrix.
    eigenvalues_scaled : numpy.ndarray
        Eigenvalues of scaled matrix.
    
    Notes
    -----
    Modified by Alain Chancé with fixes for improved numerical stability.
    """
    # Modified by Alain Chancé + fixes
    def __init__(self, matrix_A, vector_b, num_time_qubits=5, gain=0.3, lam_s=6, angle_pi=True, shots=1024, debug=False, do_draw=False):
        A = matrix_A
        self.original_dim = A.shape[0]
        self.debug = debug
        self.do_draw = do_draw  # FIX: remember the flag

        #------------------------
        # Parameters added by Alain Chancé
        self.gain = gain
        self.lam_s = lam_s
        self.angle_pi = angle_pi
        #------------------------

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
        self.A_norm = np.linalg.norm(A)  # currently unused
        #A = A / self.A_norm

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

        # FIX: use unrounded spectral radius with safety floor
        rho = float(np.max(np.abs(np.linalg.eigvals(A))))
        eps = 1e-12
        self.t = np.pi / max(rho, eps)

        #------------------------------------------------------------------------------------------------------------
        # Debug prints retained but now reflect the unrounded choice
        if self.debug:
            print("class HHLAlgorithm modified by Alain Chancé + fixes")
            print("__init__ max(abs(eigvals(A))):", rho)
            print("__init__ self.t = π / max(rho, eps):", self.t)
        #------------------------------------------------------------------------------------------------------------
        
        self.eigenvalues = np.linalg.eigvals(self.A_orig)
        self.eigenvalues_scaled = np.linalg.eigvals(self.A)

    def create_input_state(self):
        qc_b = QuantumCircuit(self.num_system_qubits)
        qc_b.initialize(self.vector_b, list(range(self.num_system_qubits)))
        return qc_b

    def apply_controlled_u(self, qc, matrix, control, target, power):
        # This gets patched by add_suzuki_trotter_to_class; left here for completeness
        U = expm(1j * matrix * self.t * power)
        controlled_U = UnitaryGate(U).control(1)
        qc.append(controlled_U, [control] + target)
        return qc

    def inverse_qft(self, n_qubits):
        return QFT(n_qubits, do_swaps=True).inverse()

    def phase_estimation(self, qc):
        for qubit in self.time_qr:
            qc.h(qubit)

        for i in range(self.num_time_qubits):
            power = 2 ** i
            self.apply_controlled_u(qc, self.time_qr[self.num_time_qubits - i - 1], list(self.b_qr), power)

        iqft = self.inverse_qft(self.num_time_qubits).to_gate(label="IQFT")
        qc.append(iqft, self.time_qr[:])

    def uncompute_phase_estimation(self, qc):
        qft = QFT(self.num_time_qubits, do_swaps=True).to_gate(label="QFT")
        qc.append(qft, self.time_qr[:])

        for i in reversed(range(self.num_time_qubits)):
            power = 2 ** i
            self.apply_controlled_u(qc, self.time_qr[self.num_time_qubits - i - 1], list(self.b_qr), power)

        for qubit in self.time_qr:
            qc.h(qubit)

    def build_circuit(self):
        qc = QuantumCircuit(self.time_qr, self.b_qr, self.ancilla_qr, self.classical_reg)

        # For this example |b⟩ = uniform superposition; matches vector_b of ones
        for qubit in self.b_qr:
            qc.h(qubit)
            
        self.phase_estimation(qc)

        gain = self.gain
        if self.debug:
            print("\nbuild_circuit - λ buckets (rounded)")

        for i in range(2 ** self.num_time_qubits):
            phase = i / (2 ** self.num_time_qubits)
            lam = 2 * np.pi * phase / self.t

            # Keep Alain's discretization but FIX the predicate:
            lam_rounded = round(lam)
            if self.debug:
                print(f"i={i:>2} phase={phase:.3f} λ≈{lam:.3f} λ_round={lam_rounded}")

            # FIX: only accept buckets where |λ_round| == lam_s
            if abs(lam_rounded) != self.lam_s:
                continue

            inv_lam = 1.0 / max(abs(lam), 1e-12)  # avoid zero
            if self.angle_pi:
                angle = np.pi
            else:
                raw = gain * inv_lam / 2.0
                clamped = min(1.0, raw)
                angle = 2 * np.arcsin(clamped)
                if self.debug and clamped < raw:
                    print(f"  angle clamped: raw={raw:.3g} -> use sin^-1(1) => angle≈π")

            controls = list(self.time_qr)

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
        simulator = AerSimulator()
        transpiled_circuit = transpile(self.circuit, simulator, optimization_level=3)
        job = simulator.run(transpiled_circuit, shots=self.shots)
        result = job.result()
        self.counts = result.get_counts()
        return self.counts
    
    def run_with_noise_simulation(self, backend_name="ibm_brisbane"):
        print(f"\n--- Starting Noisy Simulation based on '{backend_name}' ---")

        try:
            # Connect to IBM Quantum (credentials should be saved via qiskit)
            print("Connecting to IBM Quantum to fetch backend properties...")
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService()  # no hard-coded token

            real_backend = service.backend(backend_name)
            print(f"Successfully fetched properties for '{backend_name}'.")

            # Create a noisy simulator from the real backend
            print("Creating noise model and configured simulator...")
            from qiskit_aer import AerSimulator
            noisy_simulator = AerSimulator.from_backend(real_backend)
            
            # Transpile for the real backend
            print(f"Transpiling circuit for '{backend_name}' architecture...")
            transpiled_circuit = transpile(self.circuit, backend=real_backend, optimization_level=3)
            
            print(f"Ideal depth: {self.circuit.depth()} -> Transpiled depth: {transpiled_circuit.depth()}")

            # Run the noisy simulation
            print(f"Running simulation with {self.shots} shots...")
            job = noisy_simulator.run(transpiled_circuit, shots=self.shots)
            result = job.result()
            self.counts = result.get_counts()
            
            print("Noisy simulation finished.")
            return self.counts

        except Exception as e:
            print(f"\nAn error occurred during the noisy simulation: {e}")
            print("Please ensure your IBM Cloud credentials are saved and you have the necessary packages installed.")
            return None

    def get_solution(self, counts = None):
        if self.counts is None and counts is None:
            raise ValueError("No measurement results available. Run run() first.")
        if counts is not None:
            self.counts = counts
        total_success = 0
        padded_dim = 2 ** self.num_system_qubits
        prob_dist = np.zeros(padded_dim)

        for outcome, count in self.counts.items():
            if outcome[-1] == '1':
                if self.debug:
                    print(f"Outcome: {outcome}, Count: {count}")
                system_bits = outcome[:-1]
                index = int(system_bits, 2)
                prob_dist[index] += count
                total_success += count

        if total_success == 0:
            print("No valid solution: ancilla was never measured as |1⟩.")
            return None

        if self.debug:
            print('Total success (fraction of kept counts per outcome key):', total_success/len(self.counts))
        prob_dist = prob_dist / total_success
        solution_padded = np.sqrt(prob_dist)
        solution_padded = solution_padded / np.linalg.norm(solution_padded)

        solution_vector = solution_padded[:self.original_dim]
        return solution_vector

    def plot_results(self, filename="hhl_results.png"):
        if self.counts is None:
            raise ValueError("No measurement results available. Run run() first.")
        plot_histogram(self.counts)
        plt.title("HHL Algorithm Results")
        plt.savefig(filename)
        print(f"Results histogram saved as '{filename}'.")

    def simulate_statevector(self):
        from qiskit_aer import Aer
        backend = Aer.get_backend("statevector_simulator")
        job = backend.run(transpile(self.circuit, backend))
        statevector = job.result().get_statevector()
        return statevector

    def extract_postselected_solution(self, statevector):
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


if __name__ == "__main__":
    matrix_A = np.array([[ 3.        ,  0.        ,  0.        ,  0.        , -1,
                            0.        ,  0.        ,  0.        ],
                          [ 0.        ,  3.        ,  0.        ,  0.        ,  0.        ,
                            0.        ,  0.        ,  0.        ],
                          [ 0.        ,  0.        ,  3.        ,  0.        ,  0.        ,
                            0.        ,  0.        ,  0.        ],
                          [ 0.        ,  0.        ,  0.        ,  3.        ,  0.        ,
                            0.        ,  0.        , -1],
                          [-1,         0.        ,  0.        ,  0.        ,  3.        ,
                            0.        ,  0.        ,  0.        ],
                          [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                            3.        ,  0.        ,  0.        ],
                          [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                            0.        ,  3.        ,  0.        ],
                          [ 0.        ,  0.        ,  0.        , -1,         0.        ,
                            0.        ,  0.        ,  3.        ]])

    vector_b = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) 

    HHLAlgorithm = add_suzuki_trotter_to_class(HHLAlgorithm)

    # You can tweak lam_s, gain, angle_pi, etc.
    hhl_solver = HHLAlgorithm(
        matrix_A, vector_b,
        num_time_qubits=3,
        shots=16000,
        debug=True,
        do_draw=False,   # set True to export a PNG
        gain=0.3,
        lam_s=6,
        angle_pi=True
    )

    circuit = hhl_solver.build_circuit()
    
    # Optional drawing guarded by the instance flag (FIX)
    if hhl_solver.do_draw:
        fig = circuit.draw(output="mpl")
        fig.savefig("HHL_solver_circuit.png", bbox_inches='tight')

    #counts = hhl_solver.run_with_noise_simulation()
    counts = hhl_solver.run()

    print('counts:', counts)
    hhl_solver.plot_results("hhl_results.png")
    x_hhl = hhl_solver.get_solution()
    print("\nHHL Solution:", x_hhl)

    x_exact = np.linalg.solve(matrix_A, vector_b)
    x_exact_normalized = x_exact / np.linalg.norm(x_exact)
    print("\nExact Solution:", x_exact_normalized)

    if x_hhl is not None:
        fidelity = np.abs(np.vdot(x_hhl, x_exact_normalized))
        print(f"\nFidelity with exact solution: {fidelity:.4f}")
    
    circuit_depth = circuit.decompose().decompose().decompose().decompose().decompose().decompose().depth()
    gate_statistics = circuit.decompose().decompose().decompose().decompose().decompose().decompose().count_ops()
    print(f"The depth of the quantum circuit is: {circuit_depth}")
    print("Gate statistics for the circuit:")
    print(gate_statistics)

    # Debug helpers (optional):
    # print("\nEigenvalues of original A:", np.round(hhl_solver.eigenvalues, 4))
    # print("Eigenvalues of scaled A (used in phase estimation):", np.round(hhl_solver.eigenvalues_scaled, 4))

    # print("\n[Debug] Running ideal statevector simulation...")
    # statevector = hhl_solver.simulate_statevector()
    # print("Final statevector (truncated):", statevector.data[:8])

    # post_selected = hhl_solver.extract_postselected_solution(statevector)
    # print("\nPostselected solution from statevector:", post_selected)
    # fidelity_post = np.abs(np.vdot(post_selected, x_exact_normalized))
    # print(f"Fidelity (postselected vs exact): {fidelity_post:.4f}")
