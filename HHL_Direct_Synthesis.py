import numpy as np
import math
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT, RYGate, RXGate, RZGate
from scipy.linalg import expm

class HHLAlgorithm_General:
    def __init__(self, matrix_A, vector_b, num_time_qubits=2, shots=1024, debug=False):
        # --- Standard Initialization ---
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

        # --- NEW: Automated Analysis of Matrix Structure ---
        # This section automatically finds the interaction pairs from the matrix A.
        # It assumes A = c*I - B, where B describes the interactions.
        if not np.all(np.diag(A) == np.diag(A)[0]):
            raise ValueError("Matrix A must have a constant diagonal for this scheme.")
        
        self.diagonal_val = np.diag(A)[0]
        B = self.diagonal_val * np.identity(self.system_dim) - self.A
        
        # Find non-zero elements in the upper triangle to get unique interaction pairs
        rows, cols = np.where(np.triu(B) != 0)
        self.interaction_pairs = list(zip(rows, cols))
        
        if self.debug:
            print("--- Automated Matrix Analysis ---")
            print(f"Diagonal Value (c): {self.diagonal_val}")
            print(f"Found {len(self.interaction_pairs)} interaction pair(s): {self.interaction_pairs}")
            print("---------------------------------")


    def _apply_direct_controlled_u(self, qc, control_qubit, target_qubits, power, inverse=False):
        """
        Implements the controlled evolution for any multi-qubit Pauli-X type interaction.
        The 'inverse' flag correctly handles the un-computation step.
        """
        evolution_time = self.t * power
        # Base angles for the forward evolution e^(-iAt) = e^(-ictI)e^(iBt)
        theta = 2 * evolution_time
        phase = -self.diagonal_val * evolution_time
        
        # To get the inverse e^(iAt), we simply negate the angles of rotation.
        if inverse:
            theta = -theta
            phase = -phase

        for i, j in self.interaction_pairs:
            xor_val = i ^ j
            
            target_indices = [k for k, bit in enumerate(bin(xor_val)[2:][::-1]) if bit == '1']
            targets = [target_qubits[k] for k in target_indices]
            control_sys_qubits = [q for q in target_qubits if q not in targets]
            
            i_bin_reversed = format(i, f'0{self.num_system_qubits}b')[::-1]
            qubits_to_flip = []
            for q_idx, q in enumerate(target_qubits):
                if q in control_sys_qubits and i_bin_reversed[q_idx] == '0':
                    qubits_to_flip.append(q)
            
            # --- Synthesis for the controlled multi-X evolution ---
            if qubits_to_flip: qc.x(qubits_to_flip)
            qc.h(targets)
            if len(targets) > 1: qc.cx(targets[:-1], targets[-1])
            
            num_total_controls = 1 + len(control_sys_qubits)
            mcrz_gate = RZGate(theta).control(num_total_controls)
            all_controls = [control_qubit] + control_sys_qubits
            qc.append(mcrz_gate, all_controls + [targets[-1]])

            if len(targets) > 1: qc.cx(targets[:-1], targets[-1])
            qc.h(targets)
            if qubits_to_flip: qc.x(qubits_to_flip)
            # --- End of Synthesis ---

        qc.p(phase, control_qubit)


    def apply_controlled_u(self, qc, control_qubit, target_qubits, power, inverse=False):
        self._apply_direct_controlled_u(qc, control_qubit, target_qubits, power, inverse=inverse)

    # --- The rest of the class remains the same ---
    def inverse_qft(self, n_qubits):
        return QFT(n_qubits, do_swaps=True).inverse()

    def phase_estimation(self, qc):
        qc.h(self.time_qr)
        for i in range(self.num_time_qubits):
            power = 2**i
            self.apply_controlled_u(self.circuit, self.time_qr[self.num_time_qubits - 1 - i], list(self.b_qr), power)
        qc.append(self.inverse_qft(self.num_time_qubits).to_gate(label="IQFT"), self.time_qr)

    def uncompute_phase_estimation(self, qc):
        qc.append(QFT(self.num_time_qubits, do_swaps=True).to_gate(label="QFT"), self.time_qr)
        for i in reversed(range(self.num_time_qubits)):
            power = 2**i
            # We now call the function with inverse=True instead of using negative power
            self.apply_controlled_u(self.circuit, self.time_qr[self.num_time_qubits - 1 - i], list(self.b_qr), power, inverse=True)
        qc.h(self.time_qr)

    def build_circuit(self):
        self.circuit = QuantumCircuit(self.time_qr, self.b_qr, self.ancilla_qr, self.classical_reg)
        self.circuit.h(self.b_qr)
        self.phase_estimation(self.circuit)

        for i in range(2**self.num_time_qubits):
            phase = i / (2**self.num_time_qubits)
            lam = 2 * np.pi * phase / self.t
            print(lam)
            if  abs(lam) < 6: continue
            angle = np.pi
            #angle = 2 * np.arcsin(min(1.0, 0.3 * lam / 2))
            bits = format(i, f"0{self.num_time_qubits}b")
            for j, bit in enumerate(bits):
                if bit == '0': self.circuit.x(self.time_qr[j])
            cry = RYGate(angle).control(num_ctrl_qubits=self.num_time_qubits)
            self.circuit.append(cry, [*self.time_qr, self.ancilla_qr[0]])
            for j, bit in enumerate(bits):
                if bit == '0': self.circuit.x(self.time_qr[j])

        self.uncompute_phase_estimation(self.circuit)
        self.circuit.measure(self.ancilla_qr[0], self.classical_reg[0])
        self.circuit.measure(self.b_qr, self.classical_reg[1:])
        return self.circuit

    def run(self):
        simulator = AerSimulator()
        transpiled_circuit = transpile(self.circuit, simulator, optimization_level=2)
        job = simulator.run(transpiled_circuit, shots=self.shots)
        result = job.result()
        self.counts = result.get_counts()
        return self.counts

    def get_solution(self, counts=None):
        if counts: self.counts = counts
        if not self.counts: raise ValueError("No measurement results available.")
        total_success, prob_dist = 0, np.zeros(2**self.num_system_qubits)
        for outcome, count in self.counts.items():
            if outcome[-1] == '1':
                index = int(outcome[:-1], 2)
                prob_dist[index] += count
                total_success += count
        if total_success == 0: return np.zeros(self.original_dim)
        prob_dist /= np.sum(prob_dist)
        solution_padded = np.sqrt(prob_dist)
        solution_padded /= np.linalg.norm(solution_padded)
        return solution_padded[:self.original_dim]
    


matrix_A = np.array([[ 3.        ,  0.        ,  0.        ,  0.        , -1,
                        0.        ,  0.        ,  0.        ],
                        [ 0.        ,  3.        ,  0.        ,  0.        ,  0.        ,
                        0.        ,  0.        ,  0.        ],
                        [ 0.        ,  0.        ,  3.        ,  0.        ,  0.        ,
                        0.        ,  0.        ,  0.        ],
                        [ 0.        ,  0.        ,  0.        ,  3.        ,  0.        ,
                        0.        ,  0.        , -1],
                        [-1,  0.        ,  0.        ,  0.        ,  3.        ,
                        0.        ,  0.        ,  0.        ],
                        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                        3.        ,  0.        ,  0.        ],
                        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                        0.        ,  3.        ,  0.        ],
                        [ 0.        ,  0.        ,  0.        , -1,  0.        ,
                        0.        ,  0.        ,  3.        ]])

vector_b = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) 

# --- Using the new class ---
# REMOVED: The line `HHLAlgorithm = add_suzuki_trotter_to_class(HHLAlgorithm)` is gone.

# The HHLAlgorithm class now uses the efficient method by default.
hhl_solver = HHLAlgorithm_General(matrix_A, vector_b, num_time_qubits=2, shots=16000, debug=True)
circuit = hhl_solver.build_circuit()

print("--- Circuit using Gate-Efficient Direct Synthesis ---")
print(circuit.draw(output="text"))

counts = hhl_solver.run()
print("\n--- Results ---")

print('counts:', counts)
#hhl_solver.plot_results("hhl_results.png")
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

'''
from pytket import Circuit, OpType
from pytket.utils import gate_counts
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.circuit.display import render_circuit_jupyter

from pytket.passes import RemoveRedundancies, CliffordSimp, AutoRebase, KAKDecomposition, SequencePass, FullPeepholeOptimise

circuit_decomposed = circuit.decompose().decompose().decompose().decompose().decompose().decompose()
tk = qiskit_to_tk(circuit_decomposed)
circuit_tk = qiskit_to_tk(circuit_decomposed)
print(gate_counts(circuit_tk))

#RemoveRedundancies().apply(circuit_tk)
#CliffordSimp().apply(circuit_tk)
#KAKDecomposition().apply(circuit_tk)
#rebase_quil = AutoRebase({OpType.CZ, OpType.Rz, OpType.Rx})
FullPeepholeOptimise().apply(circuit_tk)


print(gate_counts(circuit_tk))

render_circuit_jupyter(circuit_tk)

from pytket.extensions.qiskit import AerBackend
from pytket.circuit import Circuit, BasisOrder
backend = AerBackend()
compiled_circ = backend.get_compiled_circuit(circuit_tk)
handle = backend.process_circuit(compiled_circ, n_shots=10000)
result = backend.get_result(handle)

print(result.get_counts(basis=BasisOrder.dlo)) 

print("--- Circuit pytket Gate-Efficient Direct Synthesis ---")
qiskit_to_tk_circ = tk_to_qiskit(compiled_circ)
#print(qiskit_to_tk_circ.draw(output="text"))

print("\n--- Results ---")

print('counts:', counts)
#hhl_solver.plot_results("hhl_results.png")
x_hhl = hhl_solver.get_solution()
print("\nHHL Solution:", x_hhl)

x_exact = np.linalg.solve(matrix_A, vector_b)
x_exact_normalized = x_exact / np.linalg.norm(x_exact)
print("\nExact Solution:", x_exact_normalized)

if x_hhl is not None:
    fidelity = np.abs(np.vdot(x_hhl, x_exact_normalized))
    print(f"\nFidelity with exact solution: {fidelity:.4f}")

circuit_depth = qiskit_to_tk_circ.decompose().decompose().decompose().decompose().decompose().decompose().depth()
gate_statistics = qiskit_to_tk_circ.decompose().decompose().decompose().decompose().decompose().decompose().count_ops()
print(f"The depth of the quantum circuit is: {circuit_depth}")
print("Gate statistics for the circuit:")
print(gate_statistics)'''