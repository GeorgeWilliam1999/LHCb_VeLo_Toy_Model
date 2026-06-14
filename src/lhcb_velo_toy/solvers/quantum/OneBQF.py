import numpy as np
import math
from scipy import sparse as _sp
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT, RXGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

class OneBQF:
    def __init__(self, matrix_A, vector_b, num_time_qubits=1, shots=1024, debug=False,
                 readout='sampling'):
        if readout not in ('sampling', 'statevector'):
            raise ValueError(f"readout must be 'sampling' or 'statevector', got {readout!r}")
        self.readout = readout
        self.original_dim = matrix_A.shape[0]
        self.debug = debug

        d = self.original_dim
        n_needed = max(1, math.ceil(np.log2(d)))
        padded_dim = 2 ** n_needed

        is_sparse_input = _sp.issparse(matrix_A)

        if is_sparse_input:
            # ---- Sparse path: never materialise an O(N^2) dense matrix. -----
            # OneBQF only uses A through (diagonal_val, t, interaction_pairs).
            # Padding adds diagonal entries only and therefore does not change
            # interaction_pairs; we just track padded_dim as a scalar.
            A_sp = matrix_A.tocsr()
            diag = np.asarray(A_sp.diagonal()).ravel()
            if not np.allclose(diag, diag[0]):
                raise ValueError("Matrix A must have a constant diagonal for this scheme.")
            self.diagonal_val = diag[0]
            self.t = np.pi / self.diagonal_val

            # Symmetrised off-diagonal support: triu of (A + A.T) over k>=1.
            A_sym_offdiag = _sp.triu(A_sp + A_sp.T, k=1).tocoo()
            self.interaction_pairs = list(zip(A_sym_offdiag.row.tolist(),
                                              A_sym_offdiag.col.tolist()))

            # Pad b cheaply (vector only).
            if padded_dim != d:
                b_padded = np.ones(padded_dim)
                b_padded[:d] = vector_b
                vector_b = b_padded
            # We never store dense A in the sparse path.
            self.A = None
        else:
            # ---- Dense path (legacy, kept for backward compat). -------------
            A = matrix_A
            if padded_dim != d:
                A_padded = np.zeros((padded_dim, padded_dim))
                A_padded[:d, :d] = A
                for i in range(d, padded_dim):
                    diagonal_value = np.diag(A)[0]
                    A_padded[i, i] = diagonal_value
                A = (A_padded + A_padded.conj().T) / 2

                b_padded = np.ones(padded_dim)
                b_padded[:d] = vector_b
                vector_b = b_padded

            if not np.all(np.diag(A) == np.diag(A)[0]):
                raise ValueError("Matrix A must have a constant diagonal for this scheme.")
            self.A = A
            self.diagonal_val = np.diag(A)[0]
            self.t = np.pi / A[0, 0]

            B = self.diagonal_val * np.identity(padded_dim) - A
            rows, cols = np.where(np.triu(B) != 0)
            self.interaction_pairs = list(zip(rows, cols))

        b_normalized = vector_b / np.linalg.norm(vector_b)
        self.vector_b = b_normalized
        self.num_time_qubits = num_time_qubits
        self.shots = shots

        self.system_dim = padded_dim
        self.num_system_qubits = n_needed

        self.time_qr = QuantumRegister(self.num_time_qubits, "time")
        self.b_qr = QuantumRegister(self.num_system_qubits, "b")
        self.ancilla_qr = QuantumRegister(1, "ancilla")
        # Classical register only used for sampling mode; harmless to keep otherwise.
        self.classical_reg = ClassicalRegister(1 + self.num_system_qubits, "c")

        self.circuit = None
        self.counts = None
        
        if self.debug:
            print("--- Automated Matrix Analysis ---")
            print(f"Diagonal Value (c): {self.diagonal_val}")
            print(f"Found {len(self.interaction_pairs)} interaction pair(s): {self.interaction_pairs}")
            print("---------------------------------")
    
    def _apply_direct_controlled_u(self, qc, control_qubit, target_qubits, power, inverse=False):
        """
        Implements e^{-i H_{ij} t} exactly using Two-Level Unitary decomposition (Givens Rotation).
        This works for ANY Hamming distance and prevents spectral leakage (Ghost Couplings).
        """
        evolution_time = self.t * power
        theta = 2 * evolution_time
        
        if inverse:
            theta = -theta

        for i, j in self.interaction_pairs:
            xor_val = i ^ j
            differing_indices = [k for k in range(self.num_system_qubits) if (xor_val >> k) & 1]
            if not differing_indices: continue
            pivot = differing_indices[0]
            rest_diff = differing_indices[1:]

            for k in rest_diff:
                qc.cx(target_qubits[pivot], target_qubits[k])

            i_transformed = i
            for k in rest_diff:
                if (i_transformed >> pivot) & 1:
                    i_transformed ^= (1 << k)

            qubits_to_flip = []
            full_control_list = [control_qubit]
            
            for k in range(self.num_system_qubits):
                if k == pivot: continue 
                
                bit_val = (i_transformed >> k) & 1
                
                full_control_list.append(target_qubits[k])
                
                if bit_val == 0:
                    qubits_to_flip.append(target_qubits[k])

            if qubits_to_flip: qc.x(qubits_to_flip)
            mcrx = RXGate(theta).control(len(full_control_list))
            qc.append(mcrx, full_control_list + [target_qubits[pivot]])
            if qubits_to_flip: qc.x(qubits_to_flip)
            for k in reversed(rest_diff):
                qc.cx(target_qubits[pivot], target_qubits[k])
        phase = -self.diagonal_val * evolution_time
        if inverse: phase = -phase
        qc.p(phase, control_qubit)

    def apply_controlled_u(self, qc, control_qubit, target_qubits, power, inverse=False):
        self._apply_direct_controlled_u(qc, control_qubit, target_qubits, power, inverse=inverse)

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
            self.apply_controlled_u(self.circuit, self.time_qr[self.num_time_qubits - 1 - i], list(self.b_qr), power, inverse=True)
        qc.h(self.time_qr)

    def build_circuit(self):
        if self.readout == 'statevector':
            # No classical register / no measurements: leave the final state
            # coherent so the caller can extract amplitudes via save_statevector.
            self.circuit = QuantumCircuit(self.time_qr, self.b_qr, self.ancilla_qr)
        else:
            self.circuit = QuantumCircuit(self.time_qr, self.b_qr, self.ancilla_qr, self.classical_reg)
        self.circuit.h(self.b_qr)
        self.phase_estimation(self.circuit)
        self.circuit.x(self.time_qr[0])
        self.circuit.cx(self.time_qr[0], self.ancilla_qr[0])
        self.circuit.x(self.time_qr[0])
        self.uncompute_phase_estimation(self.circuit)
        if self.readout == 'statevector':
            # Save the full statevector for exact, shot-free post-processing.
            self.circuit.save_statevector()
        else:
            self.circuit.measure(self.ancilla_qr[0], self.classical_reg[0])
            self.circuit.measure(self.b_qr, self.classical_reg[1:])
        return self.circuit

    def run(self, use_noise_model=False, backend_name='ibm_torino'):
        """
        Run the circuit with optional noise model.
        
        Args:
            use_noise_model (bool): If True, uses the noise model from the specified backend
            backend_name (str): Name of the IBM backend to get noise model from
        """
        simulator = AerSimulator()
        
        if use_noise_model:
            # Load your IBM account (make sure you've saved your API token)
            
            service = QiskitRuntimeService()
            backend = service.backend(backend_name)
            noise_model = NoiseModel.from_backend(backend)
            basis_gates = noise_model.basis_gates
            
            print(f"\n--- Using {backend_name} Noise Model ---")
            print(f"Basis gates: {basis_gates}")
            print(f"Number of qubits: {backend.num_qubits}")
            
            pm = generate_preset_pass_manager(
                optimization_level=3,
                backend=backend
            )
            transpiled_circuit = pm.run(self.circuit)
            
            simulator = AerSimulator(noise_model=noise_model)
            job = simulator.run(transpiled_circuit, shots=self.shots)
                
        else:
            transpiled_circuit = transpile(self.circuit, simulator, optimization_level=3)
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
                system_bits = outcome[:-1]#[::-1]
                index = int(system_bits, 2)
                prob_dist[index] += count
                total_success += count

        if total_success == 0: return np.zeros(self.original_dim)
        prob_dist /= np.sum(prob_dist)
        solution_padded = np.sqrt(prob_dist)
        solution_padded /= np.linalg.norm(solution_padded)
        return solution_padded[:self.original_dim], total_success

    def get_solution_from_statevector(self, statevector):
        """Exact, shot-free solution extraction.

        Marginalises over the time qubits, selects the ancilla=1 branch, and
        returns ``(sqrt(P_i)/||sqrt(P_i)||, total_success)`` so the output has
        the same convention as :meth:`get_solution` but without any sampling
        noise. ``total_success`` here is the probability mass of the ancilla=1
        branch (P(ancilla=1)) rather than a shot count.

        Qubit ordering in the circuit is ``[time | system | ancilla]`` (time
        register added first). In Qiskit's little-endian statevector indexing
        this means a basis-state index ``k`` decomposes as::

            time  = k & (N_time - 1)
            sys   = (k >> num_time_qubits) & (N_sys - 1)
            anc   = (k >> (num_time_qubits + num_system_qubits)) & 1

        which is equivalent to reshaping psi as (2, N_sys, N_time).
        """
        psi = np.asarray(statevector).ravel()
        n_tq, n_sq = self.num_time_qubits, self.num_system_qubits
        N_time = 1 << n_tq
        N_sys = 1 << n_sq
        expected = 2 * N_sys * N_time
        if psi.size != expected:
            raise ValueError(
                f"Statevector size {psi.size} != expected {expected} "
                f"(2 ancilla * {N_sys} system * {N_time} time)"
            )
        psi3 = psi.reshape(2, N_sys, N_time)
        # Sum |amp|^2 over time qubits, select ancilla=1.
        prob_dist = np.sum(np.abs(psi3[1]) ** 2, axis=1)
        total_success = float(prob_dist.sum())
        if total_success == 0.0:
            return np.zeros(self.original_dim), 0.0
        prob_dist = prob_dist / total_success
        solution_padded = np.sqrt(prob_dist)
        solution_padded /= np.linalg.norm(solution_padded)
        return solution_padded[:self.original_dim], total_success