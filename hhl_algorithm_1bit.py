import numpy as np
import math
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator, Aer
from qiskit.circuit.library import QFT, UnitaryGate
from scipy.linalg import expm


class HHLAlgorithm:
    def __init__(self, matrix_A, vector_b, num_time_qubits=3, shots=1024):
        """
        Initialize the HHL algorithm parameters.

        This constructor:
          1. (Optionally) Ensures matrix_A is Hermitian.
          2. Pads matrix_A and vector_b so that their dimension is 2^n.
          3. Stores the original dimension.

        Args:
            matrix_A (numpy.ndarray): The input matrix A.
            vector_b (numpy.ndarray): The vector b.
            num_time_qubits (int): Number of qubits for phase estimation.
            shots (int): Number of measurement shots.
        """
        # Optionally ensure A is Hermitian.
        A = matrix_A  # Alternatively: (matrix_A + matrix_A.conj().T) / 2

        # Store original dimension.
        self.original_dim = A.shape[0]

        # Pad A and vector_b to be 2^n-dimensional.
        d = self.original_dim
        n_needed = math.ceil(np.log2(d))
        padded_dim = 2 ** n_needed

        if padded_dim != d:
            # Pad A into a padded_dim x padded_dim zero matrix.
            A_padded = np.zeros((padded_dim, padded_dim), dtype=complex)
            A_padded[:d, :d] = A
            # Make padded A Hermitian.
            A = (A_padded + A_padded.conj().T) / 2

            # Pad vector_b similarly.
            b_padded = np.zeros(padded_dim, dtype=complex)
            b_padded[:d] = vector_b
            vector_b = b_padded

        # Normalize vector b.
        b_normalized = vector_b / np.linalg.norm(vector_b)

        self.A = A
        self.vector_b = b_normalized
        self.num_time_qubits = num_time_qubits
        self.shots = shots

        # System dimension is the padded dimension.
        self.system_dim = A.shape[0]
        self.num_system_qubits = int(np.log2(self.system_dim))
        if 2 ** self.num_system_qubits != self.system_dim:
            raise ValueError("Padded dimension is not a power of 2. Check padding.")

        # Create quantum registers.
        self.time_qr = QuantumRegister(self.num_time_qubits, "time")
        self.b_qr = QuantumRegister(self.num_system_qubits, "b")
        self.ancilla_qr = QuantumRegister(1, "ancilla")
        # Create a classical register with 1 bit for ancilla and num_system_qubits bits for the system.
        self.classical_reg = ClassicalRegister(1 + self.num_system_qubits, "c")

        self.circuit = None
        self.counts = None

        # Set evolution time t (here assumed to be 1)
        self.t = 1

    def create_input_state(self):
        """
        Create a circuit to initialize the system register with the state |b⟩.
        For a one-qubit (2d) system use a rotation; for higher dimensions use initialize().
        """
        if self.num_system_qubits == 1:
            qc_b = QuantumCircuit(1)
            theta = 2 * np.arccos(self.vector_b[0])
            qc_b.h(0)
        else:
            qc_b = QuantumCircuit(self.num_system_qubits)
            qc_b.initialize(self.vector_b, list(range(self.num_system_qubits)))
        return qc_b

    def apply_controlled_u_trotter(self, qc, matrix, control, target, power=1, steps=2):
        """
        Apply a controlled-U operation corresponding to 
            U = exp(i * matrix * t_eff)
        with effective time t_eff = power * 2*pi, using a Suzuki–Trotter approximation
        with the given number of steps.

        That is, approximate:
            U ≈ (exp(i * matrix * (t_eff/steps)))^steps

        Then, create the controlled version of exp(i * matrix * (t_eff/steps))
        and apply it 'steps' times sequentially.

        Args:
            qc (QuantumCircuit): The circuit to update.
            matrix (numpy.ndarray): The matrix whose exponential is being approximated.
            control (Qubit): The control qubit.
            target (list): The target qubits (the entire system register).
            power (int): The multiplicative factor in the effective time.
            steps (int): Number of Trotter steps.
        """
        t_eff = power * 2 * np.pi * self.t
        # Compute the exponential for one step.
        U_step = expm(1j * matrix * (t_eff / steps))
        # Create its controlled version.
        controlled_U_step = UnitaryGate(U_step, label="U_step").control(1)
        # Append the controlled gate 'steps' times.
        for _ in range(steps):
            qc.append(controlled_U_step, [control] + target)
        return qc

    def inverse_qft(self, n_qubits):
        """
        Build an inverse QFT circuit on n_qubits.
        """
        qc = QuantumCircuit(n_qubits)
        # Reverse qubit order.
        for qubit in range(n_qubits // 2):
            qc.swap(qubit, n_qubits - qubit - 1)
        for j in range(n_qubits):
            for m in range(j):
                qc.cp(-np.pi / (2 ** (j - m)), m, j)
            qc.h(j)
        return qc

    def controlled_unitary(self, U):
        n = U.shape[0]
        I = np.eye(n, dtype=U.dtype)
        # Construct the controlled unitary as a block-diagonal matrix.
        controlled_U = np.block([
            [I,             np.zeros((n, n), dtype=U.dtype)],
            [np.zeros((n, n), dtype=U.dtype),         U]
        ])
        return controlled_U
    

    def apply_controlled_u_trotter_(self, matrix, power=1, trotter_steps=1):
        t_eff = power * np.pi

        U_step = expm(1j * matrix * (t_eff / trotter_steps))
        
        U_total = np.linalg.matrix_power(U_step, trotter_steps)
        
        # Build the controlled-U operator manually.
        ctrl_U = self.controlled_unitary(U_total)
        
        # Determine how many qubits the operator acts on.
        # 'matrix' is assumed to be 2^k x 2^k.
        n = matrix.shape[0]
        k = int(np.log2(n))
        
        # The controlled operator acts on one control qubit + k target qubits.
        total_qubits = 1 + k
        
        # Create a QuantumCircuit with the required number of qubits.
        qc = QuantumCircuit(total_qubits)
        
        # Convert the controlled unitary into a UnitaryGate.
        # The gate's matrix is of size 2^(k+1) x 2^(k+1).
        controlled_gate = UnitaryGate(ctrl_U, label="C-Trotter")
        
        # Append the controlled gate.
        # Here we assign qubit 0 as the control and qubits 1 to total_qubits-1 as the targets.
        qc.append(controlled_gate, list(range(total_qubits)))
        
        return qc

    def build_circuit(self):
        """
        Build the full HHL circuit:
          1. Prepare |b⟩.
          2. Apply Hadamard gates to time qubits.
          3. Perform phase estimation via Trotterized controlled-U operations.
          4. Apply the inverse QFT on time qubits.
          5. Controlled rotation using ancilla.
          6. Uncompute phase estimation via Trotterized controlled-U operations.
          7. Measure ancilla and system (b) qubits.
        """
        qc = QuantumCircuit(self.time_qr, self.b_qr, self.ancilla_qr, self.classical_reg)

        # Step 1: Prepare |b⟩.
        b_circ = self.create_input_state()
        qc.compose(b_circ, qubits=list(self.b_qr), inplace=True)

        # Step 2: Apply Hadamard to time qubits.
        for qubit in range(self.num_time_qubits):
            qc.h(self.time_qr[qubit])

        # Step 3: Phase estimation with Trotterized controlled-U operations.
        power = 2 ** (self.num_time_qubits - 1)
        # Pass the entire system register as target.
        qc = self.apply_controlled_u_trotter_(self.A)
        #qc = self.apply_controlled_u_trotter(qc, self.A, self.time_qr[0], list(self.b_qr), power, steps=1)

        # Step 4: Apply inverse QFT on time qubits.
        iqft = self.inverse_qft(self.num_time_qubits)
        qc.compose(iqft.to_gate(), qubits=self.time_qr, inplace=True)

        # Step 5: Controlled rotation using ancilla.
        qc.h(self.ancilla_qr[0])
        eigenvalues = np.linalg.eigvalsh(self.A)
        C = 0.5 / max(abs(eigenvalues))
        for i in range(2 ** self.num_time_qubits):
            binary_i = format(i, f'0{self.num_time_qubits}b')
            if i < len(eigenvalues):
                if abs(eigenvalues[i]) > 1e-10:
                    angle = 2 * np.arcsin(C / abs(eigenvalues[i]))
                    for j in range(self.num_time_qubits):
                        if binary_i[j] == '1':
                            qc.x(self.time_qr[j])
                    if self.num_time_qubits == 1:
                        qc.cry(angle, self.time_qr[0], self.ancilla_qr[0])
                    else:
                        for j in range(self.num_time_qubits):
                            qc.cry(angle / self.num_time_qubits, self.time_qr[j], self.ancilla_qr[0])
                    for j in range(self.num_time_qubits):
                        if binary_i[j] == '1':
                            qc.x(self.time_qr[j])

        # Step 6: Uncompute phase estimation using Trotterized controlled-U operations.
        qft_gate = self.inverse_qft(self.num_time_qubits).inverse()
        qc.compose(qft_gate, qubits=self.time_qr, inplace=True)
        power = 2 ** (self.num_time_qubits - 1)
        self.apply_controlled_u_trotter(qc, -self.A, self.time_qr[0], list(self.b_qr), power, steps=2)
        for qubit in range(self.num_time_qubits):
            qc.h(self.time_qr[qubit])

        # Step 7: Measure ancilla and system register.
        qc.measure(self.ancilla_qr[0], self.classical_reg[0])
        qc.measure(self.b_qr, self.classical_reg[1:])
        self.circuit = qc
        return qc

    def run(self):
        """
        Run the HHL circuit on AerSimulator.
        """
        simulator = AerSimulator()
        transpiled_circuit = transpile(self.circuit, simulator)
        job = simulator.run(transpiled_circuit, shots=self.shots)
        result = job.result()
        self.counts = result.get_counts()
        return self.counts

    def get_all_counts(self):
        """
        Return all raw measurement counts.
        """
        if self.counts is None:
            raise ValueError("No measurement results available. Run run() first.")
        return self.counts

    def get_solution(self):
        """
        Extract the HHL solution as a state vector.
        
        The classical register returns 1 bit for ancilla and self.num_system_qubits bits for the system.
        We filter measurement outcomes where the ancilla is '1' (successful post-selection).
        The measured system bits (returned in reverse order) are converted to an index.
        Then we construct a probability distribution over the padded system basis states.
        Finally, we trim the padded solution to the original system dimension.
        """
        if self.counts is None:
            raise ValueError("No measurement results available. Run run() first.")

        total_success = 0
        padded_dim = 2 ** self.num_system_qubits
        prob_dist = np.zeros(padded_dim)

        # The classical register string has length (1 + num_system_qubits): first bit = ancilla.
        for outcome, count in self.counts.items():
            if outcome[0] == '1':
                system_bits = outcome[1:][::-1]  # Reverse to fix ordering.
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
        return solution_vector

    def plot_results(self, filename="hhl_results.png"):
        """
        Plot a histogram of all raw measurement counts and save it as an image.
        """
        if self.counts is None:
            raise ValueError("No measurement results available. Run run() first.")
        plot_histogram(self.counts)
        plt.title("HHL Algorithm Results")
        plt.savefig(filename)
        print(f"Results histogram saved as '{filename}'.")


# ==========================
# Example usage of the HHLAlgorithm 1bit class.
# ==========================
if __name__ == "__main__":
    print("HHL Algorithm Implementation")
    print("=" * 60)

    matrix_A = np.array([[1.0, 0.0, -3.0, 0.0],
                         [0.0, 1.0, 0.0, -3.0],
                         [-3.0, 0.0, -3.0, 0.0],
                         [0.0, -3.0, 0.0, 1.0]])
    vector_b = np.array([1.0,1.0,1.0,1.0])
    
    print("\nSolving Ax = b with:")
    print("A =")
    print(matrix_A)
    print("b =")
    print(vector_b)
    
    hhl_solver = HHLAlgorithm(matrix_A, vector_b, num_time_qubits=1, shots=1024)
    circuit = hhl_solver.build_circuit()
    print("\nHHL Circuit:")
    print(circuit.draw(output="text"))
    
    counts = hhl_solver.run()
    print("\nRaw Measurement Counts:")
    print(counts)
    
    hhl_solver.plot_results("hhl_results.png")
    
    x_hhl = hhl_solver.get_solution()
    print("\nExtracted HHL solution (normalized):")
    print(x_hhl)
    
    x_exact = np.linalg.solve(matrix_A, vector_b)
    x_exact_normalized = x_exact / np.linalg.norm(x_exact)
    print("\nTheoretical solution (normalized):")
    print(x_exact_normalized)
