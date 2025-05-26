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

    def apply_controlled_u(self, qc, matrix, control, target, power):
        """
        Apply a controlled-U operation corresponding to 
            U = exp(i * matrix * t)
        with effective time t = 1.

        Args:
            qc (QuantumCircuit): The circuit to update.
            matrix (numpy.ndarray): The matrix whose exponential is being approximated.
            control (Qubit): The control qubit.
            target (list): The target qubits (the entire system register).
        """
        U = expm(1j * matrix * self.t * power)
        controlled_U = UnitaryGate(U, label="U").control(1)
        print(f"Applying controlled-U with power {power} on qubit {control} to qubits {target}.")
        qc.append(controlled_U, [control] + target)
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

    def build_circuit(self):
        """
        Build the full HHL circuit:
          1. Prepare |b⟩.
          2. Apply Hadamard gates to time qubits.
          3. Perform phase estimation via controlled-U operations.
          4. Apply the inverse QFT on time qubits.
          5. Controlled rotation using ancilla.
          6. Uncompute phase estimation.
          7. Measure ancilla and system (b) qubits.
        """
        qc = QuantumCircuit(self.time_qr, self.b_qr, self.ancilla_qr, self.classical_reg)

        # Step 1: Prepare |b⟩.
        b_circ = self.create_input_state()
        qc.compose(b_circ, qubits=list(self.b_qr), inplace=True)

        # Step 2: Hadamard on time qubits.
        for qubit in range(self.num_time_qubits):
            qc.h(self.time_qr[qubit])

        # Step 3: Phase estimation with controlled-U operations.
        for i in range(self.num_time_qubits):
            power = 2 ** (self.num_time_qubits - i - 1)
            qc = self.apply_controlled_u(qc, self.A, self.time_qr[i], list(self.b_qr), power)

        # Step 4: Inverse QFT on time qubits.
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

        # Step 6: Uncompute phase estimation.
        qft_gate = self.inverse_qft(self.num_time_qubits).inverse()
        qc.compose(qft_gate, qubits=self.time_qr, inplace=True)
        for i in reversed(range(self.num_time_qubits)):
            power = 2 ** (self.num_time_qubits - i - 1)
            self.apply_controlled_u(qc, -self.A, self.time_qr[i], list(self.b_qr), power)
        for qubit in range(self.num_time_qubits):
            qc.h(self.time_qr[qubit])

        # Step 7: Measure ancilla and system register.
        qc.measure(self.ancilla_qr[0], self.classical_reg[0])
        # Measure the entire system register. Note: Qiskit preserves the order when using list(self.b_qr)
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
        The measured system bits (which are returned in reverse order) are converted to an index.
        Then we construct a padded probability distribution over the system basis states.
        Finally, we trim the padded solution to the original system dimension.
        """
        if self.counts is None:
            raise ValueError("No measurement results available. Run run() first.")

        # Total number of shots from successful post-selection.
        total_success = 0
        # Create a probability vector of length equal to the padded system dimension.
        padded_dim = 2 ** self.num_system_qubits
        prob_dist = np.zeros(padded_dim)

        # The classical register is a string of length (1 + num_system_qubits) where the first bit is ancilla.
        for outcome, count in self.counts.items():
            # If ancilla is measured as '1'
            if outcome[0] == '1':
                # The remaining bits correspond to the system register.
                # Reverse the system bits to correct the ordering.
                system_bits = outcome[1:][::-1]
                index = int(system_bits, 2)
                prob_dist[index] += count
                total_success += count

        if total_success == 0:
            print("No valid solution: ancilla was never measured as |1⟩.")
            return None

        # Normalize the probability distribution.
        prob_dist = prob_dist / total_success
        # The solution vector amplitudes (up to a global phase) are the square roots of these probabilities.
        solution_padded = np.sqrt(prob_dist)
        solution_padded = solution_padded / np.linalg.norm(solution_padded)

        # Trim the padded solution to the original dimension.
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
# Example usage of the HHLAlgorithm class.
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
    
    # Create an instance of the HHL algorithm.
    hhl_solver = HHLAlgorithm(matrix_A, vector_b, num_time_qubits=4, shots=2048)
    circuit = hhl_solver.build_circuit()
    print("\nHHL Circuit:")
    print(circuit.draw(output="text"))
    
    # Run the circuit.
    counts = hhl_solver.run()
    print("\nRaw Measurement Counts:")
    print(counts)
    
    # Plot the histogram of all measurement results.
    hhl_solver.plot_results("hhl_results.png")
    
    # Extract the HHL solution (trimmed to the original dimension).
    x_hhl = hhl_solver.get_solution()
    print("\nExtracted HHL solution (normalized):")
    print(x_hhl)
    
    # Compute the theoretical solution for comparison (using the original system, not the padded one).
    x_exact = np.linalg.solve(matrix_A, vector_b)
    x_exact_normalized = x_exact / np.linalg.norm(x_exact)
    print("\nTheoretical solution (normalized):")
    print(x_exact_normalized)
