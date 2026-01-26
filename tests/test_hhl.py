"""
Tests for the HHL quantum algorithm implementation.
"""

import pytest
import numpy as np


class TestHHLAlgorithm:
    """Tests for HHL algorithm."""

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit not available"
    )
    def test_hhl_initialization(self, small_matrix, small_vector):
        """Test HHL algorithm can be initialized."""
        from hhl_algorithm import HHLAlgorithm
        
        hhl = HHLAlgorithm(
            matrix_A=small_matrix,
            vector_b=small_vector,
            num_time_qubits=3,
            shots=1024,
        )
        
        assert hhl.A is not None
        assert hhl.vector_b is not None

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit not available"
    )
    def test_hhl_circuit_creation(self, small_matrix, small_vector):
        """Test HHL circuit can be built."""
        from hhl_algorithm import HHLAlgorithm
        
        hhl = HHLAlgorithm(
            matrix_A=small_matrix,
            vector_b=small_vector,
            num_time_qubits=3,
            shots=1024,
        )
        hhl.build_circuit()
        
        assert hhl.circuit is not None

    def test_classical_solution_comparison(self, small_matrix, small_vector):
        """Test classical solution for reference."""
        # Classical solution using numpy
        classical_solution = np.linalg.solve(small_matrix, small_vector)
        
        # Verify the solution
        residual = np.linalg.norm(small_matrix @ classical_solution - small_vector)
        assert residual < 1e-10


class TestMatrixProperties:
    """Tests for matrix properties used in HHL."""

    def test_matrix_hermiticity(self, small_matrix):
        """Test that the test matrix is Hermitian."""
        assert np.allclose(small_matrix, small_matrix.T.conj())

    def test_matrix_eigenvalues(self, small_matrix):
        """Test eigenvalue computation."""
        eigenvalues = np.linalg.eigvals(small_matrix)
        
        # All eigenvalues should be real for Hermitian matrix
        assert np.allclose(eigenvalues.imag, 0)
        
        # All eigenvalues should be positive (positive definite)
        assert np.all(eigenvalues.real > 0)
