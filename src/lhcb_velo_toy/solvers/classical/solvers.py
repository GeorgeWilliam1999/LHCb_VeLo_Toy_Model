"""
Classical solvers for the linear system A x = b.

Provides conjugate gradient and direct solve methods for the Hamiltonian
linear system.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg, spsolve


def solve_conjugate_gradient(
    A: csc_matrix,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    maxiter: Optional[int] = None,
) -> tuple[np.ndarray, int]:
    """
    Solve A x = b using conjugate gradient method.
    
    This is an iterative method suitable for large sparse symmetric
    positive-definite systems.
    
    Parameters
    ----------
    A : csc_matrix
        The system matrix (should be symmetric positive-definite).
    b : ndarray
        The right-hand side vector.
    x0 : ndarray, optional
        Initial guess for the solution. If None, uses zeros.
    tol : float, default 1e-10
        Convergence tolerance.
    maxiter : int, optional
        Maximum number of iterations. If None, uses 10 * n.
    
    Returns
    -------
    tuple[ndarray, int]
        The solution vector and convergence info (0 = success).
    
    Examples
    --------
    >>> solution, info = solve_conjugate_gradient(A, b)
    >>> if info == 0:
    ...     print("Converged!")
    
    Notes
    -----
    Uses scipy.sparse.linalg.cg internally.
    """
    if maxiter is None:
        maxiter = 10 * A.shape[0]
    solution, info = cg(A, b, x0=x0, rtol=tol, maxiter=maxiter, atol=0)
    return solution, info


def solve_direct(
    A: csc_matrix,
    b: np.ndarray,
) -> np.ndarray:
    """
    Solve A x = b using direct LU factorization.
    
    This method is faster for small to medium systems but uses more
    memory than iterative methods.
    
    Parameters
    ----------
    A : csc_matrix
        The system matrix.
    b : ndarray
        The right-hand side vector.
    
    Returns
    -------
    ndarray
        The solution vector.
    
    Warnings
    --------
    Can consume significant memory for large matrices.
    Consider using solve_conjugate_gradient for n > 5000.
    
    Notes
    -----
    Uses scipy.sparse.linalg.spsolve internally.
    """
    return spsolve(A, b)


def select_solver(
    A: csc_matrix,
    b: np.ndarray,
    threshold: int = 5000,
) -> np.ndarray:
    """
    Automatically select and apply the best solver.
    
    Chooses between direct and iterative methods based on matrix size.
    
    Parameters
    ----------
    A : csc_matrix
        The system matrix.
    b : ndarray
        The right-hand side vector.
    threshold : int, default 5000
        Matrix dimension threshold for solver selection.
        Direct solve for n < threshold, CG for n >= threshold.
    
    Returns
    -------
    ndarray
        The solution vector.
    """
    n = A.shape[0]
    if n < threshold:
        return solve_direct(A, b)
    solution, _ = solve_conjugate_gradient(A, b)
    return solution
