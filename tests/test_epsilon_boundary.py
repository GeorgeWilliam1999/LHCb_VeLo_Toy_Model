"""Epsilon-boundary convention test (centralise-2.1 Phase 0).

Canonical ruling (2026-08-26): a segment pair whose opening angle is
EXACTLY epsilon is EXCLUDED — the compatibility test is the strict
``cos(angle) > cos(epsilon)``, on both the dense and cKDTree paths of
``SimpleHamiltonianFast.construct_hamiltonian``.

Known historical deviation: Verify_new_results/scripts/run_fig14_extend.py
used ``>=`` (inclusive); its stored outputs predate this ruling.

The test injects synthetic segment arrays directly (bypassing
construct_segments) so the pair's dot product is bit-exactly
``np.cos(epsilon)``: v1 = (0, 0, 1) and v2 = (sin a, 0, cos a) give
v1 . v2 == np.cos(a) exactly (multiplications by 0/1 and additions of 0
are exact in IEEE754), and construct_hamiltonian computes its threshold
with the same ``np.cos(epsilon)`` call.
"""
import numpy as np

from lhcb_velo_toy.solvers.hamiltonians.fast import SimpleHamiltonianFast


def _ham_with_pair(angle_between, epsilon):
    """Two segments in adjacent groups sharing middle hit id 1."""
    ham = SimpleHamiltonianFast(epsilon=epsilon, gamma=3.0, delta=1.0)
    ham.n_segments = 2
    ham._group_boundaries = [0, 1, 2]
    ham._segment_to_hit_ids = np.array([[0, 1], [1, 2]], dtype=np.int64)
    ham._segment_vectors = np.array(
        [
            [0.0, 0.0, 1.0],
            [np.sin(angle_between), 0.0, np.cos(angle_between)],
        ],
        dtype=np.float64,
    )
    return ham


def test_pair_exactly_at_epsilon_is_excluded():
    eps = 0.01
    ham = _ham_with_pair(angle_between=eps, epsilon=eps)
    A, b = ham.construct_hamiltonian(event=None)
    # Only the two diagonal entries — the exactly-at-threshold pair is out.
    assert A.nnz == 2
    assert np.all(A.diagonal() == 4.0)


def test_pair_just_inside_epsilon_is_included():
    eps = 0.01
    ham = _ham_with_pair(angle_between=0.999 * eps, epsilon=eps)
    A, b = ham.construct_hamiltonian(event=None)
    # Diagonal plus the symmetric off-diagonal coupling.
    assert A.nnz == 4
    assert A[0, 1] == -1.0 and A[1, 0] == -1.0


def test_pair_outside_epsilon_is_excluded():
    eps = 0.01
    ham = _ham_with_pair(angle_between=1.5 * eps, epsilon=eps)
    A, b = ham.construct_hamiltonian(event=None)
    assert A.nnz == 2
