"""Tests for the Denby–Peterson Hamiltonian subclasses (centralise-2.1 Phase 1).

Each term is verified against an independent brute-force O(n^2) construction
on a seeded event, so the sparse group-wise implementation is never its own
reference. A cross-check against the originating study code
(Bifurification/dp_terms.dp_system) runs when that folder is present.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
from lhcb_velo_toy.solvers.hamiltonians import (
    ExtendedHamiltonian,
    ForkHamiltonian,
    OccupancyHamiltonian,
    SimpleHamiltonianFast,
)
from lhcb_velo_toy.analysis.segment_metrics import compute_epsilon

N_MODULES, DZ, Z0, HALF = 5, 33.0, 100.0, 50.0
GAMMA, DELTA = 3.0, 1.0
ME, CN, SEED, T = 0.01, 1e-4, 202, 12


@pytest.fixture(scope="module")
def event():
    geo = PlaneGeometry(
        module_id=list(range(N_MODULES)),
        lx=[HALF] * N_MODULES,
        ly=[HALF] * N_MODULES,
        z=[Z0 + i * DZ for i in range(N_MODULES)],
    )
    gen = StateEventGenerator(
        detector_geometry=geo, events=1, n_particles=[T],
        measurement_error=ME, collision_noise=CN, seed=SEED,
    )
    gen.generate_random_primary_vertices({"x": 0.1, "y": 0.1, "z": 50.0})
    gen.generate_particles([[{"type": "pion", "mass": 139.6, "q": 1}] * T])
    gen.generate_complete_events()
    return gen


EPS = compute_epsilon(ME, CN, dz=DZ)


def brute_force_cohit(ham, eps_window=None):
    """O(n^2) reference: 0/1 co-hit adjacency, optionally angle-windowed.

    Same-role sharing only (start-with-start or end-with-end): forks are
    competing continuations from a common slot, not chain links — the
    end-to-start relation is the compatibility already encoded in A0.
    """
    hits = np.asarray(ham._segment_to_hit_ids)
    vecs = np.asarray(ham._segment_vectors)
    n = len(hits)
    B = np.zeros((n, n))
    cos_eps = np.cos(eps_window) if eps_window is not None else None
    for i in range(n):
        for j in range(i + 1, n):
            share = hits[i][0] == hits[j][0] or hits[i][1] == hits[j][1]
            if not share:
                continue
            if cos_eps is not None:
                if not (np.clip(vecs[i] @ vecs[j], -1, 1) > cos_eps):
                    continue
            B[i, j] = B[j, i] = 1.0
    return B


def base(event):
    ham = SimpleHamiltonianFast(epsilon=EPS, gamma=GAMMA, delta=DELTA)
    A0, b0 = ham.construct_hamiltonian(event)
    return ham, A0, b0


def test_fork_matches_brute_force(event):
    beta, eps_b = 2.0, 1.0
    ham0, A0, b0 = base(event)
    ham = ForkHamiltonian(epsilon=EPS, gamma=GAMMA, delta=DELTA, beta=beta, eps_B=eps_b)
    A, b = ham.construct_hamiltonian(event)
    B_ref = brute_force_cohit(ham0, eps_window=eps_b)
    assert B_ref.sum() > 0  # the term is actually exercised
    expect = A0.toarray() + beta * B_ref
    assert np.allclose(A.toarray(), expect, atol=1e-12)
    assert np.allclose(b, b0)  # fork leaves b unchanged
    assert ham.term_info["Bfork_nnz"] == int(B_ref.sum())


def test_occupancy_matches_brute_force(event):
    alpha = 0.05
    ham0, A0, b0 = base(event)
    ham = OccupancyHamiltonian(epsilon=EPS, gamma=GAMMA, delta=DELTA, alpha=alpha)
    A, b = ham.construct_hamiltonian(event)
    B_ref = brute_force_cohit(ham0)
    assert B_ref.sum() > 0
    n = A0.shape[0]
    expect = A0.toarray() + 4 * alpha * np.eye(n) + 2 * alpha * B_ref
    assert np.allclose(A.toarray(), expect, atol=1e-12)
    assert np.allclose(b, b0 + 4 * alpha)
    # attractor-aware working point
    attr = (DELTA + 4 * alpha) / (GAMMA + DELTA + 4 * alpha)
    assert ham.working_point_tau() == pytest.approx(attr + 0.10)


def test_extended_composes_both_terms(event):
    beta, eps_b, alpha = 2.0, 1.0, 0.05
    ham0, A0, b0 = base(event)
    ham = ExtendedHamiltonian(
        epsilon=EPS, gamma=GAMMA, delta=DELTA, beta=beta, eps_B=eps_b, alpha=alpha
    )
    A, b = ham.construct_hamiltonian(event)
    n = A0.shape[0]
    expect = (
        A0.toarray()
        + beta * brute_force_cohit(ham0, eps_window=eps_b)
        + 4 * alpha * np.eye(n)
        + 2 * alpha * brute_force_cohit(ham0)
    )
    assert np.allclose(A.toarray(), expect, atol=1e-12)
    assert np.allclose(b, b0 + 4 * alpha)
    # symmetric, solvable, bounded solution
    assert (A != A.T).nnz == 0
    sol = ham.solve_classicaly()
    assert np.all(np.isfinite(sol))


def test_zero_strengths_reduce_to_base(event):
    ham0, A0, b0 = base(event)
    ham = ExtendedHamiltonian(epsilon=EPS, gamma=GAMMA, delta=DELTA)
    A, b = ham.construct_hamiltonian(event)
    assert (A != A0).nnz == 0
    assert np.array_equal(b, b0)
    assert ham.working_point_tau() == pytest.approx(DELTA / (GAMMA + DELTA) + 0.10)


def test_fork_term_stays_sparse(event):
    ham = ForkHamiltonian(epsilon=EPS, gamma=GAMMA, delta=DELTA, beta=2.0)
    A, _ = ham.construct_hamiltonian(event)
    # sparse-A invariant: the windowed fork adds ~O(n_seg) entries, never O(n^2)
    n = A.shape[0]
    assert A.nnz < 20 * n


def test_matches_originating_study_dp_system(event):
    """Cross-check against Bifurification/dp_terms.dp_system when available."""
    import importlib.util
    import sys

    dp_path = (
        "/data/bfys/gscriven/Quantum_Track_Reconstruction/"
        "Toy_Characterisation/Bifurification/dp_terms.py"
    )
    spec = importlib.util.spec_from_file_location("dp_terms", dp_path)
    if spec is None:
        pytest.skip("Bifurification study folder not present")
    dp_terms = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(dp_terms)
    except Exception as exc:  # study folder moved/absent
        pytest.skip(f"dp_terms not importable: {exc}")

    beta, eps_b, alpha = 2.0, 1.0, 0.05
    ham0, A0, b0 = base(event)
    A_ref, b_ref, tau_ref, info = dp_terms.dp_system(
        ham0, beta=beta, eps_B=eps_b, alpha=alpha, gamma=GAMMA, delta=DELTA
    )
    ham = ExtendedHamiltonian(
        epsilon=EPS, gamma=GAMMA, delta=DELTA, beta=beta, eps_B=eps_b, alpha=alpha
    )
    A, b = ham.construct_hamiltonian(event)
    assert np.allclose(A.toarray(), A_ref.toarray(), atol=1e-12)
    assert np.allclose(b, b_ref)
    assert ham.working_point_tau() == pytest.approx(tau_ref)
