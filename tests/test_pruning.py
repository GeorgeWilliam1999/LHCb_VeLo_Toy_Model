"""Pruning of decoupled segments (2026-10-13).

The claim under test: removing the segments that have no compatible
neighbour (or whole small components) changes neither the classical solution
nor the one-bit filter nor the QSVT comb on the segments that remain; it only
shrinks the register and raises the post-selection probability.  Every check
compares the pruned solve against the full solve restricted to the kept
segments, and the kept set itself against a brute-force recount.
"""
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from lhcb_velo_toy.analysis.segment_metrics import compute_epsilon, segment_truth_mask
from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
from lhcb_velo_toy.solvers.hamiltonians import (
    ForkHamiltonian,
    OccupancyHamiltonian,
    SimpleHamiltonian,
    SimpleHamiltonianFast,
    decoupled_mask,
    expand_solution,
    qubits_for,
    segment_degree,
)
from lhcb_velo_toy.solvers.quantum import QSVT, solve_one_bit_matrixfree, spectral_bounds

N_MODULES, DZ, Z0, HALF = 5, 33.0, 33.0, 40.0
GAMMA, DELTA = 3.0, 1.0
ME, CN, SEED, T = 0.005, 1.5e-4, 7, 30
EPS = compute_epsilon(ME, CN, dz=DZ)


@pytest.fixture(scope="module")
def event():
    geo = PlaneGeometry(module_id=list(range(N_MODULES)), lx=[HALF] * N_MODULES,
                        ly=[HALF] * N_MODULES, z=[Z0 + i * DZ for i in range(N_MODULES)])
    gen = StateEventGenerator(detector_geometry=geo, events=1, n_particles=[T],
                              measurement_error=ME, collision_noise=CN, seed=SEED,
                              phi_min=-0.2, phi_max=0.2, theta_min=-0.2, theta_max=0.2)
    gen.generate_random_primary_vertices({"x": 0.0, "y": 0.0, "z": 1.0})
    gen.generate_particles([[{"type": "pion", "mass": 139.6, "q": 1}] * T])
    gen.generate_complete_events()
    return gen


@pytest.fixture(scope="module")
def full(event):
    h = SimpleHamiltonianFast(EPS, GAMMA, DELTA)
    h.construct_hamiltonian(event)
    return h


@pytest.fixture(scope="module")
def pruned(event):
    h = SimpleHamiltonianFast(EPS, GAMMA, DELTA)
    h.construct_hamiltonian(event, prune=True)
    return h


def test_kept_set_is_the_coupled_set(full, pruned):
    A = full.A.toarray()
    np.fill_diagonal(A, 0.0)
    coupled = np.nonzero((A != 0).any(axis=1))[0]
    assert np.array_equal(pruned.kept_index, coupled)
    assert pruned.n_segments == len(coupled) < full.n_segments
    assert pruned.n_segments_full == full.n_segments
    assert pruned.prune_info["n_pruned"] == full.n_segments - len(coupled)
    assert np.array_equal(segment_degree(full.A)[coupled] >= 1, np.ones(len(coupled), bool))
    assert pruned.A.shape == (len(coupled), len(coupled))
    assert np.allclose(pruned.A.toarray(), full.A.toarray()[np.ix_(coupled, coupled)])
    assert np.allclose(pruned.b, full.b[coupled])


def test_per_segment_arrays_follow_the_prune(full, pruned):
    k = pruned.kept_index
    assert np.array_equal(pruned._segment_to_hit_ids, full._segment_to_hit_ids[k])
    assert np.array_equal(pruned._segment_track_ids, full._segment_track_ids[k])
    assert np.allclose(pruned._segment_vectors, full._segment_vectors[k])
    assert len(pruned.segments) == pruned.n_segments
    assert [s.segment_id for s in pruned.segments] == k.tolist()
    assert len(pruned.segments_full) == full.n_segments
    assert sum(len(g) for g in pruned.segments_grouped) == pruned.n_segments
    tf = segment_truth_mask(full)
    assert np.array_equal(segment_truth_mask(pruned, full_space=True), tf)
    assert np.array_equal(segment_truth_mask(pruned), tf[k])


def test_no_complete_track_segment_is_pruned(full, pruned):
    # every track in this clean-geometry event has all five hits, so every
    # true segment has a true continuation and must survive
    tf = segment_truth_mask(full)
    assert tf.sum() == 4 * T
    assert segment_truth_mask(pruned).sum() == tf.sum()


def test_classical_solution_is_unchanged_on_kept_segments(full, pruned):
    x = spsolve(full.A, full.b)
    xp = spsolve(pruned.A, pruned.b)
    assert np.max(np.abs(x[pruned.kept_index] - xp)) < 1e-12
    pruned_pos = np.setdiff1d(np.arange(full.n_segments), pruned.kept_index)
    assert np.allclose(x[pruned_pos], DELTA / (GAMMA + DELTA))   # the attractor
    assert np.allclose(pruned.expand(xp, fill=DELTA / (GAMMA + DELTA)), x)
    assert np.allclose(expand_solution(xp, pruned.kept_index, full.n_segments)[pruned_pos], 0.0)


def test_one_bit_filter_is_unchanged_and_success_scales_with_register(full, pruned):
    s, P, n_sys = solve_one_bit_matrixfree(full.A)
    sp_, Pp, n_sys_p = solve_one_bit_matrixfree(pruned.A)
    assert n_sys_p == qubits_for(pruned.n_segments) < n_sys == qubits_for(full.n_segments)
    sk = s[pruned.kept_index].astype(np.float64)
    assert np.max(np.abs(sk / np.linalg.norm(sk) - sp_ / np.linalg.norm(sp_))) < 1e-6
    # the decoupled segments are zeroed by the notch; the padded register is
    # uniform, so P grows exactly by the register ratio
    assert Pp / P == pytest.approx(2.0 ** (n_sys - n_sys_p), rel=1e-6)


def test_qsvt_comb_is_unchanged_on_kept_segments(full, pruned):
    bd = spectral_bounds(full.A, s=GAMMA + DELTA, pad=0.10)
    q = QSVT.comb(full.A, full.b, spectral_bounds=bd)
    qp = QSVT.comb(pruned.A, pruned.b, spectral_bounds=bd)
    y, P = q.solve_statevector()
    yp, Pp = qp.solve_statevector()
    assert qp.total_qubits < q.total_qubits
    yk = y[pruned.kept_index]
    assert np.max(np.abs(yk / np.linalg.norm(yk) - yp / np.linalg.norm(yp))) < 1e-9
    # same polynomial (same domain) -> the signed vectors agree up to the
    # global norm ratio sqrt(n_kept / n_full), plus the comb's leak p(s) at
    # the decoupled line
    n_full, n_kept = full.n_segments, pruned.n_segments
    leak = (n_full - n_kept) / n_full * (DELTA * float(q.poly(GAMMA + DELTA)) / q.l1) ** 2
    assert P == pytest.approx(Pp * n_kept / n_full + leak, rel=1e-9, abs=1e-12)


def test_component_prune_matches_connected_components(full):
    C = full.A.copy().tocsr()
    C.setdiag(0.0)
    C.eliminate_zeros()
    _, lab = connected_components(C, directed=False)
    sizes = np.bincount(lab)
    for m in (2, 3, 4):
        keep, info = decoupled_mask(full.A, min_degree=1, min_component_size=m)
        assert np.array_equal(keep, sizes[lab] >= m)
        assert info["n_kept"] == int(keep.sum())
    keep1, _ = decoupled_mask(full.A)
    assert np.array_equal(keep1, sizes[lab] >= 2)        # degree >= 1 == component >= 2
    keep0, info0 = decoupled_mask(full.A, min_degree=0)
    assert keep0.all() and info0["n_pruned"] == 0


def test_reference_builder_prunes_identically(event, pruned):
    r = SimpleHamiltonian(EPS, GAMMA, DELTA)
    r.construct_hamiltonian(event, prune=True)
    assert np.array_equal(r.kept_index, pruned.kept_index)
    assert np.allclose(r.A.toarray(), pruned.A.toarray())
    assert [s.segment_id for s in r.segments] == pruned.kept_index.tolist()


def test_extended_hamiltonians_prune_after_their_terms(event, pruned):
    f = ForkHamiltonian(EPS, GAMMA, DELTA, beta=0.5)
    f.construct_hamiltonian(event, prune=True)
    # the fork term couples same-slot competitors, which already have a
    # compatible neighbour in A0 whenever they fall inside the window, so
    # nothing new is coupled: the kept set is the base kept set
    assert np.array_equal(f.kept_index, pruned.kept_index)
    o = OccupancyHamiltonian(EPS, GAMMA, DELTA, alpha=0.05)
    o.construct_hamiltonian(event, prune=True)
    # the occupancy term couples every co-hit pair: the prune (applied to the
    # final operator, as it must be) removes nothing of substance
    assert o.n_segments >= pruned.n_segments
    assert o.n_segments_full == pruned.n_segments_full
    assert o.n_segments >= 0.99 * o.n_segments_full


def test_prune_twice_is_refused_and_unbuilt_is_refused(event, pruned):
    with pytest.raises(ValueError):
        pruned.prune()
    with pytest.raises(ValueError):
        SimpleHamiltonianFast(EPS, GAMMA, DELTA).prune()


def test_dict_options_and_rebuild_after_prune(event):
    h = SimpleHamiltonianFast(EPS, GAMMA, DELTA)
    h.construct_hamiltonian(event, prune=dict(min_component_size=3))
    assert h.prune_info["min_component_size"] == 3
    n3 = h.n_segments
    # a rebuild starts from the segments again and gives the full system back
    h2 = SimpleHamiltonianFast(EPS, GAMMA, DELTA)
    h2.construct_hamiltonian(event)
    h.kept_index = None
    h.construct_hamiltonian(event)
    assert h.n_segments == h2.n_segments and h.A.nnz == h2.A.nnz
    assert n3 <= h2.n_segments


def test_expand_matches_truth_layout(full, pruned):
    v = np.arange(pruned.n_segments, dtype=float) + 1.0
    e = pruned.expand(v)
    assert e.shape == (full.n_segments,)
    assert np.array_equal(e[pruned.kept_index], v)
    assert np.count_nonzero(e) == pruned.n_segments
    assert np.array_equal(full.expand(v[: full.n_segments]), v[: full.n_segments])
