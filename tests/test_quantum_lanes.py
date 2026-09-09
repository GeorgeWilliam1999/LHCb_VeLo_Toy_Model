"""Phase 2 tests: matrix-free 1BQF, QSVT eigh backend, post-selection.

Equivalence contracts:
- QSVT.solve_eigh == QSVT.solve_statevector (same y = p(A) b / ||c||_1,
  computed via per-component eigendecomposition vs Chebyshev recursion).
- solve_one_bit_matrixfree == OneBQF circuit + exact statevector readout
  (both carry the 2026-09-07 uncompute fix: inverse pair rotations in
  reverse order).
- solvers.postselect == the originating qtrk_pipeline.postselect (when the
  study tree is present).
"""
import numpy as np
import pytest

from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
from lhcb_velo_toy.solvers.hamiltonians.fast import SimpleHamiltonianFast
from lhcb_velo_toy.solvers.postselect import (
    hit_uniqueness_filter,
    uniqueness_frontier,
)
from lhcb_velo_toy.solvers.quantum import (
    QSVT,
    OneBQF,
    solve_one_bit_matrixfree,
    spectral_bounds,
)
from lhcb_velo_toy.analysis.segment_metrics import compute_epsilon

N_MODULES, DZ, Z0, HALF = 5, 33.0, 100.0, 50.0
GAMMA, DELTA = 3.0, 1.0


def build(T, me, cn, seed):
    geo = PlaneGeometry(
        module_id=list(range(N_MODULES)),
        lx=[HALF] * N_MODULES,
        ly=[HALF] * N_MODULES,
        z=[Z0 + i * DZ for i in range(N_MODULES)],
    )
    gen = StateEventGenerator(
        detector_geometry=geo, events=1, n_particles=[T],
        measurement_error=me, collision_noise=cn, seed=seed,
    )
    gen.generate_random_primary_vertices({"x": 0.1, "y": 0.1, "z": 50.0})
    gen.generate_particles([[{"type": "pion", "mass": 139.6, "q": 1}] * T])
    gen.generate_complete_events()
    ham = SimpleHamiltonianFast(
        epsilon=compute_epsilon(me, cn, dz=DZ), gamma=GAMMA, delta=DELTA
    )
    A, b = ham.construct_hamiltonian(gen)
    return gen, ham, A, b


@pytest.mark.parametrize("T,me,cn,seed", [(5, 0.0, 1e-8, 101), (20, 0.01, 1e-4, 202)])
def test_qsvt_eigh_equals_statevector(T, me, cn, seed):
    _, _, A, b = build(T, me, cn, seed)
    q = QSVT(A, b, degree=40)
    sol_sv, p_sv = q.solve_statevector()
    signed_sv = q.solution_signed.copy()
    q2 = QSVT(A, b, degree=40)
    sol_e, p_e = q2.solve_eigh()
    assert p_e == pytest.approx(p_sv, rel=1e-9, abs=1e-12)
    assert np.allclose(sol_e, sol_sv, atol=1e-9)
    # signed vectors agree up to a global sign per definition of y
    assert np.allclose(np.abs(q2.solution_signed), np.abs(signed_sv), atol=1e-9)


def test_qsvt_eigh_reuses_decomposition():
    _, _, A, b = build(5, 0.0, 1e-8, 101)
    q = QSVT(A, b, degree=40)
    q.solve_eigh()
    ms = q._mode_system
    q.solve_eigh()
    assert q._mode_system is ms  # cached, not rebuilt


def test_matrixfree_equals_onebqf_statevector():
    qiskit = pytest.importorskip("qiskit")
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    _, _, A, b = build(5, 0.0, 1e-8, 101)
    ref = OneBQF(A, b, readout="statevector")
    qc = ref.build_circuit()
    sim = AerSimulator(method="statevector")
    sv = sim.run(transpile(qc, sim)).result().get_statevector()
    sol_ref, p_ref = ref.get_solution_from_statevector(sv)

    sol_mf, p_mf, n_sys = solve_one_bit_matrixfree(A)
    assert p_mf == pytest.approx(p_ref, rel=1e-9, abs=1e-12)
    # sol_mf is float32 by design — cosine agreement at float32 precision
    cos = float(np.dot(sol_mf.astype(np.float64), np.asarray(sol_ref)))
    assert cos == pytest.approx(1.0, abs=1e-6)


def test_matrixfree_rejects_nonconstant_diagonal():
    import scipy.sparse as sp

    A = sp.diags([1.0, 2.0, 3.0, 4.0]).tocsr()
    with pytest.raises(ValueError):
        solve_one_bit_matrixfree(A)


def test_spectral_bounds_contains_spectrum():
    _, _, A, b = build(20, 0.01, 1e-4, 202)
    lo, hi = spectral_bounds(A, pad=0.10)
    w = np.linalg.eigvalsh(A.toarray())
    assert lo <= w[0] and w[-1] <= hi


def test_hit_uniqueness_greedy_basic():
    # Segments 0 and 1 compete for start-hit 0 (same role); 2 is a chain
    # continuation of 0 (opposite roles at hit 1) and must NOT compete.
    seg_hits = np.array([[0, 1], [0, 2], [1, 3]])
    sol = np.array([0.9, 0.5, 0.8])
    active = np.array([True, True, True])
    out = hit_uniqueness_filter(sol, seg_hits, active, mode="greedy")
    assert out.tolist() == [True, False, True]


def test_hit_uniqueness_margin_spares_close_contest():
    seg_hits = np.array([[0, 1], [0, 2]])
    sol = np.array([0.50, 0.48])  # near-equal true-vs-true contest
    active = np.array([True, True])
    out = hit_uniqueness_filter(sol, seg_hits, active, mode="margin", kappa=0.8)
    assert out.tolist() == [True, True]
    out2 = hit_uniqueness_filter(sol, seg_hits, active, mode="strict")
    assert out2.tolist() == [True, False]


def test_uniqueness_frontier_shapes_and_monotone_eff():
    _, ham, A, b = build(20, 0.01, 1e-4, 202)
    sol = ham.solve_classicaly()
    truth = np.array(
        [ts == te and ts >= 0 for ts, te in np.asarray(ham._segment_track_ids)]
    )
    taus = np.linspace(0.05, 0.45, 9)
    eff, far = uniqueness_frontier(sol, ham._segment_to_hit_ids, truth, taus)
    assert eff.shape == far.shape == taus.shape
    assert np.all(np.diff(eff) <= 1e-12)  # eff non-increasing in tau


def test_matches_originating_qtrk_postselect():
    import importlib.util

    path = (
        "/data/bfys/gscriven/Quantum_Track_Reconstruction/"
        "Toy_Characterisation/_shared/qtrk_pipeline/postselect.py"
    )
    spec = importlib.util.spec_from_file_location("qtrk_postselect", path)
    if spec is None:
        pytest.skip("qtrk_pipeline not present")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        pytest.skip(f"qtrk postselect not importable: {exc}")

    rng = np.random.default_rng(7)
    _, ham, A, b = build(20, 0.01, 1e-4, 202)
    sol = ham.solve_classicaly() + 0.01 * rng.standard_normal(ham.n_segments)
    seg_hits = np.asarray(ham._segment_to_hit_ids)
    active = np.abs(sol) > 0.2
    for mode in ("greedy", "strict", "margin", "anyrole"):
        ours = hit_uniqueness_filter(sol, seg_hits, active, mode=mode)
        theirs = mod.hit_uniqueness_filter(sol, seg_hits, active, mode=mode)
        assert np.array_equal(ours, theirs), mode
