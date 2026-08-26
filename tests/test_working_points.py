"""Phase 4 tests: working-point policy, truth-from-event, Run3 adapter.

Cross-checked against the originating qtrk_pipeline.metrics module when the
study tree is importable.
"""
import json

import numpy as np
import pytest

from lhcb_velo_toy.analysis.working_points import (
    ABS_THRESHOLD,
    WP_TAU_FLOOR,
    metrics_at,
    metrics_at_wp,
    quantum_metrics,
    quantum_metrics_wp,
    rescale_to_signal,
    threshold_for,
    truth_from_event,
    working_point_threshold,
)
from lhcb_velo_toy.analysis.segment_metrics import (
    segment_truth_mask,
    solver_segment_metrics,
)
from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
from lhcb_velo_toy.generation.adapters import load_run3_event
from lhcb_velo_toy.solvers.hamiltonians.fast import SimpleHamiltonianFast
from lhcb_velo_toy.analysis import compute_epsilon

N_MODULES, DZ, Z0, HALF = 5, 33.0, 100.0, 50.0


@pytest.fixture(scope="module")
def solved():
    geo = PlaneGeometry(
        module_id=list(range(N_MODULES)),
        lx=[HALF] * N_MODULES, ly=[HALF] * N_MODULES,
        z=[Z0 + i * DZ for i in range(N_MODULES)],
    )
    gen = StateEventGenerator(
        detector_geometry=geo, events=1, n_particles=[20],
        measurement_error=0.01, collision_noise=1e-4, seed=202,
    )
    gen.generate_random_primary_vertices({"x": 0.1, "y": 0.1, "z": 50.0})
    gen.generate_particles([[{"type": "pion", "mass": 139.6, "q": 1}] * 20])
    gen.generate_complete_events()
    ham = SimpleHamiltonianFast(
        epsilon=compute_epsilon(0.01, 1e-4, dz=DZ), gamma=3.0, delta=1.0
    )
    ham.construct_hamiltonian(gen)
    sol = ham.solve_classicaly()
    truth = segment_truth_mask(ham)
    return gen, ham, sol, truth


def test_threshold_for_gamma_aware():
    assert threshold_for(3.0, 1.0) == pytest.approx(ABS_THRESHOLD)
    assert threshold_for(1.0, 1.0) == pytest.approx(0.6)  # attractor 0.5 + 0.10


def test_metrics_at_is_solver_segment_metrics(solved):
    _, _, sol, truth = solved
    assert metrics_at(sol, truth) == solver_segment_metrics(truth, sol, threshold=0.35)


def test_truth_from_event_matches_built_ham(solved):
    gen, ham, _, truth = solved
    assert np.array_equal(truth_from_event(gen), truth)


def test_working_point_threshold_holds_target_eff(solved):
    _, _, sol, truth = solved
    tau = working_point_threshold(sol, truth, target_eff=0.99)
    eff = (np.abs(sol[truth]) > tau).mean()
    assert eff >= 0.99


def test_working_point_threshold_floors_on_structural_zeros():
    sol = np.array([0.0, 0.0, 0.0, 0.5])
    truth = np.array([True, True, True, False])
    tau = working_point_threshold(sol, truth, target_eff=0.99)
    assert tau == WP_TAU_FLOOR


def test_wp_metrics_scale_invariant(solved):
    _, _, sol, truth = solved
    m1 = metrics_at_wp(sol, truth)
    m2 = metrics_at_wp(37.0 * sol, truth)
    for k in ("segment_efficiency", "segment_false_rate"):
        if k in m1:
            assert m1[k] == pytest.approx(m2[k])


def test_quantum_metrics_signal_rescale(solved):
    _, _, sol, truth = solved
    q = sol / np.linalg.norm(sol)  # unit-norm 'quantum' partner, perfect solve
    out = quantum_metrics(q, sol, truth)
    assert out["cos_QC"] == pytest.approx(1.0, abs=1e-12)
    base = metrics_at(sol, truth)
    for k in ("segment_efficiency", "segment_false_rate"):
        if k in base:
            assert out[k] == pytest.approx(base[k])
    out_wp = quantum_metrics_wp(q, sol, truth)
    assert out_wp["cos_QC"] == pytest.approx(1.0, abs=1e-12)


def test_matches_originating_qtrk_metrics(solved):
    import sys

    shared = ("/data/bfys/gscriven/Quantum_Track_Reconstruction/"
              "Toy_Characterisation/_shared")
    if shared not in sys.path:
        sys.path.insert(0, shared)
    qp_metrics = pytest.importorskip("qtrk_pipeline.metrics")

    _, _, sol, truth = solved
    q = 0.7 * sol / np.linalg.norm(sol)
    assert threshold_for(2.0, 1.0) == qp_metrics.threshold_for(2.0, 1.0)
    assert metrics_at(sol, truth) == qp_metrics.metrics_at(sol, truth)
    assert working_point_threshold(sol, truth) == \
        qp_metrics.working_point_threshold(sol, truth)
    ours, theirs = quantum_metrics(q, sol, truth), qp_metrics.quantum_metrics(q, sol, truth)
    assert ours.keys() == theirs.keys()
    for k in ours:
        assert ours[k] == pytest.approx(theirs[k], rel=1e-12), k
    ours_wp = quantum_metrics_wp(q, sol, truth)
    theirs_wp = qp_metrics.quantum_metrics_wp(q, sol, truth)
    for k in ours_wp:
        assert ours_wp[k] == pytest.approx(theirs_wp[k], rel=1e-12), k


def test_run3_adapter_halves_and_hamiltonian(tmp_path):
    # Synthetic 4-module Allen-format event: modules interleave left/right in z
    # (z-rank even = left, odd = right); two straight tracks, one per half.
    #   left half:  modules z=100, 160  — track 0 hits 0, 4
    #   right half: modules z=130, 190  — track 1 hits 2, 6
    # plus one noise hit per module (indices 1, 3, 5, 7).
    x = [0.0, 5.0, 1.0, 6.0, 0.0, 7.0, 1.0, 8.0]
    y = [0.0] * 8
    z = [100.0, 100.0, 130.0, 130.0, 160.0, 160.0, 190.0, 190.0]
    data = {
        "x": x, "y": y, "z": z,
        "module_prefix_sum": [0, 2, 4, 6, 8],
        "montecarlo": {
            "description": ["key", "hits"],
            "particles": [[0, [0, 4]], [1, [2, 6]]],
        },
    }
    p = tmp_path / "velo_event_0.json"
    p.write_text(json.dumps(data))

    ev, meta = load_run3_event(p)
    assert meta["n_hits"] == 8
    assert meta["n_left"] == 2 and meta["n_right"] == 2
    # module list = left half, separator (empty), right half
    assert [m.hit_ids for m in ev.modules] == [[0, 1], [4, 5], [], [2, 3], [6, 7]]

    ham = SimpleHamiltonianFast(epsilon=1.0, gamma=3.0, delta=1.0)
    ham.construct_segments(ev, materialize_segments=False)
    # 2x2 hits per adjacent in-half pair, no cross-half segments: 4 + 4
    assert ham.n_segments == 8
    truth = segment_truth_mask(ham)
    assert int(truth.sum()) == 2  # one true segment per track
