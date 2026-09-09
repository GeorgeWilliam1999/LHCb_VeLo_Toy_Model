"""Golden-baseline tests (centralise-2.1 Phase 0).

Pins today's physics on seeded events across three regimes so the
centralisation refactor cannot silently change results. Values captured
2026-08-26 on Q_env (numpy/scipy as installed) with a double-pass
determinism check.

Conventions pinned here (do not change without a deliberate physics
decision; two were taken on 2026-09-07 and are re-pinned below — the QSVT
default design is the comb, and the 1BQF uncompute applies the inverse pair
rotations in reverse order):
- geometry: 5 planes, dz = 33 mm, half-width 50 mm, z0 = 100 mm
- Hamiltonian: gamma = 3, delta = 1  =>  A diagonal = +4, chain bulk
  activation delta/(delta+gamma-2) bounded by 1/2
- epsilon from analysis.segment_metrics.compute_epsilon with its
  canonical defaults (scale = 3.0, dz = 33.0, theta_min = 1.5e-5)
"""
import numpy as np
import pytest

from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
from lhcb_velo_toy.solvers.hamiltonians.fast import SimpleHamiltonianFast
from lhcb_velo_toy.solvers.reconstruction.track_finder import get_tracks
from lhcb_velo_toy.analysis import EventValidator
from lhcb_velo_toy.analysis.segment_metrics import (
    DEFAULT_DZ,
    DEFAULT_SCALE,
    DEFAULT_THETA_MIN,
    compute_epsilon,
)

N_MODULES, DZ, Z0, HALF = 5, 33.0, 100.0, 50.0
GAMMA, DELTA = 3.0, 1.0

REGIMES = {
    "clean": dict(T=5, me=0.0, cn=1e-8, seed=101),
    "moderate": dict(T=20, me=0.01, cn=1e-4, seed=202),
    "heavy": dict(T=50, me=0.01, cn=1e-4, seed=303),
}

GOLDEN = {
    "clean": dict(
        n_segments=100, A_nnz=130, A_sum=370.0, b_sum=100.0,
        sol_sum=28.181818181818183, sol_max=0.4545454545454546,
    ),
    "moderate": dict(
        n_segments=1446, A_nnz=1554, A_sum=5676.0, b_sum=1446.0,
        sol_sum=372.70670995671, sol_max=0.45454545454545453,
    ),
    "heavy": dict(
        n_segments=9600, A_nnz=9884, A_sum=38116.0, b_sum=9600.0,
        sol_sum=2429.87012987013, sol_max=0.45454545454545453,
    ),
}


def build(regime):
    r = REGIMES[regime]
    geo = PlaneGeometry(
        module_id=list(range(N_MODULES)),
        lx=[HALF] * N_MODULES,
        ly=[HALF] * N_MODULES,
        z=[Z0 + i * DZ for i in range(N_MODULES)],
    )
    gen = StateEventGenerator(
        detector_geometry=geo, events=1, n_particles=[r["T"]],
        measurement_error=r["me"], collision_noise=r["cn"], seed=r["seed"],
    )
    gen.generate_random_primary_vertices({"x": 0.1, "y": 0.1, "z": 50.0})
    gen.generate_particles([[{"type": "pion", "mass": 139.6, "q": 1}] * r["T"]])
    gen.generate_complete_events()
    ham = SimpleHamiltonianFast(
        epsilon=compute_epsilon(r["me"], r["cn"], dz=DZ), gamma=GAMMA, delta=DELTA
    )
    A, b = ham.construct_hamiltonian(gen)
    return gen, ham, A, b


def test_compute_epsilon_canonical_defaults():
    # Guards the scale=1.0-vs-3.0 landmine class found in the 2026-08-26 audit.
    assert DEFAULT_SCALE == 3.0
    assert DEFAULT_DZ == 33.0
    assert DEFAULT_THETA_MIN == 1.5e-5
    assert compute_epsilon(0.01, 1e-4) == pytest.approx(0.0031777, rel=1e-4)


@pytest.mark.parametrize("regime", list(REGIMES))
def test_hamiltonian_and_classical_solve_golden(regime):
    gen, ham, A, b = build(regime)
    g = GOLDEN[regime]
    assert ham.n_segments == g["n_segments"]
    assert A.nnz == g["A_nnz"]
    assert A.diagonal()[0] == DELTA + GAMMA
    assert A.sum() == pytest.approx(g["A_sum"], rel=1e-12)
    assert b.sum() == pytest.approx(g["b_sum"], rel=1e-12)
    sol = ham.solve_classicaly()
    assert sol.sum() == pytest.approx(g["sol_sum"], rel=1e-9)
    assert sol.max() == pytest.approx(g["sol_max"], rel=1e-9)


def test_track_level_metrics_golden_moderate():
    gen, ham, A, b = build("moderate")
    sol = ham.solve_classicaly()
    tracks = get_tracks(ham, sol, gen, threshold=0.35)
    validator = EventValidator(truth_event=gen.true_event, rec_tracks=tracks)
    matches, metrics = validator.match_tracks(purity_min=0.7)
    assert len(tracks) == 19
    assert metrics["efficiency"] == pytest.approx(0.95, abs=1e-12)
    assert metrics["ghost_rate"] == pytest.approx(0.0, abs=1e-12)


def test_quantum_lanes_golden_clean():
    qiskit = pytest.importorskip("qiskit")
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    from lhcb_velo_toy.solvers.quantum import OneBQF, QSVT

    gen, ham, A, b = build("clean")

    q = OneBQF(A, b, readout="statevector")
    qc = q.build_circuit()
    sim = AerSimulator(method="statevector")
    sv = sim.run(transpile(qc, sim)).result().get_statevector()
    amps_q, p_q = q.get_solution_from_statevector(sv)
    # 1BQF with the exact-inverse uncompute (2026-09-07). The success
    # probability is unchanged by the fix (4.4783e-2 before and after); the
    # amplitude sum moved from 4.206959752194961 (forward-order uncompute).
    assert p_q == pytest.approx(0.044783070679944914, rel=1e-6)
    assert np.sum(amps_q) == pytest.approx(4.281635210208872, rel=1e-6)

    # the bare constructor is the comb (the QSVT of the papers) since 2026-09-07
    amps_s, p_s = QSVT(A, b, degree=40).solve_statevector()
    assert p_s == pytest.approx(0.08796160232901191, rel=1e-6)
    assert np.sum(amps_s) == pytest.approx(4.530657664955381, rel=1e-6)
    # the retired band design, pinned under its explicit name
    amps_b, p_b = QSVT(A, b, degree=40, filter_design="band").solve_statevector()
    assert p_b == pytest.approx(0.03702804240383503, rel=1e-6)
    assert np.sum(amps_b) == pytest.approx(4.57577599172363, rel=1e-6)
