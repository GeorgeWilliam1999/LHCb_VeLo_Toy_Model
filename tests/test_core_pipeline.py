"""Core pipeline contracts: generation, Hamiltonians, tracks, validation,
persistence, plotting.

These are the non-quantum lanes the Condor workers and notebooks rely on;
they had no unit tests before 2026-09-07 (only the golden baselines and the
integration script).
"""
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

from lhcb_velo_toy.analysis import EventValidator, threshold_for  # noqa: E402
from lhcb_velo_toy.analysis.plotting.event_display import (  # noqa: E402
    plot_event_3d,
    plot_reco_vs_truth,
)
from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator  # noqa: E402
from lhcb_velo_toy.persistence import (  # noqa: E402
    load_pipeline,
    load_study,
    save_pipeline,
    save_study,
)
from lhcb_velo_toy.solvers import SimpleHamiltonian, SimpleHamiltonianFast, get_tracks  # noqa: E402
from lhcb_velo_toy.solvers.classical import solve_direct  # noqa: E402
from tests.test_golden_baseline import build  # noqa: E402


def pathlib_root():
    import pathlib
    return pathlib.Path(__file__).resolve().parents[1]


def _gen(T, seed, me=0.0, cn=1e-8):
    geo = PlaneGeometry(module_id=list(range(5)), lx=[50.0] * 5, ly=[50.0] * 5,
                        z=[100.0 + 33.0 * i for i in range(5)])
    gen = StateEventGenerator(detector_geometry=geo, events=1, n_particles=[T],
                              measurement_error=me, collision_noise=cn, seed=seed)
    gen.generate_random_primary_vertices({"x": 0.1, "y": 0.1, "z": 50.0})
    gen.generate_particles([[{"type": "pion", "mass": 139.6, "q": 1}] * T])
    gen.generate_complete_events()
    return gen


def _hits_xyz(event):
    return np.array([[h.x, h.y, h.z] for h in event.hits])


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def test_seeded_generation_is_reproducible():
    a, b, c = _gen(8, 11, 0.01, 1e-4), _gen(8, 11, 0.01, 1e-4), _gen(8, 12, 0.01, 1e-4)
    assert np.allclose(_hits_xyz(a.true_event), _hits_xyz(b.true_event))
    assert not np.allclose(_hits_xyz(a.true_event), _hits_xyz(c.true_event))
    assert len(a.true_event.tracks) == 8
    assert all(len(t.hit_ids) == 5 for t in a.true_event.tracks)


def test_noisy_event_drops_and_adds_hits():
    gen = _gen(10, 21)
    n0 = len(gen.true_event.hits)
    dropped = gen.make_noisy_event(drop_rate=0.2, ghost_rate=0.0)
    assert len(dropped.hits) == n0 - int(0.2 * n0)
    ghosted = gen.make_noisy_event(drop_rate=0.0, ghost_rate=0.1)
    ghosts = [h for h in ghosted.hits if h.track_id == -1]
    assert len(ghosted.hits) > n0 and len(ghosts) == len(ghosted.hits) - n0
    assert len(gen.true_event.hits) == n0   # the truth event is untouched


# ---------------------------------------------------------------------------
# Hamiltonians and classical solvers
# ---------------------------------------------------------------------------
def test_simple_and_fast_hamiltonians_agree():
    gen = _gen(6, 31, 0.01, 1e-4)
    kw = dict(epsilon=0.01, gamma=3.0, delta=1.0)
    A1, b1 = SimpleHamiltonian(**kw).construct_hamiltonian(gen)
    fast = SimpleHamiltonianFast(**kw)
    A2, b2 = fast.construct_hamiltonian(gen)
    assert A1.shape == A2.shape
    assert np.allclose(A1.toarray(), A2.toarray())
    assert np.allclose(b1, b2)
    x_direct = solve_direct(A2, b2)
    assert np.allclose(x_direct, fast.solve_classicaly(), atol=1e-8)
    assert A2.diagonal()[0] == pytest.approx(4.0)
    assert threshold_for(3.0, 1.0) == pytest.approx(0.35)


def test_clean_event_reconstructs_perfectly():
    gen, ham, A, b = build("clean")
    sol = ham.solve_classicaly()
    tracks = get_tracks(ham, sol, gen, threshold=threshold_for(3.0, 1.0))
    matches, metrics = EventValidator(gen.true_event, tracks).match_tracks(purity_min=0.7)
    assert len(tracks) == 5
    assert metrics["efficiency"] == pytest.approx(1.0)
    assert metrics["ghost_rate"] == pytest.approx(0.0)
    assert metrics["mean_purity"] == pytest.approx(1.0)


def test_package_imports_without_numba_and_kernels_agree():
    """numba is an OPTIONAL accelerator, not a hidden hard dependency.

    Regression (2026-09-09): ``analysis/segment_metrics.py`` imported numba at
    module level and numba was declared nowhere, so ``import lhcb_velo_toy``
    failed outright on any install without it. It is now guarded, with a pure
    Python fallback that must give the SAME numbers as the compiled kernel.
    Needs a fresh interpreter to hide numba.
    """
    code = textwrap.dedent("""
        import sys, importlib.abc, json
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "numba" or name.startswith("numba."):
                    raise ImportError("numba hidden for the test")
                return None
        sys.meta_path.insert(0, Block())
        import lhcb_velo_toy                       # must not raise
        from lhcb_velo_toy.analysis.segment_metrics import HAS_NUMBA, fast_segment_metrics
        assert HAS_NUMBA is False
        sys.path.insert(0, %r)
        from tests.test_core_pipeline import _gen
        from lhcb_velo_toy.analysis.segment_metrics import compute_epsilon
        gen = _gen(8, 909, 0.01, 1e-4)
        print(json.dumps(fast_segment_metrics(gen.true_event, compute_epsilon(0.01, 1e-4, dz=33.0))))
    """) % str(pathlib_root())
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    import json as _json
    without = _json.loads(out.stdout.strip().splitlines()[-1])

    from lhcb_velo_toy.analysis.segment_metrics import (
        HAS_NUMBA, compute_epsilon, fast_segment_metrics,
    )
    gen = _gen(8, 909, 0.01, 1e-4)
    with_numba = fast_segment_metrics(gen.true_event, compute_epsilon(0.01, 1e-4, dz=33.0))
    assert without == with_numba, (without, with_numba)
    if HAS_NUMBA:                       # this env has it; prove the accelerator ran
        assert with_numba["n_true"] > 0


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def test_pipeline_round_trip(tmp_path):
    gen, ham, A, b = build("moderate")
    sol = ham.solve_classicaly()
    tracks = get_tracks(ham, sol, gen, threshold=0.35)
    matches, metrics = EventValidator(gen.true_event, tracks).match_tracks(purity_min=0.7)
    out = save_pipeline(tmp_path / "run", event=gen.true_event, ham=ham, solution=sol,
                        threshold=0.35, reco_tracks=tracks, matches=matches, metrics=metrics,
                        extra_config={"note": "test"})
    res = load_pipeline(out)
    assert len(res.event.hits) == len(gen.true_event.hits)
    assert len(res.event.tracks) == len(gen.true_event.tracks)
    assert res.A.shape == A.shape and abs(res.A - A).max() == 0
    assert np.allclose(res.solution, sol)
    assert res.metrics["efficiency"] == pytest.approx(metrics["efficiency"])
    assert res.config["epsilon"] == pytest.approx(ham.epsilon)


def test_study_round_trip(tmp_path):
    scan = {(0.1, 1.0): [dict(n_tracks=5, eff_mean=1.0, eff_se=0.0),
                         dict(n_tracks=10, eff_mean=0.9, eff_se=0.05)]}
    hist = {(0.1, 1.0): {"purity": np.array([1.0, 0.8, 0.9])}}
    out = save_study(tmp_path / "study", config={"sizes": [5, 10]}, scan_results=scan,
                     hist_data=hist, extra_arrays={"extra": np.arange(3.0)})
    res = load_study(out)
    assert res.config["sizes"] == [5, 10]
    key = next(iter(res.scan_results))
    assert np.allclose(res.scan_results[key]["eff_mean"], [1.0, 0.9])
    assert np.allclose(res.extra_arrays["extra"], np.arange(3.0))


# ---------------------------------------------------------------------------
# plotting smoke
# ---------------------------------------------------------------------------
def test_plotting_smoke():
    import matplotlib.pyplot as plt

    gen, ham, A, b = build("clean")
    tracks = get_tracks(ham, ham.solve_classicaly(), gen, threshold=0.35)
    fig1 = plot_event_3d(gen.true_event, title="t")
    fig2 = plot_reco_vs_truth(gen.true_event, tracks)
    assert fig1.axes and fig2.axes
    plt.close("all")
