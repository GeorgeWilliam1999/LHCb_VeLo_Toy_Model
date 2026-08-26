"""Phase 3 tests: the Hough solver lane.

- Point- and locus-vote pipelines recover clean, well-separated tracks.
- Candidate sets equal the originating study library's (when present).
- The adapter routes candidates through Track objects + EventValidator; on
  the easy regime (clean, well-separated) the canonical matcher and the
  study's majority rule must agree exactly (eff 1, ghosts 0).
"""
import numpy as np
import pytest

from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
from lhcb_velo_toy.solvers.hough import (
    candidates_to_tracks,
    event_arrays,
    hit_directions,
    run_locus,
    run_point,
    validate_hough,
)

# Hough-study geometry: 5 planes at z = 33..165 mm, half-width 40 mm,
# PV pinned to x = y = 0 with only z spreading (sigma = 1 mm).
Z_PLANES = [33.0, 66.0, 99.0, 132.0, 165.0]


def make_event(T, seed, sigma_scatt=1e-4, sigma_res=0.0):
    geo = PlaneGeometry(
        module_id=list(range(5)),
        lx=[40.0] * 5, ly=[40.0] * 5, z=Z_PLANES,
    )
    gen = StateEventGenerator(
        detector_geometry=geo, events=1, n_particles=[T],
        measurement_error=sigma_res, collision_noise=sigma_scatt, seed=seed,
    )
    gen.generate_random_primary_vertices({"x": 0.0, "y": 0.0, "z": 1.0})
    gen.generate_particles([[{"type": "pion", "mass": 139.6, "q": 1}] * T])
    gen.generate_complete_events()
    return gen.true_event


@pytest.fixture(scope="module")
def clean_event():
    return make_event(T=10, seed=515)


def test_point_vote_recovers_clean_tracks(clean_event):
    D, planes, truth = hit_directions(clean_event)
    cands, centres, sm = run_point(D, planes, n_bins=256)
    tracks, matches, metrics = validate_hough(cands, clean_event)
    assert metrics["efficiency"] >= 0.9
    assert metrics["ghost_rate"] == pytest.approx(0.0, abs=1e-12)


def test_locus_vote_recovers_clean_tracks(clean_event):
    xs, ys, zs, planes, truth = event_arrays(clean_event)
    cands, centres, sm = run_locus((xs, ys, zs, planes), n_bins=1024)
    _, _, metrics = validate_hough(cands, clean_event)
    assert metrics["efficiency"] >= 0.9
    assert metrics["ghost_rate"] == pytest.approx(0.0, abs=1e-12)


def test_adapter_track_objects(clean_event):
    D, planes, _ = hit_directions(clean_event)
    cands, _, _ = run_point(D, planes, n_bins=256)
    tracks = candidates_to_tracks(cands, clean_event)
    assert len(tracks) == len(cands)
    hits_by_id = {h.hit_id: h for h in clean_event.hits}
    for t, c in zip(tracks, cands):
        assert t.n_hits == len(c["hit_idx"])
        zs = [hits_by_id[i].z for i in t.hit_ids]
        assert zs == sorted(zs)  # z-ordered per Track convention
        assert "tx" in t.extra and "ty" in t.extra


def test_matches_originating_study_run_point(clean_event):
    import importlib.util
    import sys

    hough_dir = "/data/bfys/gscriven/Quantum_Track_Reconstruction/Hough"
    for p in (hough_dir,):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import hough_study_lib as lib
    except Exception as exc:
        pytest.skip(f"Hough study lib not importable: {exc}")

    D, planes, truth = hit_directions(clean_event)
    ours, _, _ = run_point(D, planes, n_bins=256, smooth_sigma=1.0)
    theirs, _, _ = lib.run_point(D, planes, 256, 1.0)
    assert len(ours) == len(theirs)
    for a, b in zip(ours, theirs):
        assert np.array_equal(np.sort(a["hit_idx"]), np.sort(b["hit_idx"]))
        assert a["tx"] == pytest.approx(b["tx"])
        assert a["ty"] == pytest.approx(b["ty"])


def test_canonical_matcher_agrees_with_majority_on_easy_regime(clean_event):
    """On clean well-separated tracks the metric-change caveat must vanish."""
    import importlib.util
    import sys

    D, planes, truth = hit_directions(clean_event)
    cands, _, _ = run_point(D, planes, n_bins=256)
    _, _, metrics = validate_hough(cands, clean_event)

    hough_dir = "/data/bfys/gscriven/Quantum_Track_Reconstruction/Hough"
    if hough_dir not in sys.path:
        sys.path.insert(0, hough_dir)
    try:
        import hough_prototype as proto
    except Exception as exc:
        pytest.skip(f"hough_prototype not importable: {exc}")
    study = proto.evaluate(cands, truth, clean_event)
    assert metrics["efficiency"] == pytest.approx(study["eff"])
    assert metrics["ghost_rate"] == pytest.approx(study["ghost_rate"])
