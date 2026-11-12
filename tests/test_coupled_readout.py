"""Coupling-following readout (2026-11-12).

The claim under test: linking active segments only along the Hamiltonian
couplings never joins two tracks through a segment that is not angle-compatible
with them (the get_tracks failure on dense events), hit uniqueness removes the
branches that remain, and on a clean toy event with the true segments active
every group is a piece of one truth track.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from lhcb_velo_toy.analysis.segment_metrics import compute_epsilon
from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
from lhcb_velo_toy.solvers.hamiltonians import SimpleHamiltonianFast
from lhcb_velo_toy.solvers.reconstruction.coupled_readout import (
    coupled_hit_groups,
    get_tracks_coupled,
)
from lhcb_velo_toy.solvers.reconstruction.track_finder import get_tracks

N_MODULES, DZ, Z0, HALF = 5, 33.0, 33.0, 40.0
GAMMA, DELTA = 3.0, 1.0
ME, CN, SEED, T = 0.005, 1.5e-4, 7, 30
EPS = compute_epsilon(ME, CN, dz=DZ)


def _ham(n, couplings):
    """Hand-made Hamiltonian: diagonal -4, +1 on each coupled pair."""
    i, j = (np.array(c) for c in zip(*couplings)) if couplings else (np.zeros(0, int),) * 2
    A = sp.coo_matrix((np.r_[np.full(n, -4.0), np.ones(2 * len(i))],
                       (np.r_[np.arange(n), i, j], np.r_[np.arange(n), j, i])), shape=(n, n))
    return A.tocsr()


def _sets(groups):
    return sorted(tuple(sorted(int(h) for h in g)) for g in groups)


def test_false_bridge_does_not_merge():
    # track 1: hits 0-1-2, track 2: hits 3-4-5; segment 4 = 1 -> 4 is a false bridge,
    # shares hits with both tracks but is coupled to neither
    seg_hits = np.array([[0, 1], [1, 2], [3, 4], [4, 5], [1, 4]])
    A = _ham(5, [(0, 1), (2, 3)])
    act = np.ones(5, bool)
    g = coupled_hit_groups(A, seg_hits, act)
    assert _sets(g) == [(0, 1, 2), (1, 4), (3, 4, 5)]
    # the bridge alone is dropped by a 3-hit minimum
    assert _sets(coupled_hit_groups(A, seg_hits, act, min_hits=3)) == [(0, 1, 2), (3, 4, 5)]


def test_uniqueness_removes_a_coupled_fork():
    # hit 1 has two outgoing segments, both angle-compatible with 0 -> 1
    seg_hits = np.array([[0, 1], [1, 2], [1, 3], [2, 4]])
    A = _ham(4, [(0, 1), (0, 2), (1, 3)])
    x = np.array([0.5, 0.45, 0.30, 0.5])
    act = np.ones(4, bool)
    assert _sets(coupled_hit_groups(A, seg_hits, act)) == [(0, 1, 2, 3, 4)]
    assert _sets(coupled_hit_groups(A, seg_hits, act, solution=x, uniqueness="greedy")) == [(0, 1, 2, 4)]
    with pytest.raises(ValueError):
        coupled_hit_groups(A, seg_hits, act, uniqueness="greedy")


def test_empty_active_set():
    seg_hits = np.array([[0, 1]])
    assert coupled_hit_groups(_ham(1, []), seg_hits, np.zeros(1, bool)) == []


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


def test_true_segments_give_pure_groups(event):
    h = SimpleHamiltonianFast(EPS, GAMMA, DELTA)
    h.construct_segments(event, materialize_segments=False)
    h.construct_hamiltonian(event)
    tid = np.asarray(h._segment_track_ids)
    truth = (tid[:, 0] == tid[:, 1]) & (tid[:, 0] >= 0)
    x = truth.astype(float)
    tracks = get_tracks_coupled(h, x, event, threshold=0.5)
    track_of_hit = {hit.hit_id: hit.track_id for hit in event.hits}
    for t in tracks:
        assert len({track_of_hit[k] for k in t.hit_ids}) == 1
    # the coupled readout never builds a group larger than get_tracks does
    h2 = SimpleHamiltonianFast(EPS, GAMMA, DELTA)
    h2.construct_hamiltonian(event)
    assert max(len(t.hit_ids) for t in tracks) <= max(len(t.hit_ids) for t in get_tracks(h2, x, event, 0.5))


def test_matches_get_tracks_when_every_shared_hit_is_coupled():
    # a single 4-hit chain, all consecutive pairs coupled: both rules give one track
    seg_hits = np.array([[0, 1], [1, 2], [2, 3]])
    A = _ham(3, [(0, 1), (1, 2)])
    assert _sets(coupled_hit_groups(A, seg_hits, np.ones(3, bool))) == [(0, 1, 2, 3)]
