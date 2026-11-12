"""Segment windows (solvers/hamiltonians/windows.py).

Contracts:
- slopes, d0 and z0 of a line are exact for hand-made segments;
- the window keeps prompt segments of a toy event and drops a wide-slope or
  far-pointing one;
- restricting the system equals building the Hamiltonian on an event that
  only has the kept segments' hits, when every segment of that event is kept;
- a window never enlarges a connected component.
"""
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from lhcb_velo_toy.analysis.segment_metrics import compute_epsilon
from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
from lhcb_velo_toy.solvers.hamiltonians import SimpleHamiltonianFast
from lhcb_velo_toy.solvers.hamiltonians.windows import restrict_system, segment_geometry, segment_window_mask


def seg(p, q):
    return np.array([[p, q]], dtype=float)


def test_geometry_of_a_line():
    # line through (1, 0, 10) with slope (0.1, 0): closest to the z axis at z = 0, d0 = 0
    g = segment_geometry(seg((1.0, 0.0, 10.0), (2.0, 0.0, 20.0)))
    assert g["tx"][0] == pytest.approx(0.1) and g["ty"][0] == 0
    assert g["d0"][0] == pytest.approx(0.0, abs=1e-12) and g["z0"][0] == pytest.approx(0.0)
    # offset line: x = 1 + 0.1 (z - 10), y = 3 -> d0 = 3
    g = segment_geometry(seg((1.0, 3.0, 10.0), (2.0, 3.0, 20.0)))
    assert g["d0"][0] == pytest.approx(3.0)
    # parallel to the beam: d0 = transverse radius, z0 NaN
    g = segment_geometry(seg((3.0, 4.0, 10.0), (3.0, 4.0, 20.0)))
    assert g["d0"][0] == pytest.approx(5.0) and np.isnan(g["z0"][0])


def test_mask_cuts():
    e = np.concatenate([seg((1, 0, 10), (2, 0, 20)),          # prompt, slope 0.1
                        seg((10, 0, 10), (20, 0, 20)),        # prompt, slope 1.0
                        seg((1, 3, 10), (2, 3, 20)),          # slope 0.1, d0 = 3
                        seg((1, 0, 110), (2, 0, 120))])       # slope 0.1, points to z = 100
    assert segment_window_mask(e).tolist() == [True] * 4
    assert segment_window_mask(e, max_slope=0.5).tolist() == [True, False, True, True]
    assert segment_window_mask(e, max_d0=1.0).tolist() == [True, True, False, True]
    assert segment_window_mask(e, z_range=(-50.0, 50.0)).tolist() == [True, True, True, False]


@pytest.fixture(scope="module")
def ham():
    n, dz = 5, 33.0
    geo = PlaneGeometry(module_id=list(range(n)), lx=[40.0] * n, ly=[40.0] * n, z=[33.0 + i * dz for i in range(n)])
    gen = StateEventGenerator(detector_geometry=geo, events=1, n_particles=[40], measurement_error=0.005,
                              collision_noise=1.5e-4, seed=11, phi_min=-0.2, phi_max=0.2, theta_min=-0.2,
                              theta_max=0.2)
    gen.generate_random_primary_vertices({"x": 0.0, "y": 0.0, "z": 1.0})
    gen.generate_particles([[{"type": "pion", "mass": 139.6, "q": 1}] * 40])
    gen.generate_complete_events()
    h = SimpleHamiltonianFast(epsilon=compute_epsilon(0.005, 1.5e-4, dz=dz), gamma=3.0, delta=1.0)
    h.construct_segments(gen, materialize_segments=False)
    h.construct_hamiltonian(gen)
    return h


def test_true_segments_inside_a_prompt_window(ham):
    tid = np.asarray(ham._segment_track_ids)
    truth = (tid[:, 0] == tid[:, 1]) & (tid[:, 0] >= 0)
    keep = segment_window_mask(ham._segment_endpoints_xyz, max_slope=0.3, max_d0=0.5, z_range=(-50, 50))
    assert keep[truth].all()
    assert keep.sum() < len(keep)          # it does remove combinatorial segments


def test_restriction_is_a_subgraph(ham):
    keep = segment_window_mask(ham._segment_endpoints_xyz, max_slope=0.15)
    Aw, bw, idx = restrict_system(ham.A, ham.b, keep)
    A = ham.A.tocsr()
    assert np.allclose(Aw.diagonal(), A.diagonal()[idx]) and np.allclose(bw, np.asarray(ham.b)[idx])
    assert (abs(Aw - A[idx][:, idx]).sum()) == 0

    def comps(M):
        off = M.copy().tolil()
        off.setdiag(0)
        return connected_components(sp.csr_matrix(off), directed=False)[1]
    full, sub = comps(A), comps(Aw)
    for c in np.unique(sub):
        assert len(np.unique(full[idx[sub == c]])) == 1     # each windowed component lies in one full one
