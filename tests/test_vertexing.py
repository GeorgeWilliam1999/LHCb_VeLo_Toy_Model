"""Direct PV readout (solvers/vertexing).

Contracts:
- the POCA of a line through a known vertex is that vertex (two points and a fit);
- the doublet / triplet tables flag exactly the pointing items, with the stated widths;
- a statevector simulation of amplitude amplification on a small oracle gives the
  success probability of `amplification` and, after the flag, the vertex-register
  distribution of `output_distribution` (the shortcut the studies rely on);
- the peak finder returns the planted vertices whatever the shot count scale;
- the track-based vertex fit recovers an off-beamline vertex that the z0 median misses.
"""
import math

import numpy as np
import scipy.sparse as sp

from lhcb_velo_toy.solvers.vertexing import (
    amplification,
    doublet_table,
    find_vertex_peaks,
    fit_line_poca,
    line_poca,
    match_vertices,
    output_distribution,
    sample_counts,
    sampler_resources,
    triplet_table,
    vertices_from_tracks,
    zeta_bins,
)

ZP = np.array([33.0, 66.0, 99.0])


def _track_points(v, t, zs=ZP):
    return np.array([[v[0] + t[0] * (z - v[2]), v[1] + t[1] * (z - v[2]), z] for z in zs])


def test_poca_two_points_and_fit():
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = np.array([0.0, 0.0, rng.uniform(-100, 20)])
        t = rng.uniform(-0.2, 0.2, 2)
        P = _track_points(v, t)
        z0, d, tn = line_poca(P[None, 0], P[None, 1])
        assert abs(z0[0] - v[2]) < 1e-9 and d[0] < 1e-9 and abs(tn[0] - np.hypot(*t)) < 1e-12
        z0, d, _ = fit_line_poca(P[None])
        assert abs(z0[0] - v[2]) < 1e-9 and d[0] < 1e-9
    # an off-beam line: doca equals the transverse offset for a line along x
    z0, d, _ = line_poca(np.array([[0.0, 0.5, 10.0]]), np.array([[1.0, 0.5, 20.0]]))
    assert abs(d[0] - 0.5) < 1e-12 and abs(z0[0] - 10.0) < 1e-12


def _toy_segments():
    """Two vertices, three tracks each, plus one non-pointing segment."""
    segs, hits, hid = [], [], 0
    for vz in (-20.0, 5.0):
        for t in ((0.1, 0.0), (0.0, -0.12), (-0.08, 0.05)):
            P = _track_points((0, 0, vz), t)
            ids = []
            for p in P:
                hits.append(p)
                ids.append(hid)
                hid += 1
            segs += [((P[0], P[1]), (ids[0], ids[1])), ((P[1], P[2]), (ids[1], ids[2]))]
    junk = (np.array([3.0, 3.0, 33.0]), np.array([3.0, 6.0, 66.0]))
    segs.append((junk, (hid, hid + 1)))
    xyz = np.array([[a, b] for (a, b), _ in segs])
    sh = np.array([ids for _, ids in segs])
    return xyz, sh


def test_doublet_and_triplet_tables():
    xyz, sh = _toy_segments()
    tab = doublet_table(xyz, sh, dmax=0.05, tmin=0.02, zwin=(-50, 30))
    assert tab.n_items == 13 and tab.n_flagged == 12 and tab.n_index_qubits == 4
    assert np.allclose(np.sort(tab.z0), [-20] * 6 + [5] * 6)
    # couplings: segment 2k -> 2k+1 for each track (forward), plus a backward copy
    n = len(sh)
    rows = np.arange(0, 12, 2)
    A = sp.coo_matrix((np.ones(12), (np.r_[rows, rows + 1], np.r_[rows + 1, rows])), shape=(n, n))
    tt = triplet_table(xyz, sh, A, dmax=0.05, tmin=0.02, zwin=(-50, 30))
    assert tt.n_items == 6 and tt.n_flagged == 6 and tt.cuts["d_max"] == 1
    assert tt.n_index_qubits == 4 and tt.items.shape == (6, 3)


def _grover_statevector(flag, zeta, r):
    """Statevector amplitude amplification on the index register; returns
    P(flag) and the vertex-register distribution conditional on flag = 1."""
    N = len(flag)
    psi = np.full(N, 1 / math.sqrt(N))
    s = psi.copy()
    for _ in range(r):
        psi = np.where(flag, -psi, psi)               # oracle phase on flagged items
        psi = 2 * s * (s @ psi) - psi                 # reflection about the start state
    pr = psi ** 2
    pf = pr[flag].sum()
    dist = np.bincount(zeta[flag], weights=pr[flag], minlength=zeta.max() + 1) / pf
    return pf, dist


def test_amplification_matches_statevector():
    rng = np.random.default_rng(3)
    for N, nf in ((64, 1), (256, 5), (1024, 40), (16, 6)):
        flag = np.zeros(N, bool)
        flag[rng.choice(N, nf, replace=False)] = True
        zeta = rng.integers(0, 7, N)
        r, ps, qps = amplification(nf / N)
        pf, dist = _grover_statevector(flag, zeta, r)
        assert abs(pf - ps) < 1e-12
        exact = np.bincount(zeta[flag], minlength=zeta.max() + 1) / nf
        assert np.allclose(dist, exact, atol=1e-12)
        assert abs(qps - (2 * r + 1) / ps) < 1e-12
    assert amplification(0.0)[2] == math.inf
    assert amplification(1.0)[0] == 0


def test_peaks_recover_planted_vertices_at_any_scale():
    rng = np.random.default_rng(7)
    true = np.array([-60.0, -20.0, -17.0, 10.0])
    n_per = np.array([40, 25, 30, 12])
    z = np.concatenate([rng.normal(v, 0.1, n) for v, n in zip(true, n_per)] + [rng.uniform(-100, 30, 30)])
    edges, q = zeta_bins((-100, 30), 0.05)
    assert q == 12
    from lhcb_velo_toy.solvers.vertexing import VertexOracleTable
    tab = VertexOracleTable("doublet", 10 ** 5, 17, z, np.zeros((len(z), 2), int), np.zeros(len(z)))
    p = output_distribution(tab, edges)
    exact = find_vertex_peaks(np.histogram(z, edges)[0], edges, 1.0, 0.15, 6, 1.0)
    m, res, merges, fakes = match_vertices([e[0] for e in exact], true, 1.0)
    assert (m >= 0).all() and fakes == 0 and np.abs(res).max() < 0.1
    K = 20000
    c = sample_counts(p, K, rng)
    shots = find_vertex_peaks(c, edges, tab.n_flagged / K, 0.15, 6, 1.0)
    m, res, _, fakes = match_vertices([e[0] for e in shots], true, 1.0)
    assert (m >= 0).all() and fakes == 0
    R = sampler_resources(tab, q, K)
    assert R["q_data"] == 17 + 12 + 1 and R["queries"] >= K and R["classical_exhaustive"] == 10 ** 5


def test_track_vertex_fit_off_beamline():
    rng = np.random.default_rng(11)
    v = np.array([0.03, -0.02, -12.0])               # 30 um off the beamline
    lines = []
    for _ in range(20):
        t = rng.uniform(-0.1, 0.1, 2)
        if np.hypot(*t) < 0.03:
            continue
        lines.append((v[0] - t[0] * v[2], v[1] - t[1] * v[2], t[0], t[1]))
    lines = np.array(lines)
    med = vertices_from_tracks(lines, fit=False)
    fit = vertices_from_tracks(lines, fit=True)
    assert len(fit) == 1 and abs(fit[0][0] - v[2]) < 1e-6
    assert abs(fit[0][1] - v[0]) < 1e-6 and abs(fit[0][2] - v[1]) < 1e-6
    assert abs(med[0][0] - v[2]) > 1e-3              # the beamline z0 median is biased
