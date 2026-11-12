"""Hit-register clique filters (solvers/quantum/clique_filters.py).

Contracts:
- a clique K_m loads only its line m - 1 under the uniform probe: the step
  passes it with weight 1 for m >= 3; pairs and isolated hits are rejected;
- the {3, 4} projector rejects a 3-hit clique, which the step passes;
- the step polynomial is bounded, |p| <= 1 on [-alpha, alpha];
- filter_hit_operator equals a dense eigh of the whole matrix on a random
  union of cliques with extra couplings;
- padding: P_padded = n P / N + (N - n) f(0)^2 / N;
- the readout returns the cliques of >= 3 surviving hits.
"""
import math

import numpy as np
import pytest
import scipy.sparse as sp

from lhcb_velo_toy.solvers.quantum import (
    design_step_polynomial,
    filter_hit_operator,
    ideal_line_projector,
    ideal_step,
    one_bit_clique,
    soft_band,
    tracks_from_filtered_hits,
)


def union_of_cliques(sizes, extra=0, seed=0):
    """Block-diagonal union of cliques K_m, plus ``extra`` random couplings."""
    rng = np.random.default_rng(seed)
    n = int(sum(sizes))
    M = np.zeros((n, n))
    o = 0
    for m in sizes:
        M[o:o + m, o:o + m] = 1.0
        o += m
    np.fill_diagonal(M, 0.0)
    for _ in range(extra):
        i, j = rng.choice(n, 2, replace=False)
        M[i, j] = M[j, i] = 1.0
    return sp.csr_matrix(M)


@pytest.mark.parametrize("m", [3, 4, 5, 8, 13])
def test_clique_passes_at_line_m_minus_1(m):
    C = union_of_cliques([m])
    r = filter_hit_operator(C, ideal_step(1.5))
    assert r["P"] == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(r["weight"], 1.0)
    w, V = np.linalg.eigh(C.toarray())
    u = np.full(m, 1 / math.sqrt(m))
    load = (V.T @ u) ** 2
    assert load[np.isclose(w, m - 1)].sum() == pytest.approx(1.0)


def test_pair_and_isolated_rejected():
    C = union_of_cliques([2, 1, 1, 3])
    for f in (ideal_step(1.5), soft_band(1.5)):
        r = filter_hit_operator(C, f)
        assert np.allclose(r["weight"][:4], 0.0)
        assert np.allclose(r["weight"][4:], 1.0)
        assert r["P"] == pytest.approx(3 / 7)


def test_line_projector_34_rejects_three_hit_clique():
    C = union_of_cliques([3, 4, 5])
    r34 = filter_hit_operator(C, ideal_line_projector((3, 4)))
    assert np.allclose(r34["weight"][:3], 0.0)
    assert np.allclose(r34["weight"][3:], 1.0)
    assert filter_hit_operator(C, ideal_step())["P"] == pytest.approx(1.0)


@pytest.mark.parametrize("degree", [6, 10, 16])
@pytest.mark.parametrize("alpha", [5.0, 13.0])
def test_step_polynomial_bounded(degree, alpha):
    p = design_step_polynomial(degree, alpha)
    xs = np.linspace(-alpha, alpha, 20001)
    assert np.max(np.abs(p(xs))) <= 1.0 + 1e-12
    assert p.degree() == degree
    assert p(4.0) > p(1.0) > p(-1.0) - 0.2   # increasing through the step


def test_step_polynomial_shifted_domain_bounded_and_sharper():
    a, lo, d = 13.0, -3.0, 16
    p = design_step_polynomial(d, a, lo=lo)
    xs = np.linspace(lo, a, 20001)
    assert np.max(np.abs(p(xs))) <= 1.0 + 1e-12
    sym = design_step_polynomial(d, a)
    assert p(2.0) - p(1.0) > sym(2.0) - sym(1.0)


def test_matches_dense_eigh_on_random_union():
    C = union_of_cliques([3, 4, 5, 5, 2, 1, 6], extra=6, seed=3)
    n = C.shape[0]
    u = np.full(n, 1 / math.sqrt(n))
    w, V = np.linalg.eigh(C.toarray())
    for f in (design_step_polynomial(10, 6.0), soft_band(1.5), one_bit_clique(4.0),
              ideal_line_projector((3, 4))):
        ref = V @ (np.asarray(f(w)) * (V.T @ u))
        r = filter_hit_operator(C, f)
        assert np.allclose(r["y"], ref, atol=1e-12)
        assert r["P"] == pytest.approx(float(ref @ ref), abs=1e-12)


def test_padding_formula():
    C = union_of_cliques([5, 4, 2])       # n = 11 -> N = 16
    p = design_step_polynomial(6, 5.0)
    r = filter_hit_operator(C, p)
    f0 = float(p(0.0))
    assert r["n_sys"] == 4
    assert r["P_padded"] == pytest.approx(11 / 16 * r["P"] + 5 / 16 * f0 ** 2)
    r1 = filter_hit_operator(C, one_bit_clique(4.0))       # f(0) = 0 exactly
    assert r1["P_padded"] == pytest.approx(11 / 16 * r1["P"])


def test_one_bit_clique_matches_cosine_on_A():
    c = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    s = 4.0
    lam = s - c
    assert np.allclose(one_bit_clique(s)(c), np.cos(lam * (np.pi / s) / 2))


def test_readout_returns_cliques():
    C = union_of_cliques([3, 5, 2, 1, 4])
    r = filter_hit_operator(C, ideal_step())
    tr = tracks_from_filtered_hits(C, r["weight"], threshold=0.5)
    assert sorted(len(t) for t in tr) == [3, 4, 5]
