"""Component split, per-component Givens price and shots (solvers/quantum/components.py).

Contracts:
- the split covers every row once and matches the connected components;
- a filter applied per component, each with its own uniform probe, equals the
  whole-register filter output after the probe rescaling sqrt(m / n);
- Givens price of one component: pairs x (2 (dH - 1) + MCRX(n_s));
- shot allocation is ceil(beta m ln m) >= 1;
- per-component counts sum to each budget, are nested, and stay inside the
  component.
"""
import math

import numpy as np
import pytest
import scipy.sparse as sp

from lhcb_velo_toy.solvers.quantum import design_step_polynomial, filter_hit_operator, one_bit_clique
from lhcb_velo_toy.solvers.quantum.components import (
    ComponentShotSampler,
    component_givens_price,
    popcount_mean,
    shot_allocation,
    split_components,
)


def cliques(sizes, extra=0, seed=0):
    rng = np.random.default_rng(seed)
    n = sum(sizes)
    M = np.zeros((n, n))
    o = 0
    for m in sizes:
        M[o:o + m, o:o + m] = 1
        o += m
    np.fill_diagonal(M, 0)
    for _ in range(extra):
        i, j = rng.choice(n, 2, replace=False)
        M[i, j] = M[j, i] = 1
    perm = rng.permutation(n)                  # scatter the blocks over the register
    return sp.csr_matrix(M[np.ix_(perm, perm)])


def test_split_covers_rows():
    C = cliques([3, 1, 5, 2, 4], extra=0, seed=1)
    comps, lab = split_components(C)
    allrows = np.sort(np.concatenate(comps))
    assert np.array_equal(allrows, np.arange(C.shape[0]))
    assert sorted(len(c) for c in comps) == [1, 2, 3, 4, 5]
    for k, c in enumerate(comps):
        assert np.all(lab[c] == k)


@pytest.mark.parametrize("make", [lambda: one_bit_clique(8.0), lambda: design_step_polynomial(6, 6.0, lo=-2.0)])
def test_per_component_equals_whole(make):
    f = make()
    C = cliques([3, 5, 2, 1, 4, 6], extra=3, seed=4)
    n = C.shape[0]
    whole = filter_hit_operator(C, f)["y"]
    comps, _ = split_components(C)
    for c in comps:
        part = filter_hit_operator(C[c][:, c], f)["y"]          # its own uniform probe 1/sqrt(m)
        assert np.allclose(part * math.sqrt(len(c) / n), whole[c], atol=1e-12)


def test_givens_price():
    C = cliques([4, 2], seed=0)
    comps, _ = split_components(C)
    reg = np.arange(C.shape[0])
    mc = lambda m: 10.0 * m  # noqa: E731
    pr = component_givens_price(C, comps, reg, mc)
    for c, idx in enumerate(comps):
        m = len(idx)
        assert pr["m"][c] == m and pr["k"][c] == m * (m - 1) / 2
        ns = max(1, math.ceil(math.log2(m)))
        assert pr["n_s"][c] == ns
        assert pr["cx_step"][c] == pytest.approx(pr["k"][c] * (2 * (pr["dH"][c] - 1) + mc(ns)))
    assert popcount_mean(np.array([0, 1]), np.array([3, 1])) == 1.0
    assert popcount_mean(np.array([], int), np.array([], int)) == 1.0


def test_shot_allocation():
    a = shot_allocation(np.array([1, 2, 10]), 1.0)
    assert np.array_equal(a, [1, 2, 24])   # ceil(ln 2), ceil(2 ln 2), ceil(10 ln 10)
    assert np.all(shot_allocation(np.array([1, 3]), 0.01) >= 1)


def test_component_sampler_nested_and_local():
    C = cliques([3, 4, 1], seed=2)
    comps, _ = split_components(C)
    p = np.zeros(C.shape[0])
    for c in comps:
        if len(c) > 1:
            p[c] = np.arange(1, len(c) + 1) / np.arange(1, len(c) + 1).sum()
    s = ComponentShotSampler(p, comps, seed=0)
    prev = np.zeros(C.shape[0], int)
    for b in (5, 50, 500):
        budget = np.full(len(comps), b)
        cnt = s.counts_at(budget).copy()
        for c in comps:
            assert cnt[c].sum() == (b if len(c) > 1 else 0)
        assert np.all(cnt >= prev)
        prev = cnt
