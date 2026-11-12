"""Shot sampling from an exact output state (solvers/quantum/shots.py).

Contracts:
- the distribution is |a|^2 normalised;
- nested counts sum to N and never decrease; the marginal mean is N p;
- the estimate rule converges to the exact reading p >= p_min, the count rule
  to p > 0;
- the grid is increasing and spans the requested range.
"""
import numpy as np
import pytest

from lhcb_velo_toy.solvers.quantum.shots import ShotSampler, on_items, output_distribution, shot_grid


def test_output_distribution():
    p = output_distribution(np.array([1.0, -1.0, 2j, 0.0]))
    assert np.allclose(p, [1 / 6, 1 / 6, 4 / 6, 0])
    assert np.all(output_distribution(np.zeros(3)) == 0)


def test_nested_counts_monotone_and_unbiased():
    p = np.array([0.5, 0.3, 0.2, 0.0])
    s = ShotSampler(p, seed=1)
    prev = np.zeros(4, int)
    for N in (10, 100, 1000, 100000):
        c = s.counts_at(N).copy()
        assert c.sum() == N and np.all(c >= prev) and c[3] == 0
        prev = c
    assert np.allclose(prev / 100000, p, atol=0.01)
    with pytest.raises(ValueError):
        s.counts_at(10)


def test_rules_converge_to_exact_reading():
    rng = np.random.default_rng(0)
    p = output_distribution(rng.random(50) * (rng.random(50) > 0.3))
    p_min = np.quantile(p[p > 0], 0.5)
    s = ShotSampler(p, seed=2)
    N = 10 ** 7
    c = s.counts_at(N)
    est = on_items(c, N, k=1, p_min=p_min)
    margin = np.abs(p - p_min) > 0.05 * p_min
    assert np.array_equal(est[margin], (p >= p_min)[margin])
    assert np.array_equal(on_items(c, N, k=1), p > 0)


def test_grid():
    g = shot_grid(10, 1e6, 2)
    assert g[0] == 10 and g[-1] == 10 ** 6 and np.all(np.diff(g) > 0) and len(g) == 11
