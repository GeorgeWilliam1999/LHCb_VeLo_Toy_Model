"""Measurement shots from an exact post-selected output state.

Provenance
----------
New, 2026-11-12 (experiment G of the research plan "cheap and effective VELO
track finding": the readout shot budget). Every quantum solver of the package
(one-bit filter, QSVT, the hit-register clique filters) is evaluated exactly,
which gives the post-selected output amplitudes ``psi_i`` per register item
(segment or hit). A measurement of the system register then returns item i
with probability ``p_i = |psi_i|^2 / sum_j |psi_j|^2``. This module draws those
shots and turns the counts into an "on" set that a classical readout
(``get_tracks_coupled`` for segments, ``clique_filters.tracks_from_filtered_hits``
for hits) can use. No circuit is simulated: the shots are drawn from the exact
distribution.

Two "on" rules
--------------
* count rule: item i is on when it was seen at least ``k`` times;
* estimate rule: additionally ``c_i / N >= p_min``, i.e. the exact-output
  reading ``p_i >= p_min`` applied to the estimated probability. As ``N`` grows
  the estimate rule reproduces the exact reading; the count rule switches on
  every item with ``p_i > 0``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["output_distribution", "ShotSampler", "on_items", "shot_grid"]


def output_distribution(amplitudes: np.ndarray) -> np.ndarray:
    """Measurement probabilities of the post-selected state.

    Parameters
    ----------
    amplitudes : (n,) array
        Output amplitudes per register item (real or complex, any
        normalisation). Padding items of a ``2^k`` register are excluded by
        passing only the real items.

    Returns
    -------
    (n,) array
        ``|a_i|^2 / sum |a|^2``. All zeros if every amplitude is zero.
    """
    w = np.abs(np.asarray(amplitudes)) ** 2
    s = float(w.sum())
    return w / s if s > 0 else w


def shot_grid(lo: float = 1e1, hi: float = 1e8, per_decade: int = 2) -> np.ndarray:
    """Logarithmic grid of shot counts (integers, strictly increasing)."""
    n = int(round(np.log10(hi / lo) * per_decade)) + 1
    g = np.unique(np.round(np.logspace(np.log10(lo), np.log10(hi), n)).astype(np.int64))
    return g


class ShotSampler:
    """Nested multinomial shots: counts for N_1 < N_2 < ... from one stream.

    Drawing the extra ``N_{j+1} - N_j`` shots on top of the counts at ``N_j``
    gives exactly the multinomial distribution at every ``N_j`` and keeps the
    counts of one seed monotone in N, so curves against N are not noisier than
    the shots themselves.

    Parameters
    ----------
    p : (n,) array
        Measurement probabilities (:func:`output_distribution`).
    seed : int
        Seed of the numpy Generator.

    Examples
    --------
    >>> s = ShotSampler(output_distribution(psi), seed=0)
    >>> for N in shot_grid():
    ...     c = s.counts_at(N)
    """

    def __init__(self, p: np.ndarray, seed: int = 0):
        self.p = np.asarray(p, dtype=np.float64)
        tot = self.p.sum()
        if tot <= 0:
            raise ValueError("no probability mass")
        self.p = self.p / tot
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(self.p.size, dtype=np.int64)
        self.n_shots = 0

    def counts_at(self, n_shots: int) -> np.ndarray:
        """Counts after ``n_shots`` shots in total (must not decrease)."""
        n_shots = int(n_shots)
        if n_shots < self.n_shots:
            raise ValueError("shot counts must be non-decreasing")
        extra = n_shots - self.n_shots
        if extra:
            self.counts += self.rng.multinomial(extra, self.p)
            self.n_shots = n_shots
        return self.counts


def on_items(counts: np.ndarray, n_shots: int, k: int = 1,
             p_min: Optional[float] = None) -> np.ndarray:
    """Items switched on by the shots.

    Parameters
    ----------
    counts : (n,) int array
        Shot counts per item.
    n_shots : int
        Total shots N.
    k : int
        Minimum count (count rule).
    p_min : float, optional
        Minimum estimated probability ``c_i / N`` (estimate rule); None for
        the count rule alone.

    Returns
    -------
    (n,) bool array
    """
    c = np.asarray(counts)
    on = c >= k
    if p_min is not None and p_min > 0:
        on &= c >= p_min * n_shots
    return on
