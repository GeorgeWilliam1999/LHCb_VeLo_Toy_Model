"""Threshold / working-point policy as API — metrics as a VIEW.

Provenance
----------
Promoted from ``Toy_Characterisation/_shared/qtrk_pipeline/metrics.py``
(centralise-2.1 Phase 4, 2026-08-26). The policy history it encodes:

* The relative-threshold bug: a metric frozen into a pickle with a
  ``tau * max(sol)`` cut. Metrics here are a pure function of
  (solution, truth, threshold) computed by ONE code path.
* ``ABS_THRESHOLD = 0.35`` is the validated absolute operating point FOR
  gamma=3, delta=1 (false attractor delta/(delta+gamma) = 0.25 + 0.10
  margin); :func:`threshold_for` generalises it gamma-aware.
* The wp99 refresh (user ruling 2026-06-14): the HEADLINE operating point
  for filtered solvers (1BQF / QSVT) is the efficiency-first working point
  (:func:`working_point_threshold`, target eff 0.99) — the fixed cut chops
  genuine true segments on filtered spectra (the "~75 % efficiency
  plateau" artefact).
* The quantum rescale convention: unit-norm quantum solutions are rescaled
  onto the classical SIGNAL support, not the full norm (which is
  false-bulk dominated and multiplicity-dependent) — see
  :func:`rescale_to_signal`.
"""
from __future__ import annotations

import numpy as np

from lhcb_velo_toy.analysis.segment_metrics import (
    segment_truth_mask,
    solver_segment_metrics,
)

__all__ = [
    "ABS_THRESHOLD", "WP_TARGET_EFF", "WP_TAU_FLOOR",
    "threshold_for", "truth_from_event", "segments_only",
    "rescale_quantum", "rescale_to_signal", "cos_sim",
    "metrics_at", "quantum_metrics",
    "working_point_threshold", "metrics_at_wp", "quantum_metrics_wp",
]

# The validated absolute operating point FOR gamma=3, delta=1 (Hopfield false
# attractor delta/(delta+gamma)=0.25, true plateau ~0.375).  Do NOT use a
# relative tau*max.
ABS_THRESHOLD = 0.35

# Margin above the false attractor (0.25 + 0.10 = 0.35 at gamma=3).
_THRESHOLD_MARGIN = 0.10

# Default efficiency target for the HEADLINE operating point (user, 2026-06-14).
WP_TARGET_EFF = 0.99

WP_TAU_FLOOR = 1e-6   # never below this: a tau <= 0 would re-admit the segments
                      # the 1BQF STRUCTURALLY zeroes (isolated clusters sit at
                      # lambda = s exactly -> notch kills them to ~1e-17)


def threshold_for(gamma: float = 3.0, delta: float = 1.0,
                  margin: float = _THRESHOLD_MARGIN) -> float:
    """gamma-aware absolute segment threshold.

    The false/isolated Hopfield attractor sits at ``delta/(delta+gamma)``; the
    operating point is a fixed ``margin`` above it.  Reduces to the validated
    0.35 at gamma=3, delta=1.  For gamma!=3 the fixed 0.35 is wrong (e.g. at
    gamma=1 the attractor is 0.5 > 0.35, so every false segment reads active).
    """
    return float(delta) / (float(delta) + float(gamma)) + float(margin)


def segments_only(event, epsilon: float = 1.0, gamma: float = 3.0,
                  delta: float = 1.0):
    """Build only the segment skeleton (no A) — enough for the truth mask.

    Much cheaper than a full Hamiltonian build; ``materialize_segments=False``
    skips the O(n_seg) Python ``Segment`` objects — the truth mask only needs
    the cached ``_segment_track_ids`` numpy array.
    """
    from lhcb_velo_toy.solvers import SimpleHamiltonianFast

    ham = SimpleHamiltonianFast(epsilon=float(epsilon), gamma=float(gamma),
                                delta=float(delta))
    ham.construct_segments(event, materialize_segments=False)
    return ham


def truth_from_event(event) -> np.ndarray:
    """Per-segment truth mask, recomputed from the event (no A build)."""
    return segment_truth_mask(segments_only(event))


def rescale_quantum(sol_Q: np.ndarray, sol_C: np.ndarray) -> np.ndarray:
    """Rescale ``sol_Q`` so its L2 norm matches ``sol_C`` (canonical convention)."""
    nQ = float(np.linalg.norm(sol_Q))
    nC = float(np.linalg.norm(sol_C))
    return sol_Q * (nC / nQ) if nQ > 0 else np.zeros_like(sol_Q)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb + 1e-30))


def metrics_at(sol: np.ndarray, truth: np.ndarray,
               threshold: float = ABS_THRESHOLD) -> dict:
    """Segment efficiency / purity / false-rate at an ABSOLUTE threshold."""
    return solver_segment_metrics(np.asarray(truth, bool), np.asarray(sol),
                                  threshold=float(threshold))


def rescale_to_signal(sol_Q_raw: np.ndarray, sol_C: np.ndarray,
                      threshold: float = ABS_THRESHOLD) -> np.ndarray:
    """Put a unit-norm quantum solution on the classical SIGNAL scale.

    Matching the *full* L2 norm is WRONG: ``||sol_C||`` is dominated by the
    O(n_seg) false segments pinned at the Hopfield attractor (95-99 % of the
    norm), so it grows ~sqrt(n_seg) — a multiplicity-dependent rescale that
    inflates the quantum amplitudes and made the quantum false rate diverge
    from the classical one (~44 % vs ~2 % at T=400) while the underlying
    solve was faithful. Fix: rescale on the classical SIGNAL support
    (``sol_C > threshold``) — truth-free and n-independent.
    """
    sol_Q_raw = np.asarray(sol_Q_raw, np.float64)
    sol_C = np.asarray(sol_C, np.float64)
    mask = sol_C > float(threshold)
    if not mask.any():                              # nothing active -> fall back
        return rescale_quantum(sol_Q_raw, sol_C)
    denom = float(np.linalg.norm(sol_Q_raw[mask]))
    if denom == 0.0:
        return np.zeros_like(sol_Q_raw)
    return sol_Q_raw * (float(np.linalg.norm(sol_C[mask])) / denom)


def quantum_metrics(sol_Q_raw: np.ndarray, sol_C: np.ndarray,
                    truth: np.ndarray, threshold: float = ABS_THRESHOLD) -> dict:
    """Metrics for a quantum solve: rescale on the classical SIGNAL, then threshold.

    Returns the segment metrics dict plus ``cos_QC``, the fidelity to the
    classical partner measured on the same signal support (the full-vector
    cosine is dragged down to ~0.1 by the false bulk and understates the
    agreement).
    """
    sol_C = np.asarray(sol_C, np.float64)
    sol_Q = rescale_to_signal(sol_Q_raw, sol_C, threshold)
    out = metrics_at(sol_Q, truth, threshold)
    mask = sol_C > float(threshold)
    if mask.any():
        out["cos_QC"] = cos_sim(sol_Q[mask], sol_C[mask])
    else:
        out["cos_QC"] = cos_sim(sol_Q, sol_C)
    return out


# ---------------------------------------------------------------------------
# Efficiency-first working point (wp99) — the HEADLINE operating point.
# ---------------------------------------------------------------------------

def working_point_threshold(sol: np.ndarray, truth: np.ndarray,
                            target_eff: float = WP_TARGET_EFF) -> float:
    """Efficiency-first per-solve segment threshold (the wp99 working point).

    Place ``tau`` just BELOW the ``(1 - target_eff)`` quantile of the
    TRUE-segment amplitudes, so >= ``target_eff`` of the true segments stay
    active. Scale-invariant in eff/false-rate.

    DEGENERATE CORNER (found on Run-3, 2026-07-06): if more than
    (1 - target_eff) of the TRUE segments have ~zero amplitude — the 1BQF
    structurally zeroes ISOLATED clusters, and real events have isolated-TRUE
    segments from missing hits — the quantile lands at ~0 and the threshold
    would go negative, re-admitting every structurally-killed segment. It is
    therefore floored at ``WP_TAU_FLOOR``: the working point then reads
    "admit every segment the solver can PHYSICALLY recover", and the achieved
    efficiency honestly reports the structural miss instead of fabricating
    100 %.
    """
    s = np.abs(np.asarray(sol, np.float64))
    t = np.asarray(truth, bool)
    st = np.sort(s[t])
    if st.size == 0:
        return WP_TAU_FLOOR
    k = int(np.floor((1.0 - float(target_eff)) * st.size))
    k = min(max(k, 0), st.size - 1)
    return float(max(st[k] - 1e-9, WP_TAU_FLOOR))


def metrics_at_wp(sol: np.ndarray, truth: np.ndarray,
                  target_eff: float = WP_TARGET_EFF) -> dict:
    """Segment metrics at the efficiency-first working point.

    Drop-in replacement for :func:`metrics_at` for the HEADLINE numbers.
    Adds ``tau_wp`` (the threshold used) and ``target_eff``.
    """
    sol = np.abs(np.asarray(sol, np.float64))
    tau = working_point_threshold(sol, truth, target_eff)
    out = metrics_at(sol, truth, threshold=tau)
    out["tau_wp"] = tau
    out["target_eff"] = float(target_eff)
    return out


def quantum_metrics_wp(sol_Q_raw: np.ndarray, sol_C: np.ndarray,
                       truth: np.ndarray,
                       target_eff: float = WP_TARGET_EFF,
                       signal_threshold: float = ABS_THRESHOLD) -> dict:
    """Quantum-solve metrics at the efficiency-first working point.

    Like :func:`quantum_metrics` but the threshold is the per-solve wp99
    working point on the (signal-rescaled) quantum amplitudes, not the fixed
    cut. Pass the gamma-aware ``signal_threshold`` for gamma != 3 so
    ``tau_wp`` / ``cos_QC`` sit on the right classical scale; eff/false-rate
    are rescale-invariant either way.
    """
    sol_C = np.asarray(sol_C, np.float64)
    sol_Q = rescale_to_signal(sol_Q_raw, sol_C, signal_threshold)
    out = metrics_at_wp(sol_Q, truth, target_eff)
    mask = sol_C > float(signal_threshold)
    if mask.any():
        out["cos_QC"] = cos_sim(np.asarray(sol_Q, np.float64)[mask], sol_C[mask])
    else:
        out["cos_QC"] = cos_sim(sol_Q, sol_C)
    return out
