"""
Analysis submodule for the LHCb VELO Toy Model.

This module provides validation metrics and plotting utilities for
evaluating track reconstruction performance.

Submodules
----------
validation
    Track matching and LHCb-style metrics
plotting
    Event displays and performance plots
"""

# Validation
from lhcb_velo_toy.analysis.validation import (
    Match,
    EventValidator,
)

# Segment-level acceptance metrics (used by the segment-analysis experiments
# under Toy_Characterisation/).
from lhcb_velo_toy.analysis.segment_metrics import (
    compute_epsilon,
    pairwise_angle,
    collect_segment_pair_angles,
    compute_segment_metrics,
    fast_segment_metrics,
    segment_truth_mask,
    solver_segment_metrics,
)

# Threshold / working-point policy (metrics as a view; wp99 headline point)
from lhcb_velo_toy.analysis.working_points import (
    ABS_THRESHOLD,
    WP_TARGET_EFF,
    threshold_for,
    truth_from_event,
    metrics_at,
    quantum_metrics,
    working_point_threshold,
    metrics_at_wp,
    quantum_metrics_wp,
)

__all__ = [
    # Validation
    "Match",
    "EventValidator",
    # Segment metrics
    "compute_epsilon",
    "pairwise_angle",
    "collect_segment_pair_angles",
    "compute_segment_metrics",
    "fast_segment_metrics",
    "segment_truth_mask",
    "solver_segment_metrics",
    # Working points
    "ABS_THRESHOLD",
    "WP_TARGET_EFF",
    "threshold_for",
    "truth_from_event",
    "metrics_at",
    "quantum_metrics",
    "working_point_threshold",
    "metrics_at_wp",
    "quantum_metrics_wp",
]
