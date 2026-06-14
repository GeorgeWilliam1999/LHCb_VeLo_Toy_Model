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
]
