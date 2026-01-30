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

__all__ = [
    # Validation
    "Match",
    "EventValidator",
]
