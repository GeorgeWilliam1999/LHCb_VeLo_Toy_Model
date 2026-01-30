"""Plotting utilities for the LHCb VELO Toy Model."""

from lhcb_velo_toy.analysis.plotting.event_display import (
    plot_event_3d,
    plot_segments_3d,
)
from lhcb_velo_toy.analysis.plotting.performance import (
    plot_efficiency_vs_parameter,
    plot_ghost_rate_vs_parameter,
    plot_purity_distribution,
    generate_performance_report,
)

__all__ = [
    # Event display
    "plot_event_3d",
    "plot_segments_3d",
    # Performance plots
    "plot_efficiency_vs_parameter",
    "plot_ghost_rate_vs_parameter",
    "plot_purity_distribution",
    "generate_performance_report",
]
