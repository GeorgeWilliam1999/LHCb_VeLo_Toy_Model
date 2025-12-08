"""
Analysis utilities for Velo Toy experiments.

Provides functions for:
- Loading and processing experiment results
- Computing statistics with error bars
- Generating publication-quality plots
"""

from .loader import (
    load_metrics,
    load_events,
    filter_data,
)
from .statistics import (
    compute_mean_rms,
    compute_efficiency_vs_parameter,
)
from .plotting import (
    plot_efficiency_vs_scattering,
    plot_ghost_rate_vs_scattering,
    plot_hit_efficiency_vs_drop,
    plot_efficiency_vs_ghost_scatter,
    plot_parameter_scan,
    plot_cost_function,
)

__all__ = [
    # Loader
    'load_metrics', 'load_events', 'filter_data',
    # Statistics
    'compute_mean_rms', 'compute_efficiency_vs_parameter',
    # Plotting
    'plot_efficiency_vs_scattering',
    'plot_ghost_rate_vs_scattering', 
    'plot_hit_efficiency_vs_drop',
    'plot_efficiency_vs_ghost_scatter',
    'plot_parameter_scan',
    'plot_cost_function',
]
