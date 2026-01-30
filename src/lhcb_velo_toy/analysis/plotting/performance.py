"""
Performance plotting utilities.

Functions for creating LHCb-style performance plots for track reconstruction.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from pathlib import Path


def plot_efficiency_vs_parameter(
    parameter_values: Sequence[float],
    efficiencies: Sequence[float],
    parameter_name: str = "Parameter",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Reconstruction Efficiency",
    figsize: tuple[float, float] = (8, 6),
    color: str = "blue",
    marker: str = "o",
    add_error_bars: bool = False,
    errors: Optional[Sequence[float]] = None,
) -> "Figure":
    """
    Plot reconstruction efficiency vs. a parameter.
    
    Parameters
    ----------
    parameter_values : Sequence[float]
        X-axis values (e.g., multiple scattering strength, drop rate).
    efficiencies : Sequence[float]
        Y-axis efficiency values (0 to 1).
    parameter_name : str, default "Parameter"
        Name of the parameter for axis label.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label. If None, uses parameter_name.
    ylabel : str, default "Reconstruction Efficiency"
        Y-axis label.
    figsize : tuple[float, float], default (8, 6)
        Figure size in inches.
    color : str, default "blue"
        Line/marker color.
    marker : str, default "o"
        Marker style.
    add_error_bars : bool, default False
        Whether to add error bars.
    errors : Sequence[float], optional
        Error bar values (required if add_error_bars=True).
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    
    Examples
    --------
    >>> fig = plot_efficiency_vs_parameter(
    ...     drop_rates,
    ...     efficiencies,
    ...     parameter_name="Drop Rate"
    ... )
    >>> plt.savefig("efficiency.pdf")
    """
    raise NotImplementedError


def plot_ghost_rate_vs_parameter(
    parameter_values: Sequence[float],
    ghost_rates: Sequence[float],
    parameter_name: str = "Parameter",
    title: Optional[str] = None,
    figsize: tuple[float, float] = (8, 6),
    color: str = "red",
) -> "Figure":
    """
    Plot ghost rate vs. a parameter.
    
    Parameters
    ----------
    parameter_values : Sequence[float]
        X-axis values.
    ghost_rates : Sequence[float]
        Y-axis ghost rate values (0 to 1).
    parameter_name : str, default "Parameter"
        Name of the parameter.
    title : str, optional
        Plot title.
    figsize : tuple[float, float], default (8, 6)
        Figure size.
    color : str, default "red"
        Line/marker color.
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    raise NotImplementedError


def plot_purity_distribution(
    purities: Sequence[float],
    title: Optional[str] = None,
    bins: int = 20,
    figsize: tuple[float, float] = (8, 6),
) -> "Figure":
    """
    Plot histogram of track purities.
    
    Parameters
    ----------
    purities : Sequence[float]
        List of purity values for all accepted tracks.
    title : str, optional
        Plot title.
    bins : int, default 20
        Number of histogram bins.
    figsize : tuple[float, float], default (8, 6)
        Figure size.
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    raise NotImplementedError


def plot_comparison(
    parameter_values: Sequence[float],
    classical_metric: Sequence[float],
    quantum_metric: Sequence[float],
    metric_name: str = "Efficiency",
    parameter_name: str = "Parameter",
    title: Optional[str] = None,
    figsize: tuple[float, float] = (10, 6),
) -> "Figure":
    """
    Plot comparison between classical and quantum methods.
    
    Parameters
    ----------
    parameter_values : Sequence[float]
        X-axis values.
    classical_metric : Sequence[float]
        Metric values for classical method.
    quantum_metric : Sequence[float]
        Metric values for quantum method.
    metric_name : str, default "Efficiency"
        Name of the metric being compared.
    parameter_name : str, default "Parameter"
        Name of the parameter.
    title : str, optional
        Plot title.
    figsize : tuple[float, float], default (10, 6)
        Figure size.
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure with both curves.
    """
    raise NotImplementedError


def generate_performance_report(
    results_df: "pd.DataFrame",
    output_dir: "str | Path",
    prefix: str = "performance",
) -> dict[str, str]:
    """
    Generate a complete set of performance plots.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame containing performance results with columns:
        - parameter (the varied parameter)
        - efficiency
        - ghost_rate
        - clone_fraction
        - mean_purity
    output_dir : str or Path
        Directory to save plots.
    prefix : str, default "performance"
        Prefix for output filenames.
    
    Returns
    -------
    dict[str, str]
        Dictionary mapping plot names to file paths.
    
    Examples
    --------
    >>> paths = generate_performance_report(df, "plots/")
    >>> print(paths['efficiency'])
    'plots/performance_efficiency.pdf'
    """
    raise NotImplementedError


def set_lhcb_style() -> None:
    """
    Apply LHCb publication-style settings to matplotlib.
    
    Sets font sizes, line widths, and other parameters to match
    LHCb style guidelines for publications.
    """
    raise NotImplementedError
