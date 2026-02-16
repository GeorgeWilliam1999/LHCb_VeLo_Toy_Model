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
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    plot_kwargs: dict = dict(color=color, marker=marker, linewidth=1.5, markersize=6)
    if add_error_bars and errors is not None:
        ax.errorbar(parameter_values, efficiencies, yerr=errors, capsize=3, **plot_kwargs)
    else:
        ax.plot(parameter_values, efficiencies, **plot_kwargs)
    ax.set_xlabel(xlabel or parameter_name)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{ylabel} vs {parameter_name}")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


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
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(parameter_values, ghost_rates, color=color, marker="s", linewidth=1.5, markersize=6)
    ax.set_xlabel(parameter_name)
    ax.set_ylabel("Ghost Rate")
    ax.set_title(title or f"Ghost Rate vs {parameter_name}")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


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
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(purities, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel("Purity")
    ax.set_ylabel("Count")
    ax.set_title(title or "Track Purity Distribution")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=np.mean(purities), color="red", linestyle="--", label=f"Mean = {np.mean(purities):.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


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
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(parameter_values, classical_metric, "o-", color="blue", label="Classical", linewidth=1.5)
    ax.plot(parameter_values, quantum_metric, "s--", color="red", label="Quantum", linewidth=1.5)
    ax.set_xlabel(parameter_name)
    ax.set_ylabel(metric_name)
    ax.set_title(title or f"{metric_name}: Classical vs Quantum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


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
    import matplotlib.pyplot as plt
    from pathlib import Path as _Path

    out = _Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    param_col = results_df.columns[0]  # first column as parameter
    params = results_df[param_col].values

    for metric_col, label in [
        ("efficiency", "Reconstruction Efficiency"),
        ("ghost_rate", "Ghost Rate"),
        ("clone_fraction", "Clone Fraction"),
        ("mean_purity", "Mean Purity"),
    ]:
        if metric_col not in results_df.columns:
            continue
        fig = plot_efficiency_vs_parameter(
            params,
            results_df[metric_col].values,
            parameter_name=param_col,
            ylabel=label,
        )
        fpath = out / f"{prefix}_{metric_col}.pdf"
        fig.savefig(str(fpath), bbox_inches="tight")
        plt.close(fig)
        paths[metric_col] = str(fpath)

    return paths


def set_lhcb_style() -> None:
    """
    Apply LHCb publication-style settings to matplotlib.
    
    Sets font sizes, line widths, and other parameters to match
    LHCb style guidelines for publications.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "figure.figsize": (8, 6),
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
    })
