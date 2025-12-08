"""
Plotting utilities for Velo Toy analysis.

Creates publication-quality plots for track reconstruction performance.
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf


# Default style
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 100,
})


def plot_efficiency_vs_scattering(
    df: pd.DataFrame,
    resolution_values: List[float],
    fixed_params: Optional[Dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot track efficiency vs multiple scattering for different resolutions.
    
    Args:
        df: DataFrame with results
        resolution_values: Hit resolution values to plot
        fixed_params: Additional parameters to fix (ghost=0, drop=0, etc.)
        ax: Existing axes to plot on
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    from .statistics import compute_mean_rms
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    fixed_params = fixed_params or {'p_ghost_rate': 0, 'p_drop_rate': 0}
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(resolution_values)))
    
    for i, res in enumerate(resolution_values):
        # Filter data
        subset = df.copy()
        for k, v in fixed_params.items():
            if k in subset.columns:
                subset = subset[subset[k] == v]
        subset = subset[subset['p_hit_res'] == res]
        
        if len(subset) == 0:
            continue
        
        # Compute mean/RMS per scattering value
        stats = compute_mean_rms(
            subset,
            group_by=['p_multi_scatter'],
            metrics=['m_m_reconstruction_efficiency']
        )
        
        ax.errorbar(
            stats['p_multi_scatter'],
            stats['m_m_reconstruction_efficiency_mean'],
            yerr=stats['m_m_reconstruction_efficiency_rms'],
            marker='o',
            capsize=3,
            label=f'σ_res = {res:.0e}',
            color=colors[i],
        )
    
    ax.set_xlabel('Multiple Scattering Parameter')
    ax.set_ylabel('Track Reconstruction Efficiency')
    ax.set_title('Efficiency vs Multiple Scattering')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_ghost_rate_vs_scattering(
    df: pd.DataFrame,
    resolution_values: List[float],
    fixed_params: Optional[Dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot ghost rate vs multiple scattering for different resolutions.
    """
    from .statistics import compute_mean_rms
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    fixed_params = fixed_params or {'p_ghost_rate': 0, 'p_drop_rate': 0}
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(resolution_values)))
    
    for i, res in enumerate(resolution_values):
        subset = df.copy()
        for k, v in fixed_params.items():
            if k in subset.columns:
                subset = subset[subset[k] == v]
        subset = subset[subset['p_hit_res'] == res]
        
        if len(subset) == 0:
            continue
        
        stats = compute_mean_rms(
            subset,
            group_by=['p_multi_scatter'],
            metrics=['m_m_ghost_rate']
        )
        
        ax.errorbar(
            stats['p_multi_scatter'],
            stats['m_m_ghost_rate_mean'],
            yerr=stats['m_m_ghost_rate_rms'],
            marker='s',
            capsize=3,
            label=f'σ_res = {res:.0e}',
            color=colors[i],
        )
    
    ax.set_xlabel('Multiple Scattering Parameter')
    ax.set_ylabel('Ghost Track Rate')
    ax.set_title('Ghost Rate vs Multiple Scattering')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_hit_efficiency_vs_drop(
    df: pd.DataFrame,
    resolution_values: List[float],
    ms_fixed: float = 0.0002,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot hit efficiency vs drop rate (inefficiency).
    """
    from .statistics import compute_mean_rms
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(resolution_values)))
    
    for i, res in enumerate(resolution_values):
        subset = df[
            (df['p_hit_res'] == res) &
            (df['p_multi_scatter'] == ms_fixed) &
            (df['p_ghost_rate'] == 0)
        ]
        
        if len(subset) == 0:
            continue
        
        stats = compute_mean_rms(
            subset,
            group_by=['p_drop_rate'],
            metrics=['m_m_hit_efficiency_mean']
        )
        
        ax.errorbar(
            stats['p_drop_rate'],
            stats['m_m_hit_efficiency_mean_mean'],
            yerr=stats['m_m_hit_efficiency_mean_rms'],
            marker='^',
            capsize=3,
            label=f'σ_res = {res:.0e}',
            color=colors[i],
        )
    
    ax.set_xlabel('Hit Drop Rate (Inefficiency)')
    ax.set_ylabel('Hit Efficiency')
    ax.set_title(f'Hit Efficiency vs Drop Rate (MS = {ms_fixed})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_efficiency_vs_ghost_scatter(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Scatter plot of efficiency vs ghost rate colored by parameters.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    scatter = ax.scatter(
        df['m_m_ghost_rate'],
        df['m_m_reconstruction_efficiency'],
        c=df['p_multi_scatter'],
        cmap='viridis',
        alpha=0.6,
        s=30,
    )
    
    plt.colorbar(scatter, ax=ax, label='Multiple Scattering')
    
    ax.set_xlabel('Ghost Rate')
    ax.set_ylabel('Track Efficiency')
    ax.set_title('Efficiency vs Ghost Rate')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_parameter_scan(
    df: pd.DataFrame,
    x_param: str,
    y_metric: str,
    group_param: Optional[str] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Path] = None,
    x_log: bool = False,
) -> plt.Figure:
    """
    Generic parameter scan plot.
    
    Args:
        df: DataFrame with results
        x_param: X-axis parameter column
        y_metric: Y-axis metric column  
        group_param: Optional parameter to create multiple lines
        fixed_params: Parameters to hold fixed
        ax: Existing axes
        save_path: Path to save figure
        x_log: Use log scale for x-axis
    """
    from .statistics import compute_mean_rms
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Apply fixed params filter
    subset = df.copy()
    if fixed_params:
        for k, v in fixed_params.items():
            if k in subset.columns:
                subset = subset[subset[k] == v]
    
    if group_param and group_param in subset.columns:
        group_values = sorted(subset[group_param].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(group_values)))
        
        for i, gval in enumerate(group_values):
            gsub = subset[subset[group_param] == gval]
            stats = compute_mean_rms(gsub, group_by=[x_param], metrics=[y_metric])
            
            ax.errorbar(
                stats[x_param],
                stats[f'{y_metric}_mean'],
                yerr=stats[f'{y_metric}_rms'],
                marker='o',
                capsize=3,
                label=f'{group_param}={gval}',
                color=colors[i],
            )
    else:
        stats = compute_mean_rms(subset, group_by=[x_param], metrics=[y_metric])
        ax.errorbar(
            stats[x_param],
            stats[f'{y_metric}_mean'],
            yerr=stats[f'{y_metric}_rms'],
            marker='o',
            capsize=3,
        )
    
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_metric)
    if x_log:
        ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_cost_function(
    epsilon: float,
    sigma_values: List[float],
    r_range: Tuple[float, float] = (-0.001, 0.001),
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot the ERF cost function for different sigma values.
    
    Args:
        epsilon: Threshold angle
        sigma_values: List of sigma values to plot
        r_range: Range of angles to plot
        ax: Existing axes
        save_path: Path to save figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    r = np.linspace(r_range[0], r_range[1], 1000)
    
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(sigma_values)))
    
    for i, sigma in enumerate(sigma_values):
        cost = 0.5 * (1 + erf((epsilon - np.abs(r)) / (sigma * np.sqrt(2))))
        ax.plot(r, cost, label=f'σ = {sigma:.0e}', color=colors[i])
    
    ax.axvline(x=epsilon, color='black', linestyle='--', alpha=0.7, label=f'ε = {epsilon:.2e}')
    ax.axvline(x=-epsilon, color='black', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Angle (radians)')
    ax.set_ylabel('Cost')
    ax.set_title(f'Cost Function: C(r) = ½(1 + erf((ε-|r|)/(σ√2)))')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_summary_figure(
    df: pd.DataFrame,
    resolution_values: List[float],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create a 2x2 summary figure with key performance plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    plot_efficiency_vs_scattering(df, resolution_values, ax=axes[0, 0])
    plot_ghost_rate_vs_scattering(df, resolution_values, ax=axes[0, 1])
    plot_hit_efficiency_vs_drop(df, resolution_values, ax=axes[1, 0])
    plot_efficiency_vs_ghost_scatter(df, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
