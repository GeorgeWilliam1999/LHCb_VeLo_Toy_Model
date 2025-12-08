"""
Statistical analysis functions.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd


def compute_mean_rms(
    df: pd.DataFrame,
    group_by: List[str],
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute mean and RMS (standard deviation) for metrics grouped by parameters.
    
    Args:
        df: DataFrame with experiment results
        group_by: Parameter columns to group by
        metrics: Metric columns to compute (defaults to all m_* columns)
    
    Returns:
        DataFrame with _mean and _rms columns for each metric
    """
    if metrics is None:
        metrics = [c for c in df.columns if c.startswith('m_')]
    
    # Build aggregation dict
    agg_funcs = {}
    for m in metrics:
        if m in df.columns:
            agg_funcs[f'{m}_mean'] = (m, 'mean')
            agg_funcs[f'{m}_rms'] = (m, 'std')
            agg_funcs[f'{m}_count'] = (m, 'count')
    
    if not agg_funcs:
        raise ValueError("No valid metrics found")
    
    return df.groupby(group_by, as_index=False).agg(**agg_funcs)


def compute_efficiency_vs_parameter(
    df: pd.DataFrame,
    x_param: str,
    fixed_params: Dict[str, Any],
    efficiency_col: str = 'm_m_reconstruction_efficiency',
    n_repeats: int = 10,
) -> pd.DataFrame:
    """
    Compute efficiency vs a parameter with RMS error bars.
    
    Args:
        df: DataFrame with results
        x_param: X-axis parameter (e.g., 'p_multi_scatter')
        fixed_params: Dict of parameters to hold fixed
        efficiency_col: Which efficiency column to use
        n_repeats: Expected number of repeats per point
    
    Returns:
        DataFrame with columns: x_param, mean, rms, count
    """
    # Filter to fixed parameters
    filtered = df.copy()
    for col, val in fixed_params.items():
        if col in filtered.columns:
            filtered = filtered[filtered[col] == val]
    
    # Group by x parameter and compute stats
    grouped = filtered.groupby(x_param).agg(
        mean=(efficiency_col, 'mean'),
        rms=(efficiency_col, 'std'),
        count=(efficiency_col, 'count'),
    ).reset_index()
    
    return grouped


def compute_correlation_matrix(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute correlation matrix between metrics.
    
    Args:
        df: DataFrame with results
        metrics: Which metrics to include (defaults to all m_* columns)
    
    Returns:
        Correlation matrix DataFrame
    """
    if metrics is None:
        metrics = [c for c in df.columns if c.startswith('m_')]
    
    return df[metrics].corr()


def find_optimal_parameters(
    df: pd.DataFrame,
    efficiency_col: str = 'm_m_reconstruction_efficiency',
    ghost_col: str = 'm_m_ghost_rate',
    efficiency_threshold: float = 0.9,
    ghost_threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Find parameter combinations meeting efficiency and ghost rate criteria.
    
    Args:
        df: DataFrame with results
        efficiency_col: Efficiency column name
        ghost_col: Ghost rate column name
        efficiency_threshold: Minimum required efficiency
        ghost_threshold: Maximum allowed ghost rate
    
    Returns:
        Filtered DataFrame with optimal configurations
    """
    mask = (
        (df[efficiency_col] >= efficiency_threshold) &
        (df[ghost_col] <= ghost_threshold)
    )
    return df[mask].copy()
