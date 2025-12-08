"""
Data loading utilities for analysis.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import gzip

import pandas as pd

try:
    import dill as _pickle
except ImportError:
    import pickle as _pickle


def load_metrics(
    runs_dir: Path,
    merged: bool = True,
) -> pd.DataFrame:
    """
    Load metrics from a runs directory.
    
    Args:
        runs_dir: Path to runs_X directory
        merged: If True, load merged file; otherwise combine batch files
    
    Returns:
        DataFrame with all metrics
    """
    runs_dir = Path(runs_dir)
    
    if merged:
        merged_file = runs_dir / "metrics_merged.csv"
        if merged_file.exists():
            return pd.read_csv(merged_file)
    
    # Load from individual batch files
    batch_files = sorted(runs_dir.glob("batch_*/metrics.csv"))
    
    dfs = []
    for f in batch_files:
        try:
            if f.stat().st_size > 0:
                dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"⚠️ Skipping {f}: {e}")
    
    if not dfs:
        raise FileNotFoundError(f"No metrics files found in {runs_dir}")
    
    return pd.concat(dfs, ignore_index=True)


def load_events(
    runs_dir: Path,
    max_events: Optional[int] = None,
    filter_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load event snapshots from a runs directory.
    
    Args:
        runs_dir: Path to runs_X directory
        max_events: Maximum events to load
        filter_params: Only load events matching these parameters
    
    Returns:
        Dictionary mapping filename to event data
    """
    runs_dir = Path(runs_dir)
    event_files = sorted(runs_dir.glob("batch_*/events_*.pkl.gz"))
    
    if max_events:
        event_files = event_files[:max_events]
    
    events = {}
    for f in event_files:
        try:
            with gzip.open(f, "rb") as fp:
                data = _pickle.load(fp)
            
            # Apply filter if specified
            if filter_params:
                params = data.get('params', {})
                match = all(
                    params.get(k) == v 
                    for k, v in filter_params.items()
                )
                if not match:
                    continue
            
            events[f.name] = data
            
        except Exception as e:
            print(f"⚠️ Error loading {f}: {e}")
    
    return events


def filter_data(
    df: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """
    Filter DataFrame by parameter values.
    
    Args:
        df: Input DataFrame
        **kwargs: Column-value pairs to filter by
    
    Example:
        filtered = filter_data(df, p_ghost_rate=0, p_drop_rate=0)
    """
    result = df
    for col, val in kwargs.items():
        if col in result.columns:
            result = result[result[col] == val]
    return result


def get_parameter_columns(df: pd.DataFrame) -> List[str]:
    """Get all parameter columns (p_* prefix)."""
    return [c for c in df.columns if c.startswith('p_')]


def get_metric_columns(df: pd.DataFrame) -> List[str]:
    """Get all metric columns (m_* prefix)."""
    return [c for c in df.columns if c.startswith('m_')]
