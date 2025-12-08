"""
Result aggregation for batch experiments.

This module provides utilities for combining results from multiple
Condor batch jobs into unified metrics files.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Dict, List, Any, Optional
import csv
from itertools import chain

import pandas as pd
import numpy as np

# Prefer dill for broader object support
try:
    import dill as _pickle
except ImportError:
    import pickle as _pickle


def aggregate_batch(
    batch_dir: Path,
    output_dir: Optional[Path] = None,
    recompute_metrics: bool = False,
) -> pd.DataFrame:
    """
    Aggregate results from a single batch directory.
    
    Args:
        batch_dir: Path to batch directory containing event files
        output_dir: Where to save aggregated results (defaults to batch_dir)
        recompute_metrics: Whether to recompute metrics from saved events
    
    Returns:
        DataFrame with all metrics for this batch
    """
    from ..core import EventValidator
    from ..core import state_event_model
    
    batch_dir = Path(batch_dir)
    output_dir = output_dir or batch_dir
    
    event_files = sorted(batch_dir.glob("**/events_*.pkl.gz"))
    
    if not event_files:
        print(f"⚠️ No event files found in {batch_dir}")
        return pd.DataFrame()
    
    rows = []
    for event_file in event_files:
        try:
            with gzip.open(event_file, "rb") as f:
                data = _pickle.load(f)
            
            # Extract parameters
            params = data.get('params', {})
            row = {'file': event_file.name}
            
            # Add parameters with p_ prefix
            for k, v in params.items():
                if not isinstance(v, (list, dict)):
                    row[f'p_{k}'] = v
            
            # Compute or extract metrics
            if recompute_metrics or 'metrics' not in data:
                truth = data['truth_event']
                reco = data['reco_event']
                
                # Reconstruct reco_event if needed
                if hasattr(reco, 'tracks') and hasattr(reco.tracks, '__iter__'):
                    reco_event = state_event_model.Event(
                        detector_geometry=truth.detector_geometry,
                        tracks=reco.tracks if hasattr(reco, 'tracks') else reco,
                        hits=list(chain.from_iterable(
                            getattr(reco, 'hits', []) or []
                        )),
                        segments=list(chain.from_iterable(
                            getattr(reco, 'segments', []) or []
                        )),
                        modules=truth.detector_geometry.module_id
                    )
                else:
                    reco_event = reco
                
                validator = EventValidator(truth, reco_event)
                metrics = validator.compute_metrics()
            else:
                metrics = data['metrics']
            
            # Add metrics with m_ prefix
            for k, v in metrics.items():
                if not isinstance(v, (list, dict)):
                    row[f'm_{k}'] = v
            
            rows.append(row)
            
        except Exception as e:
            print(f"⚠️ Error processing {event_file}: {e}")
            continue
    
    df = pd.DataFrame(rows)
    
    # Save to output directory
    if len(df) > 0:
        output_path = output_dir / "metrics.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Saved {len(df)} rows to {output_path}")
    
    return df


def merge_all_batches(
    runs_dir: Path,
    output_file: Optional[str] = "metrics_merged.csv",
) -> pd.DataFrame:
    """
    Merge metrics from all batch directories in a run.
    
    Args:
        runs_dir: Root directory containing batch_* subdirectories
        output_file: Name of merged output file
    
    Returns:
        Merged DataFrame with all metrics
    """
    runs_dir = Path(runs_dir)
    
    # Find all metrics.csv files
    metrics_files = sorted(runs_dir.glob("batch_*/metrics.csv"))
    
    if not metrics_files:
        print(f"⚠️ No metrics.csv files found in {runs_dir}")
        return pd.DataFrame()
    
    dfs = []
    for f in metrics_files:
        try:
            if f.stat().st_size > 0:
                dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"⚠️ Skipping {f}: {e}")
    
    if not dfs:
        print("⚠️ No valid metrics files found")
        return pd.DataFrame()
    
    merged = pd.concat(dfs, ignore_index=True)
    
    if output_file:
        output_path = runs_dir / output_file
        merged.to_csv(output_path, index=False)
        print(f"✅ Merged {len(dfs)} files ({len(merged)} rows) -> {output_path}")
    
    return merged


def load_all_events(
    runs_dir: Path,
    max_events: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Load all event snapshots from a run directory.
    
    Args:
        runs_dir: Root directory containing batch_* subdirectories
        max_events: Maximum number of events to load (None = all)
    
    Returns:
        Dictionary mapping filename to event data
    """
    runs_dir = Path(runs_dir)
    event_files = sorted(runs_dir.glob("batch_*/**/events_*.pkl.gz"))
    
    if max_events:
        event_files = event_files[:max_events]
    
    events = {}
    for f in event_files:
        try:
            with gzip.open(f, "rb") as fp:
                events[f.name] = _pickle.load(fp)
        except Exception as e:
            print(f"⚠️ Error loading {f}: {e}")
    
    print(f"✅ Loaded {len(events)} events")
    return events


def compute_statistics(
    df: pd.DataFrame,
    group_by: List[str],
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute mean and RMS statistics grouped by parameters.
    
    Args:
        df: DataFrame with metrics
        group_by: Columns to group by (e.g., ['p_meas_error', 'p_coll_noise'])
        metrics: Metric columns to aggregate (defaults to all m_* columns)
    
    Returns:
        DataFrame with mean and std for each metric
    """
    if metrics is None:
        metrics = [c for c in df.columns if c.startswith('m_')]
    
    # Compute aggregations
    agg_dict = {}
    for m in metrics:
        agg_dict[f'{m}_mean'] = (m, 'mean')
        agg_dict[f'{m}_std'] = (m, 'std')
        agg_dict[f'{m}_count'] = (m, 'count')
    
    stats = df.groupby(group_by, as_index=False).agg(**agg_dict)
    
    return stats
