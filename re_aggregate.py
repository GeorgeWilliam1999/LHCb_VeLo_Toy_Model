#!/usr/bin/env python
"""
re_aggregate.py - Re-compute metrics with fixed validator for existing runs

Usage:
    python re_aggregate.py runs_8
    python re_aggregate.py runs_9
"""

import sys
import gzip
import pickle
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Import the fixed validator
from LHCB_Velo_Toy_Models.toy_validator import EventValidator


def recompute_metrics_for_run(runs_dir: str):
    """Re-compute metrics for all batches in a run directory."""
    runs_path = Path(runs_dir)
    
    if not runs_path.exists():
        print(f"Error: {runs_path} does not exist")
        return
    
    # Find all batch directories
    batch_dirs = sorted(runs_path.glob("batch_*"))
    print(f"Found {len(batch_dirs)} batch directories in {runs_dir}")
    
    all_metrics = []
    
    for batch_dir in tqdm(batch_dirs, desc="Processing batches"):
        event_store_path = batch_dir / "event_store.pkl.gz"
        
        if not event_store_path.exists():
            print(f"  Skipping {batch_dir.name} - no event_store.pkl.gz")
            continue
        
        try:
            with gzip.open(event_store_path, 'rb') as f:
                store = pickle.load(f)
        except Exception as e:
            print(f"  Error loading {event_store_path}: {e}")
            continue
        
        batch_metrics = []
        
        for fname, entry in store.items():
            params = entry.get('params', {})
            noisy = entry.get('noisy', None)
            reco = entry.get('reco', None)
            
            if noisy is None or reco is None:
                continue
            
            try:
                # Re-compute metrics with fixed validator (purity-based)
                validator = EventValidator(noisy, reco)
                metrics = validator.compute_metrics()  # Uses purity_min=0.7 by default now
            except Exception as e:
                print(f"  Validator error on {fname}: {e}")
                metrics = {}
            
            # Build flat row
            flat = {"file": fname, "batch": batch_dir.name}
            flat.update({f"p_{k}": v for k, v in params.items()})
            if isinstance(metrics, dict):
                flat.update({f"m_{k}": v for k, v in metrics.items()})
            
            batch_metrics.append(flat)
        
        all_metrics.extend(batch_metrics)
        
        # Also save per-batch metrics
        if batch_metrics:
            batch_df = pd.DataFrame(batch_metrics)
            batch_df.to_csv(batch_dir / "metrics_fixed.csv", index=False)
    
    # Save merged metrics
    if all_metrics:
        merged_df = pd.DataFrame(all_metrics)
        output_path = runs_path / "metrics_fixed.csv"
        merged_df.to_csv(output_path, index=False)
        print(f"\nSaved {len(merged_df)} rows to {output_path}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Total events: {len(merged_df)}")
        if 'm_track_efficiency_good_over_true' in merged_df.columns:
            print(f"  Mean efficiency: {100*merged_df['m_track_efficiency_good_over_true'].mean():.1f}%")
        if 'm_track_ghost_rate_over_rec' in merged_df.columns:
            print(f"  Mean ghost rate: {100*merged_df['m_track_ghost_rate_over_rec'].mean():.1f}%")
    else:
        print("No metrics computed!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python re_aggregate.py <runs_dir>")
        print("Example: python re_aggregate.py runs_8")
        sys.exit(1)
    
    runs_dir = sys.argv[1]
    recompute_metrics_for_run(runs_dir)
