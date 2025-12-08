"""
Experiment runner - core execution logic for track reconstruction experiments.

This module provides the main entry point for running experiments, either
locally or as part of a Condor batch job.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Dict, Any, Optional, List
import csv

import numpy as np

from .config import ExperimentConfig, DEFAULT_DETECTOR

# Prefer dill for broader object support; fall back to pickle
try:
    import dill as _pickle
except ImportError:
    import pickle as _pickle


def run_experiment(
    config: ExperimentConfig,
    output_dir: Optional[Path] = None,
    save_events: bool = True,
    compute_metrics: bool = True,
) -> Dict[str, Any]:
    """
    Run a single track reconstruction experiment.
    
    Args:
        config: Experiment configuration
        output_dir: Directory to save results (optional)
        save_events: Whether to save full event snapshots
        compute_metrics: Whether to compute validation metrics
    
    Returns:
        Dictionary containing:
            - 'config': The experiment configuration
            - 'truth_event': Ground truth event
            - 'noisy_event': Event with noise applied
            - 'reco_event': Reconstructed event
            - 'classical_solution': Raw Hamiltonian solution
            - 'metrics': Validation metrics (if computed)
            - 'filename': Output filename (if saved)
    """
    # Import here to avoid circular imports
    from ..core import (
        StateEventGenerator, SimpleHamiltonian,
        get_tracks, construct_event, EventValidator
    )
    from ..core.state_event_model import PlaneGeometry
    
    # Set random seed for reproducibility
    np.random.seed(config.seed)
    
    # Build detector geometry
    detector = PlaneGeometry(
        module_id=config.detector.module_ids,
        lx=config.detector.lx_list,
        ly=config.detector.ly_list,
        z=config.detector.zs_mm,
    )
    
    # Create event generator
    generator = StateEventGenerator(
        detector,
        phi_max=config.phi_max,
        phi_min=config.phi_max,  # Same for now
        theta_max=config.theta_max,
        events=config.n_events,
        n_particles=config.n_particles_per_event,
        measurement_error=config.meas_error,
        collision_noise=config.coll_noise,
    )
    
    # Generate primary vertices
    generator.generate_random_primary_vertices({"x": 1, "y": 1, "z": 1})
    
    # Generate particles
    event_particles = [
        [{"type": "MIP", "mass": 0.511, "q": 1} 
         for _ in range(config.total_particles)]
        for _ in range(config.n_events)
    ]
    generator.generate_particles(event_particles)
    
    # Generate truth tracks
    truth_event = generator.generate_complete_events()
    
    # Apply noise (ghost hits, dropped hits)
    noisy_event = generator.make_noisy_event(
        drop_rate=config.drop_rate,
        ghost_rate=config.ghost_rate
    )
    
    # Build and solve Hamiltonian
    hamiltonian = SimpleHamiltonian(
        epsilon=config.epsilon,
        gamma=config.gamma,
        delta=config.delta,
        theta_d=config.erf_sigma,
    )
    hamiltonian.construct_hamiltonian(
        event=truth_event,
        convolution=config.thresh_flag
    )
    
    classical_solution = hamiltonian.solve_classicaly()
    discretized_solution = (classical_solution > 0.45).astype(int)
    
    # Reconstruct tracks
    rec_tracks = get_tracks(hamiltonian, discretized_solution, noisy_event)
    
    reco_event = construct_event(
        truth_event.detector_geometry,
        rec_tracks,
        [t.hits for t in rec_tracks],
        [t.segments for t in rec_tracks],
        truth_event.detector_geometry.module_id
    )
    
    # Build result dictionary
    result = {
        'config': config,
        'params': config.to_dict(),
        'truth_event': truth_event,
        'noisy_event': noisy_event,
        'reco_event': reco_event,
        'reco_tracks': rec_tracks,
        'classical_solution': classical_solution,
        'discretized_solution': discretized_solution,
        'hamiltonian': hamiltonian,
    }
    
    # Compute metrics if requested
    if compute_metrics:
        validator = EventValidator(truth_event, reco_event)
        result['metrics'] = validator.compute_metrics()
    
    # Save if output directory specified
    if output_dir is not None and save_events:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"events_{config.get_filename_tag()}.pkl.gz"
        filepath = output_dir / filename
        
        _dump_pickle(result, filepath)
        result['filename'] = filename
        
        # Append to index CSV
        _append_index_row(
            output_dir / "events_index.csv",
            {
                'file': filename,
                **{f'p_{k}': v for k, v in config.to_dict().items() 
                   if not isinstance(v, (list, dict))}
            }
        )
    
    return result


def run_batch(
    configs: List[ExperimentConfig],
    output_dir: Path,
    batch_id: int = 0,
) -> List[Dict[str, Any]]:
    """
    Run a batch of experiments (typically called from Condor job).
    
    Args:
        configs: List of experiment configurations
        output_dir: Base output directory
        batch_id: Batch identifier
    
    Returns:
        List of result dictionaries
    """
    batch_dir = output_dir / f"batch_{batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for config in configs:
        result = run_experiment(
            config,
            output_dir=batch_dir,
            save_events=True,
            compute_metrics=True
        )
        results.append(result)
    
    return results


def _dump_pickle(obj: Any, path: Path) -> None:
    """Save object to gzipped pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        _pickle.dump(obj, f, protocol=_pickle.HIGHEST_PROTOCOL)


def _append_index_row(index_csv: Path, row_dict: dict) -> None:
    """Append a row to the index CSV file."""
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not index_csv.exists()
    with open(index_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row_dict.keys())
        if new_file:
            w.writeheader()
        w.writerow(row_dict)


def load_experiment(path: Path) -> Dict[str, Any]:
    """Load a saved experiment result."""
    with gzip.open(path, "rb") as f:
        return _pickle.load(f)
