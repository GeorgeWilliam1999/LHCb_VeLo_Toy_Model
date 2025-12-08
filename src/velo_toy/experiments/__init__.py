"""
Experiment infrastructure for running batch jobs on the Nikhef cluster.

This module provides:
- ExperimentConfig: Configuration dataclass for experiments
- run_experiment: Main entry point for running single experiments
- aggregate_results: Combine results from batch jobs
"""

from .config import ExperimentConfig, DetectorConfig, DEFAULT_DETECTOR
from .runner import run_experiment, run_batch
from .aggregator import aggregate_batch, merge_all_batches

__all__ = [
    'ExperimentConfig',
    'DetectorConfig', 
    'DEFAULT_DETECTOR',
    'run_experiment',
    'run_batch',
    'aggregate_batch',
    'merge_all_batches',
]
