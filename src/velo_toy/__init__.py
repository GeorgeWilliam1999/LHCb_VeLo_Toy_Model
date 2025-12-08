"""
Velo Toy Model - Quantum-inspired Track Reconstruction

A framework for simulating and analyzing Hamiltonian-based track reconstruction
for the LHCb VELO detector, designed for batch experiments on the Nikhef cluster.

References:
    - Nicotra 2023, J. Inst. 18 P11028
    - arXiv:2511.11458v1
"""

__version__ = "1.0.0"
__author__ = "George Scriven"

from .core import *
from .experiments import run_experiment, ExperimentConfig
