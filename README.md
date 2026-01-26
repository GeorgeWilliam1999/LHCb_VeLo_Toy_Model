# Quantum Track Reconstruction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A quantum-inspired and quantum computing framework for charged particle track reconstruction in High Energy Physics detectors, with a focus on the LHCb VELO (Vertex Locator) detector.

## Overview

This package implements **Hamiltonian-based track reconstruction** algorithms that formulate the combinatorial track-finding problem as an optimization problem solvable by both classical and quantum methods. The approach is based on:

- **Classical**: Iterative solvers for the Hamiltonian ground state
- **Quantum**: HHL (Harrow-Hassidim-Lloyd) algorithm implementation using Qiskit

### Key Features

- 🔬 **Toy detector simulation** with configurable geometry and physics parameters
- 🧮 **Hamiltonian formulation** with ERF-smoothed cost functions
- ⚛️ **Quantum HHL algorithm** implementation for linear system solving
- 📊 **Comprehensive validation** metrics (efficiency, ghost rate, clone fraction, purity)
- 🚀 **Scalable experiment framework** with HTCondor batch processing support
- 📈 **Publication-quality plotting** and analysis tools

## References

- Nicotra et al., *"Track finding and fitting with a quantum-inspired algorithm for the LHCb VELO"*, J. Inst. **18** P11028 (2023)
- arXiv:2511.11458v1

## Physics Background

The track reconstruction problem is formulated as minimizing a Hamiltonian:

$$H = -\frac{1}{2}\sum_{i,j} A_{ij} z_i z_j + \sum_i b_i z_i$$

where:
- $z_i \in \{0, 1\}$ indicates whether segment $i$ belongs to a reconstructed track
- $A_{ij}$ encodes geometric compatibility between segments
- The cost function uses ERF-smoothed angular thresholds:

$$C(r) = \frac{1}{2}\left(1 + \text{erf}\left(\frac{\varepsilon - |r|}{\sigma\sqrt{2}}\right)\right)$$

## Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Quick Install

```bash
# Clone the repository
git clone https://github.com/GeorgeWilliam1999/Quantum_Track_Reconstruction.git
cd Quantum_Track_Reconstruction

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

### Dependencies

Core dependencies (automatically installed):
- `numpy>=1.20` - Numerical computing
- `scipy>=1.7` - Scientific computing and sparse matrices
- `pandas>=1.3` - Data analysis
- `matplotlib>=3.4` - Visualization
- `dill>=0.3` - Serialization
- `tqdm>=4.60` - Progress bars

For quantum algorithms:
```bash
pip install qiskit qiskit-aer
```

For Jupyter notebooks:
```bash
pip install ipykernel jupyterlab
```

## Quick Start

### 1. Generate and Reconstruct a Simple Event

```python
from LHCB_Velo_Toy_Models.state_event_generator import StateEventGenerator
from LHCB_Velo_Toy_Models.simple_hamiltonian import SimpleHamiltonian, get_tracks

# Create a toy detector event
generator = StateEventGenerator(
    n_modules=26,           # Number of detector planes
    n_tracks=5,             # Number of particle tracks
    hit_resolution=0.0001,  # Position resolution (m)
    multi_scatter=0.0002,   # Multiple scattering angle (rad)
)
event = generator.generate()

# Build the Hamiltonian
hamiltonian = SimpleHamiltonian(
    epsilon=0.001,  # Angular threshold
    gamma=1.0,      # Track length penalty
    delta=1.0,      # Segment activation reward
)
A, b = hamiltonian.construct_hamiltonian(event, convolution=True)

# Solve classically
solution = hamiltonian.solve_classically()

# Extract reconstructed tracks
reconstructed_tracks = get_tracks(hamiltonian, solution, event)
print(f"Found {len(reconstructed_tracks)} tracks")
```

### 2. Run a Parameter Scan Experiment

```python
from velo_toy.experiments import ExperimentConfig, run_experiment

# Configure experiment
config = ExperimentConfig(
    n_modules=26,
    n_tracks=5,
    hit_resolution=0.0001,
    multi_scatter=0.0002,
    ghost_rate=0.0,
    drop_rate=0.0,
)

# Run and get metrics
result = run_experiment(config)
print(f"Reconstruction Efficiency: {result['m_m_reconstruction_efficiency']:.1%}")
print(f"Ghost Rate: {result['m_m_ghost_rate']:.1%}")
print(f"Track Purity: {result['m_m_purity_all_matched']:.1%}")
```

### 3. Use the Quantum HHL Algorithm

```python
from hhl_algorithm import HHLAlgorithm
import numpy as np

# Example: Solve a small linear system Ax = b
A = np.array([[1, -1/3], [-1/3, 1]])
b = np.array([1, 0])

# Initialize and run HHL
hhl = HHLAlgorithm(A, b, num_time_qubits=5, shots=10240)
hhl.build_circuit()
result = hhl.execute()

print(f"Quantum solution: {result['solution']}")
print(f"Classical solution: {np.linalg.solve(A, b)}")
```

## Project Structure

```
Quantum_Track_Reconstruction/
├── src/velo_toy/              # Main Python package
│   ├── core/                  # Core physics models
│   │   ├── state_event_model.py      # Hit, Track, Segment dataclasses
│   │   ├── state_event_generator.py  # Event generation
│   │   ├── simple_hamiltonian.py     # Hamiltonian construction
│   │   └── toy_validator.py          # Validation metrics
│   ├── experiments/           # Experiment infrastructure
│   │   ├── config.py          # Configuration dataclasses
│   │   ├── runner.py          # Experiment execution
│   │   └── aggregator.py      # Result aggregation
│   └── analysis/              # Analysis tools
│       ├── loader.py          # Data loading utilities
│       ├── statistics.py      # Statistical analysis
│       └── plotting.py        # Visualization
├── LHCB_Velo_Toy_Models/      # Core algorithm implementations
├── scripts/condor/            # HTCondor job submission
├── helpful/                   # Utility data and scripts
├── hhl_algorithm.py           # Quantum HHL implementation
├── hhl_algorithm_1bit.py      # Simplified 1-bit HHL
└── *.ipynb                    # Analysis notebooks
```

## Configuration Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `n_modules` | Number of detector planes | 26 (LHCb VELO) |
| `n_tracks` | Tracks per event | 1-100 |
| `hit_resolution` | Position measurement error (m) | 10-200 µm |
| `multi_scatter` | Multiple scattering angle (rad) | 0.1-2 mrad |
| `ghost_rate` | Fraction of noise hits | 0-30% |
| `drop_rate` | Hit inefficiency | 0-30% |
| `epsilon` | Angular compatibility threshold | ~1 mrad |
| `scale` | Hamiltonian parameter scaling | 1-5 |

## Performance Metrics

The framework computes standard HEP tracking metrics:

- **Reconstruction Efficiency**: Fraction of true tracks successfully reconstructed
- **Ghost Rate**: Fraction of reconstructed tracks that are fake
- **Clone Fraction**: Duplicate track rate
- **Purity**: Fraction of hits on a reconstructed track from the same true track
- **Hit Efficiency**: Fraction of true hits included in reconstruction

## Batch Processing (HTCondor)

For large-scale parameter studies, use the HTCondor submission system:

```bash
# Generate parameter configurations
python gen_params.py

# Submit batch jobs
condor_submit scripts/condor/velo_experiment.sub

# Monitor progress
condor_q

# Aggregate results after completion
python re_aggregate.py
```

## Analysis Notebooks

| Notebook | Description |
|----------|-------------|
| `track_density_study.ipynb` | Main analysis notebook |
| `track_density_study_runs*.ipynb` | Individual run analyses |
| `experiment_analysis_organized.ipynb` | Comprehensive analysis |
| `George_Sandbox.ipynb` | Development and testing |

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**George William Scriven**

## Acknowledgments

- LHCb Collaboration for the VELO detector design inspiration
- Nikhef computing cluster for batch processing capabilities
- Qiskit team for the quantum computing framework
