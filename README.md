# LHCb VELO Toy Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for simulating and analyzing particle tracking in the LHCb VELO (Vertex Locator) detector. This package provides tools for event generation, Hamiltonian-based track reconstruction, quantum algorithm exploration, and performance validation.

## Overview

The LHCb VELO is the silicon vertex detector closest to the interaction point at the LHCb experiment at CERN's Large Hadron Collider. This toy model simulates:

- **Particle collision events** with configurable detector geometry
- **Multiple scattering effects** as particles traverse detector material
- **Track reconstruction** using Hamiltonian-based optimization
- **Quantum algorithms** (HHL) for solving the track-finding linear system
- **Validation metrics** following LHCb conventions

## Installation

### Basic Installation

```bash
git clone https://github.com/YourUsername/LHCb_VeLo_Toy_Model.git
cd LHCb_VeLo_Toy_Model
pip install -e .
```

### Dependencies

```bash
pip install numpy scipy matplotlib pandas
```

For quantum algorithm features:
```bash
pip install qiskit qiskit-aer
```

For IBM Quantum hardware simulation:
```bash
pip install qiskit-ibm-runtime
```

## Quick Start

### 1. Generate a Simulated Event

```python
from LHCB_Velo_Toy_Models.state_event_generator import StateEventGenerator
from LHCB_Velo_Toy_Models.state_event_model import PlaneGeometry

# Define detector geometry (10 planes)
geometry = PlaneGeometry(
    module_id=list(range(10)),
    lx=[50.0] * 10,      # Half-width in x (mm)
    ly=[50.0] * 10,      # Half-width in y (mm)
    z=[100 + i * 30 for i in range(10)]  # z positions (mm)
)

# Create event generator
particles = [[{'type': 'pion', 'mass': 139.6, 'q': 1}] * 5]  # 5 pions
generator = StateEventGenerator(
    detector_geometry=geometry,
    events=1,
    n_particles=[5],
    measurement_error=0.01,
    collision_noise=1e-3
)

# Generate primary vertices and particles
generator.generate_random_primary_vertices({'x': 0.01, 'y': 0.01, 'z': 50})
generator.generate_particles(particles)

# Generate complete event with hits
true_event = generator.generate_complete_events()

# Visualize
true_event.plot_segments()
```

### 2. Reconstruct Tracks Using Hamiltonian Method

```python
from LHCB_Velo_Toy_Models.simple_hamiltonian import SimpleHamiltonian, get_tracks

# Create Hamiltonian with parameters
ham = SimpleHamiltonian(
    epsilon=0.01,  # Angular tolerance (radians)
    gamma=1.0,     # Self-interaction penalty
    delta=1.0      # Bias term
)

# Build and solve the linear system
A, b = ham.construct_hamiltonian(generator, convolution=False)
solution = ham.solve_classicaly()

# Extract reconstructed tracks
reco_tracks = get_tracks(ham, solution, generator)
print(f"Reconstructed {len(reco_tracks)} tracks")
```

### 3. Validate Reconstruction Performance

```python
from LHCB_Velo_Toy_Models.toy_validator import EventValidator

# Create validator
validator = EventValidator(
    truth_event=true_event,
    rec_tracks=reco_event
)

# Compute LHCb-style metrics
metrics = validator.compute_metrics(
    purity_min=0.7,
    completeness_min=0.7
)

# Print results
validator.print_metrics()
```

### 4. Solve with Quantum Algorithm (HHL)

```python
import numpy as np
from hhl_algorithm import HHLAlgorithm

# Get the Hamiltonian matrix (must be small for quantum simulation)
A_dense = ham.A.toarray()
b_vec = ham.b

# Create HHL solver
hhl = HHLAlgorithm(
    A_dense, b_vec,
    num_time_qubits=4,
    shots=10000
)

# Build and run circuit
circuit = hhl.build_circuit()
counts = hhl.run()
quantum_solution = hhl.get_solution()

# Compare with classical solution
classical_solution = np.linalg.solve(A_dense, b_vec)
classical_normalized = classical_solution / np.linalg.norm(classical_solution)
fidelity = np.abs(np.vdot(quantum_solution, classical_normalized))
print(f"Quantum-Classical Fidelity: {fidelity:.4f}")
```

## Package Structure

```
LHCb_VeLo_Toy_Model/
├── LHCB_Velo_Toy_Models/
│   ├── __init__.py                 # Package initialization
│   ├── state_event_model.py        # Core data structures (Hit, Segment, Track, Event)
│   ├── state_event_generator.py    # Event generation with LHCb state vectors
│   ├── multi_scattering_generator.py  # Legacy generator with multiple scattering
│   ├── hamiltonian.py              # Abstract Hamiltonian interface
│   ├── simple_hamiltonian.py       # Reference Hamiltonian implementation
│   ├── simple_hamiltonian_fast.py  # Optimized implementation (vectorized)
│   ├── simple_hamiltonian_cpp.py   # C++/CUDA wrapper (optional)
│   ├── toy_validator.py            # Track reconstruction validation
│   └── lhcb_tracking_plots.py      # Visualization utilities
├── hhl_algorithm.py                # Basic HHL quantum algorithm
├── hhl_algorithm_1bit.py           # HHL with Suzuki-Trotter decomposition
└── README.md
```

## Key Modules

### Event Generation

| Module | Description |
|--------|-------------|
| `state_event_generator` | Generate events using LHCb state vectors (x, y, tx, ty, p/q) |
| `multi_scattering_generator` | Simpler generator focused on multiple scattering physics |

### Track Finding

| Module | Description |
|--------|-------------|
| `simple_hamiltonian` | Reference implementation of Hamiltonian-based track finding |
| `simple_hamiltonian_fast` | Optimized version with vectorized numpy operations |
| `simple_hamiltonian_cpp` | C++/CUDA accelerated version for large events |

### Validation & Visualization

| Module | Description |
|--------|-------------|
| `toy_validator` | LHCb-style track matching and metrics (efficiency, ghost rate, purity) |
| `lhcb_tracking_plots` | Comprehensive plotting for performance analysis |

### Quantum Algorithms

| Module | Description |
|--------|-------------|
| `hhl_algorithm` | Standard HHL implementation for linear systems |
| `hhl_algorithm_1bit` | Enhanced HHL with Suzuki-Trotter decomposition and noise simulation |

## Hamiltonian Track Finding

The track finding problem is formulated as a quadratic optimization:

```
H(x) = -0.5 * xᵀAx + bᵀx
```

Where:
- **x**: Segment activation vector (continuous relaxation)
- **A**: Interaction matrix encoding segment compatibility
- **b**: Bias vector encouraging segment activation

**Algorithm:**
1. Construct all possible segments between adjacent detector layers
2. Build matrix A where A[i,j] = 1 if segments i,j are compatible (share a hit and are nearly collinear)
3. Solve the linear system Ax = b
4. Threshold the solution to identify active segments
5. Group connected segments into tracks

## Validation Metrics

Following LHCb conventions:

| Metric | Definition |
|--------|------------|
| **Track Efficiency** | Fraction of truth tracks correctly reconstructed |
| **Ghost Rate** | Fraction of reconstructed tracks not matching any truth |
| **Clone Rate** | Fraction of duplicate reconstructions of the same truth |
| **Purity** | |R ∩ T| / |R| - fraction of reco hits from the matched truth |
| **Completeness** | |R ∩ T| / |T| - fraction of truth hits present in reco |

## Configuration Parameters

### Hamiltonian Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `epsilon` | Angular tolerance for segment compatibility | 0.001 - 0.1 rad |
| `gamma` | Self-interaction penalty | 1.0 - 10.0 |
| `delta` | Bias for segment activation | 1.0 - 10.0 |
| `theta_d` | ERF smoothing width (for convolution mode) | 1e-4 - 1e-2 |

### Event Generation Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `measurement_error` | Position resolution (mm) | 0.001 - 0.1 |
| `collision_noise` | Multiple scattering strength | 1e-4 - 1e-2 |
| `drop_rate` | Hit inefficiency fraction | 0.0 - 0.2 |
| `ghost_rate` | Noise hit fraction | 0.0 - 0.2 |

## Performance

### Classical Solver

| Implementation | Events/sec | Notes |
|----------------|------------|-------|
| `simple_hamiltonian` | ~10 | Reference implementation |
| `simple_hamiltonian_fast` | ~100 | Vectorized operations |
| `simple_hamiltonian_cpp` | ~1000 | Requires C++ compilation |

### Quantum (HHL) Limitations

- Limited to small systems (< 16 segments for practical simulation)
- Exponential circuit depth with system size
- Useful for proof-of-concept and algorithm research

## Examples

See the scripts in the repository root for complete examples:

- `hhl_algorithm.py` - Run HHL on a sample matrix
- `hhl_algorithm_1bit.py` - Run HHL with Trotter decomposition

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Authors

- George William
- Marcel Kunze
- Alain Chancé
- Contributors

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. LHCb Collaboration, "LHCb VELO Technical Design Report", CERN-LHCC-2001-011
2. Harrow, A. W., Hassidim, A., & Lloyd, S. (2009). "Quantum algorithm for linear systems of equations", Physical Review Letters, 103(15), 150502.
3. Denby, B. (1988). "Neural networks and cellular automata in experimental high energy physics", Computer Physics Communications, 49(3), 429-448.

## Acknowledgments

- CERN and the LHCb Collaboration for detector specifications
- IBM Quantum for Qiskit framework
- The quantum computing research community
