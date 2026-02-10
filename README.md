# LHCb VELO Toy Model

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![pip installable](https://img.shields.io/badge/pip-installable-brightgreen.svg)](https://pip.pypa.io/)

A Python package for simulating and analyzing particle tracking in the LHCb VELO
(Vertex Locator) detector. This package provides tools for event generation,
Hamiltonian-based track reconstruction, quantum algorithm exploration, and
performance validation.

**Authors:** George William Scriven · Xenofon Chiotopoulos · Alain Chancé  
**Institutes:** Maastricht University · UHasselt · Nikhef

## Overview

The LHCb VELO is the silicon vertex detector closest to the interaction point at
the LHCb experiment at CERN's Large Hadron Collider. This toy model simulates:

- **Particle collision events** with configurable detector geometry
- **Multiple scattering effects** as particles traverse detector material
- **Track reconstruction** using Hamiltonian-based optimization
- **Quantum algorithms** (HHL) for solving the track-finding linear system
- **Validation metrics** following LHCb conventions

## Installation

### From source (recommended)

```bash
git clone https://github.com/GeorgeWilliam1999/LHCb_VeLo_Toy_Model.git
cd LHCb_VeLo_Toy_Model
pip install -e .
```

### With optional dependencies

```bash
# Quantum algorithm support (Qiskit)
pip install -e ".[quantum]"

# Development tools (pytest, mypy, ruff)
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

## Quick Start

### 1. Generate a Simulated Event

```python
from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator

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
```

### 2. Reconstruct Tracks Using Hamiltonian Method

```python
from lhcb_velo_toy.solvers import SimpleHamiltonian, get_tracks
from lhcb_velo_toy.solvers.classical import solve_direct

# Create Hamiltonian with parameters
ham = SimpleHamiltonian(
    epsilon=0.01,  # Angular tolerance (radians)
    gamma=1.0,     # Self-interaction penalty
    delta=1.0      # Bias term
)

# Build the linear system
A, b = ham.construct_hamiltonian(generator, convolution=False)

# Solve classically
solution = solve_direct(A, b)

# Extract reconstructed tracks
reco_tracks = get_tracks(ham, solution, generator)
print(f"Reconstructed {len(reco_tracks)} tracks")
```

### 3. Validate Reconstruction Performance

```python
from lhcb_velo_toy.analysis import EventValidator

# Create validator
validator = EventValidator(
    truth_event=true_event,
    rec_tracks=reco_tracks
)

# Compute LHCb-style metrics
metrics = validator.compute_metrics(
    purity_min=0.7,
    hit_efficiency_min=0.7
)

# Print results
validator.print_metrics()
```

### 4. Visualise an Event

```python
from lhcb_velo_toy.analysis.plotting import plot_event_display

plot_event_display(generator, true_event, show_modules=True)
```

## Package Structure

```
LHCb_VeLo_Toy_Model/
├── pyproject.toml              # Package metadata & dependencies
├── LICENSE                     # MIT licence
├── README.md
├── src/
│   └── lhcb_velo_toy/         # Installable package
│       ├── core/               # Shared type aliases
│       ├── generation/         # Event simulation
│       │   ├── generators/     #   StateEventGenerator
│       │   ├── geometry/       #   PlaneGeometry, RectangularVoidGeometry
│       │   └── models/         #   Hit, Track, Module, Event, Segment
│       ├── solvers/            # Track reconstruction
│       │   ├── hamiltonians/   #   Hamiltonian ABC, Simple, Fast
│       │   ├── classical/      #   solve_direct, solve_conjugate_gradient
│       │   ├── quantum/        #   HHL, OneBitHHL
│       │   └── reconstruction/ #   track_finder, get_tracks
│       └── analysis/           # Validation & plots
│           ├── validation/     #   EventValidator, Match
│           └── plotting/       #   event_display, performance
├── deprecated/                 # Old monolithic code (reference only)
│   ├── README.md               #   Migration guide
│   ├── LHCB_Velo_Toy_Models/   #   Original flat package
│   ├── HHL.py                  #   Standalone HHL script
│   └── OneBQF.py               #   Standalone 1-bit HHL script
├── docs/                       # Documentation & presentation
└── tests/                      # Test suite
```

## Key Modules

### Event Generation (`lhcb_velo_toy.generation`)

| Module | Description |
|--------|-------------|
| `generators.state_event` | Generate events using LHCb state vectors (x, y, tx, ty, p/q) |
| `geometry.plane` | Planar detector module geometry |
| `geometry.rectangular_void` | Geometry with a rectangular beam-pipe void |
| `models.event` | `Event` dataclass — hits, tracks, modules, primary vertices |

### Solvers (`lhcb_velo_toy.solvers`)

| Module | Description |
|--------|-------------|
| `hamiltonians.simple` | Reference Hamiltonian-based track finding |
| `hamiltonians.fast` | Optimised vectorised implementation |
| `classical.solvers` | Direct (LU) and conjugate-gradient solvers |
| `quantum.hhl` | Full HHL algorithm for linear systems |
| `quantum.one_bit_hhl` | 1-bit HHL with Suzuki-Trotter decomposition |
| `reconstruction.track_finder` | Extract tracks from solution vectors |

### Analysis (`lhcb_velo_toy.analysis`)

| Module | Description |
|--------|-------------|
| `validation.validator` | LHCb-style track matching and metrics |
| `plotting.event_display` | 3D event visualisation with detector planes |
| `plotting.performance` | Efficiency, ghost-rate, and purity plots |

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
| **Hit Efficiency** | |R ∩ T| / |T| - fraction of truth hits present in reco |

**Non-Greedy Matching:** When multiple reco tracks match the same truth,
the algorithm compares match quality and reassigns to find the optimal
global matching (not first-come-first-served).

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

- `demo_workflow.ipynb` — End-to-end notebook: generate → solve → validate → visualise
- `test_pipeline.py` — Automated integration test of the full pipeline
- `deprecated/HHL.py` — Standalone HHL example (old code, for reference)

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Authors

- George William Scriven (Maastricht University)
- Xenofon Chiotopoulos (UHasselt)
- Alain Chancé (Nikhef)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## References

1. LHCb Collaboration, "LHCb VELO Technical Design Report", CERN-LHCC-2001-011
2. Harrow, A. W., Hassidim, A., & Lloyd, S. (2009). "Quantum algorithm for linear systems of equations", Physical Review Letters, 103(15), 150502.
3. Denby, B. (1988). "Neural networks and cellular automata in experimental high energy physics", Computer Physics Communications, 49(3), 429-448.

## Acknowledgments

- CERN and the LHCb Collaboration for detector specifications
- IBM Quantum for Qiskit framework
- The quantum computing research community
