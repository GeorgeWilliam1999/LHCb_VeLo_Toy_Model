# LHCb VELO Toy Model - Dependencies & Requirements

This document details all dependencies, input/output specifications, and requirements for each component of the LHCb VELO Toy Model package.

---

## Table of Contents

1. [Package Dependencies](#package-dependencies)
2. [Installation Profiles](#installation-profiles)
3. [Module Dependencies](#module-dependencies)
4. [Input/Output Specifications](#inputoutput-specifications)
5. [Environment Requirements](#environment-requirements)
6. [Optional Accelerators](#optional-accelerators)

---

## Package Dependencies

### Core Dependencies

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `numpy` | ≥1.20.0 | Numerical arrays, linear algebra | All modules |
| `scipy` | ≥1.7.0 | Sparse matrices, solvers, special functions | Hamiltonians, validation |
| `matplotlib` | ≥3.5.0 | Visualization, 3D plotting | Event plotting, analysis |

### Analysis Dependencies

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `pandas` | ≥1.3.0 | DataFrames, CSV I/O | Validation, plotting |

### Quantum Dependencies (Optional)

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `qiskit` | ≥1.0.0 | Quantum circuit construction | HHL, OneBQF |
| `qiskit-aer` | ≥0.13.0 | Quantum simulation | HHL, OneBQF |
| `qiskit-ibm-runtime` | ≥0.20.0 | IBM backend access, noise models | OneBQF |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | ≥7.0.0 | Unit testing |
| `pytest-cov` | ≥4.0.0 | Coverage reporting |
| `sphinx` | ≥6.0.0 | Documentation generation |
| `black` | ≥23.0.0 | Code formatting |
| `isort` | ≥5.12.0 | Import sorting |
| `mypy` | ≥1.0.0 | Static type checking |

---

## Installation Profiles

### pyproject.toml Configuration

```toml
[project]
name = "lhcb-velo-toy"
version = "2.0.0"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "matplotlib>=3.5.0",
]

[project.optional-dependencies]
analysis = [
    "pandas>=1.3.0",
]

quantum = [
    "qiskit>=1.0.0",
    "qiskit-aer>=0.13.0",
    "qiskit-ibm-runtime>=0.20.0",
]

dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "sphinx>=6.0.0",
    "sphinx-rtd-theme>=2.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.0.0",
]

all = [
    "lhcb-velo-toy[analysis,quantum,dev]",
]
```

### Installation Commands

```bash
# Minimal installation (generation + classical solving)
pip install lhcb-velo-toy

# With validation and plotting
pip install lhcb-velo-toy[analysis]

# With quantum algorithms
pip install lhcb-velo-toy[quantum]

# Full installation
pip install lhcb-velo-toy[all]

# Development installation
pip install -e ".[all]"
```

---

## Module Dependencies

### Dependency Matrix

| Module | numpy | scipy | matplotlib | pandas | qiskit | qiskit-aer | qiskit-ibm-runtime |
|--------|:-----:|:-----:|:----------:|:------:|:------:|:----------:|:------------------:|
| `state_event_model` | ✓ | | ✓ | | | | |
| `state_event_generator` | ✓ | | | | | | |
| `multi_scattering_generator` | ✓ | | | | | | |
| `hamiltonian` | ✓ | ✓ | | | | | |
| `simple_hamiltonian` | ✓ | ✓ | | | | | |
| `simple_hamiltonian_fast` | ✓ | ✓ | | | | | |
| `simple_hamiltonian_cpp` | ✓ | ✓ | | | | | |
| `hhl_algorithm` | ✓ | | | | ✓ | ✓ | |
| `OneBQF` | ✓ | | | | ✓ | ✓ | ✓ |
| `toy_validator` | ✓ | | | ✓ | | | |
| `lhcb_tracking_plots` | ✓ | | ✓ | ✓ | | | |

### Import Graph

```
state_event_model
    └── numpy, matplotlib

state_event_generator
    ├── numpy
    └── state_event_model (Hit, Module, Segment, Track, Event, Geometry)

multi_scattering_generator
    ├── numpy
    └── state_event_model (Hit, Module, Segment, Track)

hamiltonian (ABC)
    ├── numpy
    └── scipy.sparse

simple_hamiltonian
    ├── numpy
    ├── scipy.sparse
    ├── scipy.special (erf)
    ├── scipy.sparse.linalg (cg)
    ├── hamiltonian (Hamiltonian)
    └── state_event_model (Segment, Track, Hit, Event)

simple_hamiltonian_fast
    ├── numpy
    ├── scipy.sparse (coo_matrix, csc_matrix)
    ├── scipy.sparse.linalg (cg, spsolve)
    └── hamiltonian (Hamiltonian)

simple_hamiltonian_cpp
    ├── numpy
    ├── scipy.sparse
    ├── hamiltonian (Hamiltonian)
    └── cpp_hamiltonian (optional C++ module)

hhl_algorithm
    ├── numpy
    ├── qiskit (QuantumCircuit, QuantumRegister, ClassicalRegister)
    ├── qiskit.circuit.library (UnitaryGate)
    ├── qiskit.quantum_info (Statevector)
    └── qiskit_aer (AerSimulator)

OneBQF
    ├── numpy
    ├── qiskit (QuantumCircuit, QuantumRegister, ClassicalRegister)
    ├── qiskit.circuit.library (UnitaryGate)
    ├── qiskit_aer (AerSimulator)
    ├── qiskit_aer.noise (NoiseModel)
    └── qiskit_ibm_runtime (QiskitRuntimeService)

toy_validator
    ├── numpy
    ├── pandas
    └── state_event_model (Event, Track, Hit)

lhcb_tracking_plots
    ├── numpy
    ├── pandas
    └── matplotlib.pyplot
```

---

## Input/Output Specifications

### StateEventGenerator

#### Constructor Inputs

| Parameter | Type | Constraints | Default |
|-----------|------|-------------|---------|
| `detector_geometry` | `Geometry` | Must be non-empty | Required |
| `phi_min` | `float` | -π < phi_min < phi_max < π | -0.2 |
| `phi_max` | `float` | -π < phi_min < phi_max < π | 0.2 |
| `theta_min` | `float` | -π/2 < theta_min < theta_max < π/2 | -0.2 |
| `theta_max` | `float` | -π/2 < theta_min < theta_max < π/2 | 0.2 |
| `events` | `int` | ≥ 1 | 3 |
| `n_particles` | `list[int]` | len == events, all > 0 | None |
| `measurement_error` | `float` | ≥ 0 | 0.0 |
| `collision_noise` | `float` | ≥ 0 | 1e-4 |

#### Method Inputs/Outputs

| Method | Input | Output |
|--------|-------|--------|
| `generate_random_primary_vertices(variance)` | `dict[str, float]`: {x, y, z} variances | `list[tuple[float, float, float]]` |
| `generate_particles(particles)` | `list[list[dict]]`: particle configs per event | `list[dict]`: flattened particles |
| `generate_complete_events()` | (uses internal state) | `Event` |
| `make_noisy_event(drop_rate, ghost_rate)` | `float, float`: rates ∈ [0, 1] | `Event` |

---

### SimpleHamiltonian

#### Constructor Inputs

| Parameter | Type | Constraints | Recommended |
|-----------|------|-------------|-------------|
| `epsilon` | `float` | > 0, typically 1e-7 to 0.1 | 0.001 - 0.01 |
| `gamma` | `float` | > 0 | 1.0 - 2.0 |
| `delta` | `float` | > 0 | 1.0 |
| `theta_d` | `float` | > 0, for ERF smoothing | 1e-4 |

#### Method Inputs/Outputs

| Method | Input | Output |
|--------|-------|--------|
| `construct_segments(event)` | `StateEventGenerator` or `Event` | `None` (mutates self.segments) |
| `construct_hamiltonian(event, convolution)` | `StateEventGenerator`, `bool` | `tuple[csc_matrix, ndarray]` |
| `solve_classicaly()` | (uses internal A, b) | `ndarray`: solution vector |
| `evaluate(solution)` | `ndarray[n_segments]` | `float`: Hamiltonian energy |

---

### HHLAlgorithm

#### Constructor Inputs

| Parameter | Type | Constraints | Default |
|-----------|------|-------------|---------|
| `matrix_A` | `np.ndarray` | 2D, square, Hermitian | Required |
| `vector_b` | `np.ndarray` | 1D, len = matrix dimension | Required |
| `num_time_qubits` | `int` | ≥ 1, typically 4-8 | 5 |
| `shots` | `int` | > 0 | 10240 |
| `debug` | `bool` | | False |

#### Method Inputs/Outputs

| Method | Input | Output |
|--------|-------|--------|
| `build_circuit()` | | `QuantumCircuit` |
| `run()` | | `dict`: measurement counts |
| `get_solution()` | | `np.ndarray`: solution vector |
| `simulate_statevector()` | | `Statevector` |
| `extract_postselected_solution(sv)` | `Statevector` | `np.ndarray` |

---

### OneBQF

#### Constructor Inputs

| Parameter | Type | Constraints | Default |
|-----------|------|-------------|---------|
| `matrix_A` | `np.ndarray` | 2D, square | Required |
| `vector_b` | `np.ndarray` | 1D | Required |
| `num_time_qubits` | `int` | ≥ 1 | 1 |
| `shots` | `int` | > 0 | 1024 |
| `debug` | `bool` | | False |

#### Method Inputs/Outputs

| Method | Input | Output |
|--------|-------|--------|
| `build_circuit()` | | `QuantumCircuit` |
| `run(use_noise_model, backend_name)` | `bool`, `str` | `dict`: counts |
| `get_solution(counts)` | `dict` (optional) | `tuple[ndarray, int]`: (solution, success_count) |

---

### EventValidator

#### Constructor Inputs

| Parameter | Type | Constraints |
|-----------|------|-------------|
| `truth_event` | `Event` | Must have tracks |
| `rec_tracks` | `list[Track]` | |
| `reconstructible_filter` | `Callable[[Track], bool]` | Optional |

#### Method Inputs/Outputs

| Method | Input | Output |
|--------|-------|--------|
| `match_tracks(purity_min, completeness_min, min_rec_hits)` | `float, float, int` | `tuple[list[Match], dict]` |
| `summary_table()` | | `pd.DataFrame` |
| `truth_table()` | | `pd.DataFrame` |

---

## Environment Requirements

### Python Version

| Version | Status | Notes |
|---------|--------|-------|
| 3.9 | ✓ Supported | Minimum version |
| 3.10 | ✓ Supported | Recommended |
| 3.11 | ✓ Supported | Best performance |
| 3.12 | ⚠️ Testing | Check qiskit compatibility |

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 16 GB |
| CPU | 2 cores | 4+ cores |
| Disk | 500 MB | 2 GB |
| GPU | - | CUDA-capable (for cpp_hamiltonian) |

### Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux (x86_64) | ✓ Full | Primary development platform |
| macOS (Intel) | ✓ Full | |
| macOS (ARM) | ✓ Full | Rosetta not required |
| Windows (x86_64) | ✓ Full | |

---

## Optional Accelerators

### C++ Hamiltonian Extension

The `simple_hamiltonian_cpp` module wraps a C++ implementation for improved performance.

#### Requirements

| Component | Version |
|-----------|---------|
| C++ Compiler | C++17 compatible (GCC 9+, Clang 10+, MSVC 2019+) |
| CMake | ≥ 3.18 |
| pybind11 | ≥ 2.10.0 |

#### CUDA Support

| Component | Version |
|-----------|---------|
| CUDA Toolkit | ≥ 11.0 |
| cuBLAS | Included with CUDA |
| cuSPARSE | Included with CUDA |

#### Installation

```bash
cd LHCB_Velo_Toy_Models/cpp_hamiltonian

# CPU-only build
pip install .

# With CUDA support
CMAKE_ARGS="-DUSE_CUDA=ON" pip install .
```

#### Fallback Behavior

```python
from lhcb_velo_toy.solvers import SimpleHamiltonianCPPWrapper

try:
    ham = SimpleHamiltonianCPPWrapper(epsilon=0.01, gamma=1.0, delta=1.0)
except ImportError:
    # Fall back to pure Python
    from lhcb_velo_toy.solvers import SimpleHamiltonian
    ham = SimpleHamiltonian(epsilon=0.01, gamma=1.0, delta=1.0)
```

---

## Version Compatibility Matrix

### Qiskit Versions

| lhcb-velo-toy | qiskit | qiskit-aer | qiskit-ibm-runtime | Notes |
|---------------|--------|------------|-------------------|-------|
| 1.x | 0.45 | 0.12 | 0.15 | Legacy |
| 2.0 | ≥1.0 | ≥0.13 | ≥0.20 | Current |

### NumPy/SciPy Versions

| lhcb-velo-toy | numpy | scipy | Notes |
|---------------|-------|-------|-------|
| 1.x | ≥1.19 | ≥1.6 | Legacy |
| 2.0 | ≥1.20 | ≥1.7 | Current |

---

## Dependency Conflicts

### Known Issues

| Conflict | Description | Resolution |
|----------|-------------|------------|
| qiskit < 1.0 | API changes | Use qiskit ≥ 1.0 |
| numpy 2.0 | ABI changes | Pin numpy < 2.0 if issues |
| scipy.sparse | API deprecations | Use csr_array/csc_array for scipy ≥ 1.11 |

### Recommended Constraints

```txt
# requirements.txt
numpy>=1.20.0,<2.0.0
scipy>=1.7.0,<1.14.0
qiskit>=1.0.0,<2.0.0
qiskit-aer>=0.13.0
qiskit-ibm-runtime>=0.20.0
```

---

## Runtime Checks

### Dependency Verification Script

```python
#!/usr/bin/env python
"""Verify lhcb-velo-toy dependencies."""

def check_dependencies():
    """Check all dependencies and report status."""
    results = {}
    
    # Core
    try:
        import numpy as np
        results['numpy'] = np.__version__
    except ImportError:
        results['numpy'] = 'MISSING'
    
    try:
        import scipy
        results['scipy'] = scipy.__version__
    except ImportError:
        results['scipy'] = 'MISSING'
    
    try:
        import matplotlib
        results['matplotlib'] = matplotlib.__version__
    except ImportError:
        results['matplotlib'] = 'MISSING'
    
    # Analysis
    try:
        import pandas
        results['pandas'] = pandas.__version__
    except ImportError:
        results['pandas'] = 'Not installed (optional)'
    
    # Quantum
    try:
        import qiskit
        results['qiskit'] = qiskit.__version__
    except ImportError:
        results['qiskit'] = 'Not installed (optional)'
    
    try:
        import qiskit_aer
        results['qiskit-aer'] = qiskit_aer.__version__
    except ImportError:
        results['qiskit-aer'] = 'Not installed (optional)'
    
    try:
        from qiskit_ibm_runtime import __version__ as ibm_version
        results['qiskit-ibm-runtime'] = ibm_version
    except ImportError:
        results['qiskit-ibm-runtime'] = 'Not installed (optional)'
    
    # C++ extension
    try:
        import cpp_hamiltonian
        results['cpp_hamiltonian'] = 'Available'
    except ImportError:
        results['cpp_hamiltonian'] = 'Not built (optional)'
    
    return results


if __name__ == '__main__':
    deps = check_dependencies()
    print("LHCb VELO Toy Model - Dependency Check")
    print("=" * 45)
    for pkg, ver in deps.items():
        status = "✓" if "MISSING" not in ver else "✗"
        print(f"{status} {pkg:25s} {ver}")
```

---

## See Also

- [API_REFERENCE.md](API_REFERENCE.md) - Detailed API documentation
- [FLOW_DIAGRAMS.md](FLOW_DIAGRAMS.md) - Architecture diagrams
- [RESTRUCTURING_PROPOSAL.md](../RESTRUCTURING_PROPOSAL.md) - Package restructuring plan
