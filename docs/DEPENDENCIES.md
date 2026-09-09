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
| `scipy` | ≥1.7.0 | Sparse matrices, solvers, special functions, `linprog` (minimax comb), `eigsh` (spectral bounds), `minimize` (phase factors) | Hamiltonians, QSVT, spectral filter, refit |
| `matplotlib` | ≥3.5.0 | Visualization, 3D plotting | Event plotting, analysis |

### Analysis Dependencies

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `pandas` | ≥1.4.0 | DataFrames, CSV I/O | Validation, plotting |

### Quantum Dependencies (Optional)

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `qiskit` | ≥2.1 | Quantum circuit construction (`QFTGate`, `StatePreparation`, `UnitaryGate`, `DiagonalGate`) | HHL, OneBQF, `QSVT.build_circuit` / `run_circuit(streaming=False)` |
| `qiskit-aer` | ≥0.17 | Quantum simulation (`AerSimulator`, statevector method, noise models) | HHL, OneBQF |
| `qiskit-ibm-runtime` | ≥0.40 | IBM backend access (noise-model source), imported only inside `OneBQF.run(use_noise_model=True)` | OneBQF |

> **Lazy (2.2.0).** `lhcb_velo_toy.solvers.quantum` only imports qiskit when `HHLAlgorithm`, `OneBQF` / `OneBitHHL` are first accessed. `QSVT` (statevector / eigh / phase-sequence backends), `solve_one_bit_matrixfree`, `spectral_filter`, `responses`, `qsp_phases` and `refit` are numpy/scipy-only and work in a base install.

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | ≥7.0.0 | Unit testing |
| `pytest-cov` | ≥4.0.0 | Coverage reporting |
| `mypy` | ≥1.0.0 | Static type checking |
| `ruff` | ≥0.1.0 | Fast linting and formatting |

---

## Installation Profiles

### pyproject.toml Configuration

```toml
[project]
name = "lhcb-velo-toy"
version = "2.2.0"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.20",
    "scipy>=1.7",
    "matplotlib>=3.5",
]

[project.optional-dependencies]
quantum = [
    "qiskit>=2.1",
    "qiskit-aer>=0.17",
    "qiskit-ibm-runtime>=0.40",
]

analysis = [
    "pandas>=1.4",
]

dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "mypy>=1.0",
    "ruff>=0.1",
]

all = ["lhcb-velo-toy[quantum,analysis,dev]"]
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
| `generation.entities` | ✓ | | | | | | |
| `generation.geometry` | ✓ | | | | | | |
| `generation.generators` | ✓ | | | | | | |
| `solvers.hamiltonians` | ✓ | ✓ | | | | | |
| `solvers.classical` | ✓ | ✓ | | | | | |
| `solvers.hamiltonians.extended` | ✓ | ✓ | | | | | |
| `solvers.quantum.HHL` (`hhl` shim) | ✓ | ✓ | ✓ | | ✓ | ✓ | |
| `solvers.quantum.one_bqf` (`one_bit_hhl` shim) | ✓ | ✓ | | | ✓ | ✓ | ✓ |
| `solvers.quantum.matrixfree` | ✓ | ✓ | | | | | |
| `solvers.quantum.QSVT` | ✓ | ✓ | | | (circuit only) | | |
| `solvers.quantum.spectral_filter` | ✓ | ✓ | | | | | |
| `solvers.quantum.responses` | ✓ | ✓ | | | | | |
| `solvers.quantum.qsp_phases` | ✓ | ✓ | | | | | |
| `solvers.quantum.refit` | ✓ | ✓ | | | | | |
| `solvers.hough` | ✓ | | | | | | |
| `solvers.postselect` | ✓ | | | | | | |
| `solvers.reconstruction` | ✓ | | | | | | |
| `analysis.segment_metrics` | ✓ | ✓ | | | | | |
| `analysis.working_points` | ✓ | | | | | | |
| `generation.adapters.run3` | ✓ | | | | | | |
| `analysis.validation` | ✓ | | | ✓ | | | |
| `analysis.plotting` | ✓ | | ✓ | ✓ | | | |
| `persistence.pipeline` | ✓ | ✓ | | | | | |
| `persistence.study` | ✓ | | | | | | |

### Import Graph

```
generation.entities
    └── numpy

generation.geometry
    └── numpy

generation.generators.state_event
    ├── numpy
    ├── generation.geometry (Geometry)
    └── generation.entities (Hit, Track, Module, Event, PrimaryVertex)

solvers.hamiltonians.base (ABC)
    ├── numpy
    └── scipy.sparse

solvers.hamiltonians.simple
    ├── numpy
    ├── scipy.sparse
    ├── scipy.special (erf)
    ├── scipy.sparse.linalg (cg)
    ├── solvers.hamiltonians.base (Hamiltonian)
    └── solvers.reconstruction.segment (Segment)

solvers.hamiltonians.fast
    ├── numpy
    ├── scipy.sparse (coo_matrix, csc_matrix)
    ├── scipy.sparse.linalg (cg, spsolve)
    └── solvers.hamiltonians.base (Hamiltonian)

solvers.classical.solvers
    ├── numpy
    └── scipy.sparse.linalg (spsolve, cg)

solvers.hamiltonians.extended
    ├── numpy, scipy.sparse
    └── solvers.hamiltonians.fast (SimpleHamiltonianFast)

solvers.quantum.HHL            [lazy from solvers.quantum; hhl.py is a shim]
    ├── numpy, matplotlib, scipy.linalg (expm)
    ├── qiskit (QuantumCircuit, QuantumRegister, ClassicalRegister, transpile)
    ├── qiskit.circuit.library (QFTGate, RYGate, UnitaryGate)
    └── qiskit_aer (AerSimulator)

solvers.quantum.one_bqf        [lazy from solvers.quantum; one_bit_hhl.py is a shim]
    ├── numpy, scipy.sparse
    ├── qiskit (QuantumCircuit, QuantumRegister, ClassicalRegister, transpile)
    ├── qiskit.circuit.library (QFTGate, RXGate)
    ├── qiskit_aer (AerSimulator), qiskit_aer.noise (NoiseModel)
    └── qiskit_ibm_runtime (QiskitRuntimeService)

solvers.quantum.matrixfree
    └── numpy, scipy.sparse

solvers.quantum.QSVT
    ├── numpy (numpy.polynomial.chebyshev), scipy.sparse, scipy.sparse.linalg (eigsh)
    ├── scipy.optimize (linprog)          — design_minimax_comb only
    ├── solvers.quantum.spectral_filter   — solve_eigh
    ├── solvers.quantum.responses         — filter_design="cosine"
    ├── solvers.quantum.qsp_phases        — solve_phase_sequence
    └── qiskit (QuantumCircuit, StatePreparation, DiagonalGate, UnitaryGate),
        qiskit.quantum_info (Statevector) — build_circuit / run_circuit(streaming=False) ONLY

solvers.quantum.spectral_filter
    └── numpy, scipy.sparse, scipy.linalg (eigh), scipy.sparse.csgraph, scipy.sparse.linalg (eigsh)

solvers.quantum.responses
    └── numpy (numpy.polynomial.chebyshev), scipy.sparse

solvers.quantum.qsp_phases
    └── numpy, scipy.sparse, scipy.optimize (minimize)

solvers.quantum.refit
    └── numpy, scipy.sparse

solvers.hough
    ├── numpy
    └── analysis.validation (EventValidator), generation.entities (Track)

solvers.postselect
    └── numpy, scipy.sparse

analysis.segment_metrics
    └── numpy, scipy

analysis.working_points
    ├── numpy
    └── analysis.segment_metrics

solvers.reconstruction.track_finder
    ├── numpy
    └── generation.entities (Hit, Track, Event)

analysis.validation.validator
    ├── numpy
    ├── pandas
    └── generation.entities (Event, Track, Hit)

analysis.plotting
    ├── numpy
    ├── pandas
    └── matplotlib.pyplot

persistence.pipeline
    ├── json
    ├── numpy
    ├── scipy.sparse (load_npz, save_npz, csc_matrix)
    ├── generation.entities.event (Event)
    ├── generation.geometry (geometry_from_dict)
    └── analysis.validation.match (Match)

persistence.study
    ├── json
    └── numpy
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
| `match_tracks(purity_min, hit_efficiency_min, min_rec_hits)` | `float, float, int` | `tuple[list[Match], dict]` |
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
| 3.12 | ✓ Supported | Tested and working |

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 16 GB |
| CPU | 2 cores | 4+ cores |
| Disk | 500 MB | 2 GB |
| GPU | - | - |

### Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux (x86_64) | ✓ Full | Primary development platform |
| macOS (Intel) | ✓ Full | |
| macOS (ARM) | ✓ Full | Rosetta not required |
| Windows (x86_64) | ✓ Full | |

---

## Version Compatibility Matrix

### Qiskit Versions

| lhcb-velo-toy | qiskit | qiskit-aer | qiskit-ibm-runtime | Notes |
|---------------|--------|------------|-------------------|-------|
| 1.x | 0.45 | 0.12 | 0.15 | Legacy |
| 2.0 | ≥1.0 | ≥0.14 | ≥0.20 | Modular package |
| 2.1 / 2.2 | ≥2.1 (tested 2.4.1) | ≥0.17 (tested 0.17.2) | ≥0.40 (tested 0.47.0) | Current; qiskit optional for QSVT / matrix-free 1BQF. Code uses `QFTGate`, `Gate.control(..., annotated=False)`, `qiskit.transpiler.generate_preset_pass_manager`; the pytest config fails on any deprecation attributed to `lhcb_velo_toy` |

### NumPy/SciPy Versions

| lhcb-velo-toy | numpy | scipy | Notes |
|---------------|-------|-------|-------|
| 1.x | ≥1.19 | ≥1.6 | Legacy |
| 2.0 | ≥1.20 | ≥1.7 | Modular package |
| 2.1 / 2.2 | ≥1.20 | ≥1.7 (`linprog(method="highs")` needs ≥1.6; `minres(rtol=)` ≥1.12 in the studies only) | Current |

---

## Dependency Conflicts

### Known Issues

| Conflict | Description | Resolution |
|----------|-------------|------------|
| qiskit < 2.1 | `QFTGate`, `Gate.control(annotated=)` and `qiskit.transpiler.generate_preset_pass_manager` are unavailable or moved | Use qiskit ≥ 2.1 |
| numpy 2.0 | ABI changes in third-party wheels | The package is developed and tested on numpy 2.3.5; pin only if a dependency forces it |
| scipy.sparse | API deprecations | Use csr_array/csc_array for scipy ≥ 1.11 |

### Recommended Constraints

```txt
# requirements.txt — matches pyproject.toml (tested: numpy 2.3.5, scipy 1.16.3,
# qiskit 2.4.1, qiskit-aer 0.17.2, qiskit-ibm-runtime 0.47.0)
numpy>=1.20
scipy>=1.7
qiskit>=2.1
qiskit-aer>=0.17
qiskit-ibm-runtime>=0.40
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
- [WORKFLOW_OVERVIEW.md](WORKFLOW_OVERVIEW.md) - End-to-end workflow guide
