# LHCb VELO Toy Model - Restructuring Proposal

## Executive Summary

This document proposes restructuring the `LHCb_VeLo_Toy_Model` repository into three clear submodules to improve modularity, maintainability, and usability.

---

## Proposed Package Structure

```
lhcb_velo_toy/                          # Main package
├── __init__.py                         # Top-level exports & version
├── core/                               # Shared core utilities
│   ├── __init__.py
│   └── types.py                        # Type definitions, Protocols
│
├── generation/                         # SUBMODULE 1: Data Generation
│   ├── __init__.py
│   ├── models/                         # Data structures
│   │   ├── __init__.py
│   │   ├── hit.py                      # Hit dataclass
│   │   ├── segment.py                  # Segment dataclass
│   │   ├── track.py                    # Track dataclass
│   │   ├── module.py                   # Module dataclass
│   │   └── event.py                    # Event container
│   ├── geometry/                       # Detector geometries
│   │   ├── __init__.py
│   │   ├── base.py                     # Abstract Geometry ABC
│   │   ├── plane.py                    # PlaneGeometry
│   │   └── rectangular_void.py         # RectangularVoidGeometry
│   └── generators/                     # Event generators
│       ├── __init__.py
│       ├── base.py                     # Abstract EventGenerator
│       └── state_event.py              # StateEventGenerator
│
├── solvers/                            # SUBMODULE 2: Track Reconstruction
│   ├── __init__.py
│   ├── hamiltonians/                   # Hamiltonian formulations
│   │   ├── __init__.py
│   │   ├── base.py                     # Abstract Hamiltonian ABC
│   │   ├── simple.py                   # SimpleHamiltonian (reference)
│   │   └── fast.py                     # SimpleHamiltonianFast (optimized)
│   ├── classical/                      # Classical solvers
│   │   ├── __init__.py
│   │   └── solvers.py                  # CG and direct solvers
│   ├── quantum/                        # Quantum algorithm implementations
│   │   ├── __init__.py
│   │   ├── hhl.py                      # Full HHL algorithm
│   │   └── one_bit_hhl.py              # 1-Bit HHL (OneBQF)
│   └── reconstruction/                 # Track reconstruction utilities
│       ├── __init__.py
│       └── track_finder.py             # get_tracks(), find_segments()
│
└── analysis/                           # SUBMODULE 3: Analysis & Validation
    ├── __init__.py
    ├── validation/                     # Reconstruction metrics
    │   ├── __init__.py
    │   ├── match.py                    # Match dataclass
    │   └── validator.py                # EventValidator
    └── plotting/                       # Visualization
        ├── __init__.py
        ├── event_display.py            # 3D event visualization
        └── performance.py              # Efficiency/ghost rate plots
```

---

## Development Flow

### Phase 1: Foundation
- [x] Create `src/lhcb_velo_toy/` directory structure
- [x] Create skeleton files with signatures and documentation
- [ ] Set up `pyproject.toml` with dependencies

### Phase 2: Generation Module
- [ ] Implement `models/` dataclasses (Hit, Segment, Track, Module, Event)
- [ ] Implement `geometry/` classes (Geometry ABC, PlaneGeometry, RectangularVoidGeometry)
- [ ] Implement `generators/` (StateEventGenerator)

### Phase 3: Solvers Module
- [ ] Implement `hamiltonians/` (Hamiltonian ABC, SimpleHamiltonian, SimpleHamiltonianFast)
- [ ] Implement `classical/` solvers
- [ ] Implement `quantum/` algorithms (HHL, OneBitHHL)
- [ ] Implement `reconstruction/` track finder

### Phase 4: Analysis Module
- [ ] Implement `validation/` (Match, EventValidator)
- [ ] Implement `plotting/` utilities

### Phase 5: Integration & Testing
- [ ] Integration tests
- [ ] Example notebooks
- [ ] Release v2.0.0
