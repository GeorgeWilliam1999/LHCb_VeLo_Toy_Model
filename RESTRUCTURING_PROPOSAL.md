# LHCb VELO Toy Model - Restructuring Proposal

## Executive Summary

This document proposes restructuring the `LHCb_VeLo_Toy_Model` repository into three clear submodules to improve modularity, maintainability, and usability.

---

## Data Hierarchy

The data model follows this hierarchy:

```
Event
├── Primary Vertices (PV)
│   └── track_ids → references to Tracks
├── Tracks
│   ├── hit_ids → references to Hits
│   └── pv_id → reference to parent PV
├── Hits
│   ├── track_id → back-reference to parent Track (-1 for ghosts)
│   └── module_id → reference to Module
└── Modules
    └── hit_ids → references to Hits on this module
```

**Key Design Decisions:**
- All cross-references use **IDs** (not object references) for JSON serialization
- **Segments are NOT stored** in the Event - they are computed on-demand in `solvers/reconstruction/`
- The entire Event is serializable to JSON for storage and retrieval

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
│   │   ├── hit.py                      # Hit dataclass (with track_id)
│   │   ├── track.py                    # Track dataclass (with hit_ids, pv_id)
│   │   ├── primary_vertex.py           # PrimaryVertex dataclass
│   │   ├── module.py                   # Module dataclass
│   │   └── event.py                    # Event container (JSON-serializable)
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
│       ├── segment.py                  # Segment class & generation functions
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

## JSON Serialization

All models support JSON serialization via `to_dict()` / `from_dict()` methods:

```python
# Save event to JSON
event.to_json("my_event.json")

# Load event from JSON (geometry provided separately)
loaded_event = Event.from_json("my_event.json", geometry)

# Individual model serialization
track_dict = track.to_dict()
track = Track.from_dict(track_dict)
```

---

## Segment Generation (On-Demand)

Segments are computed when needed, not stored:

```python
from lhcb_velo_toy.solvers.reconstruction import (
    Segment,
    get_segments_from_event,
    get_segments_from_track,
    get_candidate_segments,
)

# Get true segments from known tracks
segments = get_segments_from_event(event)

# Get all possible candidate segments for reconstruction
candidates = get_candidate_segments(event, max_delta_z=100)
```

---

## Development Flow

### Phase 1: Foundation
- [x] Create `src/lhcb_velo_toy/` directory structure
- [x] Create skeleton files with signatures and documentation
- [ ] Set up `pyproject.toml` with dependencies

### Phase 2: Generation Module
- [x] Implement `models/` dataclasses (Hit, Track, Module, Event, PrimaryVertex)
- [x] Implement JSON serialization for all models
- [ ] Implement `geometry/` classes (Geometry ABC, PlaneGeometry, RectangularVoidGeometry)
- [ ] Implement `generators/` (StateEventGenerator)

### Phase 3: Solvers Module
- [ ] Implement `hamiltonians/` (Hamiltonian ABC, SimpleHamiltonian, SimpleHamiltonianFast)
- [ ] Implement `classical/` solvers
- [ ] Implement `quantum/` algorithms (HHL, OneBitHHL)
- [x] Implement `reconstruction/` segment generation
- [ ] Implement `reconstruction/` track finder

### Phase 4: Analysis Module
- [ ] Implement `validation/` (Match, EventValidator)
- [ ] Implement `plotting/` utilities

### Phase 5: Integration & Testing
- [ ] Integration tests
- [ ] Example notebooks
- [ ] Release v2.0.0
