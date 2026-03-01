# LHCb VELO Toy Model - Restructuring Proposal

## Executive Summary

This document proposes restructuring the `LHCb_VeLo_Toy_Model` repository into four clear submodules to improve modularity, maintainability, and usability.

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
│   ├── entities/                       # Core data structures
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
├── analysis/                           # SUBMODULE 3: Analysis & Validation
│   ├── __init__.py
│   ├── validation/                     # Reconstruction metrics
│   │   ├── __init__.py
│   │   ├── match.py                    # Match dataclass
│   │   └── validator.py                # EventValidator
│   └── plotting/                       # Visualization
│       ├── __init__.py
│       ├── event_display.py            # 3D event visualization
│       └── performance.py              # Efficiency/ghost rate plots
│
└── persistence/                        # SUBMODULE 4: Save / Load
    ├── __init__.py                     # Re-exports all public API
    ├── pipeline.py                     # PipelineResult, save/load pipeline, batch
    └── study.py                        # StudyResult, save/load parametric studies
```

---

## JSON Serialization

All models support JSON serialization via `to_dict()` / `from_dict()` methods.
Geometry classes (`PlaneGeometry`, `RectangularVoidGeometry`) now also support
`to_dict()` / `from_dict()`, with a `"geometry_class"` discriminator for
polymorphic deserialization via `geometry_from_dict()`.

**Event geometry embedding:** `Event.to_dict()` embeds the detector geometry
dictionary.  `Event.from_dict()` / `Event.from_json()` accept an optional
`detector_geometry` (default `None`) and auto-reconstruct geometry from the
embedded dict when not provided.

```python
# Save event to JSON (geometry is embedded automatically)
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
- [x] Set up `pyproject.toml` with dependencies

### Phase 2: Generation Module
- [x] Implement `entities/` dataclasses (Hit, Track, Module, Event, PrimaryVertex)
- [x] Implement JSON serialization for all entities
- [x] Implement `geometry/` classes (Geometry ABC, PlaneGeometry, RectangularVoidGeometry)
- [x] Implement `generators/` (EventGenerator ABC, StateEventGenerator)

### Phase 3: Solvers Module
- [x] Implement `hamiltonians/` (Hamiltonian ABC, SimpleHamiltonian, SimpleHamiltonianFast)
- [x] Implement `classical/` solvers
- [x] Implement `quantum/` algorithms (HHL, OneBitHHL/OneBQF)
- [x] Implement `reconstruction/` segment generation
- [x] Implement `reconstruction/` track finder

### Phase 4: Analysis Module
- [x] Implement `validation/` (Match, EventValidator)
- [x] Implement `plotting/` utilities

### Phase 5: Integration & Testing
- [x] Integration tests (3 end-to-end notebooks)
- [x] Example notebooks (classical, HHL, 1-BQF)
- [x] Release v2.0.0

### Phase 6: Persistence Module
- [x] Geometry `to_dict()` / `from_dict()` with `geometry_class` discriminator
- [x] `geometry_from_dict()` dispatch function
- [x] Event embed geometry in `to_dict()`, auto-reconstruct in `from_dict()`
- [x] `Match.from_dict()` classmethod
- [x] `persistence/pipeline.py` — `PipelineResult`, `save_pipeline`, `load_pipeline`, batch
- [x] `persistence/study.py` — `StudyResult`, `save_study`, `load_study`
- [x] Wire into package-level exports
