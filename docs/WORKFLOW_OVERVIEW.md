# LHCb VELO Toy Model - Workflow Overview

A comprehensive guide to understanding the architecture, data flow, and usage patterns of the LHCb VELO Toy Model package.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Package Architecture](#package-architecture)
3. [Complete Workflow Pipeline](#complete-workflow-pipeline)
4. [Detailed Module Descriptions](#detailed-module-descriptions)
5. [Data Model Relationships](#data-model-relationships)
6. [Usage Examples](#usage-examples)

---

## Executive Summary

The **LHCb VELO Toy Model** is a simulation and reconstruction framework for particle tracking in the LHCb Vertex Locator (VELO) detector at CERN's Large Hadron Collider. The package provides:

| Component | Purpose |
|-----------|---------|
| **Event Generation** | Simulate particle collisions and detector responses |
| **Hamiltonian Solvers** | Formulate track reconstruction as an optimization problem |
| **Quantum Algorithms** | Explore HHL algorithm for linear system solving |
| **Validation Tools** | Measure reconstruction performance using LHCb metrics |

---

## Package Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         lhcb_velo_toy Package                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────┐    ┌─────────────────┐         │
│  │   🔵 GENERATION │───▶│   🟢 SOLVERS   │───▶│   🟡 ANALYSIS  │         │
│  │                 │     │                 │    │                 │         │
│  │ • Geometry      │     │ • Hamiltonians  │    │ • Validation    │         │
│  │ • Models        │     │ • Classical     │    │ • Plotting      │         │
│  │ • Generators    │     │ • Quantum       │    │                 │         │
│  │                 │     │ • Reconstruction│    │                 │         │
│  └─────────────────┘     └─────────────────┘    └─────────────────┘         │
│           │                     │                     │                     │
│           ▼                     ▼                     ▼                     │
│  ┌──────────────────────────────────────────────────────────────-───┐       │
│  │                       📦 core/types.py                          │       │
│  │    HitID, ModuleID, SegmentID, TrackID, Position, StateVector    │       │
│  └────────────────────────────────────────────────────────────────-─┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Three-Submodule Design

```mermaid
graph TB
    subgraph PKG["lhcb_velo_toy Package"]
        subgraph GEN["generation"]
            G1["models/"]
            G1a["Hit, Track, Module, Event, PrimaryVertex"]
            G2["geometry/"]
            G2a["PlaneGeometry, RectangularVoidGeometry"]
            G3["generators/"]
            G3a["StateEventGenerator"]
            G1 --- G1a
            G2 --- G2a
            G3 --- G3a
        end
        
        subgraph SOL["solvers"]
            S1["hamiltonians/"]
            S1a["SimpleHamiltonian, SimpleHamiltonianFast"]
            S2["classical/"]
            S2a["CG Solver, Direct Solver"]
            S3["quantum/"]
            S3a["HHL Algorithm, OneBitHHL"]
            S4["reconstruction/"]
            S4a["Segment, get_segments_from_event"]
            S1 --- S1a
            S2 --- S2a
            S3 --- S3a
            S4 --- S4a
        end
        
        subgraph ANA["analysis"]
            A1["validation/"]
            A1a["Match, EventValidator"]
            A2["plotting/"]
            A2a["Event Display, Performance Plots"]
            A1 --- A1a
            A2 --- A2a
        end
    end
    
    GEN --> SOL
    SOL --> ANA
    
    style GEN fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style SOL fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style ANA fill:#fff8e1,stroke:#fbc02d,stroke-width:2px
```

---

## Complete Workflow Pipeline

### High-Level Flow Diagram

```mermaid
flowchart LR
    subgraph INPUT["1. INPUT"]
        CONFIG["Configuration"]
    end
    
    subgraph GENERATION["2. GENERATION"]
        GEN2["StateEventGenerator"]
        EVT["Event"]
        NOISE["Add Noise"]
        GEN2 --> EVT --> NOISE
    end
    
    subgraph SOLVING["3. SOLVING"]
        HAM["Hamiltonian"]
        SOLVE{"Solver?"}
        CLASS["Classical CG"]
        QUANT["Quantum HHL"]
        SOLN["Solution"]
        HAM --> SOLVE
        SOLVE --> CLASS
        SOLVE --> QUANT
        CLASS --> SOLN
        QUANT --> SOLN
    end
    
    subgraph RECONSTRUCTION["4. RECONSTRUCTION"]
        THRESH["Threshold"]
        GROUP["Group Segments"]
        RECO["Reco Tracks"]
        THRESH --> GROUP --> RECO
    end
    
    subgraph VALIDATION["5. VALIDATION"]
        MATCH["Match Tracks"]
        METRICS["Metrics"]
        PLOTS["Plots"]
        MATCH --> METRICS --> PLOTS
    end
    
    CONFIG --> GEN2
    NOISE --> HAM
    SOLN --> THRESH
    RECO --> MATCH
    EVT -.->|truth| MATCH
    
    style INPUT fill:#f5f5f5,stroke:#9e9e9e
    style GENERATION fill:#e3f2fd,stroke:#1976d2
    style SOLVING fill:#e8f5e9,stroke:#388e3c
    style RECONSTRUCTION fill:#fff3e0,stroke:#f57c00
    style VALIDATION fill:#fff8e1,stroke:#fbc02d
```

### Detailed Step-by-Step Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE RECONSTRUCTION PIPELINE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: CONFIGURE DETECTOR GEOMETRY                                        │
│  ═══════════════════════════════════                                        │
│                                                                             │
│     PlaneGeometry                                                           │
│     ┌──────────────────────────────────────────────────────────────┐        │
│     │  Module 0    Module 1    Module 2   ...   Module N-1         │        │
│     │     │           │           │                 │              │        │
│     │  z=100mm    z=130mm     z=160mm    ...    z=z_max            │        │
│     │                                                              │        │
│     │  Each module: lx (half-width X), ly (half-width Y)           │        │
│     └──────────────────────────────────────────────────────────────┘        │
│                                                                             │
│                                  │                                          │
│                                  ▼                                          │
│                                                                             │
│  STEP 2: GENERATE PARTICLE EVENTS                                           │
│  ════════════════════════════════                                           │
│                                                                             │
│     StateEventGenerator                                                     │
│     ┌──────────────────────────────────────────────────────────────┐        │
│     │                                                              │        │
│     │   Primary Vertex (PV)                                        │        │
│     │         *                                                    │        │
│     │        /|\                                                   │        │
│     │       / | \                                                  │        │
│     │      /  |  \     ← Particles with momentum (tx, ty, p/q)     │        │
│     │     ●   ●   ●     ← Hits at each module crossing             │        │
│     │     │   │   │                                                │        │
│     │     ●   ●   ●     ← Measurement error applied                │        │
│     │     │   │   │                                                │        │
│     │     ●   ●   ●     ← Multiple scattering effects              │        │
│     │                                                              │        │
│     └──────────────────────────────────────────────────────────────┘        │
│                                                                             │
│     Output: Event containing:                                               │
│     • Truth tracks (T_1, T_2, ..., T_n)                                     │
│     • Hits on modules                                                       │
│     • Segments connecting adjacent hits                                     │
│                                                                             │
│                                  │                                          │
│                                  ▼                                          │
│                                                                             │
│  STEP 3: ADD REALISTIC NOISE (Optional)                                     │
│  ══════════════════════════════════════                                     │
│                                                                             │
│     make_noisy_event(drop_rate, ghost_rate)                                 │
│     ┌──────────────────────────────────────────────────────────────┐        │
│     │                                                              │        │
│     │   ● = Real hit                                               │        │
│     │   ○ = Dropped hit (random removal)                           │        │
│     │   ✕ = Ghost hit (random fake hit)                            │        │
│     │                                                              │        │
│     │     Before:  ●──●──●──●──●                                   │        │
│     │     After:   ●──○──●──●──●  +  ✕ ✕                           │        │
│     │                                                              │        │
│     └──────────────────────────────────────────────────────────────┘        │
│                                                                             │
│                                  │                                          │
│                                  ▼                                          │
│                                                                             │
│  STEP 4: CONSTRUCT HAMILTONIAN                                              │
│  ═════════════════════════════════                                          │
│                                                                             │
│     SimpleHamiltonian(ε, γ, δ)                                              │
│     ┌──────────────────────────────────────────────────────────────┐        │
│     │                                                              │        │
│     │   1. Build ALL possible segments between adjacent modules:   │        │
│     │                                                              │        │
│     │      Module i        Module i+1                              │        │
│     │         ●──────────────●                                     │        │
│     │         ●──────────────●     ← n_segs = hits_i × hits_{i+1}  │        │
│     │         ●──────────────●                                     │        │
│     │                                                              │        │
│     │   2. Build interaction matrix A (n_segs × n_segs):           │        │
│     │                                                              │        │
│     │      A_ij = 1  if segments i,j share endpoint AND            │        │
│     │                angle between them < ε (angular tolerance)    │        │
│     │                                                              │        │
│     │      A_ii = -(γ + δ)  (diagonal penalty)                     │        │
│     │                                                              │        │
│     │   3. Build bias vector b:                                    │        │
│     │                                                              │        │
│     │      b_i = γ + δ  (favor all segments equally)               │        │
│     │                                                              │        │
│     └──────────────────────────────────────────────────────────────┘        │
│                                                                             │
│                                  │                                          │
│                                  ▼                                          │
│                                                                             │
│  STEP 5: SOLVE LINEAR SYSTEM Ax = b                                         │
│  ══════════════════════════════════                                         │
│                                                                             │
│     ┌─────────────────────┬─────────────────────┐                           │
│     │   CLASSICAL         │   QUANTUM           │                           │
│     ├─────────────────────┼─────────────────────┤                           │
│     │                     │                     │                           │
│     │ scipy.sparse.linalg │  HHL Algorithm      │                           │
│     │ .cg(A, b)           │  ┌───────────────┐  │                           │
│     │                     │  │ |b⟩ state prep│  │                           │
│     │ Direct methods:     │  │ Phase Est.    │  │                           │
│     │ np.linalg.solve     │  │ Rotation      │  │                           │
│     │                     │  │ Uncompute     │  │                           │
│     │                     │  │ Measure       │  │                           │
│     │                     │  └───────────────┘  │                           │
│     │                     │                     │                           │
│     │                     │  OneBitHHL          │                           │
│     │                     │  (Simplified)       │                           │
│     │                     │                     │                           │
│     └─────────────────────┴─────────────────────┘                           │
│                                                                             │
│     Output: Solution vector x̄ where x̄_i ≈ 1 if segment i is real            │
│                                                                             │
│                                  │                                          │
│                                  ▼                                          │
│                                                                             │
│  STEP 6: RECONSTRUCT TRACKS                                                 │
│  ══════════════════════════                                                 │
│                                                                             │
│     get_tracks(hamiltonian, solution, event)                                │
│     ┌──────────────────────────────────────────────────────────────┐        │
│     │                                                              │        │
│     │   1. Apply threshold: keep segments where x̄_i > cutoff       │        │
│     │                                                              │        │
│     │   2. Group connected segments:                               │        │
│     │                                                              │        │
│     │      Segment A ──●── Segment B ──●── Segment C               │        │
│     │                  │                                           │        │
│     │                  └── shared endpoint → same track            │        │
│     │                                                              │        │
│     │   3. Output: List of reconstructed Track objects             │        │
│     │                                                              │        │
│     └──────────────────────────────────────────────────────────────┘        │
│                                                                             │
│                                  │                                          │
│                                  ▼                                          │
│                                                                             │
│  STEP 7: VALIDATE RECONSTRUCTION                                            │
│  ═══════════════════════════════                                            │
│                                                                             │
│     EventValidator(truth_event, reco_tracks)                                │
│     ┌──────────────────────────────────────────────────────────────┐        │
│     │                                                              │        │
│     │   For each reco track R_i, compare to each truth track T_j:  │        │
│     │                                                              │        │
│     │   • shared_hits = |R_i ∩ T_j|                                │        │
│     │   • purity = shared_hits / |R_i|                             │        │
│     │   • hit_efficiency = shared_hits / |T_j|                     │        │
│     │                                                              │        │
│     │   Classification:                                            │        │
│     │   ┌────────────────────────────────────────────────────┐     │        │
│     │   │ ACCEPTED: purity ≥ threshold                       │     │        │
│     │   │ GHOST:    purity < threshold (fake track)          │     │        │
│     │   │ CLONE:    same truth track matched multiple times  │     │        │
│     │   └────────────────────────────────────────────────────┘     │        │
│     │                                                              │        │
│     │   Metrics:                                                   │        │
│     │   • Efficiency = matched / reconstructible                   │        │
│     │   • Ghost Rate = ghosts / candidates                         │        │
│     │   • Clone Rate = clones / primaries                          │        │
│     │                                                              │        │
│     └──────────────────────────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Module Descriptions

### 🔵 Generation Module

The **generation** module creates simulated particle collision events that mimic real LHCb VELO detector data.

| Component | File | Description |
|-----------|------|-------------|
| **Hit** | `models/hit.py` | A measurement point (x, y, z) on a detector module |
| **Track** | `models/track.py` | A particle trajectory with `hit_ids` and `pv_id` |
| **PrimaryVertex** | `models/primary_vertex.py` | Collision point with associated `track_ids` |
| **Module** | `models/module.py` | A detector layer at fixed z position |
| **Event** | `models/event.py` | Container for all PVs, tracks, hits, modules (JSON-serializable) |
| **PlaneGeometry** | `geometry/plane.py` | Simple rectangular detector planes |
| **RectangularVoidGeometry** | `geometry/rectangular_void.py` | Planes with beam pipe hole |
| **StateEventGenerator** | `generators/state_event.py` | Main event simulation engine |

**Note:** Segments are NOT stored in Events. They are computed on-demand in `solvers/reconstruction/`.

#### Generation Module - Class Input/Output Diagram

```mermaid
flowchart LR
    subgraph GEOMETRY["Geometry Classes"]
        direction TB
        PG["PlaneGeometry"]
        RVG["RectangularVoidGeometry"]
    end
    
    subgraph GEO_IO["PlaneGeometry I/O"]
        direction TB
        PG_IN["INPUT:<br/>module_id: list[int]<br/>lx: list[float]<br/>ly: list[float]<br/>z: list[float]"]
        PG_OUT["OUTPUT:<br/>Geometry object<br/>with n_modules"]
    end
    
    PG_IN --> PG --> PG_OUT
    
    subgraph GENERATOR["StateEventGenerator"]
        direction TB
        SEG["StateEventGenerator"]
    end
    
    subgraph SEG_IO["StateEventGenerator I/O"]
        direction TB
        SEG_IN["INPUT:<br/>detector_geometry: Geometry<br/>events: int<br/>n_particles: list[int]<br/>measurement_error: float<br/>collision_noise: float"]
        SEG_OUT["OUTPUT:<br/>Event object with:<br/>- primary_vertices<br/>- tracks<br/>- hits<br/>- modules"]
    end
    
    SEG_IN --> SEG --> SEG_OUT
    PG_OUT -.-> SEG_IN
```

#### Data Model Class Diagram

```mermaid
classDiagram
    class Event {
        +Geometry detector_geometry
        +list~PrimaryVertex~ primary_vertices
        +list~Track~ tracks
        +list~Hit~ hits
        +list~Module~ modules
        +to_dict() dict
        +to_json(filepath)
        +from_dict(data, geometry) Event
        +from_json(filepath, geometry) Event
        +get_hit_by_id(hit_id) Hit
        +get_hits_by_ids(hit_ids) list~Hit~
        +get_track_by_id(track_id) Track
    }
    
    class PrimaryVertex {
        +int pv_id
        +float x
        +float y
        +float z
        +list~int~ track_ids
        +to_dict() dict
        +from_dict(data) PrimaryVertex
    }
    
    class Track {
        +int track_id
        +int pv_id
        +list~int~ hit_ids
        +to_dict() dict
        +from_dict(data) Track
    }
    
    class Hit {
        +int hit_id
        +float x
        +float y
        +float z
        +int module_id
        +int track_id
        +position tuple
        +is_ghost bool
        +to_dict() dict
        +from_dict(data) Hit
    }
    
    class Module {
        +int module_id
        +float z
        +float lx
        +float ly
        +list~int~ hit_ids
        +n_hits int
        +to_dict() dict
        +from_dict(data) Module
    }
    
    Event "1" *-- "*" PrimaryVertex
    Event "1" *-- "*" Track
    Event "1" *-- "*" Hit
    Event "1" *-- "*" Module
    PrimaryVertex "1" o-- "*" Track : track_ids
    Track "1" o-- "*" Hit : hit_ids
    Track "*" o-- "1" PrimaryVertex : pv_id
    Hit "*" o-- "1" Track : track_id
    Hit "*" o-- "1" Module : module_id
```

#### Event Generation Process

```mermaid
flowchart TD
    A[Initialize Generator] --> B[Define Detector Geometry]
    B --> C[Generate Primary Vertices]
    C --> D[Create Particles with Momentum]
    D --> E[Propagate Through Detector]
    
    E --> F{Hit Detector?}
    F -->|Yes| G[Record Hit Position]
    F -->|No| H[Skip Module]
    
    G --> I[Apply Measurement Error]
    I --> J[Apply Multiple Scattering]
    J --> K{More Modules?}
    
    H --> K
    K -->|Yes| E
    K -->|No| L{More Particles?}
    
    L -->|Yes| D
    L -->|No| M[Build Event Object]
    M --> N[Return Truth Event]
```

---

### 🟢 Solvers Module

The **solvers** module formulates track reconstruction as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem and provides multiple solution methods.

#### Hamiltonian Formulation

The track-finding problem is encoded in a Hamiltonian:

$$H = -\sum_{i,j} A_{ij} x_i x_j + \sum_i b_i x_i$$

Where:
- $x_i \in \{0, 1\}$ indicates if segment $i$ is part of a real track
- $A_{ij}$ encodes **compatibility** between segments (share endpoint + small angle)
- $b_i$ is a **bias term** favoring segment activation

| Component | File | Description |
|-----------|------|-------------|
| **Hamiltonian** | `hamiltonians/base.py` | Abstract base class |
| **SimpleHamiltonian** | `hamiltonians/simple.py` | Reference implementation |
| **SimpleHamiltonianFast** | `hamiltonians/fast.py` | Optimized sparse version |
| **Classical Solvers** | `classical/solvers.py` | Conjugate gradient, direct |
| **HHL** | `quantum/hhl.py` | Full HHL quantum algorithm |
| **OneBitHHL** | `quantum/one_bit_hhl.py` | Simplified 1-qubit phase est. |
| **Segment** | `reconstruction/segment.py` | Track segment (computed on-demand) |
| **get_segments_from_event** | `reconstruction/segment.py` | Generate segments from tracks |
| **get_tracks** | `reconstruction/track_finder.py` | Extract tracks from solution |

#### Solver Comparison

| Solver | Complexity | Pros | Cons |
|--------|------------|------|------|
| **Direct** | O(n³) | Exact solution | Memory intensive |
| **Conjugate Gradient** | O(n²) | Sparse-friendly | Requires SPD matrix |
| **HHL** | O(log n) | Exponential speedup* | Circuit depth, noise |
| **OneBitHHL** | O(log n) | Simpler circuit | Lower precision |

*Theoretical quantum advantage for well-conditioned systems

#### Solvers Module - Function Input/Output Diagram

```mermaid
flowchart TB
    subgraph SEGMENT_GEN["Segment Generation"]
        direction LR
        SG_IN["INPUT:<br/>event: Event"]
        SG_FN["get_segments_from_event()"]
        SG_OUT["OUTPUT:<br/>list[Segment]<br/>each with track_id, pv_id"]
        SG_IN --> SG_FN --> SG_OUT
    end
    
    subgraph HAM_CONSTRUCT["Hamiltonian Construction"]
        direction LR
        HC_IN["INPUT:<br/>event: Event<br/>epsilon: float<br/>gamma: float<br/>delta: float"]
        HC_FN["SimpleHamiltonian.<br/>construct_hamiltonian()"]
        HC_OUT["OUTPUT:<br/>A: sparse matrix<br/>b: vector"]
        HC_IN --> HC_FN --> HC_OUT
    end
    
    subgraph SOLVE["Solving"]
        direction LR
        SOL_IN["INPUT:<br/>A: matrix<br/>b: vector"]
        SOL_FN["solve_classicaly()<br/>OR HHL.solve()"]
        SOL_OUT["OUTPUT:<br/>x: solution vector<br/>(segment activations)"]
        SOL_IN --> SOL_FN --> SOL_OUT
    end
    
    subgraph TRACK_FIND["Track Finding"]
        direction LR
        TF_IN["INPUT:<br/>hamiltonian<br/>solution: ndarray<br/>event: Event<br/>threshold: float"]
        TF_FN["get_tracks()"]
        TF_OUT["OUTPUT:<br/>list[Track]<br/>(reconstructed)"]
        TF_IN --> TF_FN --> TF_OUT
    end
    
    SG_OUT --> HC_IN
    HC_OUT --> SOL_IN
    SOL_OUT --> TF_IN
```

#### Hamiltonian Class Diagram

```mermaid
classDiagram
    class Hamiltonian {
        <<abstract>>
        +float epsilon
        +float gamma
        +float delta
        +ndarray A
        +ndarray b
        +list~Segment~ segments
        +construct_hamiltonian(event)*
        +solve_classicaly()*
    }
    
    class SimpleHamiltonian {
        +construct_hamiltonian(event, convolution)
        +solve_classicaly()
        +get_segment_by_index(idx) Segment
    }
    
    class SimpleHamiltonianFast {
        +construct_hamiltonian(event, convolution)
        +solve_classicaly()
        -_build_sparse_matrix()
    }
    
    class HHL {
        +int n_qubits
        +QuantumCircuit circuit
        +solve(A, b) ndarray
        +build_circuit()
    }
    
    class OneBitHHL {
        +solve(A, b) ndarray
        +estimate_eigenvalue()
    }
    
    Hamiltonian <|-- SimpleHamiltonian
    Hamiltonian <|-- SimpleHamiltonianFast
    
    class Segment {
        +Hit hit_start
        +Hit hit_end
        +int segment_id
        +int track_id
        +int pv_id
        +to_vect() tuple
        +length() float
        +shares_hit_with(other) bool
    }
    
    SimpleHamiltonian "1" *-- "*" Segment
```

#### Quantum Solver Flow

```mermaid
flowchart TD
    subgraph HHL_ALGO["HHL Algorithm"]
        A["Input: A, b"] --> B["State Preparation<br/>|b⟩"]
        B --> C["Quantum Phase<br/>Estimation"]
        C --> D["Controlled<br/>Rotation"]
        D --> E["Inverse Phase<br/>Estimation"]
        E --> F["Measurement"]
        F --> G["Output: |x⟩"]
    end
    
    subgraph ONEBIT["OneBitHHL (Simplified)"]
        H["Input: A, b"] --> I["Single-qubit<br/>phase estimation"]
        I --> J["Approximate<br/>eigenvalue"]
        J --> K["Classical<br/>post-processing"]
        K --> L["Output: x"]
    end
```

---

### 🟡 Analysis Module

The **analysis** module evaluates reconstruction quality using standard LHCb tracking metrics.

| Component | File | Description |
|-----------|------|-------------|
| **Match** | `validation/match.py` | Single track-to-track match result |
| **EventValidator** | `validation/validator.py` | Full event validation engine |
| **Event Display** | `plotting/event_display.py` | 3D event visualization |
| **Performance Plots** | `plotting/performance.py` | Efficiency/ghost rate plots |

#### Validation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Purity** | `shared_hits / reco_hits` | Fraction of reco hits that are correct |
| **Hit Efficiency** | `shared_hits / truth_hits` | Fraction of truth hits that were found |
| **Efficiency** | `matched / reconstructible` | Track finding success rate |
| **Ghost Rate** | `ghosts / candidates` | Fake track rate |
| **Clone Rate** | `clones / primaries` | Duplicate track rate |

#### Analysis Module - Function Input/Output Diagram

```mermaid
flowchart TB
    subgraph VALIDATOR["EventValidator"]
        direction TB
        V_IN["INPUT:<br/>truth_event: Event<br/>rec_tracks: list[Track]<br/>reconstructible_filter: Callable"]
        V_INIT["EventValidator()"]
        V_IN --> V_INIT
    end
    
    subgraph MATCHING["match_tracks()"]
        direction TB
        M_IN["INPUT:<br/>purity_min: float = 0.7<br/>hit_efficiency_min: float = 0.0<br/>min_rec_hits: int = 3"]
        M_FN["match_tracks()"]
        M_OUT["OUTPUT:<br/>matches: list[Match]<br/>metrics: dict"]
        M_IN --> M_FN --> M_OUT
    end
    
    subgraph METRICS["Computed Metrics"]
        direction TB
        MET["efficiency: float<br/>ghost_rate: float<br/>clone_fraction: float<br/>mean_purity: float<br/>hit_efficiency: float<br/>n_candidates: int<br/>n_accepted: int<br/>n_ghosts: int<br/>n_clones: int"]
    end
    
    V_INIT --> M_FN
    M_OUT --> MET
```

#### Match Class Diagram

```mermaid
classDiagram
    class EventValidator {
        +Event truth_event
        +list~Track~ rec_tracks
        +list~Match~ matches
        +dict metrics
        +match_tracks(purity_min, hit_efficiency_min, min_rec_hits)
        +summary_table() DataFrame
        +truth_table() DataFrame
        +print_summary()
    }
    
    class Match {
        +int best_truth_id
        +int rec_hits
        +int truth_hits
        +int correct_hits
        +float purity
        +float hit_efficiency
        +bool candidate
        +bool accepted
        +int truth_id
        +bool is_clone
        +is_ghost bool
        +is_primary bool
    }
    
    EventValidator "1" *-- "*" Match
```

#### Validation Flow (Non-Greedy Matching)

```mermaid
flowchart TD
    A["Reco Tracks"] --> B{Min hits?}
    B -->|No| C["Reject"]
    B -->|Yes| D["Mark as Candidate"]
    
    D --> E["Find best truth match<br/>(most shared hits)"]
    E --> F["Compute purity<br/>shared/reco"]
    F --> G["Compute hit_efficiency<br/>shared/truth"]
    
    G --> H{purity >= min?}
    H -->|No| I["Mark as GHOST"]
    H -->|Yes| J{hit_eff >= min?}
    J -->|No| I
    J -->|Yes| K["ACCEPTED candidate"]
    
    K --> L{Same truth already<br/>matched?}
    L -->|No| M["Assign as PRIMARY"]
    L -->|Yes| N{New match better<br/>than existing?}
    
    N -->|Yes| O["Replace existing match"]
    O --> P["Return displaced track<br/>to candidate pool"]
    P --> E
    
    N -->|No| Q["Mark as CLONE"]
    
    I --> R["Aggregate Metrics"]
    M --> R
    Q --> R
    
    R --> S["efficiency = matched/reconstructible"]
    R --> T["ghost_rate = ghosts/candidates"]
    R --> U["clone_fraction = clones/accepted"]
```

**Non-Greedy Algorithm:**
1. For each candidate track, find the best-matching truth track
2. If the truth track is already matched to another reco track:
   - Compare match quality (e.g., by purity × hit_efficiency or shared hits)
   - If new match is better, **replace** the existing assignment
   - Return the displaced track to the candidate pool for re-evaluation
3. This ensures globally optimal matching, not first-come-first-served

#### Plotting Functions

```mermaid
flowchart LR
    subgraph EVENT_DISPLAY["event_display.py"]
        ED_IN["INPUT:<br/>event: Event<br/>show_tracks: bool<br/>show_hits: bool<br/>show_modules: bool"]
        ED_FN["plot_event_3d()"]
        ED_OUT["OUTPUT:<br/>matplotlib Figure<br/>3D visualization"]
        ED_IN --> ED_FN --> ED_OUT
    end
    
    subgraph PERFORMANCE["performance.py"]
        P_IN["INPUT:<br/>metrics: list[dict]<br/>labels: list[str]"]
        P_FN["plot_efficiency_vs_noise()<br/>plot_ghost_rate()<br/>plot_roc_curve()"]
        P_OUT["OUTPUT:<br/>matplotlib Figure<br/>performance plots"]
        P_IN --> P_FN --> P_OUT
    end
```

---

## Data Model Relationships

The data model uses **ID-based references** for JSON serialization:

```
Event (JSON-serializable)
├── Primary Vertices (list)
│   └── track_ids → references to Tracks
├── Tracks (list)
│   ├── hit_ids → references to Hits
│   └── pv_id → reference to parent PV
├── Hits (flat list)
│   ├── track_id → back-reference to parent Track (-1 for ghosts)
│   └── module_id → reference to Module
└── Modules (list)
    └── hit_ids → references to Hits on this module
```

**Segments are NOT stored in the Event.** They are computed on-demand:

```python
from lhcb_velo_toy.solvers.reconstruction import get_segments_from_event

# Generate segments only when needed (e.g., for Hamiltonian construction)
segments = get_segments_from_event(event)
```

```mermaid
erDiagram
    EVENT ||--o{ PRIMARY_VERTEX : contains
    EVENT ||--o{ TRACK : contains
    EVENT ||--o{ HIT : contains
    EVENT ||--o{ MODULE : contains
    EVENT ||--|| GEOMETRY : uses
    
    PRIMARY_VERTEX ||--o{ TRACK : "track_ids"
    TRACK ||--o{ HIT : "hit_ids"
    TRACK }o--|| PRIMARY_VERTEX : "pv_id"
    HIT }o--|| TRACK : "track_id"
    
    MODULE ||--o{ HIT : "hit_ids"
    
    MATCH ||--|| TRACK : "reco"
    MATCH }o--|| TRACK : "truth"
    
    EVENT {
        Geometry detector_geometry
        list primary_vertices
        list tracks
        list hits
        list modules
    }
    
    PRIMARY_VERTEX {
        int pv_id
        float x
        float y
        float z
        list track_ids
    }
    
    TRACK {
        int track_id
        int pv_id
        list hit_ids
    }
    
    HIT {
        int hit_id
        float x
        float y
        float z
        int module_id
        int track_id
    }
    
    MODULE {
        int module_id
        float z
        float lx
        float ly
        list hit_ids
    }
    
    MATCH {
        int reco_track_id
        int truth_track_id
        float purity
        float hit_efficiency
        bool is_ghost
        bool is_clone
    }
```

---

## Usage Examples

### Complete Workflow Example

```python
# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Import the package
# ═══════════════════════════════════════════════════════════════════════════
from lhcb_velo_toy import (
    # Generation
    PlaneGeometry, StateEventGenerator,
    # Solvers
    SimpleHamiltonian, get_tracks,
    # Analysis
    EventValidator,
)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Define detector geometry (10 modules along z-axis)
# ═══════════════════════════════════════════════════════════════════════════
geometry = PlaneGeometry(
    module_id=list(range(10)),
    lx=[50.0] * 10,              # Half-width in x (mm)
    ly=[50.0] * 10,              # Half-width in y (mm)
    z=[100 + i * 30 for i in range(10)]  # z positions (mm)
)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Create event generator
# ═══════════════════════════════════════════════════════════════════════════
particles = [[{'type': 'pion', 'mass': 139.6, 'q': 1}] * 5]  # 5 pions

generator = StateEventGenerator(
    detector_geometry=geometry,
    events=1,
    n_particles=[5],
    measurement_error=0.01,      # mm
    collision_noise=1e-3         # Multiple scattering
)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Generate event
# ═══════════════════════════════════════════════════════════════════════════
generator.generate_random_primary_vertices({'x': 0.01, 'y': 0.01, 'z': 50})
generator.generate_particles(particles)
truth_event = generator.generate_complete_events()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: (Optional) Add noise
# ═══════════════════════════════════════════════════════════════════════════
noisy_event = generator.make_noisy_event(
    event=truth_event,
    drop_rate=0.05,    # 5% hit inefficiency
    ghost_rate=0.02    # 2% ghost hit rate
)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Build Hamiltonian
# ═══════════════════════════════════════════════════════════════════════════
hamiltonian = SimpleHamiltonian(
    epsilon=0.01,    # Angular tolerance (radians)
    gamma=1.0,       # Self-interaction penalty
    delta=1.0        # Bias term
)

A, b = hamiltonian.construct_hamiltonian(generator, convolution=False)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Solve (classical)
# ═══════════════════════════════════════════════════════════════════════════
solution = hamiltonian.solve_classicaly()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: Extract reconstructed tracks
# ═══════════════════════════════════════════════════════════════════════════
reco_tracks = get_tracks(hamiltonian, solution, generator)
print(f"Reconstructed {len(reco_tracks)} tracks")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 9: Validate results
# ═══════════════════════════════════════════════════════════════════════════
validator = EventValidator(
    truth_event=truth_event,
    reco_tracks=reco_tracks
)

metrics = validator.match_tracks(purity_threshold=0.75)
print(f"Efficiency: {metrics['efficiency']:.2%}")
print(f"Ghost Rate: {metrics['ghost_rate']:.2%}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 10: Visualize
# ═══════════════════════════════════════════════════════════════════════════
truth_event.plot_segments()
```

---

## Key Design Principles

1. **Modularity**: Three independent submodules that can be used separately
2. **Type Safety**: Comprehensive type hints and dataclasses throughout
3. **Extensibility**: Abstract base classes for custom geometries, Hamiltonians
4. **LHCb Compatibility**: Metrics and conventions match official LHCb tracking
5. **Quantum-Ready**: Structured to support quantum algorithm exploration

---

## See Also

- [API_REFERENCE.md](API_REFERENCE.md) - Detailed class documentation
- [FLOW_DIAGRAMS.md](FLOW_DIAGRAMS.md) - Additional Mermaid diagrams
- [DEPENDENCIES.md](DEPENDENCIES.md) - Package dependencies

---

*Last updated: January 2026*
