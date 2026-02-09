# LHCb VELO Toy Model - Flow Diagrams

This document contains Mermaid diagrams showing the data flow, architecture, and control flow of the LHCb VELO Toy Model package.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Data Flow: Event Generation to Validation](#data-flow-event-generation-to-validation)
3. [Class Hierarchy](#class-hierarchy)
4. [Module Dependency Graph](#module-dependency-graph)
5. [Event Generation Flow](#event-generation-flow)
6. [Hamiltonian Construction Flow](#hamiltonian-construction-flow)
7. [Quantum Algorithm Flow](#quantum-algorithm-flow)
8. [Validation Flow](#validation-flow)
9. [Sequence Diagrams](#sequence-diagrams)

---

## High-Level Architecture

```mermaid
graph TB
    subgraph generation["Generation Module"]
        GEO[Geometry Classes]
        EVG[Event Generators]
        DAT[Entities]
    end
    
    subgraph solvers["Solvers Module"]
        HAM[Hamiltonians]
        CLS[Classical Solvers]
        QUA[Quantum Algorithms]
    end
    
    subgraph analysis["Analysis Module"]
        VAL[Validators]
        PLT[Plotting]
    end
    
    GEO --> EVG
    DAT --> EVG
    EVG --> HAM
    HAM --> CLS
    HAM --> QUA
    CLS --> VAL
    QUA --> VAL
    VAL --> PLT
```

---

## Data Flow: Event Generation to Validation

```mermaid
flowchart LR
    subgraph Input
        G[Geometry]
        P[Particle Config]
    end
    
    subgraph Generation
        GEN[StateEventGenerator]
        EVT["Event: PVs, tracks, hits"]
        NOISE[make_noisy_event]
    end
    
    subgraph Solving
        HAM[SimpleHamiltonian]
        MAT["A matrix + b vector"]
        SOL{Solver}
        CG[Conjugate Gradient]
        HHL[HHL Algorithm]
        ONEBQF[1-Bit Filter]
        VEC[solution vector]
    end
    
    subgraph Reconstruction
        FILT[Threshold Filter]
        GRP[Segment Grouping]
        TRK[Reconstructed Tracks]
    end
    
    subgraph Validation
        VAL[EventValidator]
        MET["Metrics: Efficiency, Ghost Rate, Purity"]
        PLT[Plots]
    end
    
    G --> GEN
    P --> GEN
    GEN --> EVT
    EVT --> NOISE
    NOISE --> HAM
    EVT -->|true tracks| VAL
    
    HAM --> MAT
    MAT --> SOL
    SOL --> CG
    SOL --> HHL
    SOL --> ONEBQF
    CG --> VEC
    HHL --> VEC
    ONEBQF --> VEC
    
    VEC --> FILT
    FILT --> GRP
    GRP --> TRK
    
    TRK --> VAL
    VAL --> MET
    MET --> PLT
```

---

## Class Hierarchy

```mermaid
classDiagram
    class Geometry {
        <<abstract>>
        +module_id: list[int]
        +__getitem__(index) tuple
        +point_on_bulk(state) bool
        +__len__() int
    }
    
    class PlaneGeometry {
        +lx: list[float]
        +ly: list[float]
        +z: list[float]
        +__getitem__(index) tuple
        +point_on_bulk(state) bool
    }
    
    class RectangularVoidGeometry {
        +void_x_boundary: list[float]
        +void_y_boundary: list[float]
        +lx: list[float]
        +ly: list[float]
        +z: list[float]
        +__getitem__(index) tuple
        +point_on_bulk(state) bool
    }
    
    Geometry <|-- PlaneGeometry
    Geometry <|-- RectangularVoidGeometry
    
    class Hamiltonian {
        <<abstract>>
        +A: sparse.csc_matrix
        +b: ndarray
        +segments: list[Segment]
        +construct_hamiltonian(event)* tuple
        +evaluate(solution)* float
    }
    
    class SimpleHamiltonian {
        +epsilon: float
        +gamma: float
        +delta: float
        +theta_d: float
        +construct_segments(event)
        +construct_hamiltonian(event, convolution) tuple
        +solve_classicaly() ndarray
        +evaluate(solution) float
    }
    
    class SimpleHamiltonianFast {
        +_segment_vectors: ndarray
        +_segment_to_hit_ids: dict
        -_construct_coo(event)
    }
    
    class SimpleHamiltonianCPPWrapper {
        +use_cuda: bool
        -_cpp_hamiltonian: module
        +construct_hamiltonian(event) tuple
    }
    
    Hamiltonian <|-- SimpleHamiltonian
    Hamiltonian <|-- SimpleHamiltonianFast
    Hamiltonian <|-- SimpleHamiltonianCPPWrapper
    
    class Hit {
        +hit_id: int
        +x: float
        +y: float
        +z: float
        +module_id: int
        +track_id: int
        +position tuple
        +is_ghost bool
        +to_dict() dict
        +from_dict(data) Hit
    }
    
    class Segment {
        +hit_start: Hit
        +hit_end: Hit
        +segment_id: int
        +track_id: int
        +pv_id: int
        +to_vect() tuple
        +__mul__(other) float
        +shares_hit_with(other) bool
    }
    
    class Track {
        +track_id: int
        +pv_id: int
        +hit_ids: list[int]
        +to_dict() dict
        +from_dict(data) Track
    }
    
    class PrimaryVertex {
        +pv_id: int
        +x: float
        +y: float
        +z: float
        +track_ids: list[int]
        +to_dict() dict
    }
    
    class Module {
        +module_id: int
        +z: float
        +lx: float
        +ly: float
        +hit_ids: list[int]
        +to_dict() dict
    }
    
    class Event {
        +detector_geometry: Geometry
        +primary_vertices: list[PrimaryVertex]
        +tracks: list[Track]
        +hits: list[Hit]
        +modules: list[Module]
        +to_dict() dict
        +to_json(filepath)
        +from_dict(data, geometry) Event
        +from_json(filepath, geometry) Event
    }
    
    Segment --> Hit : connects 2
    Track o-- Hit : hit_ids
    Track o-- PrimaryVertex : pv_id
    PrimaryVertex o-- Track : track_ids
    Module o-- Hit : hit_ids
    Event *-- PrimaryVertex : contains many
    Event *-- Track : contains many
    Event *-- Hit : contains many
    Event *-- Module : contains many
    Event --> Geometry : uses
```

---

## Module Dependency Graph

```mermaid
graph TD
    subgraph External
        NP[numpy]
        SP[scipy]
        MPL[matplotlib]
        PD[pandas]
        QK[qiskit]
        AER[qiskit-aer]
        IBM[qiskit-ibm-runtime]
    end
    
    subgraph generation
        SEM[state_event_model.py]
        SEG[state_event_generator.py]
        MSG[multi_scattering_generator.py]
    end
    
    subgraph solvers
        HAM[hamiltonian.py]
        SH[simple_hamiltonian.py]
        SHF[simple_hamiltonian_fast.py]
        SHC[simple_hamiltonian_cpp.py]
        HHL[hhl_algorithm.py]
        OBQ[OneBQF.py]
    end
    
    subgraph analysis
        TV[toy_validator.py]
        LTP[lhcb_tracking_plots.py]
    end
    
    %% External dependencies
    NP --> SEM
    NP --> SEG
    NP --> MSG
    NP --> HAM
    NP --> SH
    NP --> SHF
    NP --> SHC
    NP --> TV
    NP --> LTP
    NP --> HHL
    NP --> OBQ
    
    SP --> SH
    SP --> SHF
    SP --> HAM
    
    MPL --> SEM
    MPL --> LTP
    
    PD --> TV
    PD --> LTP
    
    QK --> HHL
    QK --> OBQ
    AER --> HHL
    AER --> OBQ
    IBM --> OBQ
    
    %% Internal dependencies
    SEM --> SEG
    SEG --> SH
    SEG --> SHF
    SEG --> SHC
    HAM --> SH
    HAM --> SHF
    HAM --> SHC
    
    SEM --> TV
    SH --> TV
    TV --> LTP
    
    style generation fill:#e3f2fd
    style solvers fill:#e8f5e9
    style analysis fill:#fff8e1
    style External fill:#f3e5f5
```

---

## Event Generation Flow

```mermaid
flowchart TD
    START([Start]) --> INIT["Initialize StateEventGenerator with Geometry"]
    
    INIT --> VTX{Primary Vertices?}
    VTX -->|Generate| GVX["generate_random_primary_vertices: Gaussian distribution"]
    VTX -->|Provided| SVX[set_primary_vertices]
    GVX --> PAR
    SVX --> PAR
    
    PAR[Define particles per event] --> GPRT[generate_particles]
    
    GPRT --> LOOP[For each event]
    LOOP --> PV[Get primary vertex]
    PV --> PLOOP[For each particle]
    
    PLOOP --> STATE["Create state vector: x, y, z, tx, ty"]
    STATE --> PROP[propagate to first module]
    
    PROP --> MLOOP[For each module]
    MLOOP --> BULK{On bulk?}
    
    BULK -->|Yes| HIT[Create Hit]
    BULK -->|No| SKIP[Skip module]
    
    HIT --> MEAS[Apply measurement error]
    MEAS --> COLL[Apply multiple scattering]
    COLL --> NEXT{More modules?}
    
    SKIP --> NEXT
    NEXT -->|Yes| PROP2[propagate to next z]
    NEXT -->|No| NEXTP{More particles?}
    
    PROP2 --> MLOOP
    
    NEXTP -->|Yes| PLOOP
    NEXTP -->|No| NEXTE{More events?}
    
    NEXTE -->|Yes| LOOP
    NEXTE -->|No| BUILD[Build Event object]
    
    BUILD --> EVT["Event with: modules, hits, tracks"]
    EVT --> NOISE{Add noise?}
    
    NOISE -->|Yes| DROP[Apply drop_rate: Remove random hits]
    DROP --> GHOST[Apply ghost_rate: Add random ghost hits]
    GHOST --> REBUILD[rebuild modules]
    
    NOISE -->|No| DONE([Return Event])
    REBUILD --> DONE
```

---

## Hamiltonian Construction Flow

```mermaid
flowchart TD
    START([Event]) --> SEGS[construct_segments]
    
    SEGS --> PAIR["For each pair of adjacent modules"]
    PAIR --> H1["For each hit h1 in module i"]
    H1 --> H2["For each hit h2 in module i+1"]
    H2 --> CREATE["Create Segment: h1 to h2"]
    CREATE --> NEXT{More pairs?}
    NEXT -->|Yes| PAIR
    NEXT -->|No| NSEG[n_segments = total segments]
    
    NSEG --> INIT["Initialize: A = sparse matrix, b = zeros"]
    
    INIT --> DIAG["Set diagonal: A_ii = -(gamma + delta)"]
    
    DIAG --> OFF[For each segment pair]
    OFF --> SHARE{Share endpoint?}
    
    SHARE -->|No| NEXTOFF
    SHARE -->|Yes| ANGLE["Compute cos angle"]
    
    ANGLE --> CONV{convolution?}
    
    CONV -->|No| HARD{angle less than epsilon?}
    HARD -->|Yes| SET1["A_ij = 1"]
    HARD -->|No| SET0["A_ij = 0"]
    
    CONV -->|Yes| ERF["ERF smoothed A_ij"]
    
    SET1 --> NEXTOFF{More pairs?}
    SET0 --> NEXTOFF
    ERF --> NEXTOFF
    
    NEXTOFF -->|Yes| OFF
    NEXTOFF -->|No| BIAS["Set bias vector: b_i = gamma + delta"]
    
    BIAS --> RET(["Return A, b"])
```

---

## Quantum Algorithm Flow

### HHL Algorithm

```mermaid
flowchart TD
    subgraph Input
        A[Matrix A]
        B[Vector b]
    end
    
    subgraph CircuitConstruction["Circuit Construction"]
        PAD[Pad to power of 2]
        NORM["Normalize b"]
        
        subgraph StatePrep["State Prep"]
            SPREP["Amplitude encoding"]
        end
        
        subgraph PhaseEstimation["Phase Estimation"]
            HGATE[Hadamard on time register]
            CEVO[Controlled evolution]
            IQFT[Inverse QFT]
        end
        
        subgraph Rotation
            ANCILLA[Ancilla qubit]
            CROT[Controlled rotation]
        end
        
        subgraph Uncompute
            QFT2[QFT]
            CEVO2[Controlled inverse evolution]
            HGATE2[Hadamard gates]
        end
    end
    
    subgraph Measurement
        MEAS[Measure ancilla + system]
        POST[Post-select ancilla = 1]
        EXTRACT[Extract solution from counts]
    end
    
    A --> PAD
    B --> NORM
    PAD --> CEVO
    NORM --> SPREP
    SPREP --> HGATE
    HGATE --> CEVO
    CEVO --> IQFT
    IQFT --> CROT
    ANCILLA --> CROT
    CROT --> QFT2
    QFT2 --> CEVO2
    CEVO2 --> HGATE2
    HGATE2 --> MEAS
    MEAS --> POST
    POST --> EXTRACT
    EXTRACT --> SOL[Solution vector]
```

### 1-Bit Quantum Filter (OneBQF)

```mermaid
flowchart TD
    subgraph Input
        A[Matrix A]
        B[Vector b]
    end
    
    subgraph SuzukiTrotter["Suzuki-Trotter"]
        DEC[Decompose A into Pauli terms]
        TROT[Build time evolution operator]
    end
    
    subgraph OneBitPhaseEst["1-Bit Phase Estimation"]
        SPREP[State prep]
        H1[Hadamard on time qubit]
        CTRL[Controlled U_A]
        H2[Hadamard on time qubit]
    end
    
    subgraph RotationMeasure["Rotation and Measure"]
        CROT[Rotation on ancilla]
        MEAS[Measure all]
        POST[Post-select ancilla = 1]
    end
    
    A --> DEC
    DEC --> TROT
    B --> SPREP
    TROT --> CTRL
    SPREP --> H1
    H1 --> CTRL
    CTRL --> H2
    H2 --> CROT
    CROT --> MEAS
    MEAS --> POST
    POST --> SOL[Solution vector]
```

---

## Validation Flow (Non-Greedy Matching)

```mermaid
flowchart TD
    TRUE[Truth Event] --> FILT{Reconstructible filter?}
    FILT -->|Yes| RECON[Filter truth tracks]
    FILT -->|No| PASS[Use all truth tracks]
    
    RECO[Reco Tracks] --> CAND[For each reco track]
    CAND --> NHIT{n_hits >= min?}
    NHIT -->|No| REJECT[Reject]
    NHIT -->|Yes| CANDOK[Mark CANDIDATE]
    
    RECON --> STARTMATCH
    PASS --> STARTMATCH
    CANDOK --> STARTMATCH[Start Matching]
    
    STARTMATCH --> LOOP[For each candidate]
    LOOP --> ASSOC[Find truth with most shared hits]
    ASSOC --> CALC[Compute purity and hit_efficiency]
    
    CALC --> PURE{purity >= thresh?}
    PURE -->|No| GHOST[Mark GHOST]
    PURE -->|Yes| ACCEPT[ACCEPTED]
    
    ACCEPT --> ALREADY{Truth already matched?}
    ALREADY -->|No| PRIMARY[Mark PRIMARY]
    ALREADY -->|Yes| BETTER{New match better?}
    BETTER -->|Yes| REPLACE[Replace existing match]
    BETTER -->|No| CLONE[Mark CLONE]
    REPLACE --> REEVAL[Re-evaluate displaced track]
    REEVAL --> LOOP
    
    GHOST --> NEXTTRACK{More candidates?}
    PRIMARY --> NEXTTRACK
    CLONE --> NEXTTRACK
    NEXTTRACK -->|Yes| LOOP
    NEXTTRACK -->|No| METRICS[Compute Metrics]
    REJECT --> NEXTTRACK
    
    METRICS --> EFF[Efficiency]
    METRICS --> GR[Ghost Rate]
    METRICS --> CR[Clone Rate]
```

**Non-Greedy Algorithm:**
- When a truth track is already matched, compare match quality
- If new match is better, replace existing and re-evaluate displaced track
- This ensures globally optimal matching, not first-come-first-served

---

## Sequence Diagrams

### Full Reconstruction Pipeline

```mermaid
sequenceDiagram
    participant User
    participant Gen as StateEventGenerator
    participant Ham as SimpleHamiltonian
    participant Sol as Solver
    participant Val as EventValidator
    participant Plot as Plotter
    
    User->>Gen: Create with Geometry
    User->>Gen: generate_random_primary_vertices()
    User->>Gen: generate_particles([...])
    Gen-->>User: particles configured
    
    User->>Gen: generate_complete_events()
    Gen->>Gen: propagate through modules
    Gen->>Gen: record hits
    Gen-->>User: truth_event
    
    User->>Gen: make_noisy_event(drop, ghost)
    Gen->>Gen: remove random hits
    Gen->>Gen: add ghost hits
    Gen-->>User: noisy_event
    
    User->>Ham: Create with epsilon, gamma, delta
    User->>Ham: construct_hamiltonian(noisy_event)
    Ham->>Ham: construct_segments()
    Ham->>Ham: build A matrix
    Ham->>Ham: build b vector
    Ham-->>User: (A, b)
    
    User->>Sol: solve(A, b)
    alt Classical
        Sol->>Sol: scipy.sparse.linalg.cg()
    else Quantum HHL
        Sol->>Sol: build_circuit()
        Sol->>Sol: run simulation
        Sol->>Sol: post-select & extract
    end
    Sol-->>User: solution x
    
    User->>Ham: get_tracks(solution, event)
    Ham->>Ham: threshold segments
    Ham->>Ham: group connected
    Ham-->>User: reco_tracks
    
    User->>Val: Create(truth_event, reco_tracks)
    User->>Val: match_tracks(purity_min, hit_efficiency_min)
    Val->>Val: compute overlaps
    Val->>Val: classify tracks
    Val-->>User: (matches, metrics)
    
    User->>Plot: generate_plots(metrics)
    Plot-->>User: figures
```

### Quantum Execution with Noise

```mermaid
sequenceDiagram
    participant User
    participant HHL as HHLAlgorithm
    participant Circ as QuantumCircuit
    participant Sim as AerSimulator
    participant IBM as IBMRuntime
    
    User->>HHL: Create(A, b, n_qubits)
    User->>HHL: build_circuit()
    HHL->>Circ: create registers
    HHL->>Circ: state preparation
    HHL->>Circ: phase estimation
    HHL->>Circ: controlled rotation
    HHL->>Circ: uncompute
    HHL->>Circ: measurements
    Circ-->>HHL: circuit ready
    
    alt Noiseless Simulation
        User->>HHL: run(use_noise=False)
        HHL->>Sim: execute(circuit, shots)
        Sim-->>HHL: counts
    else Noisy Simulation
        User->>HHL: run(use_noise=True, backend='ibm_fez')
        HHL->>IBM: get_backend('ibm_fez')
        IBM-->>HHL: backend
        HHL->>HHL: NoiseModel.from_backend()
        HHL->>Sim: execute with noise_model
        Sim-->>HHL: counts
    end
    
    User->>HHL: get_solution(counts)
    HHL->>HHL: filter ancilla=1
    HHL->>HHL: aggregate system qubits
    HHL->>HHL: normalize
    HHL-->>User: solution_vector
```

---

## State Machine: Track Matching (Non-Greedy)

```mermaid
stateDiagram-v2
    [*] --> Candidate: n_hits >= min
    [*] --> Rejected: n_hits < min
    
    Candidate --> Accepted: purity >= threshold
    Candidate --> Ghost: purity < threshold
    
    Accepted --> CheckExisting: truth matched?
    CheckExisting --> Primary: no existing
    CheckExisting --> CompareQuality: existing found
    
    CompareQuality --> Primary: new is better
    CompareQuality --> Clone: existing is better
    
    Primary --> DisplaceOld: replaced existing
    DisplaceOld --> Candidate: re-evaluate
    
    Ghost --> [*]
    Primary --> [*]
    Clone --> [*]
    Rejected --> [*]
```

---

## Data Structures Summary

```mermaid
erDiagram
    Event ||--o{ PrimaryVertex : contains
    Event ||--o{ Track : contains
    Event ||--o{ Hit : contains
    Event ||--o{ Module : contains
    Event ||--|| Geometry : uses
    
    PrimaryVertex ||--o{ Track : "track_ids"
    Track ||--o{ Hit : "hit_ids"
    Track }o--|| PrimaryVertex : "pv_id"
    Hit }o--|| Track : "track_id"
    
    Module ||--o{ Hit : "hit_ids"
    
    Match ||--|| Track : "reco track"
    Match }o--|| Track : "truth track"
    
    Event {
        Geometry detector_geometry
        list primary_vertices
        list tracks
        list hits
        list modules
    }
    
    PrimaryVertex {
        int pv_id
        float x
        float y
        float z
        list track_ids
    }
    
    Track {
        int track_id
        int pv_id
        list hit_ids
    }
    
    Hit {
        int hit_id
        float x
        float y
        float z
        int module_id
        int track_id
    }
    
    Module {
        int module_id
        float z
        float lx
        float ly
        list hit_ids
    }
    
    Match {
        int best_truth_id
        int rec_hits
        int truth_hits
        int correct_hits
        float purity
        float hit_efficiency
        bool accepted
        bool is_clone
    }
```

> **Note:** Segments are NOT stored in Events. They are computed on-demand
> using `get_segments_from_event()` from `solvers.reconstruction`.

---

## See Also

- [API_REFERENCE.md](API_REFERENCE.md) - Detailed class and method documentation
- [DEPENDENCIES.md](DEPENDENCIES.md) - Package dependencies
- [RESTRUCTURING_PROPOSAL.md](../RESTRUCTURING_PROPOSAL.md) - Package restructuring plan
