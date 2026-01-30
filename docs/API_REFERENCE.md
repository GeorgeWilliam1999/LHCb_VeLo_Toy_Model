# LHCb VELO Toy Model - API Reference

> **Future Package Name**: `lhcb-velo-toy`  
> **Version**: 2.0.0 (proposed)  
> **License**: MIT

---

## Table of Contents

1. [Package Overview](#package-overview)
2. [Module: generation](#module-generation)
   - [Data Models](#data-models)
   - [Geometry Classes](#geometry-classes)
   - [Event Generators](#event-generators)
3. [Module: solvers](#module-solvers)
   - [Hamiltonians](#hamiltonians)
   - [Classical Solvers](#classical-solvers)
   - [Quantum Algorithms](#quantum-algorithms)
   - [Track Reconstruction](#track-reconstruction)
4. [Module: analysis](#module-analysis)
   - [Validation](#validation)
   - [Plotting](#plotting)

---

## Package Overview

The LHCb VELO Toy Model provides a complete framework for simulating particle tracking in the LHCb Vertex Locator detector and testing track reconstruction algorithms, including quantum approaches.

### Installation

```bash
# Basic installation
pip install lhcb-velo-toy

# With quantum algorithms support
pip install lhcb-velo-toy[quantum]

# Full installation with all dependencies
pip install lhcb-velo-toy[all]
```

### Quick Start

```python
from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
from lhcb_velo_toy.solvers import SimpleHamiltonian, get_tracks

# Define detector
geometry = PlaneGeometry(
    module_id=[1, 2, 3, 4, 5],
    lx=[33.0] * 5, ly=[33.0] * 5,
    z=[20.0, 40.0, 60.0, 80.0, 100.0]
)

# Generate events
gen = StateEventGenerator(geometry, events=1, n_particles=[5])
gen.generate_random_primary_vertices({"z": 1.0})
gen.generate_particles([...])
event = gen.generate_complete_events()

# Reconstruct tracks
ham = SimpleHamiltonian(epsilon=0.01, gamma=1.0, delta=1.0)
ham.construct_hamiltonian(event)
solution = ham.solve_classicaly()
tracks = get_tracks(ham, solution, event)
```

---

## Module: generation

### Data Models

#### `class Hit`

A single detector hit (measurement point).

```python
@dataclass
class Hit:
    hit_id: int       # Unique identifier
    x: float          # X coordinate (mm)
    y: float          # Y coordinate (mm)
    z: float          # Z coordinate (mm), along beam axis
    module_id: int    # Detector module ID
    track_id: int     # True particle track ID (-1 for ghosts)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `__getitem__(index)` | `float` | Access coordinates: 0=x, 1=y, 2=z |
| `__eq__(other)` | `bool` | Identity comparison (same object) |

**Example:**
```python
hit = Hit(hit_id=0, x=1.5, y=-0.3, z=100.0, module_id=1, track_id=0)
print(hit[0])  # 1.5 (x coordinate)
print(hit[2])  # 100.0 (z coordinate)
```

---

#### `class Segment`

A track segment connecting two hits on adjacent detector layers.

```python
@dataclass
class Segment:
    hits: list[Hit]     # [start_hit, end_hit]
    segment_id: int     # Unique identifier
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `to_vect()` | `tuple[float, float, float]` | Direction vector (dx, dy, dz) |
| `__mul__(other)` | `float` | Cosine of angle between segments |
| `__eq__(other)` | `bool` | Identity comparison |

**Mathematical Details:**

The `__mul__` operator computes the cosine of the angle between two segments:

$$\cos(\theta) = \frac{\vec{v}_1 \cdot \vec{v}_2}{|\vec{v}_1| \cdot |\vec{v}_2|}$$

where $\vec{v}_i$ is the direction vector of segment $i$.

**Example:**
```python
seg1 = Segment([hit1, hit2], segment_id=0)
seg2 = Segment([hit2, hit3], segment_id=1)

# Check angular compatibility
cos_angle = seg1 * seg2
if abs(cos_angle - 1.0) < epsilon:
    print("Segments are nearly collinear")
```

---

#### `class Track`

A particle track through the detector.

```python
@dataclass
class Track:
    track_id: int              # Unique identifier
    hits: list[Hit]            # Ordered hits (by z)
    segments: list[Segment]    # Connecting segments
```

**Example:**
```python
track = Track(
    track_id=0,
    hits=[hit1, hit2, hit3],
    segments=[seg1, seg2]
)
print(f"Track has {len(track.hits)} hits")
```

---

#### `class Module`

A detector module (sensor plane) at a specific z position.

```python
@dataclass
class Module:
    module_id: int      # Unique identifier
    z: float            # Z position (mm)
    lx: float           # Half-width in x (mm)
    ly: float           # Half-width in y (mm)
    hits: list[Hit]     # Hits on this module
```

---

#### `class Event`

A complete collision event container.

```python
@dataclass
class Event:
    detector_geometry: Geometry     # Detector configuration
    tracks: list[Track]             # Particle tracks
    hits: list[Hit]                 # All hits
    segments: list[Segment]         # All segments
    modules: list[Module]           # Detector modules
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `plot_segments()` | `None` | Interactive 3D visualization |
| `save_plot_segments(filename, params=None)` | `None` | Save visualization to file |

**Visualization:**
- **Red dots**: Hits belonging to segments
- **Blue lines**: Track segments
- **Green X**: Ghost hits (not in any segment)
- **Gray surfaces**: Detector planes

---

### Geometry Classes

#### `class Geometry` (ABC)

Abstract base class for detector geometry specifications.

```python
@dataclass(frozen=True)
class Geometry(ABC):
    module_id: list[int]

    @abstractmethod
    def __getitem__(self, index) -> tuple: ...

    @abstractmethod
    def point_on_bulk(self, state: dict) -> bool: ...

    def __len__(self) -> int: ...
```

**Abstract Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__getitem__(index)` | `int` | `tuple` | Get module geometry data |
| `point_on_bulk(state)` | `dict` | `bool` | Check if point is in active region |

---

#### `class PlaneGeometry(Geometry)`

Simple planar detector with rectangular active areas.

```python
@dataclass(frozen=True)
class PlaneGeometry(Geometry):
    module_id: list[int]    # Module IDs
    lx: list[float]         # Half-widths in x (mm)
    ly: list[float]         # Half-widths in y (mm)
    z: list[float]          # Z positions (mm)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `__getitem__(index)` | `(module_id, lx, ly, z)` | Get module geometry |
| `point_on_bulk(state)` | `bool` | True if point within any module's active area |

**Example:**
```python
geometry = PlaneGeometry(
    module_id=[1, 2, 3, 4, 5],
    lx=[50.0, 50.0, 50.0, 50.0, 50.0],
    ly=[50.0, 50.0, 50.0, 50.0, 50.0],
    z=[100.0, 130.0, 160.0, 190.0, 220.0]
)

# Iterate over modules
for mod_id, lx, ly, z in geometry:
    print(f"Module {mod_id} at z={z}mm")
```

---

#### `class RectangularVoidGeometry(Geometry)`

Detector geometry with a rectangular beam pipe void in the center.

```python
@dataclass(frozen=True)
class RectangularVoidGeometry(Geometry):
    module_id: list[int]
    z: list[float]
    void_x_boundary: list[float]    # Half-width of void in x
    void_y_boundary: list[float]    # Half-width of void in y
    lx: list[float]                 # Half-width of sensor in x
    ly: list[float]                 # Half-width of sensor in y
```

**Active Region:**

A point $(x, y)$ is on the bulk (active region) if:
- It is **inside** the outer boundaries: $|x| < l_x$ and $|y| < l_y$
- It is **outside** the void: $|x| > v_x$ or $|y| > v_y$

---

### Event Generators

#### `class StateEventGenerator`

Primary event generator using LHCb state vectors $(x, y, t_x, t_y, p/q)$.

```python
class StateEventGenerator:
    def __init__(
        self,
        detector_geometry: Geometry,
        primary_vertices: list[tuple[float, float, float]] = None,
        phi_min: float = -0.2,
        phi_max: float = 0.2,
        theta_min: float = -0.2,
        theta_max: float = 0.2,
        events: int = 3,
        n_particles: list[int] = None,
        particles: list[dict] = None,
        measurement_error: float = 0.0,
        collision_noise: float = 0.1e-3
    )
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detector_geometry` | `Geometry` | Required | Detector configuration |
| `primary_vertices` | `list[tuple]` | `None` | Pre-defined vertex positions |
| `phi_min`, `phi_max` | `float` | ±0.2 | Range for $t_x$ slope angle (rad) |
| `theta_min`, `theta_max` | `float` | ±0.2 | Range for $t_y$ slope angle (rad) |
| `events` | `int` | 3 | Number of collision events |
| `n_particles` | `list[int]` | `None` | Particles per event |
| `measurement_error` | `float` | 0.0 | Hit position resolution σ (mm) |
| `collision_noise` | `float` | 1e-4 | Multiple scattering σ (rad) |

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate_random_primary_vertices(variance)` | `dict[str, float]` | `list[tuple]` | Generate Gaussian-distributed vertices |
| `set_primary_vertices(vertices)` | `list[tuple]` | `None` | Set explicit vertex positions |
| `generate_particles(particles)` | `list[list[dict]]` | `list[dict]` | Generate particle states |
| `generate_complete_events()` | | `Event` | Propagate particles and record hits |
| `make_noisy_event(drop_rate, ghost_rate)` | `float, float` | `Event` | Add hit inefficiency and ghosts |

**Particle State Dictionary:**
```python
{
    'type': 'MIP',       # Particle type
    'x': 0.0,            # X position (mm)
    'y': 0.0,            # Y position (mm)
    'z': 0.0,            # Z position (mm)
    'tx': 0.01,          # X slope (dx/dz)
    'ty': -0.02,         # Y slope (dy/dz)
    'p/q': 5000.0        # Momentum/charge (MeV/c)
}
```

**Complete Example:**
```python
# Create geometry
geometry = PlaneGeometry(
    module_id=[1, 2, 3, 4, 5],
    lx=[33.0] * 5, ly=[33.0] * 5,
    z=[20.0, 40.0, 60.0, 80.0, 100.0]
)

# Initialize generator
gen = StateEventGenerator(
    detector_geometry=geometry,
    phi_max=0.02,
    theta_max=0.02,
    events=1,
    n_particles=[10],
    measurement_error=0.005,
    collision_noise=0.0001
)

# Generate vertices
gen.generate_random_primary_vertices({"x": 0.1, "y": 0.1, "z": 0.5})

# Define particles
particles = [[{"type": "MIP", "mass": 0.511, "q": 1} for _ in range(10)]]
gen.generate_particles(particles)

# Generate event
true_event = gen.generate_complete_events()

# Add noise
noisy_event = gen.make_noisy_event(drop_rate=0.1, ghost_rate=0.1)

# Visualize
true_event.plot_segments()
```

---

#### `class MultiScatteringGenerator`

Legacy event generator focused on multiple scattering physics.

```python
@dataclass
class MultiScatteringGenerator:
    detector_geometry: SimpleDetectorGeometry
    primary_vertices: list = field(default_factory=list)
    phi_min: float = 0.0
    phi_max: float = 2*np.pi
    theta_min: float = 0.0
    theta_max: float = np.pi/10
    rng: np.random.Generator = np.random.default_rng()
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate_random_primary_vertices(n_events, sigma)` | `int, tuple` | `list[tuple]` | Generate vertices |
| `generate_event(n_particles, n_events, sigma)` | `int, int, tuple` | `Event` | Generate events |

---

## Module: solvers

### Hamiltonians

#### `class Hamiltonian` (ABC)

Abstract base class for track-finding Hamiltonians.

```python
class Hamiltonian(ABC):
    A: scipy.sparse.csc_matrix    # Hamiltonian matrix
    b: np.ndarray                 # Bias vector
    segments: list[Segment]       # Constructed segments
    n_segments: int               # Total segments

    @abstractmethod
    def construct_hamiltonian(self, event: StateEventGenerator) -> tuple: ...

    @abstractmethod
    def evaluate(self, solution) -> float: ...
```

**Mathematical Formulation:**

The track-finding problem is formulated as minimizing:

$$H(\mathbf{x}) = -\frac{1}{2} \mathbf{x}^T A \mathbf{x} + \mathbf{b}^T \mathbf{x}$$

where $\mathbf{x}$ is the segment activation vector and the matrix $A$ encodes:

- **Diagonal**: $A_{ii} = -(\gamma + \delta)$ (self-interaction penalty)
- **Off-diagonal**: $A_{ij} = 1$ if segments $i, j$ share a hit and are angularly compatible

---

#### `class SimpleHamiltonian(Hamiltonian)`

Reference implementation of the track-finding Hamiltonian.

```python
class SimpleHamiltonian(Hamiltonian):
    def __init__(
        self,
        epsilon: float,      # Angular tolerance (radians)
        gamma: float,        # Self-interaction penalty
        delta: float,        # Bias term
        theta_d: float = 1e-4  # ERF smoothing width
    )
```

**Parameters:**

| Parameter | Type | Typical Value | Description |
|-----------|------|---------------|-------------|
| `epsilon` | `float` | 0.001 - 0.1 | Angular tolerance for segment compatibility |
| `gamma` | `float` | 1.0 - 2.0 | Penalty for activating segments |
| `delta` | `float` | 1.0 | Bias encouraging segment activation |
| `theta_d` | `float` | 1e-4 | Width for ERF-smoothed threshold |

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `construct_segments(event)` | `StateEventGenerator` | `None` | Build all segment candidates |
| `construct_hamiltonian(event, convolution=False)` | `StateEventGenerator, bool` | `(A, b)` | Build matrix and bias |
| `solve_classicaly()` | | `np.ndarray` | Solve via conjugate gradient |
| `evaluate(solution)` | `np.ndarray` | `float` | Compute Hamiltonian energy |

**Compatibility Function:**

With `convolution=False` (hard threshold):
$$A_{ij} = \begin{cases} 1 & \text{if } |\cos^{-1}(\vec{v}_i \cdot \vec{v}_j) - 1| < \epsilon \\ 0 & \text{otherwise} \end{cases}$$

With `convolution=True` (ERF-smoothed):
$$A_{ij} = 1 + \text{erf}\left(\frac{\epsilon - \theta_{ij}}{\theta_d \sqrt{2}}\right)$$

---

#### `class SimpleHamiltonianFast(Hamiltonian)`

Optimized implementation with vectorized operations.

```python
class SimpleHamiltonianFast(Hamiltonian):
    def __init__(
        self,
        epsilon: float,
        gamma: float,
        delta: float,
        theta_d: float = 1e-4
    )
```

**Optimizations:**
- COO sparse matrix construction (faster than LIL)
- Pre-computed normalized direction vectors
- Vectorized angle calculations
- Cached segment data structures
- Automatic solver selection (direct vs iterative)

**Performance:**
| Event Size (segments) | SimpleHamiltonian | SimpleHamiltonianFast | Speedup |
|-----------------------|-------------------|----------------------|---------|
| 100 | 0.05s | 0.02s | 2.5x |
| 1,000 | 0.8s | 0.15s | 5x |
| 10,000 | 45s | 3s | 15x |

---

#### `class SimpleHamiltonianCPPWrapper(Hamiltonian)`

C++/CUDA accelerated implementation wrapper.

```python
class SimpleHamiltonianCPPWrapper(Hamiltonian):
    def __init__(
        self,
        epsilon: float,
        gamma: float,
        delta: float,
        use_cuda: bool = False
    )
```

**Requirements:**
```bash
cd LHCB_Velo_Toy_Models/cpp_hamiltonian
pip install .
```

**Performance:**
| Event Size | Python | C++ CPU | CUDA GPU |
|------------|--------|---------|----------|
| 10,000 | 3s | 0.1s | 0.02s |
| 100,000 | 300s | 5s | 0.5s |

---

### Classical Solvers

#### `solve_classicaly()`

Solve the linear system $A\mathbf{x} = \mathbf{b}$ using conjugate gradient.

```python
def solve_classicaly(self) -> np.ndarray:
    """
    Returns segment activation vector.
    
    Uses scipy.sparse.linalg.cg with automatic tolerance.
    For small systems (<5000), tries direct solve first.
    """
```

---

### Quantum Algorithms

#### `class HHLAlgorithm`

Full Harrow-Hassidim-Lloyd algorithm for solving linear systems.

```python
class HHLAlgorithm:
    def __init__(
        self,
        matrix_A: np.ndarray,      # System matrix
        vector_b: np.ndarray,      # RHS vector
        num_time_qubits: int = 5,  # QPE precision qubits
        shots: int = 10240,        # Measurement shots
        debug: bool = False
    )
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `matrix_A` | `np.ndarray` | Required | System matrix (automatically padded to power of 2) |
| `vector_b` | `np.ndarray` | Required | RHS vector (normalized internally) |
| `num_time_qubits` | `int` | 5 | Number of qubits for phase estimation |
| `shots` | `int` | 10240 | Number of measurement shots |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `build_circuit()` | `QuantumCircuit` | Construct the HHL circuit |
| `run()` | `dict` | Execute on simulator |
| `get_solution()` | `np.ndarray` | Extract solution from measurements |
| `get_quantum_only_circuit()` | `QuantumCircuit` | Circuit without measurements |
| `simulate_statevector()` | `Statevector` | Exact statevector simulation |
| `extract_postselected_solution(sv)` | `np.ndarray` | Solution from statevector |

**Circuit Structure:**
1. State preparation: $|b\rangle$
2. Phase estimation: Extract eigenvalues to time register
3. Controlled rotation: Apply $R_y(\arcsin(C/\lambda))$ to ancilla
4. Uncompute phase estimation
5. Measure ancilla and system registers

---

#### `class OneBQF`

1-Bit Quantum Filter using Suzuki-Trotter decomposition.

```python
class OneBQF:
    def __init__(
        self,
        matrix_A: np.ndarray,
        vector_b: np.ndarray,
        num_time_qubits: int = 1,   # Single bit for phase
        shots: int = 1024,
        debug: bool = False
    )
```

**Key Features:**
- Single time qubit (1-bit phase estimation)
- First-order Suzuki-Trotter for time evolution
- Designed for track reconstruction at LHCb
- Smaller circuit depth than full HHL

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `build_circuit()` | `QuantumCircuit` | Construct 1-bit HHL circuit |
| `run(use_noise_model=False, backend_name='ibm_torino')` | `dict` | Execute with optional noise |
| `get_solution(counts=None)` | `tuple[np.ndarray, int]` | Solution vector and success count |

**Noise Model Execution:**
```python
hhl = OneBQF(A, b, num_time_qubits=1, shots=10000)
hhl.build_circuit()

# Noiseless simulation
counts_clean = hhl.run(use_noise_model=False)

# Noisy simulation using IBM backend noise model
counts_noisy = hhl.run(use_noise_model=True, backend_name='ibm_fez')

solution, success_count = hhl.get_solution(counts_noisy)
```

---

### Track Reconstruction

#### `find_segments(s0, active)`

Find segments connected to a given segment.

```python
def find_segments(s0: Segment, active: list[Segment]) -> list[Segment]:
    """
    Two segments are connected if they share an endpoint hit.
    
    Parameters
    ----------
    s0 : Segment
        Reference segment
    active : list[Segment]
        Pool of segments to search
    
    Returns
    -------
    list[Segment]
        Segments connected to s0
    """
```

---

#### `get_tracks(ham, classical_solution, event)`

Extract tracks from Hamiltonian solution.

```python
def get_tracks(
    ham: SimpleHamiltonian,
    classical_solution: np.ndarray,
    event: StateEventGenerator
) -> list[Track]:
    """
    Groups connected active segments into track candidates.
    
    Algorithm:
    1. Filter segments where activation > min value
    2. Grow tracks via depth-first search from arbitrary seed
    3. Convert segment chains to Track objects
    
    Returns list of reconstructed Track objects.
    """
```

---

#### `construct_event(detector_geometry, tracks, hits, segments, modules)`

Construct an Event object from components.

```python
def construct_event(
    detector_geometry: Geometry,
    tracks: list[Track],
    hits: list[Hit],
    segments: list[Segment],
    modules: list[Module]
) -> Event:
    """Factory function for creating Event objects."""
```

---

## Module: analysis

### Validation

#### `class Match`

Association summary for a reconstructed track.

```python
@dataclass
class Match:
    best_truth_id: Optional[int]     # Best-matching truth track
    rec_hits: int                    # |R_i| (reconstructed hits)
    truth_hits: int                  # |T_j| (truth hits)
    correct_hits: int                # |R_i ∩ T_j| (shared hits)
    purity: float                    # |R_i ∩ T_j| / |R_i|
    completeness: float              # |R_i ∩ T_j| / |T_j|
    candidate: bool = True           # Passed min_rec_hits cut
    accepted: bool = False           # Passed purity/completeness cuts
    truth_id: Optional[int] = None   # Accepted truth ID
    is_clone: bool = False           # Marked as clone
```

---

#### `class EventValidator`

LHCb-style event validator for track performance.

```python
class EventValidator:
    def __init__(
        self,
        truth_event: Event,
        rec_tracks: list[Track],
        reconstructible_filter: Optional[Callable[[Track], bool]] = None
    )
```

**Conventions:**

| Term | Definition |
|------|------------|
| **Candidate** | Reco track passing minimum hit count |
| **Accepted** | Candidate with purity ≥ threshold |
| **Ghost** | Candidate that failed acceptance |
| **Clone** | Accepted track matching same truth as another |
| **Primary** | Best accepted track per truth |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `match_tracks(purity_min=0.7, ...)` | `tuple` | Match reco to truth tracks |
| `summary_table()` | `pd.DataFrame` | Per-track summary |
| `truth_table()` | `pd.DataFrame` | Per-truth summary |

**Metrics:**

| Metric | Formula | Description |
|--------|---------|-------------|
| Reconstruction Efficiency | matched_truths / reconstructible_truths | How many true tracks found |
| Ghost Rate | ghosts / candidates | Fraction of fake tracks |
| Clone Fraction | clones / candidates | Duplicate track rate |
| Purity (mean) | mean(correct_hits / rec_hits) | Hit correctness |
| Hit Efficiency | mean(correct_hits / truth_hits) | Hit completeness |

---

### Plotting

#### Core Plotting Functions

```python
def plot_all(
    df: pd.DataFrame,
    out_prefix: str = "perf",
    ms_fixed: float = 2e-4,
    save_csv: bool = True
) -> dict:
    """
    Generate all standard LHCb-style plots.
    
    Returns dict with file paths.
    """
```

**Generated Plots:**

| Plot | X-axis | Y-axis | Description |
|------|--------|--------|-------------|
| Efficiency vs MS | Multiple scattering | Reco efficiency | Track finding performance |
| Ghost rate vs MS | Multiple scattering | Ghost rate | Fake track rate |
| Efficiency vs Drop | Hit inefficiency | Reco efficiency | Robustness to dropouts |
| Purity vs ε | Epsilon threshold | Mean purity | Parameter optimization |

---

## Type Definitions

### Core Types

```python
from typing import TypeVar, Protocol

# Type aliases
HitID = int
TrackID = int
ModuleID = int
SegmentID = int

Position = tuple[float, float, float]  # (x, y, z)
State = dict[str, float]  # {'x': ..., 'y': ..., 'z': ..., 'tx': ..., 'ty': ..., 'p/q': ...}

# Geometry iteration
GeometryItem = tuple[ModuleID, float, float, float]  # (module_id, lx, ly, z)
```

### Protocols

```python
class SupportsHitPosition(Protocol):
    """Objects with position coordinates."""
    x: float
    y: float
    z: float

class SupportsIteration(Protocol):
    """Objects supporting len() and iteration."""
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...
```

---

## Error Handling

### Common Exceptions

| Exception | Raised When | Resolution |
|-----------|-------------|------------|
| `ValueError("Not initialised")` | Calling `solve_classicaly()` before `construct_hamiltonian()` | Call `construct_hamiltonian()` first |
| `ImportError("C++ module not available")` | Using `SimpleHamiltonianCPPWrapper` without C++ build | Build C++ extension |
| `AssertionError` | Invalid primary vertex count | Ensure `len(vertices) == events` |

---

## Constants

### Physics Constants

```python
# Default detector parameters (VELO-like)
DEFAULT_DZ_MM = 33.0           # Layer spacing
DEFAULT_LAYERS = 5             # Number of layers
DEFAULT_LX_MM = 80.0           # Half-width in x
DEFAULT_LY_MM = 80.0           # Half-width in y

# Default algorithm parameters
DEFAULT_EPSILON = 1e-7         # Angular tolerance
DEFAULT_GAMMA = 2.0            # Self-interaction
DEFAULT_DELTA = 1.0            # Bias term
DEFAULT_THETA_D = 1e-4         # ERF smoothing
```

---

## See Also

- [RESTRUCTURING_PROPOSAL.md](RESTRUCTURING_PROPOSAL.md) - Package restructuring plan
- [FLOW_DIAGRAMS.md](FLOW_DIAGRAMS.md) - Data flow and architecture diagrams
- [DEPENDENCIES.md](DEPENDENCIES.md) - Dependencies and requirements
