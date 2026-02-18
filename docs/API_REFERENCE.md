# LHCb VELO Toy Model - API Reference

> **Package Name**: `lhcb-velo-toy`  
> **Version**: 2.0.0  
> **License**: MIT

---

## Table of Contents

1. [Package Overview](#package-overview)
2. [Module: generation](#module-generation)
   - [Entities](#entities)
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
A, b = ham.construct_hamiltonian(event)

from lhcb_velo_toy.solvers.classical import solve_direct
solution = solve_direct(A, b)

tracks = get_tracks(ham, solution, event)
```

---

## Module: generation

### Entities

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
    extra: dict       # Arbitrary additional data (auto-captured from unknown keys)
```

**Properties:**

| Property | Returns | Description |
|----------|---------|-------------|
| `position` | `tuple[float, float, float]` | (x, y, z) coordinates |
| `is_ghost` | `bool` | True if track_id == -1 |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `__getitem__(index)` | `float` | Access coordinates: 0=x, 1=y, 2=z |
| `to_dict()` | `dict` | JSON-serializable dictionary |
| `from_dict(data)` | `Hit` | Create Hit from dictionary (unknown keys → `extra`) |

**Example:**
```python
hit = Hit(hit_id=0, x=1.5, y=-0.3, z=100.0, module_id=1, track_id=0)
print(hit[0])  # 1.5 (x coordinate)
print(hit[2])  # 100.0 (z coordinate)
```

---

#### `class Segment`

A track segment connecting two hits on adjacent detector layers.

> **Note:** Segments are NOT stored in Events. They are computed on-demand
> using functions in `solvers/reconstruction/segment.py`.

```python
@dataclass
class Segment:
    hit_start: Hit      # Starting hit (lower z)
    hit_end: Hit        # Ending hit (higher z)
    segment_id: int     # Unique identifier
    track_id: int       # Track this segment belongs to (-1 if unknown)
    pv_id: int          # Primary vertex ID (-1 if unknown)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `to_vect()` | `tuple[float, float, float]` | Direction vector (dx, dy, dz) |
| `__mul__(other)` | `float` | Cosine of angle between segments |
| `shares_hit_with(other)` | `bool` | Check if segments share a hit |
| `to_dict()` | `dict` | JSON-serializable dictionary |

**Mathematical Details:**

The `__mul__` operator computes the cosine of the angle between two segments:

$$\cos(\theta) = \frac{\vec{v}_1 \cdot \vec{v}_2}{|\vec{v}_1| \cdot |\vec{v}_2|}$$

where $\vec{v}_i$ is the direction vector of segment $i$.

**Example:**
```python
seg1 = Segment(hit_start=hit1, hit_end=hit2, segment_id=0)
seg2 = Segment(hit_start=hit2, hit_end=hit3, segment_id=1)

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
    pv_id: int                 # Primary vertex ID this track originates from
    hit_ids: list[int]         # IDs of hits on this track (ordered by z)
    extra: dict                # Arbitrary additional data (auto-captured from unknown keys)
```

**Properties:**

| Property | Returns | Description |
|----------|---------|-------------|
| `n_hits` | `int` | Number of hits on this track |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `add_hit_id(hit_id)` | `None` | Append a hit ID to this track |
| `to_dict()` | `dict` | JSON-serializable dictionary |
| `from_dict(data)` | `Track` | Create Track from dictionary (unknown keys → `extra`) |

**Example:**
```python
track = Track(
    track_id=0,
    pv_id=0,
    hit_ids=[0, 1, 2, 3, 4]
)
print(f"Track has {len(track.hit_ids)} hits")
```

---

#### `class Module`

A detector module (sensor plane) at a specific z position.

```python
@dataclass
class Module:
    module_id: int       # Unique identifier
    z: float             # Z position (mm)
    lx: float            # Half-width in x (mm)
    ly: float            # Half-width in y (mm)
    hit_ids: list[int]   # IDs of hits on this module
    extra: dict          # Arbitrary additional data (auto-captured from unknown keys)
```

**Properties:**

| Property | Returns | Description |
|----------|---------|-------------|
| `n_hits` | `int` | Number of hits on this module |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `add_hit_id(hit_id)` | `None` | Add a hit ID to this module |
| `clear_hits()` | `None` | Remove all hits from module |
| `contains_point(x, y)` | `bool` | Check if (x,y) is within module bounds |
| `to_dict()` | `dict` | JSON-serializable dictionary |
| `from_dict(data)` | `Module` | Create Module from dictionary (unknown keys → `extra`) |

---

#### `class Event`

A complete collision event container (JSON-serializable).

```python
@dataclass
class Event:
    detector_geometry: Geometry         # Detector configuration
    primary_vertices: list[PrimaryVertex]  # Collision points
    tracks: list[Track]                 # Particle tracks
    hits: list[Hit]                     # All hits
    modules: list[Module]               # Detector modules
    metadata: dict                      # Generation hyper-parameters (auto-populated)
```

> **Note:** Segments are NOT stored in Events. Use `get_segments_from_event(event)`
> from `solvers.reconstruction` to compute them on-demand.

**Data Hierarchy:**
```
Event
├── Primary Vertices (collision points)
│   └── track_ids → references to Tracks
├── Tracks
│   ├── hit_ids → references to Hits
│   └── pv_id → reference to parent PV
├── Hits
│   ├── track_id → back-reference to Track (-1 for ghosts)
│   └── module_id → reference to Module
└── Modules
    └── hit_ids → references to Hits
```

**Properties:**

| Property | Returns | Description |
|----------|---------|-------------|
| `n_primary_vertices` | `int` | Number of primary vertices |
| `n_tracks` | `int` | Number of tracks |
| `n_hits` | `int` | Number of hits |
| `n_modules` | `int` | Number of modules |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `dict` | Convert to JSON-serializable dictionary |
| `to_json(filepath, indent=2)` | `None` | Save event to JSON file |
| `from_dict(data, detector_geometry)` | `Event` | Create Event from dictionary |
| `from_json(filepath, detector_geometry)` | `Event` | Load event from JSON file |
| `from_tracks(geometry, tracks, hits)` | `Event` | Build reconstructed event (modules auto-derived, PVs empty) |
| `from_tracks(geometry, tracks, hits, *, metadata=...)` | `Event` | Build reconstructed event with explicit metadata |
| `get_hit_by_id(hit_id)` | `Hit or None` | Lookup hit by ID |
| `get_hits_by_ids(hit_ids)` | `list[Hit]` | Lookup multiple hits |
| `get_track_by_id(track_id)` | `Track or None` | Lookup track by ID |
| `get_hits_by_module(module_id)` | `list[Hit]` | Get all hits on a module |
| `get_hits_by_track(track_id)` | `list[Hit]` | Get all hits belonging to a track |
| `get_tracks_by_pv(pv_id)` | `list[Track]` | Get tracks from a primary vertex |
| `plot_event(title, show_ghosts, show_modules)` | `None` | Interactive 3D matplotlib visualisation |

---

#### `class PrimaryVertex`

A primary vertex (collision point) in the detector.

```python
@dataclass
class PrimaryVertex:
    pv_id: int              # Unique identifier
    x: float                # X position (mm)
    y: float                # Y position (mm)
    z: float                # Z position (mm)
    track_ids: list[int]    # IDs of tracks from this vertex
    extra: dict             # Arbitrary additional data (auto-captured from unknown keys)
```

**Properties:**

| Property | Returns | Description |
|----------|---------|-------------|
| `position` | `tuple[float, float, float]` | (x, y, z) vertex position |
| `n_tracks` | `int` | Number of associated tracks |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `add_track(track_id)` | `None` | Associate a track with this vertex |
| `to_dict()` | `dict` | JSON-serializable dictionary |
| `from_dict(data)` | `PrimaryVertex` | Create from dictionary (unknown keys → `extra`) |

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
        particles: list[list[dict]] = None,
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
| `measurement_error` | `float` | 0.0 | Hit position resolution σ (mm) — detector artefact only |
| `collision_noise` | `float` | 1e-4 | Multiple scattering σ (rad) — real physical deflection |

> **⚠️ Important — Measurement Error vs Multiple Scattering:**
>
> These two parameters model fundamentally different physics:
>
> | | `measurement_error` | `collision_noise` |
> |---|---|---|
> | **Nature** | Detector artefact | Real physical process |
> | **Affects** | Recorded Hit (x, y) only | True particle slopes (tx, ty) |
> | **Accumulates?** | No — each Hit is independently smeared | Yes — deflections compound through the detector |
> | **Feeds back?** | No — the true state is unchanged | Yes — trajectory is permanently altered |
>
> The correct processing order at each module is:
> 1. Propagate true state to module z → true (x, y)
> 2. Check acceptance — **not on bulk → skip entirely** (no hit, no scattering, no material). Particle continues to next module.
> 3. Record Hit at `(x + N(0, σ_meas), y + N(0, σ_meas))`
> 4. Apply scattering to true `tx += tan(N(0, σ_scatter))` (material interaction on bulk only)

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate_random_primary_vertices(variance)` | `dict[str, float]` | `list[tuple]` | Generate Gaussian-distributed vertices |
| `set_primary_vertices(vertices)` | `list[tuple]` | `None` | Set explicit vertex positions |
| `generate_particles(particles)` | `list[list[dict]]` | `list[list[dict]]` | Generate particle states |
| `propagate(state, z_target)` | `dict, float` | `dict` | Propagate true state to target z |
| `collision_update(state)` | `dict` | `dict` | Apply multiple scattering to true state |
| `measurement_error_update(x, y)` | `float, float` | `tuple[float, float]` | Smear hit coordinates (does **not** modify true state) |
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
true_event.plot_event(title="Generated Event")
```

---

## Module: solvers

### Hamiltonians

#### `class Hamiltonian` (ABC)

Abstract base class for track-finding Hamiltonians.

```python
class Hamiltonian(ABC):
    A: Optional[csc_matrix] = None    # Hamiltonian matrix (None before construction)
    b: Optional[np.ndarray] = None    # Bias vector (None before construction)
    segments: list[Segment]           # Constructed segments
    n_segments: int                   # Total segments

    @abstractmethod
    def construct_hamiltonian(
        self,
        event: Event | StateEventGenerator,
        convolution: bool = False,
    ) -> tuple[csc_matrix, np.ndarray]: ...

    @abstractmethod
    def evaluate(self, solution: np.ndarray) -> float: ...

    def solve_classicaly(self) -> np.ndarray: ...
    def get_matrix_dense(self) -> np.ndarray: ...
```

**Mathematical Formulation:**

The track-finding problem is formulated as minimizing:

$$H(\mathbf{x}) = -\frac{1}{2} \mathbf{x}^T A \mathbf{x} + \mathbf{b}^T \mathbf{x}$$

where $\mathbf{x}$ is the segment activation vector and the matrix $A$ encodes:

- **Diagonal**: $A_{ii} = -(\gamma + \delta)$ (self-interaction penalty)
- **Off-diagonal**: $A_{ij} = 1$ if segments $i, j$ share a hit and are angularly compatible
- **Bias**: $b_i = \delta$

**Concrete Methods (inherited by all subclasses):**

| Method | Returns | Description |
|--------|---------|-------------|
| `solve_classicaly()` | `np.ndarray` | Solve $Ax = b$ via conjugate gradient |
| `get_matrix_dense()` | `np.ndarray` | Return $A$ as a dense array |

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
| `construct_segments(event)` | `Event \| StateEventGenerator` | `None` | Build all segment candidates |
| `construct_hamiltonian(event, convolution=False)` | `Event \| StateEventGenerator, bool` | `(A, b)` | Build matrix and bias |
| `solve_classicaly()` | | `np.ndarray` | Solve via conjugate gradient |
| `evaluate(solution)` | `np.ndarray` | `float` | Compute Hamiltonian energy |
| `get_matrix_dense()` | | `np.ndarray` | Return A as dense array |

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

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `construct_segments(event)` | `Event \| StateEventGenerator` | `None` | Build segments with pre-computed direction vectors |
| `construct_hamiltonian(event, convolution=False)` | `Event \| StateEventGenerator, bool` | `(A, b)` | Build matrix using COO construction |
| `solve_classicaly()` | | `np.ndarray` | Solve with automatic solver selection |
| `evaluate(solution)` | `np.ndarray` | `float` | Compute Hamiltonian energy |
| `get_matrix_dense()` | | `np.ndarray` | Return A as dense array |

---

### Classical Solvers

#### `solve_conjugate_gradient(A, b, x0, tol, maxiter)`

Solve $Ax = b$ using conjugate gradient.

```python
def solve_conjugate_gradient(
    A: csc_matrix,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    maxiter: Optional[int] = None,
) -> tuple[np.ndarray, int]:
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A` | `csc_matrix` | Required | Sparse system matrix |
| `b` | `ndarray` | Required | Right-hand side vector |
| `x0` | `ndarray` | `None` | Initial guess (zeros if None) |
| `tol` | `float` | `1e-10` | Convergence tolerance |
| `maxiter` | `int` | `None` | Max iterations (default 10n) |

Returns `(solution, info)` where `info = 0` means convergence.

---

#### `solve_direct(A, b)`

Solve $Ax = b$ using direct LU factorisation.

```python
def solve_direct(A: csc_matrix, b: np.ndarray) -> np.ndarray
```

Faster for small/medium systems but memory-intensive.

---

#### `select_solver(A, b, threshold)`

Automatically choose between direct and iterative solvers.

```python
def select_solver(
    A: csc_matrix,
    b: np.ndarray,
    threshold: int = 5000,
) -> np.ndarray
```

Uses `solve_direct` when $n \le$ `threshold`, otherwise `solve_conjugate_gradient`.

---

#### `solve_classicaly()`

Solve the linear system $A\mathbf{x} = \mathbf{b}$ using conjugate gradient.

> **Note:** This is a convenience method on the `Hamiltonian` ABC, not a
> standalone function. See the standalone solvers above for direct use.

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

#### `class OneBitHHL`

1-Bit Quantum Filter using Suzuki-Trotter decomposition.

```python
class OneBitHHL:
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
| `get_success_probability()` | `float` | Post-selection success probability |

**Noise Model Execution:**
```python
hhl = OneBitHHL(A, b, num_time_qubits=1, shots=10000)
hhl.build_circuit()

# Noiseless simulation
counts_clean = hhl.run(use_noise_model=False)

# Noisy simulation using IBM backend noise model
counts_noisy = hhl.run(use_noise_model=True, backend_name='ibm_fez')

solution, success_count = hhl.get_solution(counts_noisy)
```

---

### Track Reconstruction

#### `find_segments(segment, active_segments)`

Find segments connected to a given segment.

```python
def find_segments(segment: Segment, active_segments: list[Segment]) -> list[Segment]:
    """
    Two segments are connected if they share an endpoint hit.
    
    Parameters
    ----------
    segment : Segment
        Reference segment
    active_segments : list[Segment]
        Pool of segments to search
    
    Returns
    -------
    list[Segment]
        Segments connected to the reference segment
    """
```

---

#### `get_tracks(hamiltonian, solution, event, threshold)`

Extract tracks from Hamiltonian solution.

```python
def get_tracks(
    hamiltonian: Hamiltonian,
    solution: np.ndarray,
    event: Union[Event, StateEventGenerator],
    threshold: float = 0.0,
) -> list[Track]:
    """
    Groups connected active segments into track candidates.
    
    Algorithm:
    1. Filter segments where activation > threshold
    2. Build adjacency graph of connected segments
    3. Find connected components via depth-first search
    4. Convert each component to a Track object
    5. Order hits within each track by z coordinate
    
    Returns list of reconstructed Track objects.
    """
```

---

#### `get_tracks_fast(hamiltonian, solution, event, threshold)`

Optimised track extraction using vectorized operations. Same interface
as `get_tracks`; delegates internally but may use pre-computed data
structures from `SimpleHamiltonianFast` in the future.

---

#### `construct_event(detector_geometry, tracks, hits)`

Construct a reconstructed Event from tracks and a hit pool. Modules are
derived automatically from the hits and geometry; primary vertices are
left empty (unknown after reconstruction). Convenience wrapper around
`Event.from_tracks`.

```python
def construct_event(
    detector_geometry: Geometry,
    tracks: list[Track],
    hits: list[Hit],
) -> Event:
    """Build a reconstructed event — modules auto-derived, PVs empty."""
```

---

#### Segment Generation Functions

Segments are computed on-demand using these functions from `solvers.reconstruction`:

```python
from lhcb_velo_toy.solvers.reconstruction import (
    Segment,
    get_segments_from_event,
    get_segments_from_track,
    get_candidate_segments,
)
```

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `get_segments_from_track(track, event)` | Track, Event | `list[Segment]` | Segments for one track (with track_id, pv_id) |
| `get_segments_from_event(event, include_ghosts=False)` | Event, bool | `list[Segment]` | All segments from all tracks |
| `get_candidate_segments(event, max_delta_z=None)` | Event, float (opt.) | `list[Segment]` | All possible candidate segments |
| `get_all_possible_segments(event, max_z_gap=1)` | Event, int | `list[Segment]` | All hit-to-hit segment candidates |

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
    hit_efficiency: float            # |R_i ∩ T_j| / |T_j|
    candidate: bool = True           # Passed min_rec_hits cut
    accepted: bool = False           # Passed purity/hit_efficiency cuts
    truth_id: Optional[int] = None   # Accepted truth ID
    is_clone: bool = False           # Marked as clone
```

**Properties:**

| Property | Returns | Description |
|----------|---------|-------------|
| `is_ghost` | `bool` | True if candidate but not accepted |
| `is_primary` | `bool` | True if accepted and not a clone |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `dict` | Convert to dictionary (for DataFrame construction) |

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
| `match_tracks(purity_min=0.7, hit_efficiency_min=0.0, ...)` | `tuple` | Match reco to truth tracks (non-greedy) |
| `summary_table()` | `pd.DataFrame` | Per-track summary |
| `truth_table()` | `pd.DataFrame` | Per-truth summary |

**Non-Greedy Matching Algorithm:**

When multiple reco tracks match the same truth track:
1. Compare match quality (e.g., purity × hit_efficiency or shared hits)
2. If new match is better, **replace** the existing assignment
3. Return displaced track to candidate pool for re-evaluation
4. Repeat until no more reassignments needed

This ensures globally optimal matching rather than first-come-first-served.

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

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `plot_event_3d(event, title, show_modules)` | Event, str, bool | `Figure` | 3D event visualisation with detector planes |
| `plot_segments_3d(event, segments, title)` | Event, list, str | `Figure` | 3D segment visualisation |
| `plot_efficiency_vs_parameter(params, values, ...)` | arrays | `Figure` | Efficiency vs parameter curve |
| `plot_ghost_rate_vs_parameter(params, values, ...)` | arrays | `Figure` | Ghost rate vs parameter curve |
| `plot_purity_distribution(purities, ...)` | array | `Figure` | Purity histogram |
| `plot_comparison(parameter_values, classical_metric, quantum_metric, ...)` | Sequence, Sequence, Sequence | `Figure` | Side-by-side solver comparison |
| `generate_performance_report(results_df, output_dir, prefix)` | DataFrame, str, str | `dict[str, str]` | Generate all standard plots and save to disk |
| `plot_hit_distribution(event, projection="xy", figsize=...)` | Event, str, tuple | `Figure` | Hit spatial distribution |
| `plot_reco_vs_truth(truth_event, reco_tracks, ...)` | Event, list[Track] | `Figure` | Side-by-side truth vs reco 3D comparison |
| `save_event_animation(event, filename, fps=30, duration=5.0)` | Event, str, int, float | `None` | Save rotating 3D animation |
| `set_lhcb_style()` | — | `None` | Apply LHCb publication style to matplotlib |

---

## Type Definitions

### Core Types

```python
from typing import TypeAlias, Protocol, runtime_checkable

# Type aliases
HitID = int
TrackID = int
ModuleID = int
SegmentID = int
PVID = int
StateVector = dict[str, float]  # {'x', 'y', 'z', 'tx', 'ty', 'p/q'}

Position = tuple[float, float, float]  # (x, y, z)
```

### Protocols

```python
@runtime_checkable
class SupportsPosition(Protocol):
    """Objects with position coordinates."""
    x: float
    y: float
    z: float

@runtime_checkable
class SupportsIteration(Protocol):
    """Objects supporting len() and iteration."""
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> tuple: ...
```

---

## Error Handling

### Common Exceptions

| Exception | Raised When | Resolution |
|-----------|-------------|------------|
| `ValueError("Not initialised")` | Calling `solve_classicaly()` before `construct_hamiltonian()` | Call `construct_hamiltonian()` first |
| `ImportError` | Missing optional dependency (e.g. Qiskit for quantum) | Install with `pip install -e ".[quantum]"` |
| `AssertionError` | Invalid primary vertex count | Ensure `len(vertices) == events` |

---

## See Also

- [FLOW_DIAGRAMS.md](FLOW_DIAGRAMS.md) - Data flow and architecture diagrams
- [DEPENDENCIES.md](DEPENDENCIES.md) - Dependencies and requirements
- [WORKFLOW_OVERVIEW.md](WORKFLOW_OVERVIEW.md) - End-to-end workflow guide
