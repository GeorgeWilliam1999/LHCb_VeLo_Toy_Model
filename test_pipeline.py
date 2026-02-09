"""End-to-end pipeline test for the LHCb VELO Toy Model."""
import sys
sys.path.insert(0, "src")

import numpy as np

# ===== STEP 1: Geometry =====
from lhcb_velo_toy.generation.geometry.plane import PlaneGeometry

geo = PlaneGeometry(
    module_id=list(range(10)),
    lx=[50.0] * 10,
    ly=[50.0] * 10,
    z=[100 + i * 30 for i in range(10)],
)
print(f"[OK] Geometry: {len(geo)} modules, z = {geo.get_z_positions()}")

# ===== STEP 2: Event generation =====
from lhcb_velo_toy.generation.generators.state_event import StateEventGenerator

particles = [[{"type": "pion", "mass": 139.6, "q": 1}] * 5]  # 1 event, 5 pions
gen = StateEventGenerator(
    detector_geometry=geo,
    events=1,
    n_particles=[5],
    measurement_error=0.01,
    collision_noise=1e-3,
)
gen.generate_random_primary_vertices({"x": 0.01, "y": 0.01, "z": 50})
gen.generate_particles(particles)
event = gen.generate_complete_events()
print(
    f"[OK] Event: {len(event.tracks)} tracks, "
    f"{len(event.hits)} hits, {len(event.modules)} modules"
)

# ===== STEP 3: Hamiltonian construction =====
from lhcb_velo_toy.solvers.hamiltonians.simple import SimpleHamiltonian

ham = SimpleHamiltonian(epsilon=0.01, gamma=1.5, delta=1.0)
A, b = ham.construct_hamiltonian(gen, convolution=False)
print(f"[OK] SimpleHamiltonian: {ham.n_segments} segments, A shape {A.shape}")

# ===== STEP 4: Classical solve =====
solution = ham.solve_classicaly()
print(
    f"[OK] Classical solve: solution min={solution.min():.4f}, "
    f"max={solution.max():.4f}"
)

# ===== STEP 5: Evaluate =====
energy = ham.evaluate(solution)
print(f"[OK] Hamiltonian energy: {energy:.4f}")

# ===== STEP 6: Track extraction =====
from lhcb_velo_toy.solvers.reconstruction.track_finder import get_tracks

reco_tracks = get_tracks(ham, solution, gen, threshold=np.min(solution))
print(f"[OK] Reconstructed: {len(reco_tracks)} tracks")

# ===== STEP 7: Fast Hamiltonian =====
from lhcb_velo_toy.solvers.hamiltonians.fast import SimpleHamiltonianFast

ham_fast = SimpleHamiltonianFast(epsilon=0.01, gamma=1.5, delta=1.0)
Af, bf = ham_fast.construct_hamiltonian(gen, convolution=False)
sol_fast = ham_fast.solve_classicaly()
print(
    f"[OK] SimpleHamiltonianFast: {ham_fast.n_segments} segs, "
    f"solution max={sol_fast.max():.4f}"
)

# ===== STEP 8: Classical solver functions =====
from lhcb_velo_toy.solvers.classical.solvers import (
    solve_conjugate_gradient,
    solve_direct,
    select_solver,
)

sol_cg, info = solve_conjugate_gradient(A, b)
sol_d = solve_direct(A, b)
sol_auto = select_solver(A, b)
print(
    f"[OK] Standalone solvers: CG info={info}, "
    f"direct max={sol_d.max():.4f}, auto max={sol_auto.max():.4f}"
)

# ===== STEP 9: Convolution mode =====
ham2 = SimpleHamiltonian(epsilon=0.01, gamma=1.5, delta=1.0)
A2, b2 = ham2.construct_hamiltonian(gen, convolution=True)
sol2 = ham2.solve_classicaly()
print(
    f"[OK] Convolution mode: {ham2.n_segments} segs, "
    f"solution max={sol2.max():.4f}"
)

# ===== STEP 10: Noisy event =====
noisy = gen.make_noisy_event(drop_rate=0.1, ghost_rate=0.05)
print(
    f"[OK] Noisy event: {len(noisy.hits)} hits "
    f"(was {len(event.hits)})"
)

# ===== STEP 11: Noisy event Hamiltonian =====
ham3 = SimpleHamiltonianFast(epsilon=0.01, gamma=1.5, delta=1.0)
A3, b3 = ham3.construct_hamiltonian(noisy, convolution=False)
sol3 = ham3.solve_classicaly()
reco3 = get_tracks(ham3, sol3, noisy, threshold=np.min(sol3))
print(
    f"[OK] Noisy reco: {len(reco3)} tracks from "
    f"{ham3.n_segments} segments"
)

# ===== STEP 12: Dense matrix =====
A_dense = ham.get_matrix_dense()
print(f"[OK] Dense matrix shape: {A_dense.shape}")

print()
print("=" * 50)
print("=== ALL PIPELINE TESTS PASSED ===")
print("=" * 50)
