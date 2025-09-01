import json
from LHCB_Velo_Toy_Models.state_event_generator import *
from LHCB_Velo_Toy_Models import state_event_model 
from LHCB_Velo_Toy_Models.simple_hamiltonian import SimpleHamiltonian
import qiskit
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from hhl_algorithm import HHLAlgorithm as hhl
from hhl_algorithm_1bit import HHLAlgorithm as hhl_1
from LHCB_Velo_Toy_Models.simple_hamiltonian import get_tracks
from LHCB_Velo_Toy_Models.toy_validator import EventValidator as evl
import itertools as it



# --- Constants ---
dz = 20  # mm layer spacing
layers = 5
n_particles = [20, 20, 20, 20, 20]
events = len(n_particles)
n = np.sum(n_particles)

# Detector configuration
module_id = list(range(1, layers + 1))
lx = [33] * layers
ly = [33] * layers
zs = [dz * l for l in range(1, layers + 1)]

# Noise and error parameters
measurement_errors = collision_noises = ghost_rates = drop_rates = np.round(np.linspace(0.00, 0.10, 11), 2)

# Results container
results = []

# --- Main parameter sweep ---
print("Starting parameter sweep...")

for mes_error, coll_noise, ghost, drop in it.product(measurement_errors, collision_noises, ghost_rates, drop_rates):
    print(f"\nRunning simulation with:")
    print(f"  Measurement error = {mes_error}")
    print(f"  Collision noise   = {coll_noise}")
    print(f"  Ghost rate        = {ghost}")
    print(f"  Drop rate         = {drop}")

    # --- Detector setup ---
    Detector = state_event_model.PlaneGeometry(
        module_id=module_id, lx=lx, ly=ly, z=zs
    )

    # --- State event generator setup ---
    state_event_gen = StateEventGenerator(
        Detector,
        events=events,
        n_particles=n_particles,
        measurement_error=mes_error,
        collision_noise=coll_noise
    )

    state_event_gen.generate_random_primary_vertices({'x': 1, 'y': 1, 'z': 1})

    event_particles = [[{'type': 'MIP', 'mass': 0.511, 'q': 1} for _ in range(n)] for _ in range(events)]

    state_event_gen.generate_particles(event_particles)
    event_tracks = state_event_gen.generate_complete_events()

    # --- Inject noise into events ---
    false_tracks = state_event_gen.make_noisy_event(drop_rate=drop, ghost_rate=ghost)

    # --- Hamiltonian setup and solve ---
    ham = SimpleHamiltonian(epsilon=1e-7, gamma=2.0, delta=1.0)
    ham.construct_hamiltonian(event=event_tracks, convolution=True)

    print("Solving classical Hamiltonian...")
    classical_solution = ham.solve_classicaly()
    discretized_solution = (classical_solution > 0.45).astype(int)

    # --- Track reconstruction and validation ---
    rec_tracks = get_tracks(ham, discretized_solution, false_tracks)
    validator = evl(false_tracks, rec_tracks)
    metrics = validator.compute_metrics()

    print(f"  -> Metrics: {metrics}")

    # --- Store result ---
    results.append({
        'measurement_error': mes_error,
        'collision_noise': coll_noise,
        'ghost_rate': ghost,
        'drop_rate': drop,
        'metrics': metrics
    })

print("\nParameter sweep completed. Saving results...")

# --- Save results to JSON file ---
with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to 'results.json'.")