#!/usr/bin/env python
"""
gen_params_runs12.py - High statistics run with LARGE phi window (0.2 rad)

Same as runs_11 but with phi_max = 0.2 rad instead of 0.02 rad.

Focus: σ_res = 10 µm (instruction default) with multiple scattering scan
       and LARGE angular acceptance (phi = 0.2 rad)

Parameters:
- σ_res = 10 µm (FIXED at instruction default)
- σ_scatt = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0] mrad
- Scale n = [3, 4, 5]
- phi_max = 0.2 rad (LARGE - 10x larger than runs_11)
- High statistics: 100 repeats per configuration
- Both sparse (10 tracks) and dense (100 tracks) events

Total: 11 scattering × 3 scales × 2 densities × 50 repeats = 3,300 events
Split across 66 batches of 50 events each (faster job completion).
"""

import json
import csv
import os
from itertools import product

# ============================================================================
# RUNS 12: HIGH STATISTICS WITH LARGE PHI WINDOW
# ============================================================================

# Fixed at instruction default
SIGMA_RES = 0.010  # 10 µm in mm (INSTRUCTION DEFAULT)

# LARGE phi window (compared to 0.02 in runs_11)
PHI_MAX = 0.2  # rad

# Multiple scattering scan - fine granularity around instruction values
SIGMA_SCATT_VALUES = [
    0.0,      # 0 mrad - perfect
    0.00005,  # 0.05 mrad
    0.0001,   # 0.1 mrad - INSTRUCTION DEFAULT
    0.00015,  # 0.15 mrad
    0.0002,   # 0.2 mrad - instruction value
    0.00025,  # 0.25 mrad
    0.0003,   # 0.3 mrad - instruction value
    0.0004,   # 0.4 mrad
    0.0005,   # 0.5 mrad - instruction value
    0.00075,  # 0.75 mrad
    0.001,    # 1.0 mrad - instruction value
]

# Window scale factor
SCALE_VALUES = [3, 4, 5]

# Track densities to test
DENSITY_CONFIGS = ['sparse', 'dense']  # 10 vs 100 tracks

# High statistics for publication
N_REPEATS = 50  # Reduced from 100 for faster runs

# Fixed parameters (no drop, no noise - clean measurement)
DROP_RATE = 0.0
NOISE_RATE = 0.0

# ============================================================================
# FIXED PARAMETERS (detector geometry from instructions)
# ============================================================================

FIXED_PARAMS = {
    "n_layers": 26,
    "layer_spacing_mm": 33.0,
    "layer_size_mm": 80.0,
    "theta_max_rad": 0.2,
    "phi_max_rad": PHI_MAX,
}

def get_n_particles(config):
    """Return number of particles for density config."""
    return 10 if config == 'sparse' else 100


def generate_params():
    """
    Generate parameter combinations for high-statistics run.
    """
    params_list = []
    
    for sigma_scatt, scale, density in product(
        SIGMA_SCATT_VALUES, SCALE_VALUES, DENSITY_CONFIGS
    ):
        for repeat in range(N_REPEATS):
            params = {
                **FIXED_PARAMS,
                "meas": SIGMA_RES,
                "coll": sigma_scatt,
                "scale": scale,
                "drop": DROP_RATE,
                "ghost": NOISE_RATE,
                "n_particles_config": density,
                "n_particles": get_n_particles(density),
                "repeat": repeat,
                "phi_max": PHI_MAX,
                "scan_type": f"high_stats_ms_scan_phi0.2_{density}",
            }
            params_list.append(params)
    
    return params_list


def generate_batches(events_per_batch=50):
    """
    Generate batched parameters for condor submission.
    Small batch size (50) for faster job completion.
    """
    all_params = generate_params()
    
    # Group into batches
    batches = []
    for i in range(0, len(all_params), events_per_batch):
        batch_params = all_params[i:i + events_per_batch]
        batches.append(batch_params)
    
    return batches


def main():
    """Generate params.csv and batches.txt for condor submission."""
    all_params = generate_params()
    batches = generate_batches(events_per_batch=50)  # Small batches for speed
    
    print(f"RUNS 12: High Statistics MS Scan - LARGE PHI WINDOW")
    print(f"=" * 60)
    print(f"Fixed parameters:")
    print(f"  σ_res = {SIGMA_RES * 1000:.0f} µm (instruction default)")
    print(f"  phi_max = {PHI_MAX} rad (LARGE - 10x runs_11)")
    print(f"  Drop rate = {DROP_RATE * 100:.0f}%")
    print(f"  Noise rate = {NOISE_RATE * 100:.0f}%")
    print(f"\nScanned parameters:")
    print(f"  σ_scatt = {[s*1000 for s in SIGMA_SCATT_VALUES]} mrad")
    print(f"  Scale = {SCALE_VALUES}")
    print(f"  Density = {DENSITY_CONFIGS}")
    print(f"\nStatistics:")
    print(f"  Repeats per config: {N_REPEATS}")
    print(f"  Total events: {len(all_params)}")
    print(f"  Number of batches: {len(batches)}")
    print(f"  Events per batch: 50 (small for fast completion)")
    
    os.makedirs("runs_12", exist_ok=True)
    
    # Create params.csv (format expected by velo_workflow.py)
    csv_rows = []
    for batch_id, batch in enumerate(batches):
        for params in batch:
            row = {
                'batch': batch_id,
                'meas': params['meas'],
                'coll': params['coll'],
                'ghost': params['ghost'],
                'drop': params['drop'],
                'repeat': params['repeat'],
                'e_win': params['scale'],
                'step_flag': 0,  # Use step function
                'erf_sigma': 0.001,
                'n_particles': params['n_particles_config'],
                'phi_max': params['phi_max'],  # NEW: configurable phi
                'scan_type': params['scan_type'],
            }
            csv_rows.append(row)
    
    # Write params.csv
    with open("runs_12/params.csv", 'w', newline='') as f:
        fieldnames = ['batch', 'meas', 'coll', 'ghost', 'drop', 'repeat', 'e_win', 'step_flag', 'erf_sigma', 'n_particles', 'phi_max', 'scan_type']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"\n✓ Created runs_12/params.csv with {len(csv_rows)} rows")
    
    # Create batches.txt (one batch ID per line)
    with open("runs_12/batches.txt", 'w') as f:
        for batch_id in range(len(batches)):
            f.write(f"{batch_id}\n")
    
    print(f"✓ Created runs_12/batches.txt with {len(batches)} batch IDs")
    
    # Create batch directories
    for batch_id in range(len(batches)):
        os.makedirs(f"runs_12/batch_{batch_id}", exist_ok=True)
    
    print(f"✓ Created {len(batches)} batch directories")
    
    # Save summary
    summary = {
        "run_name": "runs_12",
        "description": "High statistics MS scan with LARGE phi window (0.2 rad)",
        "fixed_sigma_res_mm": SIGMA_RES,
        "phi_max_rad": PHI_MAX,
        "sigma_scatt_values_rad": SIGMA_SCATT_VALUES,
        "scale_values": SCALE_VALUES,
        "density_configs": DENSITY_CONFIGS,
        "n_repeats": N_REPEATS,
        "total_events": len(all_params),
        "n_batches": len(batches),
    }
    with open("runs_12/run_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return batches


if __name__ == "__main__":
    main()
