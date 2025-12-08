#!/usr/bin/env python3
"""
Parameter generator for runs_9: Comprehensive parameter scan.

This generates a massive dataset scanning over ALL parameters:
- Measurement error (hit resolution)
- Collision noise (multiple scattering)
- Scale factor for epsilon window
- ERF sigma (smoothing parameter)
- Step vs ERF convolution
- Sparse vs Dense events

Strategy: For each target parameter scan, hold others at baseline values.
This allows us to isolate the effect of each parameter.

Scans:
1. Hit resolution scan: 0.005 - 0.1 mm (5-100 µm)
2. Multiple scattering scan: 1e-5 - 1e-3 rad
3. Scale scan: 1 - 50
4. ERF sigma scan: 1e-5 - 0.1 rad
5. Full grid at baseline for cross-validation

Total combinations designed for ~10,000 data points
"""

import itertools
import csv
from pathlib import Path
import numpy as np

# Output directory
OUT_DIR = Path("runs_9")
OUT_DIR.mkdir(exist_ok=True)

# ====================================================================
# BASELINE VALUES (used when scanning other parameters)
# ====================================================================
BASELINE = {
    'meas': 0.025,       # 25 µm - typical silicon detector resolution
    'coll': 1e-4,        # 0.1 mrad - typical multiple scattering
    'scale': 5,          # Good balance from runs_8
    'erf_sigma': 1e-3,   # 1 mrad - moderate smoothing
}

# Fixed parameters
GHOST_RATE = 0.0
DROP_RATE = 0.0

# Density configurations
DENSITIES = ['sparse', 'dense']

# Step vs ERF (0=step, 1=ERF)
METHODS = [0, 1]

# Number of repeats per configuration
N_REPEATS = 20  # Excellent statistics with 20 repeats

# Batch size - smaller for faster execution
BATCH_SIZE = 5  # Each job processes 5 parameter combinations

# ====================================================================
# PARAMETER SCANS - Comprehensive coverage with fine granularity
# ====================================================================

# Scan 1: Hit resolution (measurement error) in mm - 15 values
# Covers 1 µm to 200 µm (realistic detector range)
MEAS_SCAN = [0.001, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2]

# Scan 2: Multiple scattering (collision noise) in rad - 12 values
# Covers very low to high scattering scenarios
COLL_SCAN = [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]

# Scan 3: Scale factor for epsilon window - 15 values
# Fine-grained coverage from tight to loose windows
SCALE_SCAN = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 15, 20, 35, 50]

# Scan 4: ERF sigma (smoothing) in rad - 12 values
# From very sharp to very smooth transitions
ERF_SIGMA_SCAN = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2, 0.1]

# ====================================================================
# GENERATE COMBINATIONS
# ====================================================================

all_combos = []

def add_combos(meas, coll, scale, erf_sigma, scan_name):
    """Add combinations for all densities, methods, and repeats."""
    for density in DENSITIES:
        for method in METHODS:
            for repeat in range(N_REPEATS):
                all_combos.append({
                    'meas': meas,
                    'coll': coll,
                    'ghost': GHOST_RATE,
                    'drop': DROP_RATE,
                    'repeat': repeat,
                    'e_win': scale,
                    'step_flag': method,
                    'erf_sigma': erf_sigma,
                    'n_particles': density,
                    'scan_type': scan_name,
                })

# Scan 1: Hit resolution scan (vary meas, hold others at baseline)
print("Generating hit resolution scan...")
for meas in MEAS_SCAN:
    add_combos(meas, BASELINE['coll'], BASELINE['scale'], BASELINE['erf_sigma'], 'meas_scan')
n_meas = len(all_combos)
print(f"  {n_meas} combinations")

# Scan 2: Multiple scattering scan (vary coll, hold others at baseline)
print("Generating scattering scan...")
for coll in COLL_SCAN:
    add_combos(BASELINE['meas'], coll, BASELINE['scale'], BASELINE['erf_sigma'], 'coll_scan')
n_coll = len(all_combos) - n_meas
print(f"  {n_coll} combinations")

# Scan 3: Scale scan (vary scale, hold others at baseline)
print("Generating scale scan...")
for scale in SCALE_SCAN:
    add_combos(BASELINE['meas'], BASELINE['coll'], scale, BASELINE['erf_sigma'], 'scale_scan')
n_scale = len(all_combos) - n_meas - n_coll
print(f"  {n_scale} combinations")

# Scan 4: ERF sigma scan (vary erf_sigma, hold others at baseline)
# Note: Only relevant for ERF method, but we include step for comparison
print("Generating ERF sigma scan...")
for erf_sigma in ERF_SIGMA_SCAN:
    add_combos(BASELINE['meas'], BASELINE['coll'], BASELINE['scale'], erf_sigma, 'erf_sigma_scan')
n_erf = len(all_combos) - n_meas - n_coll - n_scale
print(f"  {n_erf} combinations")

# Scan 5: 2D grid - Scale vs Hit resolution (key interaction)
print("Generating scale x meas 2D grid...")
MEAS_GRID = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.075, 0.1]
SCALE_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
for meas in MEAS_GRID:
    for scale in SCALE_GRID:
        if (meas, scale) != (BASELINE['meas'], BASELINE['scale']):  # Skip baseline (already covered)
            add_combos(meas, BASELINE['coll'], scale, BASELINE['erf_sigma'], 'meas_scale_grid')
n_grid = len(all_combos) - n_meas - n_coll - n_scale - n_erf
print(f"  {n_grid} combinations")

print(f"\nTotal combinations: {len(all_combos)}")

# ====================================================================
# WRITE PARAMS FILE
# ====================================================================

params_file = OUT_DIR / "params.csv"
fieldnames = ['batch', 'meas', 'coll', 'ghost', 'drop', 'repeat', 'e_win', 
              'step_flag', 'erf_sigma', 'n_particles', 'scan_type']

with open(params_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for idx, combo in enumerate(all_combos):
        combo['batch'] = idx // BATCH_SIZE
        writer.writerow(combo)

print(f"✓ Wrote {len(all_combos)} parameter rows to {params_file}")

# ====================================================================
# GENERATE BATCH FILE
# ====================================================================

n_batches = (len(all_combos) + BATCH_SIZE - 1) // BATCH_SIZE
batches_file = OUT_DIR / "batches.txt"
with open(batches_file, "w") as f:
    for i in range(n_batches):
        f.write(f"{i}\n")

print(f"✓ Wrote {n_batches} batch indices to {batches_file}")

# ====================================================================
# SUMMARY
# ====================================================================

print("\n" + "="*70)
print("RUNS_9 PARAMETER SCAN SUMMARY")
print("="*70)
print(f"Hit resolution scan:    {len(MEAS_SCAN)} values × 2 densities × 2 methods × {N_REPEATS} repeats = {n_meas}")
print(f"Scattering scan:        {len(COLL_SCAN)} values × 2 densities × 2 methods × {N_REPEATS} repeats = {n_coll}")
print(f"Scale scan:             {len(SCALE_SCAN)} values × 2 densities × 2 methods × {N_REPEATS} repeats = {n_scale}")
print(f"ERF sigma scan:         {len(ERF_SIGMA_SCAN)} values × 2 densities × 2 methods × {N_REPEATS} repeats = {n_erf}")
print(f"Scale × Meas 2D grid:   {len(MEAS_GRID)*len(SCALE_GRID)-1} combinations × 2 densities × 2 methods × {N_REPEATS} repeats = {n_grid}")
print(f"\nTotal: {len(all_combos)} combinations in {n_batches} batches")
print(f"Batch size: {BATCH_SIZE}")
print(f"\nBaseline values: meas={BASELINE['meas']}mm, coll={BASELINE['coll']}rad, scale={BASELINE['scale']}, erf_sigma={BASELINE['erf_sigma']}rad")
