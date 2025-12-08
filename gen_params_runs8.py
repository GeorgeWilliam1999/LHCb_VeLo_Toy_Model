#!/usr/bin/env python3
"""
Parameter generator for runs_8: Corrected epsilon study.

Key corrections from runs_7:
- measurement_error: 0.01-0.05 mm (10-50 µm) instead of 1e-5 mm
- scale values: 1-5 to get epsilon in the 1-10 mrad range
- Both step function (convolution=False) and ERF (convolution=True)
- Test that erf_sigma ONLY affects convolution=True results

This generates a comprehensive grid to:
1. Verify step function has NO dependence on erf_sigma
2. Understand how scale affects metrics
3. Compare sparse vs dense events with realistic epsilon

Grid dimensions:
- 2 density configs (sparse: 5,3,2 = 10 tracks, dense: 20,20,20,20,20 = 100 tracks)
- 3 measurement errors: 0.01, 0.025, 0.05 mm (10, 25, 50 µm)
- 5 scale values: 1, 2, 3, 5, 8
- 5 erf_sigma values: 1e-4, 5e-4, 1e-3, 2e-3, 5e-3 rad
- 2 convolution flags: True (ERF), False (step)
- 10 repeats

Total: 2 × 3 × 5 × 5 × 2 × 10 = 3000 combinations
"""

import itertools
import csv
from pathlib import Path

# Output directory
OUT_DIR = Path("runs_8")
OUT_DIR.mkdir(exist_ok=True)

# ====================================================================
# PARAMETER GRID - CORRECTED VALUES
# ====================================================================

# Density configurations
DENSITY_CONFIGS = [
    "sparse",  # [5,3,2] = 10 tracks
    "dense",   # [20,20,20,20,20] = 100 tracks
]

# Measurement errors in mm (realistic: 10-50 µm)
MEASUREMENT_ERRORS = [0.01, 0.025, 0.05]  # 10, 25, 50 µm

# Multiple scattering (fixed at realistic value)
COLLISION_NOISES = [1e-4]  # ~0.1 mrad - typical for high-p tracks

# Ghost and drop rates (fixed for this study)
GHOST_RATES = [0.0]
DROP_RATES = [0.0]

# Scale factors for epsilon window
# Based on testing: small scales (1-10) work best for dense events
SCALES = [1, 2, 5, 10, 20]

# ERF sigma values (rad) - only relevant when convolution=True
# Range from sharp transition (1e-4) to very smooth (1e-2)
ERF_SIGMAS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

# Step vs ERF convolution (0=step, 1=ERF)
STEP_FLAGS = [0, 1]  # 0=step function, 1=ERF convolution

# Number of repeats per configuration
REPEATS = list(range(10))

# ====================================================================
# GENERATE PARAMETER FILE
# ====================================================================

BATCH_SIZE = 5  # rows per job (smaller for faster execution)

params_file = OUT_DIR / "params.csv"
# Column names must match what velo_workflow.py expects
fieldnames = [
    "batch",
    "meas",       # measurement_error
    "coll",       # collision_noise 
    "ghost",      # ghost_rate
    "drop",       # drop_rate
    "repeat",
    "e_win",      # scale
    "step_flag",
    "erf_sigma",
    "n_particles",
]

all_combos = list(itertools.product(
    MEASUREMENT_ERRORS,
    COLLISION_NOISES,
    GHOST_RATES,
    DROP_RATES,
    REPEATS,
    SCALES,
    STEP_FLAGS,
    ERF_SIGMAS,
    DENSITY_CONFIGS,
))

print(f"Total combinations: {len(all_combos)}")

with open(params_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for idx, combo in enumerate(all_combos):
        meas, coll, ghost, drop, rep, scale, step, erf_sig, density = combo
        batch_idx = idx // BATCH_SIZE
        writer.writerow({
            "batch": batch_idx,
            "meas": meas,
            "coll": coll,
            "ghost": ghost,
            "drop": drop,
            "repeat": rep,
            "e_win": scale,
            "step_flag": step,
            "erf_sigma": erf_sig,
            "n_particles": density,
        })

print(f"✓ Wrote {len(all_combos)} parameter rows to {params_file}")

# ====================================================================
# GENERATE BATCH FILES
# ====================================================================

n_batches = (len(all_combos) + BATCH_SIZE - 1) // BATCH_SIZE

batches_file = OUT_DIR / "batches.txt"
with open(batches_file, "w") as f:
    for i in range(n_batches):
        f.write(f"{i}\n")

print(f"✓ Wrote {n_batches} batch indices to {batches_file}")

# ====================================================================
# PREVIEW EPSILON VALUES
# ====================================================================

import numpy as np

print("\n" + "="*70)
print("EPSILON PREVIEW (with corrected parameters)")
print("="*70)

dz = 33.0  # mm
theta_min = 1.5e-5  # rad

for meas in MEASUREMENT_ERRORS:
    for scale in SCALES:
        coll = COLLISION_NOISES[0]
        theta_s = scale * coll
        theta_r = np.arctan((scale * meas) / dz)
        epsilon = np.sqrt(theta_s**2 + theta_r**2 + theta_min**2)
        print(f"meas={meas*1000:5.1f}µm, scale={scale:2d}: ε = {epsilon*1000:6.2f} mrad")

print("\nCompare to typical hit separation: ~50-200 mrad")
print("Epsilon is now in a range that should accept true tracks while rejecting ghosts!")
