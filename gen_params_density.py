#!/usr/bin/env python3
"""
gen_params_density.py - Parameter generation for track density study

This generates a parameter grid comparing sparse vs dense events:
- sparse: n_particles=[5,3,2] -> 10 total tracks across 3 events
- dense:  n_particles=[20,20,20,20,20] -> 100 total tracks across 5 events

Uses the same scale and erf_sigma variations as runs_6 for direct comparison.
"""

import itertools as it
import pandas as pd
from pathlib import Path

# --- Fixed parameters (same as runs_6) ---
MEASUREMENT_ERRORS = [1e-5]
COLLISION_NOISES   = [1e-4]
GHOST_RATES        = [0]
DROP_RATES         = [0]
REPEATS            = [i for i in range(20)]           # 20 repeats per combo
EPSILON_WIN        = [i for i in range(1, 6)]         # scale factors 1-5
ERF_SIGMA          = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
STEP_FLAG          = [True, False]

# --- NEW: Density configurations ---
# These are predefined names that velo_workflow.py will parse
N_PARTICLES_CONFIGS = ["sparse", "dense"]

BATCH_SIZE         = 10  # 10 combos per job

# Build the parameter grid
rows = [
    {
        "meas": m,
        "coll": c,
        "ghost": g,
        "drop": d,
        "repeat": r,
        "e_win": s,
        "step_flag": step,
        "erf_sigma": es,
        "n_particles": np_config,  # NEW column
    }
    for m, c, g, d, r, s, step, es, np_config in it.product(
        MEASUREMENT_ERRORS,
        COLLISION_NOISES,
        GHOST_RATES,
        DROP_RATES,
        REPEATS,
        EPSILON_WIN,
        STEP_FLAG,
        ERF_SIGMA,
        N_PARTICLES_CONFIGS,  # NEW
    )
]

df = pd.DataFrame(rows)
df.insert(0, "row_id", range(len(df)))
df.insert(1, "batch", df["row_id"] // BATCH_SIZE)

# Write into the CURRENT directory (the run folder, e.g. runs_7/)
cwd = Path(".").resolve()
params_path = cwd / "params.csv"
batches_path = cwd / "batches.txt"

df.to_csv(params_path, index=False)

batches = df["batch"].unique()
pd.Series(batches).to_csv(batches_path, index=False, header=False)

print(f"=== Density Study Parameter Generation ===")
print(f"Total combos: {len(df)}")
print(f"Batch size  : {BATCH_SIZE}")
print(f"Num batches : {len(batches)} (wrote {batches_path})")
print(f"")
print(f"Parameter breakdown:")
print(f"  Density configs : {N_PARTICLES_CONFIGS}")
print(f"  Scales          : {EPSILON_WIN}")
print(f"  ERF sigmas      : {len(ERF_SIGMA)} values")
print(f"  Step flags      : {STEP_FLAG}")
print(f"  Repeats         : {len(REPEATS)}")
print(f"")
print(f"Expected: 2 densities × 5 scales × 9 sigmas × 2 flags × 20 repeats = {2*5*9*2*20}")
print(f"Params file : {params_path}")
