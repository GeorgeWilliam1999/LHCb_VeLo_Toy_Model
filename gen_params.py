#!/usr/bin/env python3

# gen_params.py  (formerly make_params.py)
import itertools as it
import pandas as pd
import math
from pathlib import Path

# --- your grids ---
MEASUREMENT_ERRORS = [1e-5]  # [i*1e-5 for i in range(0,4)]
COLLISION_NOISES   = [1e-4]  # [i*1e-6 for i in range(0,1000,200)]
GHOST_RATES        = [0]
DROP_RATES         = [0]
REPEATS            = [i for i in range(20)]           # how many repeats per combo
EPSILON_WIN        = [i for i in range(1, 6)]        # scale factor
ERF_SIGMA          = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
STEP_FLAG          = [True, False]                          # variable epsilon or not
BATCH_SIZE         = 10                               # tune this to control “job size”

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
    }
    for m, c, g, d, r, s, step, es in it.product(
        MEASUREMENT_ERRORS,
        COLLISION_NOISES,
        GHOST_RATES,
        DROP_RATES,
        REPEATS,
        EPSILON_WIN,
        STEP_FLAG,
        ERF_SIGMA,
    )
]

df = pd.DataFrame(rows)
df.insert(0, "row_id", range(len(df)))
df.insert(1, "batch", df["row_id"] // BATCH_SIZE)

# Write into the CURRENT directory (the run folder, e.g. runs_1/)
cwd = Path(".").resolve()
params_path = cwd / "params.csv"
batches_path = cwd / "batches.txt"

df.to_csv(params_path, index=False)

batches = df["batch"].unique()
pd.Series(batches).to_csv(batches_path, index=False, header=False)

print(f"Total combos: {len(df)}")
print(f"Batch size  : {BATCH_SIZE}")
print(f"Num batches : {len(batches)} (wrote {batches_path})")
print(f"Step flag   : {STEP_FLAG}")
print(f"Params file : {params_path}")
