#!/usr/bin/env python
"""
gen_params_runs10.py - Parameter generator matching Instructions.pdf exactly

Instructions.pdf specifications:
- σ_res  = [0, 10, 20, 50] µm  (hit resolution)
- σ_scatt = [0, 0.1, 0.2, 0.3, 0.5, 1.0] mrad  (multiple scattering)  
- n (window scale) = [3, 4, 5] × max(σ_scatt, σ_res)
- drop rate = [0%, 1%, 2%]  (hit inefficiency)
- noise rate = [0, 0.0001, 0.001, 0.01]  (ghost/noise fraction)

DEFAULTS (used when scanning other parameters):
- σ_scatt = 0.1 mrad
- σ_res = 10 µm  
- drop = 0%
- noise = 0%

High statistics run with 50 repeats for reliable statistics.
"""

import json
from itertools import product

# ============================================================================
# INSTRUCTION-SPECIFIED PARAMETER VALUES
# ============================================================================

# Hit resolution in mm (instructions give µm, convert to mm for code)
SIGMA_RES_VALUES = [0.0, 0.010, 0.020, 0.050]  # 0, 10, 20, 50 µm

# Multiple scattering in radians (instructions give mrad, convert to rad)
SIGMA_SCATT_VALUES = [0.0, 0.0001, 0.0002, 0.0003, 0.0005, 0.001]  # 0, 0.1, 0.2, 0.3, 0.5, 1.0 mrad

# Window scale factor n (window = n × max(σ_scatt, σ_res))
SCALE_VALUES = [3, 4, 5]

# Hit drop rate (inefficiency)
DROP_VALUES = [0.0, 0.01, 0.02]  # 0%, 1%, 2%

# Noise/ghost hit fraction
NOISE_VALUES = [0.0, 0.0001, 0.001, 0.01]

# DEFAULTS from instructions
DEFAULT_SIGMA_RES = 0.010     # 10 µm in mm
DEFAULT_SIGMA_SCATT = 0.0001  # 0.1 mrad in rad
DEFAULT_DROP = 0.0
DEFAULT_NOISE = 0.0
DEFAULT_SCALE = 5  # Use n=5 as default (largest window)

# Number of repeats for statistical robustness
N_REPEATS = 50

# ============================================================================
# FIXED PARAMETERS (from instructions geometry)
# ============================================================================

FIXED_PARAMS = {
    "n_layers": 26,               # Number of detector layers
    "n_tracks": 10,               # Sparse event (10 tracks)
    "layer_spacing_mm": 33.0,     # 33 cm = 330 mm between layers (instructions say 33cm)
    "layer_size_mm": 80.0,        # 80x80 mm² sensor area
    "theta_max_rad": 0.2,         # [-0.2, 0.2] rad in theta
}


def generate_params():
    """
    Generate parameter combinations following the instructions methodology:
    - Use defaults when scanning other parameters
    - Full factorial scan of all parameters
    """
    params_list = []
    
    # ========================================================================
    # SCAN 1: Full factorial (as specified in instructions - all combinations)
    # Total: 4 × 6 × 3 × 3 × 4 = 864 combinations × 50 repeats = 43,200 jobs
    # ========================================================================
    
    for sigma_res, sigma_scatt, scale, drop, noise in product(
        SIGMA_RES_VALUES, SIGMA_SCATT_VALUES, SCALE_VALUES, DROP_VALUES, NOISE_VALUES
    ):
        for repeat in range(N_REPEATS):
            params = {
                **FIXED_PARAMS,
                "meas": sigma_res,           # Hit resolution (σ_res) in mm
                "coll": sigma_scatt,         # Multiple scattering (σ_scatt) in rad
                "scale": scale,              # Window scale n
                "drop": drop,                # Hit drop rate
                "ghost": noise,              # Noise hit fraction
                "repeat": repeat,
                "scan_type": "full_factorial_instructions",
            }
            params_list.append(params)
    
    return params_list


def generate_params_reduced():
    """
    Reduced scan: Vary one parameter at a time from defaults.
    Useful for understanding individual parameter effects.
    
    Total combinations much smaller for quick tests.
    """
    params_list = []
    
    # Scan σ_res with other params at default
    for sigma_res in SIGMA_RES_VALUES:
        for repeat in range(N_REPEATS):
            params = {
                **FIXED_PARAMS,
                "meas": sigma_res,
                "coll": DEFAULT_SIGMA_SCATT,
                "scale": DEFAULT_SCALE,
                "drop": DEFAULT_DROP,
                "ghost": DEFAULT_NOISE,
                "repeat": repeat,
                "scan_type": "scan_sigma_res",
            }
            params_list.append(params)
    
    # Scan σ_scatt with other params at default
    for sigma_scatt in SIGMA_SCATT_VALUES:
        for repeat in range(N_REPEATS):
            params = {
                **FIXED_PARAMS,
                "meas": DEFAULT_SIGMA_RES,
                "coll": sigma_scatt,
                "scale": DEFAULT_SCALE,
                "drop": DEFAULT_DROP,
                "ghost": DEFAULT_NOISE,
                "repeat": repeat,
                "scan_type": "scan_sigma_scatt",
            }
            params_list.append(params)
    
    # Scan scale with other params at default
    for scale in SCALE_VALUES:
        for repeat in range(N_REPEATS):
            params = {
                **FIXED_PARAMS,
                "meas": DEFAULT_SIGMA_RES,
                "coll": DEFAULT_SIGMA_SCATT,
                "scale": scale,
                "drop": DEFAULT_DROP,
                "ghost": DEFAULT_NOISE,
                "repeat": repeat,
                "scan_type": "scan_scale",
            }
            params_list.append(params)
    
    # Scan drop rate with other params at default
    for drop in DROP_VALUES:
        for repeat in range(N_REPEATS):
            params = {
                **FIXED_PARAMS,
                "meas": DEFAULT_SIGMA_RES,
                "coll": DEFAULT_SIGMA_SCATT,
                "scale": DEFAULT_SCALE,
                "drop": drop,
                "ghost": DEFAULT_NOISE,
                "repeat": repeat,
                "scan_type": "scan_drop",
            }
            params_list.append(params)
    
    # Scan noise with other params at default
    for noise in NOISE_VALUES:
        for repeat in range(N_REPEATS):
            params = {
                **FIXED_PARAMS,
                "meas": DEFAULT_SIGMA_RES,
                "coll": DEFAULT_SIGMA_SCATT,
                "scale": DEFAULT_SCALE,
                "drop": DEFAULT_DROP,
                "ghost": noise,
                "repeat": repeat,
                "scan_type": "scan_noise",
            }
            params_list.append(params)
    
    return params_list


def generate_params_best_config():
    """
    High-statistics run with best parameters from runs_9 analysis:
    - scale = 5
    - low resolution (1-10 µm)
    - low scattering (0.1 mrad)
    
    This is for validation with very high statistics.
    """
    params_list = []
    
    # Best configurations from runs_9
    best_configs = [
        {"meas": 0.001, "coll": 0.0001, "scale": 5},  # 1µm res, 0.1mrad scatt
        {"meas": 0.010, "coll": 0.0001, "scale": 5},  # 10µm res, 0.1mrad scatt (instruction default)
        {"meas": 0.010, "coll": 0.0002, "scale": 5},  # 10µm res, 0.2mrad scatt
        {"meas": 0.020, "coll": 0.0001, "scale": 5},  # 20µm res, 0.1mrad scatt
    ]
    
    N_HIGH_STAT_REPEATS = 100  # Very high statistics
    
    for config in best_configs:
        for repeat in range(N_HIGH_STAT_REPEATS):
            params = {
                **FIXED_PARAMS,
                **config,
                "drop": 0.0,
                "ghost": 0.0,
                "repeat": repeat,
                "scan_type": "best_config_validation",
            }
            params_list.append(params)
    
    return params_list


if __name__ == "__main__":
    # Default: generate full factorial scan
    import sys
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    
    if mode == "full":
        params_list = generate_params()
        desc = "Full factorial (instructions spec)"
    elif mode == "reduced":
        params_list = generate_params_reduced()
        desc = "Reduced (one-at-a-time)"
    elif mode == "best":
        params_list = generate_params_best_config()
        desc = "Best config validation"
    else:
        print(f"Unknown mode: {mode}. Use 'full', 'reduced', or 'best'")
        sys.exit(1)
    
    n_params = len(params_list)
    print(f"# {desc}: {n_params} parameter combinations")
    
    # Calculate unique combinations (excluding repeat)
    unique_combos = set()
    for p in params_list:
        key = (p["meas"], p["coll"], p["scale"], p["drop"], p["ghost"])
        unique_combos.add(key)
    
    print(f"# Unique configurations: {len(unique_combos)}")
    print(f"# Repeats per config: {n_params // len(unique_combos)}")
    
    for p in params_list:
        print(json.dumps(p))
