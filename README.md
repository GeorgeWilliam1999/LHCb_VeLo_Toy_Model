# VELO Toy Model - Hamiltonian Track Reconstruction

Quantum-inspired Hamiltonian formulation for track reconstruction in the LHCb VELO detector.

## References

- Nicotra et al., *"Track finding and fitting with a quantum-inspired algorithm for the LHCb VELO"*, J. Inst. **18** P11028 (2023)
- arXiv:2511.11458v1

## Directory Structure

```
Velo_toy/
├── src/velo_toy/           # Main Python package
│   ├── core/               # Core models (symlinks to LHCB_Velo_Toy_Models)
│   │   ├── state_event_model.py      # Hit, Track, Segment, Event dataclasses
│   │   ├── state_event_generator.py  # Generate toy events
│   │   ├── simple_hamiltonian.py     # ERF-based Hamiltonian
│   │   └── toy_validator.py          # Track validation metrics
│   ├── experiments/        # Experiment running infrastructure
│   │   ├── config.py       # ExperimentConfig, ParameterGrid dataclasses
│   │   ├── runner.py       # run_experiment(), run_batch()
│   │   └── aggregator.py   # merge results from batches
│   └── analysis/           # Analysis and plotting
│       ├── loader.py       # Load metrics from CSV/pickle
│       ├── statistics.py   # Compute mean, RMS, correlations
│       └── plotting.py     # Publication-quality figures
├── bin/                    # Command-line entry points
│   ├── run_batch.py        # Execute a batch of experiments
│   ├── aggregate_results.py # Merge batch results
│   └── submit_experiment.py # Submit jobs to Condor
├── scripts/condor/         # HTCondor job submission
│   ├── run_batch.sh        # Worker node script
│   ├── velo_experiment.sub # Main submit file
│   └── aggregate.sub       # Post-processing job
├── results/                # Output directory
│   └── archive/            # Previous experiment runs (runs_1..5)
├── LHCB_Velo_Toy_Models/   # Original library code
└── helpful/                # Utility scripts and notes
```

## Installation

```bash
# Create conda environment (on Nikhef cluster)
conda create -n Q_env python=3.10
conda activate Q_env

# Install dependencies
pip install numpy scipy pandas matplotlib dill tqdm

# Install package in development mode
cd /data/bfys/gscriven/Velo_toy
pip install -e .
```

## Quick Start

### 1. Run a Single Experiment

```python
from velo_toy.core import StateEventGenerator, SimpleHamiltonian, EventValidator
from velo_toy.experiments import ExperimentConfig, run_experiment

# Configure
config = ExperimentConfig(
    n_modules=26,
    n_tracks=5,
    hit_resolution=0.0001,
    multi_scatter=0.0002,
    ghost_rate=0.0,
    drop_rate=0.0,
)

# Run
result = run_experiment(config)
print(f"Efficiency: {result['m_m_reconstruction_efficiency']:.3f}")
print(f"Ghost Rate: {result['m_m_ghost_rate']:.3f}")
```

### 2. Submit Batch Jobs to Condor

```bash
# Generate parameter grid and submit
python bin/submit_experiment.py \
    --experiment-name scale_test \
    --scales 1 2 3 4 5 \
    --n-tracks 5 \
    --repeats 20 \
    --submit

# Monitor jobs
condor_q

# After completion, aggregate results
python bin/aggregate_results.py results/scale_test/
```

### 3. Analyze Results

```python
from velo_toy.analysis import load_metrics, compute_mean_rms, plot_efficiency_vs_scattering

# Load data
df = load_metrics("results/scale_test")

# Compute statistics
stats = compute_mean_rms(df, group_by=['p_multi_scatter'])
print(stats)

# Plot
fig = plot_efficiency_vs_scattering(df, resolution_values=[1e-4, 5e-4, 1e-3])
fig.savefig("efficiency_plot.png")
```

## Parameter Space

The experiment framework supports scanning these parameters:

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `scale` | Overall parameter scale multiplier | 1-5 |
| `hit_resolution` | Hit position resolution (σ) | 0.0001-0.001 |
| `multi_scatter` | Multiple scattering angle | 0.0001-0.001 |
| `ghost_rate` | Random noise hit fraction | 0.0-0.3 |
| `drop_rate` | Hit inefficiency fraction | 0.0-0.3 |
| `n_tracks` | Tracks per event | 1-10 |
| `n_modules` | Detector modules | 26 (default) |

## Cost Function

The Hamiltonian uses an ERF-smoothed cost function for segment compatibility:

$$C(r) = \frac{1}{2}\left(1 + \text{erf}\left(\frac{\varepsilon - |r|}{\sigma\sqrt{2}}\right)\right)$$

where:
- $r$ = angle between segments
- $\varepsilon$ = threshold angle
- $\sigma$ = smoothing width

## Condor Configuration

The Nikhef cluster uses HTCondor for job scheduling. Key files:

- `scripts/condor/velo_experiment.sub` - Main submit file
- `scripts/condor/run_batch.sh` - Worker script (activates Q_env)

Submit with:
```bash
condor_submit scripts/condor/velo_experiment.sub
```

## Output Format

### metrics.csv columns

**Parameters (p_* prefix):**
- `p_scale`, `p_hit_res`, `p_multi_scatter`, `p_ghost_rate`, `p_drop_rate`
- `p_n_tracks`, `p_n_modules`

**Metrics (m_* prefix):**
- `m_m_reconstruction_efficiency` - Track reconstruction efficiency
- `m_m_ghost_rate` - Ghost track rate  
- `m_m_clone_fraction_total` - Clone track fraction
- `m_m_purity_all_matched` - Average track purity
- `m_m_hit_efficiency_mean` - Hit efficiency (completeness)

## Experimental Runs

### Summary Table

| Run | Events | Purpose | Status |
|-----|--------|---------|--------|
| runs_6 | ~180 | ERF sigma sweep (preliminary) | Superseded |
| runs_7 | ~360 | Sparse/dense comparison (preliminary) | Superseded |
| runs_8 | 3,000 | Scale factor & ERF sigma study | ✅ Analyzed |
| runs_9 | 13,303 | Extended parameter scan | ✅ Analyzed |
| runs_10 | 1,000 | Instruction-aligned parameters | ✅ Analyzed |
| runs_11 | ~39,600 | High-stats MS scan (small φ) | ⏳ Running |
| runs_12 | ~19,800 | High-stats MS scan (large φ) | ⏳ Running |

---

### runs_6: ERF Sigma Preliminary Study
**Status:** Superseded by runs_8

**Parameters:**
- σ_res = 10 µm (fixed)
- σ_scatt = 0.1 mrad (fixed)
- erf_sigma = [10⁻⁸, 10⁻⁷, 10⁻⁶, 10⁻⁵]
- Step flag = True (ERF only)

**Purpose:** Initial exploration of ERF smoothing parameter effect.

---

### runs_7: Sparse vs Dense Preliminary
**Status:** Superseded by runs_8

**Parameters:**
- σ_res = 10 µm (fixed)
- σ_scatt = 0.1 mrad (fixed)
- erf_sigma = [10⁻⁸, 10⁻⁷, 10⁻⁶, 10⁻⁵]
- Track density = sparse (10 tracks), dense (100 tracks)

**Purpose:** First comparison of algorithm performance at different track multiplicities.

---

### runs_8: Scale Factor & ERF Sigma Study
**Status:** ✅ Analyzed (3,000 events)

**Parameters:**
| Parameter | Values |
|-----------|--------|
| σ_res | 10, 25, 50 µm |
| σ_scatt | 0.1 mrad (fixed) |
| Scale n | 1, 2, 5, 10, 20 |
| erf_sigma | 0.1, 0.5, 1, 5, 10 mrad |
| Threshold | Step (0), ERF (1) |
| Density | Sparse (10), Dense (100) |
| Repeats | 10 per config |

**Purpose:** 
1. Verify step function independence from erf_sigma
2. Optimize scale factor for sparse and dense events
3. Compare step vs ERF threshold functions

**Key Findings:**
- ✅ Step function gives identical results regardless of erf_sigma
- Optimal scale n=5 for sparse events (~94% efficiency)
- Dense events fundamentally limited to ~14% efficiency
- Step function performs as well as ERF (simpler is better)

---

### runs_9: Extended Parameter Scan
**Status:** ✅ Analyzed (13,303 events)

**Parameters:**
| Parameter | Values |
|-----------|--------|
| σ_res | 1, 2, 5, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200 µm |
| σ_scatt | 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10 mrad |
| Scale n | 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 15, 20, 30, 35, 50 |
| erf_sigma | Various |
| Threshold | Step (0), ERF (1) |
| Density | Sparse (10), Dense (100) |
| Scan types | meas_scan, coll_scan, scale_scan, erf_sigma_scan, meas_scale_grid |

**Purpose:** 
1. Comprehensive exploration of parameter space
2. Find performance boundaries
3. Generate 2D heatmaps of efficiency vs (resolution, scattering)

**Key Findings:**
- Efficiency drops sharply for σ_res > 50 µm
- Efficiency drops sharply for σ_scatt > 0.2 mrad
- Scale n=3-5 optimal across most configurations
- Dense events show 5-10x worse performance than sparse

---

### runs_10: Instruction-Aligned Parameters
**Status:** ✅ Analyzed (1,000 events)

**Parameters:**
| Parameter | Values | Default |
|-----------|--------|---------|
| σ_res | 0, 10, 20, 50 µm | 10 µm |
| σ_scatt | 0, 0.1, 0.2, 0.3, 0.5, 1.0 mrad | 0.1 mrad |
| Scale n | 3, 4, 5 | - |
| Density | Sparse (10 tracks) | - |
| Events | 50-500 per config | - |

**Purpose:**
Use exact parameter values from Instructions.pdf for publication-ready results.

**Key Findings:**
- At defaults (σ_res=10µm, σ_scatt=0.1mrad):
  - Scale 3: **81.4% efficiency**, 7.7% ghost rate ← Best
  - Scale 4: 76.6% efficiency, 11.5% ghost rate
  - Scale 5: 64.5% efficiency, 18.2% ghost rate
- Performance degrades with increasing σ_scatt due to track angular density
- Ghost rate increases from ~4% to ~40% as σ_scatt goes 0→1 mrad

---

### runs_11: High-Statistics MS Scan (Small φ_max)
**Status:** ⏳ Running (48/66 batches complete)

**Parameters:**
| Parameter | Values |
|-----------|--------|
| σ_res | 10 µm (fixed, instruction default) |
| σ_scatt | 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0 mrad |
| Scale n | 3, 4, 5 |
| φ_max | 0.02 rad (small acceptance window) |
| Density | Sparse (10), Dense (100) |
| Events | 600 per scattering value |

**Purpose:**
High-statistics scan of multiple scattering effect with small angular acceptance window for precision measurements.

---

### runs_12: High-Statistics MS Scan (Large φ_max)
**Status:** ⏳ Running (65/66 batches complete)

**Parameters:**
| Parameter | Values |
|-----------|--------|
| σ_res | 10 µm (fixed, instruction default) |
| σ_scatt | 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0 mrad |
| Scale n | 3, 4, 5 |
| φ_max | 0.2 rad (large acceptance window) |
| Density | Sparse (10), Dense (100) |
| Events | 300 per scattering value |

**Purpose:**
High-statistics scan with large angular acceptance window to compare φ_max effect on reconstruction.

---

### Previous Experiments (Archived)

Results from earlier exploratory runs are in `results/archive/`:

- `runs_1`: Initial algorithm tests
- `runs_2-4`: Early scale sweeps
- `runs_5`: Large parameter scan (superseded by runs_9)

## Analysis Notebooks

| Notebook | Description |
|----------|-------------|
| `track_density_study_runs8.ipynb` | Runs 8 analysis: scale & ERF study |
| `track_density_study_runs9.ipynb` | Runs 9 analysis: extended parameter scan |
| `track_density_study_runs10.ipynb` | Runs 10 analysis: instruction parameters |
| `track_density_study_summary.ipynb` | Combined analysis & track density study |

## Reports

- `track_density_study_report.pdf` - Comprehensive LaTeX report with all findings

## License

See LICENSE.txt
