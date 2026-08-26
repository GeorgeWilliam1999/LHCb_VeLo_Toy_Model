# Changelog

## 2.1.0 — 2026-08-26 (centralise-2.1)

The package becomes the centralised producer of results: every algorithm in
the events → Hamiltonians → solvers → tracks → metrics pipeline now lives
here. Additive release — no existing API changed. Each promoted component
carries a provenance docstring naming its originating study.

### Added
- **Hamiltonians** — `ExtendedHamiltonian`, `ForkHamiltonian`,
  `OccupancyHamiltonian` (`solvers/hamiltonians/extended.py`): Denby–Peterson
  epsilon-windowed fork and per-hit occupancy penalties composed onto
  `SimpleHamiltonianFast`, with the attractor-aware `working_point_tau()`.
  From `Toy_Characterisation/Bifurification/`.
- **1BQF** — `solve_one_bit_matrixfree` (`solvers/quantum/matrixfree.py`):
  the matrix-free statevector engine (the Aer assemble-circuit OOM fix);
  documented large-T backend. From `_shared/obqf_matrixfree.py`.
- **QSVT** — `QSVT.solve_eigh()` plus `solvers/quantum/spectral_filter.py`
  (`mode_system`, `apply_response`, `continuation_components`,
  `binned_response`, `chebyshev_realize`, `spectral_bounds`): the exact
  per-component eigendecomposition evaluator behind the Codesign /
  Efficiency-Frontier results, as an official backend equivalent to
  `solve_statevector`.
- **Hough** — `solvers/hough/`: point- and locus-vote track finding
  (directional accumulator, smoothed peak finding, height-priority locus
  claim) with `validate_hough` routing candidates through Track objects and
  `EventValidator`. From `Hough/`.
- **Post-selection** — `solvers/postselect.py`: `hit_uniqueness_filter`,
  `uniqueness_frontier` (greedy/strict/margin/anyrole). From
  `qtrk_pipeline/postselect.py`.
- **Working points** — `analysis/working_points.py`: gamma-aware
  `threshold_for` (0.35 at gamma=3), the wp99 efficiency-first headline
  point (`working_point_threshold`, `metrics_at_wp`, `quantum_metrics_wp`),
  the signal-support quantum rescale convention (`rescale_to_signal`,
  `quantum_metrics`), and `truth_from_event`. From `qtrk_pipeline/metrics.py`.
- **Run-3 adapter** — `generation/adapters/run3.py`: `load_run3_event`
  Allen-JSON duck-typed adapter incl. the left/right half-detector fix.
  From `Run3_Verification/run3_loader.py`.
- **Reproducibility** — `StateEventGenerator(seed=...)`.
- **Tests** — `tests/`: golden baselines on seeded events (three regimes,
  classical + 1BQF + QSVT), the strict `cos(angle) > cos(eps)` boundary
  ruling, brute-force-verified Hamiltonian terms, backend equivalence
  suites, and equality cross-checks against every originating study module.

### Decisions recorded
- Epsilon boundary: a pair exactly at epsilon is EXCLUDED (strict `>`);
  `run_fig14_extend.py`'s `>=` is a known historical deviation.
- 2026-08-26 benchmark: canonical `fast.py` beats the local Numba
  `AcceleratedHamiltonian` ~8x and the notebook vectorised builders ~6x at
  T=1000 with bit-identical A — the local builders are retired and no numba
  backend is needed.
