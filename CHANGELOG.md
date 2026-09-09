# Changelog

## 2.2.0 — 2026-09-07 (centralise-2.1, unreleased)

The QSVT lane becomes a complete, discoverable solver family: every response
family, realisation and evaluator used by the QSVT studies and the QSVT VeLo
paper is now in the package, importable without qiskit, tested against the
originating study code, and documented. Additive; the golden baselines are
unchanged.

### Changed (deliberate physics decisions, George 2026-09-07)
- **QSVT default design is the comb.** `QSVT(A, b)` now builds the degree-40
  line comb — the QSVT of the papers — instead of the retired band design.
  The band remains available as `filter_design="band"`; the golden baseline
  pins both. Pipeline callers always passed the polynomial explicitly, so no
  stored result changes.
- **1BQF uncompute fixed.** `OneBQF._apply_direct_controlled_u(inverse=True)`
  and `solve_one_bit_matrixfree` now apply the inverse pair rotations in
  reverse order, which makes the uncompute the exact inverse of the forward
  pair product (`test_inverse_evolution_is_exact_inverse`: 1e-17, against
  5e-2 for the old forward order). That removes the gross labelling artefact
  — a true track no longer mirrors under reversed segment labels — and
  brings the readout to a cosine of 0.99999 (clean) / 0.9996 (moderate) of
  the exact-evolution readout. A residual label-order dependence of order
  1e-2 in amplitude REMAINS: that is the forward per-pair product formula
  for `exp(-iAt)`, not the uncompute, and it is not claimed to be fixed.
  The 6,008 stored 1BQF solutions in the qtrk store and the paper's figures
  10 and 12 predate the fix (Notion to-do to re-run).
- **Qiskit 2.x.** `QFT` → `QFTGate`, controlled gates built with
  `annotated=False`, `CircuitInstruction` attribute access, HHL statevector
  path on `AerSimulator(method="statevector")`, preset pass manager from
  `qiskit.transpiler`; `qiskit-ibm-runtime` imported only when a noise model
  is requested. Extras now require `qiskit>=2.1`, `qiskit-aer>=0.17`,
  `qiskit-ibm-runtime>=0.40`. The pytest configuration turns any deprecation
  warning attributed to `lhcb_velo_toy` into a failure.

### Fixed
- **numba is optional again.** `analysis/segment_metrics.py` imported numba at
  module level while declaring it in no dependency list, so
  `import lhcb_velo_toy` failed outright on any environment without it — the
  package was uninstallable-in-practice from a clean `pip install`. The import
  is now guarded with an equivalent pure-Python fallback (verified to give
  identical counts), numba moves to a new optional `[fast]` extra, and
  `HAS_NUMBA` reports which kernel is live. Regression test added.
- **`solvers/quantum/OneBQF.py` renamed to `one_bqf.py`** (the class stays
  `OneBQF`). A submodule whose name matches a class exported from the package
  is bound over that class by the import machinery, so
  `from lhcb_velo_toy.solvers.quantum import OneBQF` returned the *module* —
  not callable — once anything had imported the submodule first, which the
  documented `one_bit_hhl` compatibility shim does. Regression test added.
  Code importing `...quantum.OneBQF` directly must now import
  `...quantum.one_bqf` (or, better, the class from the package).
- **`OneBQF.get_success_probability()` restored**: the API reference documented
  it and `notebooks/quantum_1bqf_end_to_end.ipynb` called it, but the method did
  not exist on the class — the example could not run to completion. It returns
  P(ancilla = 1) from the sampled counts (the shot-free exact value is the
  second return of `get_solution_from_statevector`).
- **Two false claims removed from the 1BQF example.** Its summary stated that
  exact Givens rotations mean "no Trotter approximation error" and that shot
  noise is the only error: the per-pair rotations are individually exact, but
  rotations sharing a segment do not commute, so their product is a
  first-order product formula and a residual ordering error of order 1e-2 in
  amplitude survives (now measured in the notebook). It also called the 1BQF
  "the recommended quantum approach ... combining quantum advantage claims";
  that is superseded by the QSVT results and by the dequantisation finding
  that every response here is classically computable.
- **Example notebooks re-executed** against this release, so every committed
  output matches the current code: `demo_workflow.ipynb` and
  `notebooks/classical_end_to_end.ipynb` (the `SimpleHamiltonian` threshold fix),
  `notebooks/quantum_1bqf_end_to_end.ipynb` (the uncompute fix) and
  `notebooks/quantum_hhl_end_to_end.ipynb`.
- **`SimpleHamiltonian` hard threshold** (reference builder): the step
  compatibility compared `1 - cos(angle)` with `epsilon` — an effective
  acceptance of `sqrt(2 epsilon)` (141 mrad at epsilon = 0.01) — and so
  coupled segment pairs far outside the window. It now tests
  `cos(angle) > cos(epsilon)` like `SimpleHamiltonianFast`; the two builders
  agree entry for entry (new test). Results produced with `SimpleHamiltonian`
  (the README quick start, `demo_workflow.ipynb`, `test_pipeline.py`) change;
  every study result was produced with the fast builder and is unaffected.
- **Plotting**: `get_cmap` (removed in matplotlib 3.11) replaced by
  `matplotlib.colormaps.get_cmap`.

### Added
- **QSVT design selector** — `QSVT(A, b, degree, filter_design=...)` with
  `"comb"` (default, **the QSVT of the papers**: the degree-40 line comb),
  `"band"` (the retired first design), `"minimax"`, `"cosine"`;
  `QSVT.comb(A, b)` shorthand; the design centre `s = gamma + delta` is now
  read off a constant diagonal (a gamma = 1 Hamiltonian gets its own lines;
  explicit `s=` overrides).
- **Response families** — `solvers/quantum/responses.py`:
  `design_normalized_comb` / `normalized_coupling` / `solve_normalized_comb`
  (the +-1/2 comb on the degree-normalised coupling, Block-Encoding III),
  `one_bit_response` / `design_one_bit_cosine` (the one-bit filter's exact
  cosine response as a degree-60 series, the `onebqf_cos` family),
  `design_trig_comb` / `TrigResponse` (the QETU cosine series over
  `exp(-iAt)`), `p4_lines`. From `QSVT/Efficiency_Frontier/*.py` and
  `QSVT/Paper_planning/section4_qsvt/02_resource_model.py`.
- **Phase-sequence realisation (QSVT proper)** — `solvers/quantum/qsp_phases.py`:
  `parity_split`, `find_phase_factors` (symmetric phase factors, Dong et al.
  2021, with the wx -> reflection convention conversion), `qsp_signal`,
  `apply_phase_sequence` (dense dilation circuit), `phase_sequence_solve`;
  `QSVT.solve_phase_sequence()` as a fourth backend. From
  `QSVT/Paper_planning/section4_qsvt/01_phase_factors.py` + `01b_convention_check.py`.
- **Resources by realisation** — `QSVT.resources(realisation="lcu"|"phase"|"qetu")`
  and `QSVT.resource_table()` (ancillas, calls per shot, the LCU one-norm
  penalty `(||c||_1 / max|p|)^2`), the paper's table of realisations.
- **Coefficient refit** — `solvers/quantum/refit.py`: `chebyshev_moments`,
  `refit_coefficients` (`fitted_moment`), `refit_binned` (`fitted_ridge`),
  `RefitResult` (with a QSVT-realisable `.poly` for the transfer test). A
  supervised diagnostic, flagged as under audit (2026-09-03 catch-up). From
  `QSVT/Efficiency_Frontier/01-03*.py`, `QSVT/Codesign/04,06*.py`.
- **Lazy qiskit** — `lhcb_velo_toy.solvers.quantum` imports `HHLAlgorithm`,
  `OneBQF` / `OneBitHHL` on first access (PEP 562); `QSVT`, `spectral_filter`,
  `responses`, `qsp_phases`, `refit` and `solve_one_bit_matrixfree` are
  numpy/scipy-only and now import in a base install. `QSVT`,
  `design_line_comb_inverse`, `solve_one_bit_matrixfree` exported at the
  top level and from `lhcb_velo_toy.solvers`.
- **Example** — `notebooks/quantum_qsvt_end_to_end.ipynb` (executed):
  responses, three backends, three realisations, a moderate event with
  efficiency-first working points and tracks, scaling with T, the other
  response families, the refit + transfer, the phase factors themselves
  (`parity_split` / `find_phase_factors` / `apply_phase_sequence`, with the
  parity penalty and the scalar-identity error), and a response fitted to the
  event's own spectrum (`refit_binned` -> `binned_response` ->
  `chebyshev_realize` at degrees 16/24/40).
- **1BQF example** — `notebooks/quantum_1bqf_end_to_end.ipynb` gains a section
  that shows the filter is *correct* and that it scales: the matrix-free engine
  against the circuit (cosine 1.0000000000), the circuit against the exact
  one-bit readout computed by diagonalising A, the uncompute demonstrated as an
  exact inverse (3e-17, against 2.6e-1 for the old forward order), and a
  6,400-segment solve in 0.01 s on a 0.5 MB statevector.
- **Tests** — `tests/test_qsvt_models.py`: selector == pipeline path,
  default is the comb, s inference, qiskit-free import, normalized comb and
  one-bit cosine == originating study code, phase-factor scalar identity +
  circuit, phase sequence == LCU amplitudes with the predicted success
  ratio, moments == recursion, refit round trip.
  `tests/test_quantum_circuits.py`: OneBQF sampling vs statevector readout,
  the uncompute as an exact inverse (and that forward order is not), the
  circuit against the exact-evolution readout, the restored
  success-probability accessor, HHL build / run / statevector consistency,
  QSVT qiskit circuit == matrix-free backend, and a no-deprecation guard for
  every circuit path. `tests/test_core_pipeline.py`:
  seeded generation, noisy events, Simple == Fast Hamiltonians, tracks +
  EventValidator, persistence round trips, plotting smoke.
- **Docs** — README, `docs/` (API reference, flow diagrams, dependencies,
  workflow) rewritten to cover every 2.1.0 / 2.2.0 lane.

### Notes
- `resources()` keeps its legacy keys and gains `ancillas`, `calls_per_shot`,
  `peak`, `lcu_penalty`, `realisation`.
- Golden baselines re-pinned for the two physics changes above (comb default,
  1BQF uncompute); the classical values are unchanged.

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
