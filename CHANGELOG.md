# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Modern `pyproject.toml` packaging configuration
- GitHub Actions CI/CD workflow
- Contributing guidelines (`CONTRIBUTING.md`)
- Type hints support with `py.typed` marker
- Comprehensive test suite structure
- Pre-commit hooks configuration

### Changed
- Updated README with production-ready documentation
- Improved project structure for better maintainability

## [1.0.0] - 2026-01-26

### Added
- **Core Framework**
  - `StateEventGenerator` for toy detector event simulation
  - `SimpleHamiltonian` for ERF-smoothed track reconstruction
  - `EventValidator` for computing tracking metrics
  - `Hit`, `Track`, `Segment`, `Module` data models

- **Quantum Algorithms**
  - HHL (Harrow-Hassidim-Lloyd) algorithm implementation using Qiskit
  - Simplified 1-bit HHL variant for testing

- **Experiment Infrastructure**
  - `ExperimentConfig` for parameter configuration
  - `run_experiment()` for single experiment execution
  - `run_batch()` for batch processing
  - HTCondor job submission scripts

- **Analysis Tools**
  - Results aggregation from batch jobs
  - Statistical analysis functions
  - Publication-quality plotting utilities

- **Documentation**
  - Comprehensive README with examples
  - Analysis notebooks for various parameter studies
  - LaTeX report for track density study

### Performance Metrics
- Reconstruction efficiency
- Ghost rate
- Clone fraction
- Track purity
- Hit efficiency

## [0.1.0] - 2025-06-15

### Added
- Initial project structure
- Basic Hamiltonian formulation
- Simple event generator
- Preliminary validation metrics

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2026-01-26 | Production-ready release with quantum HHL |
| 0.1.0 | 2025-06-15 | Initial development release |

[Unreleased]: https://github.com/GeorgeWilliam1999/LHCb_VeLo_Toy_Model/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/GeorgeWilliam1999/LHCb_VeLo_Toy_Model/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/GeorgeWilliam1999/LHCb_VeLo_Toy_Model/releases/tag/v0.1.0
