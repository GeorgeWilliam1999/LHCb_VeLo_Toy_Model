# Deprecated Code

> **Warning**: The code in this directory is **deprecated** and retained only
> for reference and backwards-compatibility testing.

## What moved where?

The monolithic flat package `LHCB_Velo_Toy_Models/` has been replaced by the
modular, installable `lhcb_velo_toy` package under `src/`.

| Old file | New location |
|----------|-------------|
| `state_event_model.py` | `src/lhcb_velo_toy/generation/models/` (split into `hit.py`, `track.py`, `module.py`, `event.py`, `segment.py`) |
| `state_event_generator.py` | `src/lhcb_velo_toy/generation/generators/state_event.py` |
| `multi_scattering_generator.py` | *Not yet migrated* |
| `hamiltonian.py` | `src/lhcb_velo_toy/solvers/hamiltonians/base.py` |
| `simple_hamiltonian.py` | `src/lhcb_velo_toy/solvers/hamiltonians/simple.py` |
| `simple_hamiltonian_fast.py` | `src/lhcb_velo_toy/solvers/hamiltonians/fast.py` |
| `simple_hamiltonian_cpp.py` | *Not yet migrated* |
| `toy_validator.py` | `src/lhcb_velo_toy/analysis/validation/validator.py` |
| `lhcb_tracking_plots.py` | `src/lhcb_velo_toy/analysis/plotting/` (split into `event_display.py`, `performance.py`) |
| `HHL.py` | `src/lhcb_velo_toy/solvers/quantum/hhl.py` |
| `OneBQF.py` | `src/lhcb_velo_toy/solvers/quantum/one_bit_hhl.py` |

## Migration guide

Replace old-style imports:

```python
# Old
from LHCB_Velo_Toy_Models.state_event_generator import StateEventGenerator
from LHCB_Velo_Toy_Models.state_event_model import PlaneGeometry
from LHCB_Velo_Toy_Models.simple_hamiltonian import SimpleHamiltonian

# New
from lhcb_velo_toy.generation import StateEventGenerator, PlaneGeometry
from lhcb_velo_toy.solvers import SimpleHamiltonian
```

Install the new package with:

```bash
pip install -e ".[all]"
```
