"""
Backwards-compatibility shim.

Previous code imported the full HHL solver as::

    from lhcb_velo_toy.solvers.quantum.hhl import HHLAlgorithm

The canonical implementation now lives in
:mod:`lhcb_velo_toy.solvers.quantum.HHL` (a byte-identical copy of the
verified reference in ``OneBQF_repo/quantum_algorithms/HHL.py``). This
module re-exports it so existing imports continue to work.
"""

from lhcb_velo_toy.solvers.quantum.HHL import HHLAlgorithm

__all__ = ["HHLAlgorithm"]
