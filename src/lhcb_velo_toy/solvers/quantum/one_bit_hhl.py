"""
Backwards-compatibility shim.

Previous code imported the 1-Bit HHL solver as::

    from lhcb_velo_toy.solvers.quantum.one_bit_hhl import OneBitHHL

The canonical implementation now lives in
:mod:`lhcb_velo_toy.solvers.quantum.one_bqf` (a byte-identical copy of
the verified reference in ``OneBQF_repo/quantum_algorithms/OneBQF.py``)
and is exposed under its reference name ``OneBQF``. This module
re-exports it as ``OneBitHHL`` so existing imports continue to work.
"""

from lhcb_velo_toy.solvers.quantum.one_bqf import OneBQF as OneBitHHL

__all__ = ["OneBitHHL"]
