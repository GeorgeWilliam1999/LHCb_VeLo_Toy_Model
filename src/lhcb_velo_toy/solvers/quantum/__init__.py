"""
Quantum algorithm implementations for track finding.

The canonical algorithms are byte-identical copies of the verified
reference implementations from
``OneBQF_repo/quantum_algorithms/{HHL.py, OneBQF.py}``. They are
re-exported here under their reference names (``HHLAlgorithm``,
``OneBQF``) and under the legacy alias ``OneBitHHL`` for backwards
compatibility with existing notebooks and scripts.
"""

from lhcb_velo_toy.solvers.quantum.HHL import HHLAlgorithm
from lhcb_velo_toy.solvers.quantum.OneBQF import OneBQF
from lhcb_velo_toy.solvers.quantum.QSVT import QSVT, design_band_limited_inverse, design_line_comb_inverse

# Backwards-compatible alias: previous code imported the 1-Bit HHL
# implementation as ``OneBitHHL``. The verified reference class is
# called ``OneBQF`` and is functionally equivalent.
OneBitHHL = OneBQF

__all__ = [
    "HHLAlgorithm",
    "OneBQF",
    "OneBitHHL",
    "QSVT",
    "design_band_limited_inverse",
    "design_line_comb_inverse",
]
