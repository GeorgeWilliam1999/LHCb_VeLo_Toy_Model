"""
Quantum algorithm implementations for track finding — the quantum solver lanes.

Three solvers act on the segment linear system ``A x = b``:

* ``HHLAlgorithm`` — full HHL (multi-qubit phase estimation + controlled
  rotation); the reference implementation, byte-identical to
  ``OneBQF_repo/quantum_algorithms/HHL.py``.
* ``OneBQF`` (alias ``OneBitHHL``) — the one-bit quantum filter (1BQF): one
  clock qubit, the cosine response ``cos(lambda t / 2)``; reference circuit
  byte-identical to ``OneBQF_repo/quantum_algorithms/OneBQF.py``, plus the
  matrix-free large-T engine ``solve_one_bit_matrixfree``.
* ``QSVT`` — the polynomial spectral filter ("the comb"): any bounded
  polynomial response ``p(A)`` applied as a linear combination of Chebyshev
  polynomials of the qubitization walk, of which the 1BQF cosine is the
  degree-one member.  Response families: ``design_line_comb_inverse`` (THE
  QSVT of the papers, ``filter_design="comb"`` — the constructor default),
  ``design_band_limited_inverse`` (the retired first design),
  ``design_minimax_comb``, ``design_one_bit_cosine``,
  ``design_normalized_comb`` (+ ``normalized_coupling`` /
  ``solve_normalized_comb``), ``design_trig_comb`` (the QETU realisation's
  response).  Realisations: LCU (the build), the phase sequence
  (``qsp_phases``: ``find_phase_factors``, ``apply_phase_sequence``,
  ``QSVT.solve_phase_sequence``), QETU.  Exact evaluators:
  ``QSVT.solve_statevector`` (Chebyshev recursion), ``QSVT.solve_eigh`` /
  ``spectral_filter`` (per-component eigendecomposition; any response
  callable through ``apply_response``).  Hit-register filters on the
  vertex-conditioned hit operator (cliques, not segment chains):
  ``clique_filters``.  Supervised refit of the
  coefficients to a measured event: ``refit`` (``chebyshev_moments``,
  ``refit_coefficients``, ``refit_binned``).

Dependency note
---------------
``QSVT`` (statevector / eigh / phase-sequence backends), ``spectral_filter``,
``responses``, ``qsp_phases``, ``refit`` and ``solve_one_bit_matrixfree`` are
pure numpy / scipy.  Only ``HHLAlgorithm``, ``OneBQF`` / ``OneBitHHL`` and
``QSVT.build_circuit`` / ``run_circuit`` need the ``[quantum]`` extras
(qiskit, qiskit-aer, qiskit-ibm-runtime); the two circuit classes are
imported lazily on first attribute access so that ``from
lhcb_velo_toy.solvers.quantum import QSVT`` works in a base install.
"""
from __future__ import annotations

from lhcb_velo_toy.solvers.quantum.QSVT import (
    DEFAULT_DESIGN,
    FILTER_DESIGNS,
    QSVT,
    design_band_limited_inverse,
    design_line_comb_inverse,
    design_minimax_comb,
)
from lhcb_velo_toy.solvers.quantum.clique_filters import (
    design_step_polynomial,
    filter_hit_operator,
    ideal_line_projector,
    ideal_step,
    one_bit_clique,
    soft_band,
    tracks_from_filtered_hits,
)
from lhcb_velo_toy.solvers.quantum.matrixfree import solve_one_bit_matrixfree
from lhcb_velo_toy.solvers.quantum.qsp_phases import (
    apply_phase_sequence,
    find_phase_factors,
    parity_split,
    phase_sequence_solve,
    qsp_signal,
    wx_to_reflection,
)
from lhcb_velo_toy.solvers.quantum.refit import (
    RefitResult,
    chebyshev_moments,
    refit_binned,
    refit_coefficients,
)
from lhcb_velo_toy.solvers.quantum.responses import (
    TrigResponse,
    design_normalized_comb,
    design_one_bit_cosine,
    design_trig_comb,
    normalized_coupling,
    one_bit_response,
    p4_lines,
    solve_normalized_comb,
)
from lhcb_velo_toy.solvers.quantum.spectral_filter import (
    apply_response,
    binned_response,
    chebyshev_realize,
    continuation_components,
    mode_system,
    spectral_bounds,
)

# The two qiskit-based reference circuits are imported lazily (PEP 562) so the
# numpy-only solvers above do not require the [quantum] extras.
#
# The 1BQF module is spelled ``one_bqf`` and NOT ``OneBQF``: a submodule whose
# name matches a class exported here is bound over that class by the import
# machinery, so ``from ...quantum import OneBQF`` returned the MODULE once
# anything had imported the submodule first (e.g. the ``one_bit_hhl`` shim).
_LAZY = {
    "HHLAlgorithm": ("lhcb_velo_toy.solvers.quantum.hhl", "HHLAlgorithm"),
    "OneBQF": ("lhcb_velo_toy.solvers.quantum.one_bqf", "OneBQF"),
    # Backwards-compatible alias: previous code imported the 1-Bit HHL
    # implementation as ``OneBitHHL``; the verified reference class is
    # called ``OneBQF`` and is functionally equivalent.
    "OneBitHHL": ("lhcb_velo_toy.solvers.quantum.one_bqf", "OneBQF"),
}


def __getattr__(name):
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        try:
            import importlib
            module = importlib.import_module(module_name)
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ImportError(
                f"{name} needs the quantum extras: pip install 'lhcb-velo-toy[quantum]' "
                f"({exc})"
            ) from exc
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    # circuits (lazy, need qiskit)
    "HHLAlgorithm",
    "OneBQF",
    "OneBitHHL",
    # QSVT solver + designs
    "QSVT",
    "DEFAULT_DESIGN",
    "FILTER_DESIGNS",
    "design_band_limited_inverse",
    "design_line_comb_inverse",
    "design_minimax_comb",
    "design_one_bit_cosine",
    "design_normalized_comb",
    "design_trig_comb",
    "normalized_coupling",
    "solve_normalized_comb",
    "one_bit_response",
    "p4_lines",
    "TrigResponse",
    # phase-sequence realisation
    "parity_split",
    "qsp_signal",
    "wx_to_reflection",
    "find_phase_factors",
    "apply_phase_sequence",
    "phase_sequence_solve",
    # refit
    "RefitResult",
    "chebyshev_moments",
    "refit_coefficients",
    "refit_binned",
    # 1BQF matrix-free engine
    "solve_one_bit_matrixfree",
    # spectral evaluator
    "apply_response",
    "binned_response",
    "chebyshev_realize",
    "continuation_components",
    "mode_system",
    "spectral_bounds",
    # hit-register clique filters
    "ideal_line_projector",
    "ideal_step",
    "soft_band",
    "design_step_polynomial",
    "one_bit_clique",
    "filter_hit_operator",
    "tracks_from_filtered_hits",
]
