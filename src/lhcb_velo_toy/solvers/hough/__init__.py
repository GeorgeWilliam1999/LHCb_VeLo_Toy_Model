"""Direct Hough-transform track finding — the fourth solver lane.

Point and locus voting on the directional accumulator (constant-in-T
registers), with candidates validated through the canonical EventValidator.
"""

from lhcb_velo_toy.solvers.hough.finder import (
    MIN_HITS,
    TX_RANGE,
    TY_RANGE,
    accumulate,
    accumulate_locus,
    assign,
    assign_locus,
    bin_width,
    claim_radius,
    event_arrays,
    find_peaks,
    hit_directions,
    locus_curves,
    n_zeta_for,
    run_locus,
    run_point,
)
from lhcb_velo_toy.solvers.hough.adapter import (
    candidates_to_tracks,
    validate_hough,
)

__all__ = [
    "TX_RANGE", "TY_RANGE", "MIN_HITS",
    "hit_directions", "event_arrays", "bin_width",
    "accumulate", "find_peaks", "claim_radius", "assign", "run_point",
    "locus_curves", "accumulate_locus", "n_zeta_for", "assign_locus",
    "run_locus",
    "candidates_to_tracks", "validate_hough",
]
