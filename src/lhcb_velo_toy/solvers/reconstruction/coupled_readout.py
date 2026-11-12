"""
Coupling-following track readout, with optional hit uniqueness.

The package rule :func:`~lhcb_velo_toy.solvers.reconstruction.track_finder.get_tracks`
puts any two active segments that share a hit into one track.  On dense events a
few false segments that share a hit with two true chains are enough to join them,
and the groups grow far beyond any single track (Run 3 minimum bias, 2026-11-11:
groups of 300 to 650 hits).

This readout links two active segments only when the Hamiltonian couples them:
the off-diagonal entry ``A[i, j]`` is nonzero.  For the segment Hamiltonians of
the package that means the end hit of one is the start hit of the other and the
angle between them is below ``epsilon``.  A track is then a connected component
of the active segments under the coupling graph, so no track can be larger than
the Hamiltonian component it sits in.

Two segments that leave (or enter) the same hit are never coupled, but both can
still join the same component through a common neighbour, so a component can
branch.  ``uniqueness`` removes the branches first with the per-(hit, role) slot
filter of :mod:`lhcb_velo_toy.solvers.postselect`: each hit keeps at most one
incoming and one outgoing active segment, the one with the largest |amplitude|.
After that every component is a simple chain.

Everything is array-based (no ``Segment`` objects), so it runs on events with
10^6 segments.  The readout is classical and acts after any solver; it does not
change the register, the qubit count or the gate count of the solve.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from scipy.sparse import spmatrix
    from lhcb_velo_toy.generation.entities.event import Event
    from lhcb_velo_toy.generation.entities.track import Track
    from lhcb_velo_toy.generation.generators.state_event import StateEventGenerator
    from lhcb_velo_toy.solvers.hamiltonians.base import Hamiltonian

__all__ = ["coupled_hit_groups", "get_tracks_coupled"]


def coupled_hit_groups(
    A: "spmatrix",
    seg_hits: np.ndarray,
    active: np.ndarray,
    solution: Optional[np.ndarray] = None,
    uniqueness: Optional[str] = None,
    kappa: float = 0.8,
    min_hits: int = 2,
    sig_digits: Optional[int] = 9,
) -> list[np.ndarray]:
    """
    Group active segments along the Hamiltonian couplings and return their hits.

    Parameters
    ----------
    A : scipy sparse matrix, shape (n_seg, n_seg)
        Segment Hamiltonian (``hamiltonian.A``).  Only the pattern of the
        off-diagonal entries is used: ``A[i, j] != 0`` links segments i and j.
    seg_hits : numpy.ndarray, shape (n_seg, 2)
        [start_hit_id, end_hit_id] per segment (``hamiltonian._segment_to_hit_ids``).
    active : numpy.ndarray of bool, shape (n_seg,)
        Segments switched on by the solver reading (e.g. ``solution > tau``).
    solution : numpy.ndarray, optional
        Segment amplitudes.  Required when ``uniqueness`` is set (the slot
        winner is the largest |amplitude|).
    uniqueness : {None, "greedy", "strict", "margin"}, default None
        Hit-uniqueness mode of :func:`~lhcb_velo_toy.solvers.postselect.hit_uniqueness_filter`,
        applied to the active set before grouping.  None groups the active set as it is.
    kappa : float, default 0.8
        Margin for ``uniqueness="margin"``.
    min_hits : int, default 2
        Drop groups with fewer hits.  2 keeps every group, as ``get_tracks`` does.
    sig_digits : int or None, default 9
        Significant digits of |solution| used by the uniqueness ranking; ties
        are then broken by (start hit id, end hit id), so the tracks do not
        depend on the segment order or on round-off below this precision.
        None restores the original ranking (ties by segment index; round-off
        can pick the winner among exact ties).

    Returns
    -------
    list[numpy.ndarray]
        One array of hit ids per group, sorted by hit id.

    Examples
    --------
    >>> groups = coupled_hit_groups(ham.A, ham._segment_to_hit_ids, x > tau,
    ...                             solution=x, uniqueness="greedy")
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    seg_hits = np.asarray(seg_hits, dtype=np.int64)
    active = np.asarray(active, dtype=bool).ravel()
    if uniqueness is not None:
        if solution is None:
            raise ValueError("uniqueness needs the solution amplitudes")
        from lhcb_velo_toy.solvers.postselect import hit_uniqueness_filter
        active = hit_uniqueness_filter(solution, seg_hits, active, mode=uniqueness, kappa=kappa,
                                       sig_digits=sig_digits)
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return []

    # coupling graph restricted to the active segments
    sub = A.tocsr()[idx][:, idx].tocoo()
    off = (sub.row != sub.col) & (sub.data != 0)
    m = idx.size
    G = coo_matrix((np.ones(int(off.sum())), (sub.row[off], sub.col[off])), shape=(m, m))
    _, lab = connected_components(G, directed=False)

    # hits of each component: both endpoints of its segments, deduplicated
    h = seg_hits[idx]
    comp = np.r_[lab, lab]
    hit = np.r_[h[:, 0], h[:, 1]]
    pairs = np.unique(np.c_[comp, hit], axis=0)
    cuts = np.flatnonzero(np.diff(pairs[:, 0])) + 1
    groups = np.split(pairs[:, 1], cuts)
    return [g for g in groups if g.size >= min_hits]


def get_tracks_coupled(
    hamiltonian: "Hamiltonian",
    solution: np.ndarray,
    event: Union["Event", "StateEventGenerator"],
    threshold: float = 0.0,
    uniqueness: Optional[str] = "greedy",
    kappa: float = 0.8,
    min_hits: int = 2,
    sig_digits: Optional[int] = 9,
) -> list["Track"]:
    """
    Extract tracks by following the Hamiltonian couplings (see module docstring).

    Same call signature as :func:`~lhcb_velo_toy.solvers.reconstruction.track_finder.get_tracks`;
    works with ``construct_segments(..., materialize_segments=False)``.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Constructed Hamiltonian (needs ``A`` and ``_segment_to_hit_ids``).
    solution : numpy.ndarray
        Segment activation vector.
    event : Event or StateEventGenerator
        The event (hits are referred to by id; kept for signature parity).
    threshold : float, default 0.0
        Segments with ``solution > threshold`` are active.
    uniqueness : {"greedy", "strict", "margin", None}, default "greedy"
        Hit-uniqueness mode applied before grouping; None turns it off.
    kappa : float, default 0.8
        Margin for ``uniqueness="margin"``.
    min_hits : int, default 2
        Minimum hits per track.
    sig_digits : int or None, default 9
        See :func:`coupled_hit_groups`.

    Returns
    -------
    list[Track]
        Reconstructed tracks, hits ordered by z.

    Examples
    --------
    >>> ham = SimpleHamiltonianFast(epsilon=0.002, gamma=3.0, delta=1.0)
    >>> ham.construct_segments(event, materialize_segments=False)
    >>> ham.construct_hamiltonian(event)
    >>> x = ham.solve_classicaly()
    >>> tracks = get_tracks_coupled(ham, x, event, threshold=0.3)
    """
    from lhcb_velo_toy.generation.entities.track import Track

    solution = np.asarray(solution).ravel()
    groups = coupled_hit_groups(
        hamiltonian.A, hamiltonian._segment_to_hit_ids, solution > threshold,
        solution=solution, uniqueness=uniqueness, kappa=kappa, min_hits=min_hits,
        sig_digits=sig_digits,
    )
    z = {h.hit_id: h.z for h in event.hits}
    return [
        Track(track_id=i, hit_ids=sorted((int(k) for k in g), key=lambda k: z[k]))
        for i, g in enumerate(groups)
    ]
