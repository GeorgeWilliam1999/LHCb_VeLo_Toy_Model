"""
Track finding utilities for extracting tracks from Hamiltonian solutions.

Functions for converting segment activation vectors into reconstructed tracks
and computing segments from events on-demand.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from lhcb_velo_toy.generation.entities.hit import Hit
    from lhcb_velo_toy.solvers.reconstruction.segment import Segment
    from lhcb_velo_toy.generation.entities.track import Track
    from lhcb_velo_toy.generation.entities.module import Module
    from lhcb_velo_toy.generation.entities.event import Event
    from lhcb_velo_toy.generation.geometry.base import Geometry
    from lhcb_velo_toy.generation.generators.state_event import StateEventGenerator
    from lhcb_velo_toy.solvers.hamiltonians.base import Hamiltonian


def find_segments(
    segment: "Segment",
    active_segments: list["Segment"],
) -> list["Segment"]:
    """
    Find all segments connected to a given segment.
    
    Two segments are connected if they share an endpoint hit.
    
    Parameters
    ----------
    segment : Segment
        The reference segment to find connections for.
    active_segments : list[Segment]
        Pool of candidate segments to search.
    
    Returns
    -------
    list[Segment]
        Segments that share a hit with the reference segment.
    
    Examples
    --------
    >>> # seg1: hit_A -> hit_B
    >>> # seg2: hit_B -> hit_C
    >>> # seg3: hit_D -> hit_E (no connection)
    >>> connected = find_segments(seg1, [seg2, seg3])
    >>> len(connected)
    1
    
    Notes
    -----
    This function is used in the track-building algorithm to group
    connected segments into track candidates.
    """
    return [s for s in active_segments if segment.shares_hit_with(s) and s != segment]


def get_tracks(
    hamiltonian: "Hamiltonian",
    solution: np.ndarray,
    event: Union["Event", "StateEventGenerator"],
    threshold: float = 0.0,
) -> list["Track"]:
    """
    Extract tracks from a Hamiltonian solution.
    
    Converts the continuous segment activation vector into discrete
    tracks by thresholding and grouping connected segments.
    
    Parameters
    ----------
    hamiltonian : Hamiltonian
        The Hamiltonian containing segment information.
    solution : numpy.ndarray
        Segment activation vector from solving A x = b.
    event : Event or StateEventGenerator
        The event containing hit and geometry information.
    threshold : float, default 0.0
        Minimum activation value for a segment to be considered active.
        Segments with solution[i] > threshold are included.
    
    Returns
    -------
    list[Track]
        Reconstructed tracks.
    
    Examples
    --------
    >>> ham = SimpleHamiltonian(epsilon=0.01, gamma=1.5, delta=1.0)
    >>> ham.construct_hamiltonian(event)
    >>> solution = ham.solve_classicaly()
    >>> tracks = get_tracks(ham, solution, event)
    
    Notes
    -----
    Algorithm:
    1. Filter segments where activation > threshold
    2. Build adjacency graph of connected segments
    3. Find connected components via depth-first search
    4. Convert each component to a Track object
    5. Order hits within each track by z coordinate
    """
    from lhcb_velo_toy.generation.entities.track import Track
    
    # Get active segments
    active_indices = np.where(solution > threshold)[0]
    active_segments = [hamiltonian.segments[i] for i in active_indices]
    
    if not active_segments:
        return []
    
    # Group segments into tracks
    segment_groups = _group_segments_into_tracks(active_segments)
    
    # Convert groups to Track objects
    tracks = []
    for track_id, group in enumerate(segment_groups):
        track = _segments_to_track(group, track_id)
        tracks.append(track)
    
    return tracks


def get_tracks_fast(
    hamiltonian: "Hamiltonian",
    solution: np.ndarray,
    event: Union["Event", "StateEventGenerator"],
    threshold: float = 0.0,
) -> list["Track"]:
    """
    Optimized track extraction using vectorized operations.
    
    Parameters
    ----------
    hamiltonian : Hamiltonian
        The Hamiltonian containing segment information.
    solution : numpy.ndarray
        Segment activation vector.
    event : Event or StateEventGenerator
        The event containing hit and geometry information.
    threshold : float, default 0.0
        Minimum activation threshold.
    
    Returns
    -------
    list[Track]
        Reconstructed tracks.
    
    Notes
    -----
    Uses pre-computed data structures from SimpleHamiltonianFast
    for improved performance.
    """
    # For now, use the same implementation as get_tracks
    # Future optimization can use pre-computed adjacency matrices
    return get_tracks(hamiltonian, solution, event, threshold)


# ---------------------------------------------------------------------------
#  Advanced reconstruction methods (ported from legacy simple_hamiltonian_fast)
# ---------------------------------------------------------------------------


def get_tracks_layered(
    hamiltonian: "Hamiltonian",
    solution: np.ndarray,
    event: Union["Event", "StateEventGenerator"],
    threshold: float = 0.45,
    min_hits: int = 3,
) -> list["Track"]:
    """
    Extract tracks using layered greedy chain-following with angle consistency.

    The algorithm:
    1. Identifies active segments (solution > *threshold*) sorted by score.
    2. Builds connectivity maps between consecutive layer-groups.
    3. Starting from the highest-scoring unused segment, extends
       forward and backward following the best angle-consistent
       continuation.
    4. A second pass attempts to recover tracks from remaining segments.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian with constructed segments and cached direction vectors.
    solution : numpy.ndarray
        Segment activation vector.
    event : Event or StateEventGenerator
        Event used for hit information.
    threshold : float, default 0.45
        Minimum solution value for an active segment.
    min_hits : int, default 3
        Minimum number of hits in a valid track.

    Returns
    -------
    list[Track]
        Reconstructed tracks.
    """
    from lhcb_velo_toy.generation.entities.track import Track as _Track
    from lhcb_velo_toy.solvers.reconstruction.segment import Segment as _Seg

    ham = hamiltonian

    # Step 1 – active segments sorted by score (descending)
    seg_scores = [
        (i, seg, solution[i])
        for i, seg in enumerate(ham.segments)
        if solution[i] > threshold
    ]
    seg_scores.sort(key=lambda x: -x[2])

    if not seg_scores:
        return []

    # Connectivity maps keyed on id(hit_object)
    seg_by_from_hit: dict[int, list[tuple[int, "Segment", float]]] = {}
    seg_by_to_hit: dict[int, list[tuple[int, "Segment", float]]] = {}

    for seg_idx, seg, score in seg_scores:
        from_hit = seg.hits[0]
        to_hit = seg.hits[1]
        seg_by_from_hit.setdefault(id(from_hit), []).append((seg_idx, seg, score))
        seg_by_to_hit.setdefault(id(to_hit), []).append((seg_idx, seg, score))

    segment_successors: dict[int, list] = {}
    segment_predecessors: dict[int, list] = {}

    for seg_idx, seg, score in seg_scores:
        to_hit = seg.hits[1]
        successors = seg_by_from_hit.get(id(to_hit), [])
        segment_successors[seg_idx] = [(i, s, sc) for i, s, sc in successors if i != seg_idx]

        from_hit = seg.hits[0]
        predecessors = seg_by_to_hit.get(id(from_hit), [])
        segment_predecessors[seg_idx] = [(i, s, sc) for i, s, sc in predecessors if i != seg_idx]

    # Helpers ---------------------------------------------------------------
    used_segments: set[int] = set()
    tracks: list["Track"] = []

    def _compute_angle(seg_info_1, seg_info_2):
        v1 = ham._segment_vectors[seg_info_1[0]]
        v2 = ham._segment_vectors[seg_info_2[0]]
        cos_a = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return np.arccos(cos_a)

    def _build_track(start_seg_info):
        seg_idx, seg, score = start_seg_info
        if seg_idx in used_segments:
            return None

        track_hits: list = []
        track_segments = [start_seg_info]
        used_modules: set[int] = set()

        for hit in seg.hits:
            mid = int(hit.module_id)
            if mid not in used_modules:
                track_hits.append(hit)
                used_modules.add(mid)

        # Forward extension
        current = start_seg_info
        while True:
            candidates = [
                (i, s, sc) for i, s, sc in segment_successors.get(current[0], [])
                if i not in used_segments
            ]
            if not candidates:
                break
            best_next, best_sc = None, -1.0
            for cand in candidates:
                if int(cand[1].hits[1].module_id) in used_modules:
                    continue
                if _compute_angle(current, cand) >= ham.epsilon:
                    continue
                if cand[2] > best_sc:
                    best_sc = cand[2]
                    best_next = cand
            if best_next is None:
                break
            track_segments.append(best_next)
            to_hit = best_next[1].hits[1]
            track_hits.append(to_hit)
            used_modules.add(int(to_hit.module_id))
            current = best_next

        # Backward extension
        current = start_seg_info
        while True:
            candidates = [
                (i, s, sc) for i, s, sc in segment_predecessors.get(current[0], [])
                if i not in used_segments
            ]
            if not candidates:
                break
            best_prev, best_sc = None, -1.0
            for cand in candidates:
                if int(cand[1].hits[0].module_id) in used_modules:
                    continue
                if _compute_angle(cand, current) >= ham.epsilon:
                    continue
                if cand[2] > best_sc:
                    best_sc = cand[2]
                    best_prev = cand
            if best_prev is None:
                break
            track_segments.insert(0, best_prev)
            from_hit = best_prev[1].hits[0]
            track_hits.insert(0, from_hit)
            used_modules.add(int(from_hit.module_id))
            current = best_prev

        return track_hits, track_segments

    def _accept_track(track_hits, track_segments):
        if len(track_hits) < min_hits:
            return
        for ts_idx, _, _ in track_segments:
            used_segments.add(ts_idx)
        hits_sorted = sorted(track_hits, key=lambda h: h.z)
        tracks.append(_Track(
            track_id=len(tracks),
            hit_ids=[h.hit_id for h in hits_sorted],
        ))

    # Main passes -----------------------------------------------------------
    for info in seg_scores:
        if info[0] in used_segments:
            continue
        result = _build_track(info)
        if result is not None:
            _accept_track(*result)

    remaining = [(i, s, sc) for i, s, sc in seg_scores if i not in used_segments]
    for info in remaining:
        if info[0] in used_segments:
            continue
        result = _build_track(info)
        if result is not None:
            _accept_track(*result)

    return tracks


def get_tracks_optimal(
    hamiltonian: "Hamiltonian",
    solution: np.ndarray,
    event: Union["Event", "StateEventGenerator"],
    threshold: float = 0.45,
    min_hits: int = 3,
    angle_weight: float = 1.0,
    score_weight: float = 1.0,
    verbose: bool = False,
) -> list["Track"]:
    """
    Extract tracks using global optimisation (weighted set packing).

    Instead of greedily assigning segments, this method:
    1. Builds ALL possible track candidates via DFS.
    2. Scores each candidate (solution values − angle penalties + length bonus).
    3. Selects the best non-overlapping set (greedy approximation to
       the NP-hard weighted set-packing problem).

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian with constructed segments.
    solution : numpy.ndarray
        Segment activation vector.
    event : Event or StateEventGenerator
        Event with hit information.
    threshold : float, default 0.45
        Minimum segment activation.
    min_hits : int, default 3
        Minimum hits for a valid track.
    angle_weight : float, default 1.0
        Penalty weight for kink angles.
    score_weight : float, default 1.0
        Weight for solution scores.
    verbose : bool, default False
        Print debug info.

    Returns
    -------
    list[Track]
        Globally-optimised reconstructed tracks.
    """
    from collections import defaultdict
    from lhcb_velo_toy.generation.entities.track import Track as _Track

    ham = hamiltonian

    seg_data = [
        (i, seg, solution[i])
        for i, seg in enumerate(ham.segments)
        if solution[i] > threshold
    ]
    if not seg_data:
        return []
    if verbose:
        print(f"[get_tracks_optimal] Found {len(seg_data)} active segments")

    # Connectivity -----------------------------------------------------------
    seg_by_from_hit: dict[int, list] = defaultdict(list)
    seg_by_to_hit: dict[int, list] = defaultdict(list)
    for seg_idx, seg, score in seg_data:
        seg_by_from_hit[id(seg.hits[0])].append((seg_idx, seg, score))
        seg_by_to_hit[id(seg.hits[1])].append((seg_idx, seg, score))

    seg_successors: dict[int, list] = {}
    seg_predecessors: dict[int, list] = {}
    for seg_idx, seg, score in seg_data:
        successors = seg_by_from_hit.get(id(seg.hits[1]), [])
        seg_successors[seg_idx] = [(i, s, sc) for i, s, sc in successors if i != seg_idx]
        predecessors = seg_by_to_hit.get(id(seg.hits[0]), [])
        seg_predecessors[seg_idx] = [(i, s, sc) for i, s, sc in predecessors if i != seg_idx]

    def _angle(idx1, idx2):
        v1 = ham._segment_vectors[idx1]
        v2 = ham._segment_vectors[idx2]
        return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    # DFS candidate generation -----------------------------------------------
    all_candidates: list[dict] = []

    def _extend_fwd(chain, used_mods, depth=50):
        if len(chain) > depth:
            return [chain]
        last_idx = chain[-1][0]
        exts = []
        for ci, cs, csc in seg_successors.get(last_idx, []):
            mod = int(cs.hits[1].module_id)
            if mod in used_mods:
                continue
            if _angle(last_idx, ci) >= ham.epsilon:
                continue
            exts.extend(_extend_fwd(chain + [(ci, cs, csc)], used_mods | {mod}, depth))
        return exts if exts else [chain]

    def _extend_bwd(chain, used_mods, depth=50):
        if len(chain) > depth:
            return [chain]
        first_idx = chain[0][0]
        exts = []
        for ci, cs, csc in seg_predecessors.get(first_idx, []):
            mod = int(cs.hits[0].module_id)
            if mod in used_mods:
                continue
            if _angle(ci, first_idx) >= ham.epsilon:
                continue
            exts.extend(_extend_bwd([(ci, cs, csc)] + chain, used_mods | {mod}, depth))
        return exts if exts else [chain]

    seen: set[frozenset[int]] = set()
    for seg_idx, seg, score in seg_data:
        init = [(seg_idx, seg, score)]
        init_mods = {int(seg.hits[0].module_id), int(seg.hits[1].module_id)}
        for fwd_chain in _extend_fwd(init, init_mods):
            fwd_mods: set[int] = set()
            for i, s, sc in fwd_chain:
                fwd_mods.add(int(s.hits[0].module_id))
                fwd_mods.add(int(s.hits[1].module_id))
            for full in _extend_bwd(fwd_chain, fwd_mods):
                sig = frozenset(i for i, s, sc in full)
                if sig in seen:
                    continue
                seen.add(sig)
                hits_in = []
                for i, s, sc in full:
                    if s.hits[0] not in hits_in:
                        hits_in.append(s.hits[0])
                    if s.hits[1] not in hits_in:
                        hits_in.append(s.hits[1])
                if len(hits_in) >= min_hits:
                    all_candidates.append({
                        'chain': full,
                        'hits': hits_in,
                        'segment_indices': sig,
                    })

    if verbose:
        print(f"[get_tracks_optimal] Generated {len(all_candidates)} track candidates")
    if not all_candidates:
        return []

    # Score candidates -------------------------------------------------------
    def _score(cand):
        chain = cand['chain']
        total = sum(sc for _, _, sc in chain) * score_weight
        penalty = sum(
            _angle(chain[k][0], chain[k + 1][0]) * angle_weight
            for k in range(len(chain) - 1)
        )
        return total - penalty + len(cand['hits']) * 0.1

    for c in all_candidates:
        c['score'] = _score(c)
    all_candidates.sort(key=lambda c: -c['score'])

    # Greedy set packing -----------------------------------------------------
    selected = []
    used: set[int] = set()
    for c in all_candidates:
        if c['segment_indices'] & used:
            continue
        selected.append(c)
        used |= c['segment_indices']

    if verbose:
        print(f"[get_tracks_optimal] Selected {len(selected)} non-overlapping tracks")

    # Build Track objects ----------------------------------------------------
    tracks: list["Track"] = []
    for tidx, cand in enumerate(selected):
        hits_sorted = sorted(cand['hits'], key=lambda h: h.z)
        tracks.append(_Track(
            track_id=tidx,
            hit_ids=[h.hit_id for h in hits_sorted],
        ))
    return tracks


def get_tracks_optimal_iterative(
    hamiltonian: "Hamiltonian",
    solution: np.ndarray,
    event: Union["Event", "StateEventGenerator"],
    threshold: float = 0.45,
    min_hits: int = 3,
    max_iterations: int = 3,
    verbose: bool = False,
) -> list["Track"]:
    """
    Iterative track finding that re-evaluates after each selection round.

    Wraps :func:`get_tracks_optimal`, zeroing used-segment activations
    between iterations so that later rounds can discover tracks that were
    previously occluded.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian with constructed segments.
    solution : numpy.ndarray
        Segment activation vector.
    event : Event or StateEventGenerator
        Event with hit information.
    threshold : float, default 0.45
        Minimum activation.
    min_hits : int, default 3
        Minimum hits per track.
    max_iterations : int, default 3
        Number of re-evaluation rounds.
    verbose : bool, default False
        Print debug info.

    Returns
    -------
    list[Track]
        Reconstructed tracks from all iterations.
    """
    all_tracks: list["Track"] = []
    remaining = solution.copy()

    for iteration in range(max_iterations):
        new = get_tracks_optimal(
            hamiltonian, remaining, event,
            threshold=threshold, min_hits=min_hits, verbose=verbose,
        )
        if not new:
            break
        all_tracks.extend(new)

        # Zero out segments touching used hits
        used_hit_ids: set[int] = set()
        for t in new:
            used_hit_ids.update(t.hit_ids)
        for seg_idx, (fid, tid) in enumerate(hamiltonian._segment_to_hit_ids):
            if fid in used_hit_ids or tid in used_hit_ids:
                remaining[seg_idx] = 0.0

        if verbose:
            print(
                f"[Iteration {iteration + 1}] Found {len(new)} tracks, "
                f"total: {len(all_tracks)}"
            )

    # Re-number track IDs
    for i, t in enumerate(all_tracks):
        t.track_id = i
    return all_tracks


def construct_event(
    detector_geometry: "Geometry",
    tracks: list["Track"],
    hits: list["Hit"],
) -> "Event":
    """
    Construct a reconstructed Event from tracks and a hit pool.
    
    Convenience wrapper around ``Event.from_tracks``.  Modules are
    derived automatically from the hits and geometry; primary vertices
    are left empty (unknown after reconstruction).
    
    Parameters
    ----------
    detector_geometry : Geometry
        The detector geometry configuration.
    tracks : list[Track]
        Reconstructed tracks (each carrying ``hit_ids``).
    hits : list[Hit]
        Pool of available hits (e.g. from the original event).
    
    Returns
    -------
    Event
        Reconstructed event with auto-derived modules.
    
    Examples
    --------
    >>> reco_tracks = get_tracks(ham, solution, event)
    >>> reco_event = construct_event(geometry, reco_tracks, event.hits)
    
    See Also
    --------
    Event.from_tracks : The underlying classmethod.
    """
    from lhcb_velo_toy.generation.entities.event import Event
    
    return Event.from_tracks(
        detector_geometry=detector_geometry,
        tracks=tracks,
        hits=hits,
    )


def _group_segments_into_tracks(
    active_segments: list["Segment"],
) -> list[list["Segment"]]:
    """
    Group connected segments into track candidates.
    
    Parameters
    ----------
    active_segments : list[Segment]
        Segments that passed the activation threshold.
    
    Returns
    -------
    list[list[Segment]]
        Groups of connected segments, each forming a track candidate.
    """
    if not active_segments:
        return []
    
    # Track visited segments
    visited: set[int] = set()
    groups: list[list["Segment"]] = []
    
    def dfs(segment: "Segment", group: list["Segment"]) -> None:
        """Depth-first search to find all connected segments."""
        if segment.segment_id in visited:
            return
        visited.add(segment.segment_id)
        group.append(segment)
        
        # Find connected segments
        for other in active_segments:
            if other.segment_id not in visited and segment.shares_hit_with(other):
                dfs(other, group)
    
    # Find all connected components
    for segment in active_segments:
        if segment.segment_id not in visited:
            group: list["Segment"] = []
            dfs(segment, group)
            if group:
                groups.append(group)
    
    return groups


def _segments_to_track(
    segment_group: list["Segment"],
    track_id: int,
) -> "Track":
    """
    Convert a group of segments into a Track object.
    
    Parameters
    ----------
    segment_group : list[Segment]
        Connected segments forming a track.
    track_id : int
        Unique identifier for the track.
    
    Returns
    -------
    Track
        The constructed track with ordered hits.
    """
    from lhcb_velo_toy.generation.entities.track import Track
    
    # Collect all unique hits from segments
    hit_set: set[int] = set()
    hits_list: list = []
    
    for segment in segment_group:
        for hit in [segment.hit_start, segment.hit_end]:
            if hit.hit_id not in hit_set:
                hit_set.add(hit.hit_id)
                hits_list.append(hit)
    
    # Sort hits by z coordinate
    hits_list.sort(key=lambda h: h.z)
    
    # Extract hit IDs
    hit_ids = [h.hit_id for h in hits_list]
    
    return Track(track_id=track_id, hit_ids=hit_ids)


def get_segments_from_track(
    track: "Track",
    event: "Event",
) -> list["Segment"]:
    """
    Compute segments for a single track.
    
    Creates segments between consecutive hits (ordered by z) on the track.
    
    Parameters
    ----------
    track : Track
        The track to compute segments for.
    event : Event
        The event containing hit data.
    
    Returns
    -------
    list[Segment]
        List of segments connecting consecutive hits.
    
    Examples
    --------
    >>> segments = get_segments_from_track(track, event)
    >>> print(f"Track has {len(segments)} segments")
    """
    from lhcb_velo_toy.solvers.reconstruction.segment import Segment
    
    # Get hits and sort by z
    hits = event.get_hits_by_ids(track.hit_ids)
    hits_sorted = sorted(hits, key=lambda h: h.z)
    
    segments = []
    for i in range(len(hits_sorted) - 1):
        segment = Segment(
            hit_start=hits_sorted[i],
            hit_end=hits_sorted[i + 1],
            segment_id=len(segments),  # Local ID within this track
            track_id=track.track_id,
            pv_id=track.pv_id,
        )
        segments.append(segment)
    
    return segments


def get_segments_from_event(
    event: "Event",
    include_ghost_tracks: bool = False,
) -> list["Segment"]:
    """
    Compute all segments from an event's tracks.
    
    This function generates segments on-demand from the event's tracks.
    Segments are NOT stored in the Event; use this function when needed.
    
    Parameters
    ----------
    event : Event
        The event containing tracks and hits.
    include_ghost_tracks : bool, default False
        If True, includes segments from ghost hits (track_id == -1).
    
    Returns
    -------
    list[Segment]
        List of all segments with globally unique segment_ids.
    
    Examples
    --------
    >>> segments = get_segments_from_event(event)
    >>> print(f"Event has {len(segments)} segments")
    
    Notes
    -----
    Segment IDs are assigned globally across the entire event to ensure
    uniqueness for Hamiltonian construction.
    """
    from lhcb_velo_toy.solvers.reconstruction.segment import Segment
    
    all_segments: list["Segment"] = []
    segment_id_counter = 0
    
    for track in event.tracks:
        if not include_ghost_tracks and track.track_id == -1:
            continue
        
        # Get hits and sort by z
        hits = event.get_hits_by_ids(track.hit_ids)
        hits_sorted = sorted(hits, key=lambda h: h.z)
        
        # Create segments between consecutive hits
        for i in range(len(hits_sorted) - 1):
            segment = Segment(
                hit_start=hits_sorted[i],
                hit_end=hits_sorted[i + 1],
                segment_id=segment_id_counter,
                track_id=track.track_id,
                pv_id=track.pv_id,
            )
            all_segments.append(segment)
            segment_id_counter += 1
    
    return all_segments


def get_all_possible_segments(
    event: "Event",
    max_z_gap: int = 1,
) -> list["Segment"]:
    """
    Generate all possible segment candidates between hits on adjacent modules.
    
    This is used for Hamiltonian construction where we need to consider
    ALL possible hit-to-hit connections, not just those from known tracks.
    
    Parameters
    ----------
    event : Event
        The event containing hits and modules.
    max_z_gap : int, default 1
        Maximum module gap between hits. 1 = adjacent modules only.
    
    Returns
    -------
    list[Segment]
        List of all possible segment candidates.
    
    Examples
    --------
    >>> candidates = get_all_possible_segments(event)
    >>> print(f"Generated {len(candidates)} segment candidates")
    """
    from lhcb_velo_toy.solvers.reconstruction.segment import Segment
    
    # Group hits by module
    hits_by_module: dict[int, list] = {}
    for hit in event.hits:
        if hit.module_id not in hits_by_module:
            hits_by_module[hit.module_id] = []
        hits_by_module[hit.module_id].append(hit)
    
    # Get sorted module IDs
    module_ids = sorted(hits_by_module.keys())
    
    segments: list["Segment"] = []
    segment_id = 0
    
    # Generate segments between adjacent modules
    for i, mod_id in enumerate(module_ids):
        for j in range(1, max_z_gap + 1):
            if i + j >= len(module_ids):
                break
            next_mod_id = module_ids[i + j]
            
            # Create all hit pairs between these modules
            for hit1 in hits_by_module[mod_id]:
                for hit2 in hits_by_module[next_mod_id]:
                    segment = Segment(
                        hit_start=hit1,
                        hit_end=hit2,
                        segment_id=segment_id,
                        track_id=-1,  # Unknown
                        pv_id=-1,     # Unknown
                    )
                    segments.append(segment)
                    segment_id += 1
    
    return segments
