"""
EventValidator: LHCb-style event validation for track reconstruction.

Computes standard reconstruction metrics including efficiency, ghost rate,
clone fraction, and purity.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from lhcb_velo_toy.generation.entities.event import Event
    from lhcb_velo_toy.generation.entities.track import Track

from lhcb_velo_toy.analysis.validation.match import Match


class EventValidator:
    """
    LHCb-style event validator for track reconstruction performance.
    
    Computes standard LHCb tracking metrics by matching reconstructed
    tracks to truth tracks based on shared hits.
    
    Parameters
    ----------
    truth_event : Event
        The truth event containing true particle tracks.
    rec_tracks : list[Track]
        List of reconstructed tracks to validate.
    reconstructible_filter : Callable[[Track], bool], optional
        Filter function to determine which truth tracks are reconstructible.
        If None, all truth tracks are considered reconstructible.
    
    Attributes
    ----------
    truth_event : Event
        The truth event.
    rec_tracks : list[Track]
        Reconstructed tracks.
    matches : list[Match]
        Match results after calling match_tracks().
    metrics : dict
        Computed metrics after calling match_tracks().
    
    Examples
    --------
    >>> validator = EventValidator(truth_event, reco_tracks)
    >>> matches, metrics = validator.match_tracks(purity_min=0.7)
    >>> print(f"Efficiency: {metrics['efficiency']:.1%}")
    >>> print(f"Ghost Rate: {metrics['ghost_rate']:.1%}")
    
    Notes
    -----
    Metric definitions (following LHCb conventions):
    
    - **Reconstruction Efficiency**: fraction of reconstructible truth tracks
      that are matched to at least one accepted reco track
    - **Ghost Rate**: fraction of candidate reco tracks that are not accepted
    - **Clone Fraction**: fraction of accepted reco tracks that are clones
    - **Mean Purity**: average purity of accepted tracks
    - **Hit Efficiency**: average hit_efficiency of accepted tracks
    """
    
    def __init__(
        self,
        truth_event: "Event",
        rec_tracks: list["Track"],
        reconstructible_filter: Optional[Callable[["Track"], bool]] = None,
    ) -> None:
        """Initialize the event validator."""
        self.truth_event = truth_event
        self.rec_tracks = rec_tracks
        self.reconstructible_filter = reconstructible_filter or (lambda t: True)
        self.matches: list[Match] = []
        self.metrics: dict[str, float] = {}
        
        # Build truth track hit sets for efficient lookup
        self._truth_hit_sets: dict[int, set[int]] = {}
        for track in truth_event.tracks:
            self._truth_hit_sets[track.track_id] = set(track.hit_ids)
        
        # Track which truth tracks are reconstructible
        self._reconstructible_truth_ids: set[int] = {
            t.track_id for t in truth_event.tracks
            if self.reconstructible_filter(t)
        }
    
    def match_tracks(
        self,
        purity_min: float = 0.7,
        hit_efficiency_min: float = 0.0,
        min_rec_hits: int = 3,
    ) -> tuple[list["Match"], dict[str, float]]:
        """
        Match reconstructed tracks to truth tracks.
        
        For each reconstructed track, finds the best-matching truth track
        based on shared hits, then applies quality cuts to classify tracks.
        
        Parameters
        ----------
        purity_min : float, default 0.7
            Minimum purity for a track to be accepted.
            Purity = (shared hits) / (reco hits).
        hit_efficiency_min : float, default 0.0
            Minimum hit efficiency for a track to be accepted.
            Hit Efficiency = (shared hits) / (truth hits).
        min_rec_hits : int, default 3
            Minimum number of hits for a track to be a candidate.
        
        Returns
        -------
        tuple[list[Match], dict[str, float]]
            List of Match objects and dictionary of computed metrics.
        
        Notes
        -----
        Matching algorithm (non-greedy):
        1. Filter reco tracks by min_rec_hits -> candidates
        2. For each candidate, find truth track with most shared hits
        3. Compute purity and hit_efficiency
        4. Apply purity_min and hit_efficiency_min cuts -> accepted
        5. If truth already matched, compare quality:
           - If new match is better, replace existing and re-evaluate displaced track
           - If existing match is better, mark new track as clone
        6. Repeat until no more reassignments needed
        7. Compute aggregate metrics
        
        This non-greedy approach ensures globally optimal matching rather
        than first-come-first-served assignments.
        """
        self.matches = []
        
        # Track truth -> best reco match mapping (for non-greedy matching)
        truth_to_match: dict[int, tuple[int, Match]] = {}  # truth_id -> (rec_idx, match)
        
        # First pass: compute all match candidates
        match_candidates: list[tuple[int, Match]] = []  # (rec_idx, match)
        
        for rec_idx, rec_track in enumerate(self.rec_tracks):
            # Check candidate status
            is_candidate = len(rec_track.hit_ids) >= min_rec_hits
            
            # Find best truth match
            best_truth_id, rec_hits, truth_hits, shared_hits = self._find_best_truth_match(rec_track)
            
            # Compute metrics
            purity = shared_hits / rec_hits if rec_hits > 0 else 0.0
            hit_efficiency = shared_hits / truth_hits if truth_hits > 0 else 0.0
            
            # Create initial match (not yet determined if accepted/clone)
            match = Match(
                best_truth_id=best_truth_id,
                rec_hits=rec_hits,
                truth_hits=truth_hits,
                correct_hits=shared_hits,
                purity=purity,
                hit_efficiency=hit_efficiency,
                candidate=is_candidate,
                accepted=False,
                truth_id=None,
                is_clone=False,
            )
            
            match_candidates.append((rec_idx, match))
        
        # Second pass: non-greedy assignment
        # Sort by quality (purity * hit_efficiency) descending
        def match_quality(item: tuple[int, Match]) -> float:
            _, m = item
            return m.purity * m.hit_efficiency
        
        sorted_candidates = sorted(match_candidates, key=match_quality, reverse=True)
        
        for rec_idx, match in sorted_candidates:
            # Check if this is an accepted match
            if not match.candidate:
                continue
            if match.purity < purity_min:
                continue
            if match.hit_efficiency < hit_efficiency_min:
                continue
            if match.best_truth_id is None:
                continue
            
            # Mark as accepted
            match.accepted = True
            match.truth_id = match.best_truth_id
            
            # Non-greedy: check if this truth is already matched
            if match.best_truth_id in truth_to_match:
                existing_idx, existing_match = truth_to_match[match.best_truth_id]
                existing_quality = existing_match.purity * existing_match.hit_efficiency
                new_quality = match.purity * match.hit_efficiency
                
                if new_quality > existing_quality:
                    # New match is better - mark old as clone
                    existing_match.is_clone = True
                    truth_to_match[match.best_truth_id] = (rec_idx, match)
                else:
                    # Existing match is better - mark new as clone
                    match.is_clone = True
            else:
                # First match to this truth
                truth_to_match[match.best_truth_id] = (rec_idx, match)
        
        # Build final matches list (in original order)
        self.matches = [match for _, match in match_candidates]
        
        # Compute aggregate metrics
        self.metrics = self._compute_metrics()
        
        return self.matches, self.metrics
    
    def _find_best_truth_match(
        self,
        rec_track: "Track",
    ) -> tuple[Optional[int], int, int, int]:
        """
        Find the best-matching truth track for a reco track.
        
        Parameters
        ----------
        rec_track : Track
            The reconstructed track.
        
        Returns
        -------
        tuple[int or None, int, int, int]
            (best_truth_id, rec_hits, truth_hits, shared_hits)
        """
        rec_hit_set = set(rec_track.hit_ids)
        rec_hits = len(rec_hit_set)
        
        best_truth_id: Optional[int] = None
        best_shared = 0
        best_truth_hits = 0
        
        for truth_id, truth_hit_set in self._truth_hit_sets.items():
            shared = len(rec_hit_set & truth_hit_set)
            if shared > best_shared:
                best_shared = shared
                best_truth_id = truth_id
                best_truth_hits = len(truth_hit_set)
        
        return best_truth_id, rec_hits, best_truth_hits, best_shared
    
    def _compute_metrics(self) -> dict[str, float]:
        """
        Compute aggregate metrics from match results.
        
        Returns
        -------
        dict[str, float]
            Dictionary containing:
            - efficiency: reco efficiency
            - ghost_rate: ghost track fraction
            - clone_fraction: clone track fraction
            - mean_purity: average purity of accepted tracks
            - hit_efficiency: average hit_efficiency of accepted tracks
            - n_candidates: number of candidate tracks
            - n_accepted: number of accepted tracks
            - n_ghosts: number of ghost tracks
            - n_clones: number of clone tracks
            - n_reconstructible: number of reconstructible truth tracks
            - n_matched: number of matched truth tracks
        """
        n_candidates = sum(1 for m in self.matches if m.candidate)
        n_accepted = sum(1 for m in self.matches if m.accepted)
        n_ghosts = sum(1 for m in self.matches if m.is_ghost)
        n_clones = sum(1 for m in self.matches if m.is_clone)
        n_primary = sum(1 for m in self.matches if m.is_primary)
        
        # Matched truth tracks (those with at least one primary match)
        matched_truth_ids = {
            m.truth_id for m in self.matches
            if m.is_primary and m.truth_id is not None
        }
        n_matched = len(matched_truth_ids & self._reconstructible_truth_ids)
        n_reconstructible = len(self._reconstructible_truth_ids)
        
        # Efficiency: fraction of reconstructible truth tracks that are matched
        efficiency = n_matched / n_reconstructible if n_reconstructible > 0 else 0.0
        
        # Ghost rate: fraction of candidates that are ghosts
        ghost_rate = n_ghosts / n_candidates if n_candidates > 0 else 0.0
        
        # Clone fraction: fraction of accepted that are clones
        clone_fraction = n_clones / n_accepted if n_accepted > 0 else 0.0
        
        # Mean purity and hit efficiency of accepted tracks
        accepted_matches = [m for m in self.matches if m.accepted]
        mean_purity = (
            np.mean([m.purity for m in accepted_matches])
            if accepted_matches else 0.0
        )
        mean_hit_efficiency = (
            np.mean([m.hit_efficiency for m in accepted_matches])
            if accepted_matches else 0.0
        )
        
        return {
            "efficiency": efficiency,
            "ghost_rate": ghost_rate,
            "clone_fraction": clone_fraction,
            "mean_purity": float(mean_purity),
            "hit_efficiency": float(mean_hit_efficiency),
            "n_candidates": n_candidates,
            "n_accepted": n_accepted,
            "n_ghosts": n_ghosts,
            "n_clones": n_clones,
            "n_primary": n_primary,
            "n_reconstructible": n_reconstructible,
            "n_matched": n_matched,
        }
    
    def summary_table(self) -> "pd.DataFrame":
        """
        Generate a per-track summary table.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with one row per reconstructed track.
        
        Columns
        -------
        rec_track_id : int
            Reconstructed track identifier.
        best_truth_id : int or None
            Best matching truth track.
        rec_hits : int
            Number of reco hits.
        truth_hits : int
            Number of truth hits (of matched track).
        correct_hits : int
            Number of shared hits.
        purity : float
            Hit purity.
        hit_efficiency : float
            Hit efficiency.
        candidate : bool
            Is a candidate track.
        accepted : bool
            Passed quality cuts.
        is_clone : bool
            Is a clone track.
        """
        import pandas as pd
        
        data = []
        for i, match in enumerate(self.matches):
            row = match.to_dict()
            row["rec_track_id"] = self.rec_tracks[i].track_id
            data.append(row)
        
        return pd.DataFrame(data)
    
    def truth_table(self) -> "pd.DataFrame":
        """
        Generate a per-truth-track summary table.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with one row per truth track.
        
        Columns
        -------
        truth_track_id : int
            Truth track identifier.
        n_hits : int
            Number of truth hits.
        reconstructible : bool
            Passes reconstructibility filter.
        matched : bool
            Has at least one accepted reco match.
        n_matches : int
            Number of accepted reco matches.
        best_purity : float
            Highest purity among matches.
        best_hit_efficiency : float
            Highest hit efficiency among matches.
        """
        import pandas as pd
        
        data = []
        for truth_track in self.truth_event.tracks:
            tid = truth_track.track_id
            
            # Find all matches to this truth
            track_matches = [
                m for m in self.matches
                if m.accepted and m.truth_id == tid
            ]
            
            n_matches = len(track_matches)
            best_purity = max((m.purity for m in track_matches), default=0.0)
            best_hit_efficiency = max((m.hit_efficiency for m in track_matches), default=0.0)
            
            data.append({
                "truth_track_id": tid,
                "n_hits": len(truth_track.hit_ids),
                "reconstructible": tid in self._reconstructible_truth_ids,
                "matched": n_matches > 0,
                "n_matches": n_matches,
                "best_purity": best_purity,
                "best_hit_efficiency": best_hit_efficiency,
            })
        
        return pd.DataFrame(data)
    
    def print_summary(self) -> None:
        """Print a formatted summary of reconstruction performance."""
        print("=" * 50)
        print("Track Reconstruction Performance Summary")
        print("=" * 50)
        print(f"Reconstruction Efficiency: {self.metrics['efficiency']:.1%}")
        print(f"Ghost Rate:                {self.metrics['ghost_rate']:.1%}")
        print(f"Clone Fraction:            {self.metrics['clone_fraction']:.1%}")
        print(f"Mean Purity:               {self.metrics['mean_purity']:.3f}")
        print(f"Mean Hit Efficiency:       {self.metrics['hit_efficiency']:.3f}")
        print("-" * 50)
        print(f"Reconstructible Tracks:    {self.metrics['n_reconstructible']}")
        print(f"Matched Tracks:            {self.metrics['n_matched']}")
        print(f"Candidate Reco Tracks:     {self.metrics['n_candidates']}")
        print(f"Accepted Reco Tracks:      {self.metrics['n_accepted']}")
        print(f"Primary Matches:           {self.metrics['n_primary']}")
        print(f"Ghost Tracks:              {self.metrics['n_ghosts']}")
        print(f"Clone Tracks:              {self.metrics['n_clones']}")
        print("=" * 50)
