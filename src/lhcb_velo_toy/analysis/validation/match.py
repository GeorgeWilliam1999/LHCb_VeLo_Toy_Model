"""
Match dataclass for track association results.

Stores information about the match between a reconstructed track and
the best-matching truth track.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from lhcb_velo_toy.core.types import TrackID


@dataclass
class Match:
    """
    Association summary for a reconstructed track.
    
    Stores all information about the match between a reconstructed
    track and its best-matching truth track, including quality metrics
    and classification flags.
    
    Attributes
    ----------
    best_truth_id : int or None
        ID of the best-matching truth track, or None if no match found.
    rec_hits : int
        Number of hits in the reconstructed track |R_i|.
    truth_hits : int
        Number of hits in the matched truth track |T_j|.
    correct_hits : int
        Number of shared hits |R_i ∩ T_j|.
    purity : float
        Fraction of reco hits that are correct: |R_i ∩ T_j| / |R_i|.
    completeness : float
        Fraction of truth hits that are found: |R_i ∩ T_j| / |T_j|.
    candidate : bool
        True if the track passed minimum hit count requirements.
    accepted : bool
        True if the track passed purity/completeness thresholds.
    truth_id : int or None
        Assigned truth ID after matching (same as best_truth_id if accepted).
    is_clone : bool
        True if another track was already matched to the same truth.
    
    Examples
    --------
    >>> match = Match(
    ...     best_truth_id=5,
    ...     rec_hits=8,
    ...     truth_hits=10,
    ...     correct_hits=7,
    ...     purity=0.875,
    ...     completeness=0.7,
    ...     candidate=True,
    ...     accepted=True,
    ...     truth_id=5,
    ...     is_clone=False
    ... )
    >>> match.is_ghost
    False
    
    Notes
    -----
    Track classification:
    - **Candidate**: Reco track passing minimum hit count
    - **Accepted**: Candidate passing purity/completeness cuts
    - **Ghost**: Candidate that failed acceptance (no good truth match)
    - **Clone**: Accepted track matching same truth as another
    - **Primary**: Best accepted track per truth track
    """
    
    best_truth_id: Optional[TrackID]
    rec_hits: int
    truth_hits: int
    correct_hits: int
    purity: float
    completeness: float
    candidate: bool = True
    accepted: bool = False
    truth_id: Optional[TrackID] = None
    is_clone: bool = False
    
    @property
    def is_ghost(self) -> bool:
        """
        Check if this is a ghost (fake) track.
        
        Returns
        -------
        bool
            True if the track is a candidate but not accepted.
        """
        raise NotImplementedError
    
    @property
    def is_primary(self) -> bool:
        """
        Check if this is the primary match for a truth track.
        
        Returns
        -------
        bool
            True if accepted and not a clone.
        """
        raise NotImplementedError
    
    def to_dict(self) -> dict:
        """
        Convert match to dictionary for DataFrame construction.
        
        Returns
        -------
        dict
            Dictionary with all match attributes.
        """
        raise NotImplementedError
