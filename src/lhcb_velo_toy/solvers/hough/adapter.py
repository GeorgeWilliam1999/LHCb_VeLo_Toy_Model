"""Hough candidates -> package Track objects -> EventValidator metrics.

This adapter closes the 2026-08-26 audit finding that the Hough studies
carried two mutually-inconsistent hand-rolled efficiency/ghost/clone
implementations (``hough_prototype.evaluate`` and
``hough_study_lib.eval_candidates``) instead of the canonical
:class:`~lhcb_velo_toy.analysis.EventValidator`. Routing Hough through
Track objects makes its track-level metrics like-for-like with every other
solver lane (classical / 1BQF / QSVT).

NOTE the deliberate metric change: ``EventValidator.match_tracks`` uses
non-greedy quality-sorted matching, whereas the study's ``evaluate()`` was a
first-come majority rule — numbers can shift slightly versus the published
study tables. That shift is the fix, not a regression — when re-baselining
against published numbers, quantify the delta explicitly rather than
absorbing it.
"""
from __future__ import annotations

import numpy as np

from lhcb_velo_toy.analysis import EventValidator
from lhcb_velo_toy.generation.entities.track import Track

__all__ = ["candidates_to_tracks", "validate_hough"]


def candidates_to_tracks(cands, event) -> list[Track]:
    """Convert Hough candidate dicts into package Track objects.

    Candidates carry ``hit_idx`` — positions in ``event.hits`` — which are
    mapped to real ``hit_id`` values, z-ordered per the Track convention.
    The fitted direction is preserved in ``Track.extra['tx'/'ty']``.
    """
    hits = event.hits
    tracks = []
    for k, c in enumerate(cands):
        idx = np.asarray(c["hit_idx"])
        zorder = np.argsort([hits[i].z for i in idx], kind="stable")
        hit_ids = [int(hits[i].hit_id) for i in idx[zorder]]
        tracks.append(Track(track_id=k, pv_id=0, hit_ids=hit_ids,
                            extra={"tx": float(c["tx"]), "ty": float(c["ty"]),
                                   "n_planes": int(c["n_planes"])}))
    return tracks


def validate_hough(cands, truth_event, purity_min: float = 0.7):
    """Canonical track-level metrics for a Hough candidate set.

    Returns ``(tracks, matches, metrics)`` where metrics come from
    ``EventValidator.match_tracks`` — the same matcher used for every other
    solver lane.
    """
    tracks = candidates_to_tracks(cands, truth_event)
    validator = EventValidator(truth_event=truth_event, rec_tracks=tracks)
    matches, metrics = validator.match_tracks(purity_min=purity_min)
    return tracks, matches, metrics
