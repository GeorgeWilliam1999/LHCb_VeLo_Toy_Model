"""
Save / load a single-event reconstruction pipeline state.

The pipeline state captures everything needed to resume analysis without
re-running event generation, Hamiltonian construction, or solving:

* **config.json** – Hamiltonian parameters (epsilon, gamma, delta, …),
  reconstruction threshold, and the full geometry specification.
* **event_truth.json** – The truth-level ``Event`` (self-contained with geometry).
* **hamiltonian_A.npz** – Sparse Hamiltonian matrix ``A``
  (``scipy.sparse.save_npz``).
* **hamiltonian_b.npy** – Bias vector ``b``.
* **solution_x.npy** – Classical solution vector ``x``.
* **event_reco.json** – *(optional)* Reconstructed event.
* **validation.json** – *(optional)* Matches list + metrics dict.

All numerical data uses native NumPy / SciPy formats — no lossy
list-of-float conversions.

Examples
--------
>>> from lhcb_velo_toy.persistence import save_pipeline, load_pipeline
>>> save_pipeline(
...     "runs/event_0",
...     event=event,
...     ham=ham,
...     solution=x,
...     threshold=THRESHOLD,
...     reco_tracks=reco_tracks,
...     matches=matches,
...     metrics=metrics,
... )
>>> result = load_pipeline("runs/event_0")
>>> result.solution   # numpy array
>>> result.event      # Event object (geometry included)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from scipy.sparse import csc_matrix, load_npz, save_npz

from lhcb_velo_toy.analysis.validation.match import Match
from lhcb_velo_toy.generation.entities.event import Event
from lhcb_velo_toy.generation.geometry import geometry_from_dict


# ── Dataclass returned by load_pipeline ────────────────────────────
@dataclass
class PipelineResult:
    """
    Container for a loaded single-event pipeline state.

    Attributes
    ----------
    config : dict[str, Any]
        Run-level configuration (Hamiltonian parameters, threshold, etc.).
    event : Event
        Truth event.
    A : csc_matrix
        Hamiltonian interaction matrix.
    b : numpy.ndarray
        Hamiltonian bias vector.
    solution : numpy.ndarray
        Classical solution vector.
    reco_event : Event | None
        Reconstructed event, if saved.
    matches : list[Match] | None
        Per-track match objects, if saved.
    metrics : dict[str, Any] | None
        Aggregate validation metrics, if saved.
    """

    config: dict[str, Any]
    event: Event
    A: csc_matrix
    b: np.ndarray
    solution: np.ndarray
    reco_event: Optional[Event] = None
    matches: Optional[list[Match]] = None
    metrics: Optional[dict[str, Any]] = None


# ── Private helpers ────────────────────────────────────────────────

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_json(obj: Any, path: Path) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def _load_json(path: Path) -> Any:
    with open(path, "r") as fh:
        return json.load(fh)


# ── Public API ─────────────────────────────────────────────────────

def save_pipeline(
    directory: str | Path,
    *,
    event: Event,
    ham: Any,  # Hamiltonian (import-free for flexibility)
    solution: np.ndarray,
    threshold: float | None = None,
    reco_tracks: Sequence[Any] | None = None,
    matches: Sequence[Match] | None = None,
    metrics: dict[str, Any] | None = None,
    extra_config: dict[str, Any] | None = None,
) -> Path:
    """
    Persist a complete single-event reconstruction pipeline.

    Parameters
    ----------
    directory : str or Path
        Output directory (created if needed).
    event : Event
        Truth event to save.
    ham : Hamiltonian
        Constructed Hamiltonian (must have ``A``, ``b``, ``epsilon``,
        ``gamma``, ``delta`` attributes).
    solution : numpy.ndarray
        Classical solution vector.
    threshold : float, optional
        Track activation threshold used for reconstruction.
    reco_tracks : sequence of Track, optional
        Reconstructed tracks (will be wrapped into an ``Event``).
    matches : sequence of Match, optional
        Per-track validation matches.
    metrics : dict, optional
        Aggregate validation metrics.
    extra_config : dict, optional
        Additional key-value pairs to store in ``config.json``.

    Returns
    -------
    Path
        The directory that was written to.
    """
    out = _ensure_dir(Path(directory))

    # ── config ──
    config: dict[str, Any] = {
        "epsilon": float(ham.epsilon),
        "gamma": float(ham.gamma),
        "delta": float(ham.delta),
        "theta_d": float(getattr(ham, "theta_d", 0.0)),
        "n_segments": int(ham.n_segments),
        "threshold": float(threshold) if threshold is not None else None,
    }
    if hasattr(event.detector_geometry, "to_dict"):
        config["geometry"] = event.detector_geometry.to_dict()
    if extra_config:
        config.update(extra_config)
    _save_json(config, out / "config.json")

    # ── truth event ──
    event.to_json(str(out / "event_truth.json"))

    # ── Hamiltonian matrix & bias ──
    if ham.A is not None:
        save_npz(str(out / "hamiltonian_A.npz"), csc_matrix(ham.A))
    if ham.b is not None:
        np.save(str(out / "hamiltonian_b.npy"), ham.b)

    # ── solution ──
    np.save(str(out / "solution_x.npy"), solution)

    # ── optional: reco event ──
    if reco_tracks is not None:
        reco_event = Event.from_tracks(
            event.detector_geometry, list(reco_tracks), event.hits,
        )
        reco_event.to_json(str(out / "event_reco.json"))

    # ── optional: validation ──
    if matches is not None or metrics is not None:
        val_data: dict[str, Any] = {}
        if matches is not None:
            val_data["matches"] = [m.to_dict() for m in matches]
        if metrics is not None:
            val_data["metrics"] = metrics
        _save_json(val_data, out / "validation.json")

    return out


def load_pipeline(directory: str | Path) -> PipelineResult:
    """
    Load a previously saved pipeline state.

    Parameters
    ----------
    directory : str or Path
        Directory produced by :func:`save_pipeline`.

    Returns
    -------
    PipelineResult
        Dataclass with all loaded artefacts.

    Raises
    ------
    FileNotFoundError
        If a required file is missing.
    """
    root = Path(directory)

    # ── config ──
    config = _load_json(root / "config.json")

    # ── truth event (geometry auto-reconstructed from embedded dict) ──
    event = Event.from_json(str(root / "event_truth.json"))

    # ── Hamiltonian ──
    A = load_npz(str(root / "hamiltonian_A.npz"))
    b = np.load(str(root / "hamiltonian_b.npy"))

    # ── solution ──
    solution = np.load(str(root / "solution_x.npy"))

    # ── optional: reco event ──
    reco_event: Event | None = None
    reco_path = root / "event_reco.json"
    if reco_path.exists():
        reco_event = Event.from_json(str(reco_path))

    # ── optional: validation ──
    matches_list: list[Match] | None = None
    metrics_dict: dict[str, Any] | None = None
    val_path = root / "validation.json"
    if val_path.exists():
        val_data = _load_json(val_path)
        if "matches" in val_data:
            matches_list = [Match.from_dict(m) for m in val_data["matches"]]
        if "metrics" in val_data:
            metrics_dict = val_data["metrics"]

    return PipelineResult(
        config=config,
        event=event,
        A=csc_matrix(A),
        b=b,
        solution=solution,
        reco_event=reco_event,
        matches=matches_list,
        metrics=metrics_dict,
    )


def save_events_batch(
    directory: str | Path,
    events_data: list[dict[str, Any]],
    *,
    threshold: float | None = None,
    extra_config: dict[str, Any] | None = None,
) -> Path:
    """
    Save a list of event-pipeline dicts (like ``events_data`` in notebooks).

    Each element is expected to have at least:

    * ``"event"`` – truth Event
    * ``"ham"`` – constructed Hamiltonian
    * ``"x"`` – solution vector

    Optional keys: ``"reco_tracks"``, ``"matches"``, ``"metrics"``.

    An ``index.json`` is written alongside the per-event subdirectories
    with summary information (track counts, metrics, etc.).

    Parameters
    ----------
    directory : str or Path
        Root output directory.
    events_data : list[dict]
        Notebook-style list of per-event result dicts.
    threshold : float, optional
        Shared activation threshold.
    extra_config : dict, optional
        Extra config entries shared across all events.

    Returns
    -------
    Path
        Root directory written to.
    """
    root = _ensure_dir(Path(directory))
    index_entries: list[dict[str, Any]] = []

    for i, ed in enumerate(events_data):
        subdir = root / f"event_{i:04d}"
        save_pipeline(
            subdir,
            event=ed["event"],
            ham=ed["ham"],
            solution=ed["x"],
            threshold=threshold,
            reco_tracks=ed.get("reco_tracks"),
            matches=ed.get("matches"),
            metrics=ed.get("metrics"),
            extra_config=extra_config,
        )
        entry: dict[str, Any] = {
            "index": i,
            "subdirectory": subdir.name,
            "n_tracks_true": ed.get("n_tracks_true", len(ed["event"].tracks)),
        }
        if "n_tracks_reco" in ed:
            entry["n_tracks_reco"] = ed["n_tracks_reco"]
        if "metrics" in ed and ed["metrics"]:
            entry["metrics"] = ed["metrics"]
        index_entries.append(entry)

    _save_json(
        {"n_events": len(events_data), "events": index_entries},
        root / "index.json",
    )
    return root


def load_events_batch(directory: str | Path) -> list[PipelineResult]:
    """
    Load a batch of events previously saved by :func:`save_events_batch`.

    Parameters
    ----------
    directory : str or Path
        Root directory containing ``index.json`` and per-event
        subdirectories.

    Returns
    -------
    list[PipelineResult]
        One result per event, in original order.
    """
    root = Path(directory)
    index = _load_json(root / "index.json")

    results: list[PipelineResult] = []
    for entry in index["events"]:
        subdir = root / entry["subdirectory"]
        results.append(load_pipeline(subdir))
    return results
