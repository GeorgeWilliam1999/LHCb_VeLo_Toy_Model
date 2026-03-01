"""
Save / load parametric sweep / scan study results.

This module handles the large-scale results produced by the parametric
scans in the characterisation and hit-competition notebooks:

* Segment efficiency / false-rate vs event size (with scattering × angle
  combinations).
* Histogram angle distributions (true vs false segment-pair angles).
* ROC / threshold sweeps, occupancy, activation, competition data.

All numerical arrays use ``numpy.savez`` for compact, type-preserving
storage.  Metadata and scalar configurations use JSON.

File layout produced by :func:`save_study`::

    <directory>/
        study_config.json       # sweep parameters, epsilons, …
        scan_results.npz        # per-(angle, mult) metric arrays
        hist_data.npz           # angle-distribution arrays (optional)
        extra_arrays.npz        # any additional named arrays (optional)

Examples
--------
>>> from lhcb_velo_toy.persistence import save_study, load_study
>>> save_study(
...     "output/sweep_001",
...     config=run_cfg,
...     scan_results=all_scan_results,
...     hist_data=hist_data,
... )
>>> study = load_study("output/sweep_001")
>>> study.scan_results[("0.2", "1")]   # dict of metric arrays
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ── Result container ───────────────────────────────────────────────

@dataclass
class StudyResult:
    """
    Container for loaded parametric-study data.

    Attributes
    ----------
    config : dict[str, Any]
        Sweep-level parameters (track sizes, repeats, scattering
        multipliers, angle settings, epsilons, etc.).
    scan_results : dict[tuple[str, str], dict[str, numpy.ndarray]]
        Keyed by ``(angle_str, mult_str)`` → dict of metric name →
        1-D array over track sizes.  Metric names match what the
        notebooks compute: ``eff_mean``, ``eff_se``, ``fr_mean``,
        ``fr_se``, ``n_true_mean``, etc.
    hist_data : dict[tuple[str, str], dict[str, numpy.ndarray]] | None
        Keyed by ``(angle_str, mult_str)`` → ``{"true": array, "false": array}``
        of pairwise segment-pair angles.
    extra_arrays : dict[str, numpy.ndarray] | None
        Arbitrary named arrays that don't fit the scan/hist schema.
    """

    config: dict[str, Any]
    scan_results: dict[tuple[str, str], dict[str, np.ndarray]] = field(
        default_factory=dict
    )
    hist_data: Optional[dict[tuple[str, str], dict[str, np.ndarray]]] = None
    extra_arrays: Optional[dict[str, np.ndarray]] = None


# ── Internal helpers ───────────────────────────────────────────────

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_json(obj: Any, path: Path) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def _load_json(path: Path) -> Any:
    with open(path, "r") as fh:
        return json.load(fh)


def _npz_key(angle: Any, mult: Any, metric: str) -> str:
    """Encode a (angle, mult, metric) triple as a flat npz key."""
    return f"a{angle}_m{mult}_{metric}"


def _parse_npz_key(key: str) -> tuple[str, str, str]:
    """Decode a flat npz key produced by :func:`_npz_key`."""
    # key format:  a<angle>_m<mult>_<metric>
    # e.g.  a0.2_m1_eff_mean  →  ("0.2", "1", "eff_mean")
    rest = key[1:]                    # drop leading 'a'
    angle_str, rest = rest.split("_m", 1)
    mult_str, metric = rest.split("_", 1)
    return angle_str, mult_str, metric


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-safe Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


# ── Public API ─────────────────────────────────────────────────────

def save_study(
    directory: str | Path,
    *,
    config: dict[str, Any],
    scan_results: dict[Any, list[dict[str, Any]]] | None = None,
    hist_data: dict[Any, dict[str, Any]] | None = None,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> Path:
    """
    Persist a parametric sweep study.

    Parameters
    ----------
    directory : str or Path
        Output directory (created if needed).
    config : dict
        Sweep-level configuration (track sizes, repeats, scattering
        multipliers, angle settings, epsilons, … ).
    scan_results : dict, optional
        Keyed by ``(angle, mult)`` (tuples, or any hashable that
        ``str()`` renders sensibly).  Values are *lists of dicts*, one
        per track-size step, each containing metric scalars like
        ``eff_mean``, ``eff_se``, ``fr_mean``, etc.  These are
        columnar-transposed into NumPy arrays and stored in
        ``scan_results.npz``.
    hist_data : dict, optional
        Keyed by ``(angle, mult)``.  Each value is a dict with at
        least ``"true"`` and ``"false"`` keys mapping to
        array-like angle distributions.
    extra_arrays : dict[str, ndarray], optional
        Arbitrary named arrays for ad-hoc data.

    Returns
    -------
    Path
        The directory that was written to.
    """
    out = _ensure_dir(Path(directory))

    # ── config → JSON ──
    _save_json(_to_jsonable(config), out / "study_config.json")

    # ── scan results → .npz ──
    if scan_results:
        npz_dict: dict[str, np.ndarray] = {}
        for key, rows in scan_results.items():
            # key is typically (angle, mult) tuple
            if isinstance(key, tuple):
                angle_s, mult_s = str(key[0]), str(key[1])
            else:
                angle_s, mult_s = str(key), "1"

            # Columnar transpose: [{k:v, …}, …] → {k: array, …}
            metric_names = list(rows[0].keys())
            for mname in metric_names:
                arr = np.array([r[mname] for r in rows])
                npz_dict[_npz_key(angle_s, mult_s, mname)] = arr

        np.savez(str(out / "scan_results.npz"), **npz_dict)

    # ── histogram data → .npz ──
    if hist_data:
        hd: dict[str, np.ndarray] = {}
        for key, arrays in hist_data.items():
            if isinstance(key, tuple):
                angle_s, mult_s = str(key[0]), str(key[1])
            else:
                angle_s, mult_s = str(key), "1"
            for sub_key, arr in arrays.items():
                flat_key = _npz_key(angle_s, mult_s, str(sub_key))
                hd[flat_key] = np.asarray(arr)
        np.savez(str(out / "hist_data.npz"), **hd)

    # ── extra arrays → .npz ──
    if extra_arrays:
        np.savez(str(out / "extra_arrays.npz"), **extra_arrays)

    return out


def load_study(directory: str | Path) -> StudyResult:
    """
    Load a parametric study previously saved by :func:`save_study`.

    Parameters
    ----------
    directory : str or Path
        Directory produced by :func:`save_study`.

    Returns
    -------
    StudyResult
        Dataclass with all loaded artefacts.
    """
    root = Path(directory)

    config = _load_json(root / "study_config.json")

    # ── scan results ──
    scan: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    scan_path = root / "scan_results.npz"
    if scan_path.exists():
        with np.load(str(scan_path)) as data:
            for key in data.files:
                angle_s, mult_s, metric = _parse_npz_key(key)
                group = scan.setdefault((angle_s, mult_s), {})
                group[metric] = data[key]

    # ── histogram data ──
    hist: dict[tuple[str, str], dict[str, np.ndarray]] | None = None
    hist_path = root / "hist_data.npz"
    if hist_path.exists():
        hist = {}
        with np.load(str(hist_path)) as data:
            for key in data.files:
                angle_s, mult_s, sub = _parse_npz_key(key)
                group = hist.setdefault((angle_s, mult_s), {})
                group[sub] = data[key]

    # ── extra arrays ──
    extra: dict[str, np.ndarray] | None = None
    extra_path = root / "extra_arrays.npz"
    if extra_path.exists():
        extra = {}
        with np.load(str(extra_path)) as data:
            for key in data.files:
                extra[key] = data[key]

    return StudyResult(
        config=config,
        scan_results=scan,
        hist_data=hist,
        extra_arrays=extra,
    )
