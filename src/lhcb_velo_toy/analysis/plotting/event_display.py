"""
3D event display utilities.

Functions for visualizing events, tracks, and segments in 3D.

Adapted from the original ``Event.plot_segments()`` /
``Event.save_plot_segments()`` in
``LHCB_Velo_Toy_Models/state_event_model.py``, rewritten as
standalone functions that accept the new-repo data structures.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from lhcb_velo_toy.generation.entities.event import Event
    from lhcb_velo_toy.generation.entities.track import Track
    from lhcb_velo_toy.generation.geometry.base import Geometry
    from lhcb_velo_toy.solvers.reconstruction.segment import Segment


# ── helpers ─────────────────────────────────────────────────────────

_MODULE_SURFACE_RESOLUTION: int = 25


def _draw_detector_planes(
    ax,
    geometry: "Geometry",
    resolution: int = _MODULE_SURFACE_RESOLUTION,
    alpha: float = 0.3,
    color: str = "gray",
) -> None:
    """
    Draw semi-transparent detector module surfaces on a 3D axis.

    For each module the active area is rendered as a ``plot_surface``
    with non-bulk regions masked to ``NaN`` so the beam-pipe void (if
    any) appears as a hole — exactly as the original code did.

    Re-maps axes to match the old conventions:
        matplotlib X ← detector Z   (beam axis, horizontal)
        matplotlib Y ← detector Y
        matplotlib Z ← detector X

    Parameters
    ----------
    ax : Axes3D
        The 3D axis to draw on.
    geometry : Geometry
        Detector geometry (PlaneGeometry or RectangularVoidGeometry).
    resolution : int
        Grid resolution per module plane.
    alpha : float
        Surface transparency.
    color : str
        Surface colour.
    """
    for mod_id, lx, ly, zpos in geometry:
        xs = np.linspace(-lx, lx, resolution)
        ys = np.linspace(-ly, ly, resolution)
        X_grid, Y_grid = np.meshgrid(xs, ys)
        Z_grid = np.full_like(X_grid, zpos, dtype=float)

        # Mask points that are NOT on the bulk (e.g. beam pipe void)
        for idx in np.ndindex(X_grid.shape):
            x_val = X_grid[idx]
            y_val = Y_grid[idx]
            if not geometry.point_on_bulk(
                {"x": x_val, "y": y_val, "z": zpos}
            ):
                X_grid[idx] = np.nan
                Y_grid[idx] = np.nan
                Z_grid[idx] = np.nan

        # Axes mapping: Z_grid → mpl-X, Y_grid → mpl-Y, X_grid → mpl-Z
        ax.plot_surface(Z_grid, Y_grid, X_grid, alpha=alpha, color=color)


# ── public API ──────────────────────────────────────────────────────


def plot_event_3d(
    event: "Event",
    title: Optional[str] = None,
    show_ghosts: bool = True,
    show_modules: bool = True,
    track_colors: Optional[dict[int, str]] = None,
    figsize: tuple[float, float] = (10, 8),
    elevation: float = 20.0,
    azimuth: float = 45.0,
    module_resolution: int = _MODULE_SURFACE_RESOLUTION,
) -> "Figure":
    """
    Create a 3D visualization of an event.

    Displays hits coloured by track, connecting lines, ghost hits,
    and — when *show_modules* is True — semi-transparent detector
    plane surfaces rendered via the geometry's ``point_on_bulk``
    (matching the original ``Event.plot_segments`` style).

    Parameters
    ----------
    event : Event
        The event to visualize.
    title : str, optional
        Title for the plot.
    show_ghosts : bool, default True
        Whether to show ghost hits (track_id == -1).
    show_modules : bool, default True
        Whether to show detector plane surfaces.
    track_colors : dict[int, str], optional
        Mapping from track_id to color.  If None, uses ``tab10``.
    figsize : tuple[float, float], default (10, 8)
        Figure size in inches.
    elevation : float, default 20.0
        Elevation angle for 3D view.
    azimuth : float, default 45.0
        Azimuth angle for 3D view.
    module_resolution : int, default 25
        Grid resolution for drawing detector planes.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # ── detector planes ──
    if show_modules:
        _draw_detector_planes(
            ax, event.detector_geometry,
            resolution=module_resolution,
        )

    # ── colour map ──
    unique_track_ids = sorted(
        {h.track_id for h in event.hits if h.track_id != -1}
    )
    cmap = plt.cm.get_cmap("tab10", max(len(unique_track_ids), 1))
    if track_colors is None:
        track_colors = {
            tid: cmap(i) for i, tid in enumerate(unique_track_ids)
        }

    # ── hits & track lines ──
    for tid in unique_track_ids:
        hits = sorted(
            [h for h in event.hits if h.track_id == tid],
            key=lambda h: h.z,
        )
        zs = [h.z for h in hits]
        xs = [h.x for h in hits]
        ys = [h.y for h in hits]
        colour = track_colors.get(tid, "blue")
        ax.scatter(zs, ys, xs, c=[colour], s=20, label=f"Track {tid}")
        ax.plot(zs, ys, xs, c=colour, alpha=0.5, linewidth=1)

    # ── ghost hits ──
    if show_ghosts:
        ghosts = [h for h in event.hits if h.track_id == -1]
        if ghosts:
            ax.scatter(
                [h.z for h in ghosts],
                [h.y for h in ghosts],
                [h.x for h in ghosts],
                c="green", marker="x", s=30, label="Ghosts",
            )

    ax.set_xlabel("Z (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("X (mm)")
    ax.view_init(elev=elevation, azim=azimuth)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def plot_segments_3d(
    segments: list["Segment"],
    event: Optional["Event"] = None,
    title: Optional[str] = None,
    highlight_segments: Optional[list[int]] = None,
    show_modules: bool = True,
    show_ghost_hits: bool = True,
    figsize: tuple[float, float] = (10, 8),
    elevation: float = 20.0,
    azimuth: float = 45.0,
    module_resolution: int = _MODULE_SURFACE_RESOLUTION,
) -> "Figure":
    """
    Create a 3D visualization of track segments.

    Closely follows the original ``Event.plot_segments()`` layout:

    * Red dots for hits that belong to segments
    * Blue lines for each segment
    * Highlighted segments drawn in red with thicker lines
    * Semi-transparent detector planes (requires *event*)
    * Green 'x' markers for ghost hits not in any segment

    Parameters
    ----------
    segments : list[Segment]
        Segments to visualize.
    event : Event, optional
        If provided, detector planes and ghost hits are drawn.
    title : str, optional
        Title for the plot.
    highlight_segments : list[int], optional
        Segment IDs to highlight (drawn in red, thicker).
    show_modules : bool, default True
        Draw detector plane surfaces (requires *event*).
    show_ghost_hits : bool, default True
        Show hits not belonging to any segment (requires *event*).
    figsize : tuple[float, float], default (10, 8)
        Figure size in inches.
    elevation : float, default 20.0
        Elevation angle for 3D view.
    azimuth : float, default 45.0
        Azimuth angle for 3D view.
    module_resolution : int, default 25
        Grid resolution for drawing detector planes.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    highlight_set = set(highlight_segments) if highlight_segments else set()

    # ── detector planes ──
    if show_modules and event is not None:
        _draw_detector_planes(
            ax, event.detector_geometry,
            resolution=module_resolution,
        )

    # ── collect all segment hits for the red-dot scatter ──
    segment_hit_ids: set[int] = set()
    all_seg_hits = []
    for seg in segments:
        for hit in (seg.hit_start, seg.hit_end):
            if hit.hit_id not in segment_hit_ids:
                segment_hit_ids.add(hit.hit_id)
                all_seg_hits.append(hit)

    # Red scatter for segment hits (matching original style)
    if all_seg_hits:
        ax.scatter(
            [h.z for h in all_seg_hits],
            [h.y for h in all_seg_hits],
            [h.x for h in all_seg_hits],
            c="r", marker="o", s=15,
        )

    # ── segment lines ──
    for seg in segments:
        z = [seg.hit_start.z, seg.hit_end.z]
        y = [seg.hit_start.y, seg.hit_end.y]
        x = [seg.hit_start.x, seg.hit_end.x]
        if seg.segment_id in highlight_set:
            ax.plot(z, y, x, c="red", linewidth=2, alpha=0.9)
        else:
            ax.plot(z, y, x, c="b", linewidth=0.8, alpha=0.4)

    # ── ghost hits (hits not part of any segment) ──
    if show_ghost_hits and event is not None:
        ghost_hits = [
            h for h in event.hits if h.hit_id not in segment_hit_ids
        ]
        if ghost_hits:
            ax.scatter(
                [h.z for h in ghost_hits],
                [h.y for h in ghost_hits],
                [h.x for h in ghost_hits],
                c="green", marker="x", s=30, label="Ghost hits",
            )
            ax.legend(fontsize=7, loc="upper left")

    ax.set_xlabel("Z (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("X (mm)")
    ax.view_init(elev=elevation, azim=azimuth)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_hit_distribution(
    event: "Event",
    projection: str = "xy",
    figsize: tuple[float, float] = (8, 6),
) -> "Figure":
    """
    Plot the 2D distribution of hits.
    
    Parameters
    ----------
    event : Event
        The event containing hits.
    projection : str, default "xy"
        Projection plane: "xy", "xz", or "yz".
    figsize : tuple[float, float], default (8, 6)
        Figure size in inches.
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    import matplotlib.pyplot as plt

    axis_map = {
        "xy": ("x", "y"),
        "xz": ("x", "z"),
        "yz": ("y", "z"),
    }
    if projection not in axis_map:
        raise ValueError(f"projection must be one of {list(axis_map)}, got {projection!r}")

    a1_name, a2_name = axis_map[projection]
    a1 = [getattr(h, a1_name) for h in event.hits]
    a2 = [getattr(h, a2_name) for h in event.hits]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(a1, a2, s=8, alpha=0.6)
    ax.set_xlabel(f"{a1_name} (mm)")
    ax.set_ylabel(f"{a2_name} (mm)")
    ax.set_title(f"Hit distribution ({projection})")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig


def save_event_animation(
    event: "Event",
    filename: str,
    fps: int = 30,
    duration: float = 5.0,
) -> None:
    """
    Create an animated rotation of the 3D event display.
    
    Parameters
    ----------
    event : Event
        The event to animate.
    filename : str
        Output filename (e.g., "event.gif", "event.mp4").
    fps : int, default 30
        Frames per second.
    duration : float, default 5.0
        Animation duration in seconds.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig = plot_event_3d(event, show_modules=True)
    ax = fig.axes[0]
    n_frames = int(fps * duration)

    def _update(frame: int) -> None:
        ax.view_init(elev=20, azim=frame * 360 / n_frames)

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=1000 / fps)
    anim.save(filename, fps=fps)
    plt.close(fig)


def plot_reco_vs_truth(
    truth_event: "Event",
    reco_tracks: list["Track"],
    title_truth: str = "Truth Tracks",
    title_reco: str = "Reconstructed Tracks",
    figsize: tuple[float, float] = (18, 7),
    show_modules: bool = True,
    elevation: float = 20.0,
    azimuth: float = 45.0,
    module_resolution: int = _MODULE_SURFACE_RESOLUTION,
) -> "Figure":
    """
    Side-by-side 3D comparison of truth and reconstructed tracks.

    Parameters
    ----------
    truth_event : Event
        The truth event containing true tracks and hits.
    reco_tracks : list[Track]
        Reconstructed tracks to compare.
    title_truth : str, default "Truth Tracks"
        Title for the truth panel.
    title_reco : str, default "Reconstructed Tracks"
        Title for the reco panel.
    figsize : tuple[float, float], default (18, 7)
        Figure size in inches.
    show_modules : bool, default True
        Whether to draw detector planes.
    elevation : float, default 20.0
        3D view elevation.
    azimuth : float, default 45.0
        3D view azimuth.
    module_resolution : int, default 25
        Grid resolution for detector planes.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with two side-by-side 3D subplots.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)

    cmap = plt.cm.get_cmap("tab10", 10)

    # ── truth panel ──────────────────────────────────────────────
    ax_t = fig.add_subplot(121, projection="3d")

    if show_modules:
        _draw_detector_planes(
            ax_t, truth_event.detector_geometry,
            resolution=module_resolution,
        )

    unique_truth_ids = sorted(
        {h.track_id for h in truth_event.hits if h.track_id != -1}
    )
    for i, tid in enumerate(unique_truth_ids):
        hits = sorted(
            [h for h in truth_event.hits if h.track_id == tid],
            key=lambda h: h.z,
        )
        zs = [h.z for h in hits]
        xs = [h.x for h in hits]
        ys = [h.y for h in hits]
        colour = cmap(i % 10)
        ax_t.scatter(zs, ys, xs, c=[colour], s=20, label=f"Track {tid}")
        ax_t.plot(zs, ys, xs, c=colour, alpha=0.5, linewidth=1)

    ax_t.set_xlabel("Z (mm)")
    ax_t.set_ylabel("Y (mm)")
    ax_t.set_zlabel("X (mm)")
    ax_t.set_title(title_truth)
    ax_t.view_init(elev=elevation, azim=azimuth)
    ax_t.legend(fontsize=7, loc="upper left")

    # ── reco panel ───────────────────────────────────────────────
    ax_r = fig.add_subplot(122, projection="3d")

    if show_modules:
        _draw_detector_planes(
            ax_r, truth_event.detector_geometry,
            resolution=module_resolution,
        )

    # Build hit lookup from truth event
    hit_map = {h.hit_id: h for h in truth_event.hits}

    for i, trk in enumerate(reco_tracks):
        hits = []
        for hid in trk.hit_ids:
            if hid in hit_map:
                hits.append(hit_map[hid])
        hits = sorted(hits, key=lambda h: h.z)

        if not hits:
            continue

        zs = [h.z for h in hits]
        xs = [h.x for h in hits]
        ys = [h.y for h in hits]
        colour = cmap(i % 10)
        ax_r.scatter(zs, ys, xs, c=[colour], s=20, label=f"Reco {trk.track_id}")
        ax_r.plot(zs, ys, xs, c=colour, alpha=0.5, linewidth=1)

    ax_r.set_xlabel("Z (mm)")
    ax_r.set_ylabel("Y (mm)")
    ax_r.set_zlabel("X (mm)")
    ax_r.set_title(title_reco)
    ax_r.view_init(elev=elevation, azim=azimuth)
    ax_r.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    return fig
