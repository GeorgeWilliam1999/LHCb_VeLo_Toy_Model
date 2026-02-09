"""
3D event display utilities.

Functions for visualizing events, tracks, and segments in 3D.
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
    from lhcb_velo_toy.solvers.reconstruction.segment import Segment


def plot_event_3d(
    event: "Event",
    title: Optional[str] = None,
    show_ghosts: bool = True,
    show_modules: bool = True,
    track_colors: Optional[dict[int, str]] = None,
    figsize: tuple[float, float] = (10, 8),
    elevation: float = 20.0,
    azimuth: float = 45.0,
) -> "Figure":
    """
    Create a 3D visualization of an event.
    
    Displays hits, tracks, and optionally detector modules in a 3D
    matplotlib figure.
    
    Parameters
    ----------
    event : Event
        The event to visualize.
    title : str, optional
        Title for the plot.
    show_ghosts : bool, default True
        Whether to show ghost hits (track_id == -1).
    show_modules : bool, default True
        Whether to show detector plane outlines.
    track_colors : dict[int, str], optional
        Mapping from track_id to color. If None, uses automatic coloring.
    figsize : tuple[float, float], default (10, 8)
        Figure size in inches.
    elevation : float, default 20.0
        Elevation angle for 3D view.
    azimuth : float, default 45.0
        Azimuth angle for 3D view.
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    
    Examples
    --------
    >>> fig = plot_event_3d(event, title="Simulated Event")
    >>> plt.show()
    
    Notes
    -----
    Color scheme:
    - Hits: colored by track_id
    - Ghost hits (track_id == -1): gray 'x' markers
    - Segments: colored by track ownership
    - Modules: light gray rectangles
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import mpl_toolkits.mplot3d.art3d as art3d  # noqa: F401

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Colour map for tracks
    unique_track_ids = sorted(
        {h.track_id for h in event.hits if h.track_id != -1}
    )
    cmap = plt.cm.get_cmap("tab10", max(len(unique_track_ids), 1))
    if track_colors is None:
        track_colors = {
            tid: cmap(i) for i, tid in enumerate(unique_track_ids)
        }

    # Plot hits grouped by track
    for tid in unique_track_ids:
        hits = [h for h in event.hits if h.track_id == tid]
        xs = [h.x for h in hits]
        ys = [h.y for h in hits]
        zs = [h.z for h in hits]
        colour = track_colors.get(tid, "blue")
        ax.scatter(zs, xs, ys, c=[colour], s=20, label=f"Track {tid}")
        # Connect hits as a line
        hits_sorted = sorted(hits, key=lambda h: h.z)
        ax.plot(
            [h.z for h in hits_sorted],
            [h.x for h in hits_sorted],
            [h.y for h in hits_sorted],
            c=colour, alpha=0.5, linewidth=1,
        )

    # Ghost hits
    if show_ghosts:
        ghosts = [h for h in event.hits if h.track_id == -1]
        if ghosts:
            ax.scatter(
                [h.z for h in ghosts],
                [h.x for h in ghosts],
                [h.y for h in ghosts],
                c="gray", marker="x", s=30, label="Ghosts",
            )

    # Module outlines
    if show_modules:
        for mod in event.modules:
            ax.axvline(x=mod.z, color="lightgray", alpha=0.3, linewidth=0.5)

    ax.set_xlabel("z (mm)")
    ax.set_ylabel("x (mm)")
    ax.set_zlabel("y (mm)")
    ax.view_init(elev=elevation, azim=azimuth)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def plot_segments_3d(
    segments: list["Segment"],
    title: Optional[str] = None,
    highlight_segments: Optional[list[int]] = None,
    figsize: tuple[float, float] = (10, 8),
) -> "Figure":
    """
    Create a 3D visualization of track segments.
    
    Parameters
    ----------
    segments : list[Segment]
        Segments to visualize.
    title : str, optional
        Title for the plot.
    highlight_segments : list[int], optional
        Indices of segments to highlight.
    figsize : tuple[float, float], default (10, 8)
        Figure size in inches.
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    highlight_set = set(highlight_segments) if highlight_segments else set()

    for seg in segments:
        z = [seg.hit_start.z, seg.hit_end.z]
        x = [seg.hit_start.x, seg.hit_end.x]
        y = [seg.hit_start.y, seg.hit_end.y]
        if seg.segment_id in highlight_set:
            ax.plot(z, x, y, c="red", linewidth=2, alpha=0.9)
        else:
            ax.plot(z, x, y, c="steelblue", linewidth=0.8, alpha=0.4)

    ax.set_xlabel("z (mm)")
    ax.set_ylabel("x (mm)")
    ax.set_zlabel("y (mm)")
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
