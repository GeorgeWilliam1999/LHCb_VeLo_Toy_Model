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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
