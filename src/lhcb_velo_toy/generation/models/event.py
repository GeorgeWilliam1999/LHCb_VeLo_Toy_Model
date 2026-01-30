"""
Event dataclass representing a complete collision event.

An Event is a container holding all information about a single collision:
detector geometry, tracks, hits, segments, and modules.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lhcb_velo_toy.generation.models.hit import Hit
    from lhcb_velo_toy.generation.models.segment import Segment
    from lhcb_velo_toy.generation.models.track import Track
    from lhcb_velo_toy.generation.models.module import Module
    from lhcb_velo_toy.generation.geometry.base import Geometry


@dataclass
class Event:
    """
    A complete collision event container.
    
    An Event holds all information related to a single proton-proton
    collision (or simulated event): the detector geometry, true particle
    tracks, recorded hits, track segments, and detector modules.
    
    Attributes
    ----------
    detector_geometry : Geometry
        The detector geometry configuration.
    tracks : list[Track]
        List of particle tracks in this event.
    hits : list[Hit]
        All hits in this event (from all tracks + ghosts).
    segments : list[Segment]
        All track segments constructed from hits.
    modules : list[Module]
        Detector modules with their recorded hits.
    
    Examples
    --------
    >>> event = Event(
    ...     detector_geometry=geometry,
    ...     tracks=tracks,
    ...     hits=hits,
    ...     segments=segments,
    ...     modules=modules
    ... )
    >>> event.n_tracks
    10
    >>> event.n_hits
    50
    >>> event.plot_segments()  # Interactive 3D visualization
    
    Notes
    -----
    Events can be visualized using the `plot_segments()` method, which
    creates an interactive 3D plot showing hits, segments, and detector
    planes.
    """
    
    detector_geometry: "Geometry"
    tracks: list["Track"] = field(default_factory=list)
    hits: list["Hit"] = field(default_factory=list)
    segments: list["Segment"] = field(default_factory=list)
    modules: list["Module"] = field(default_factory=list)
    
    @property
    def n_tracks(self) -> int:
        """
        Get the number of tracks in this event.
        
        Returns
        -------
        int
            Number of tracks.
        """
        raise NotImplementedError
    
    @property
    def n_hits(self) -> int:
        """
        Get the total number of hits in this event.
        
        Returns
        -------
        int
            Number of hits.
        """
        raise NotImplementedError
    
    @property
    def n_segments(self) -> int:
        """
        Get the number of segments in this event.
        
        Returns
        -------
        int
            Number of segments.
        """
        raise NotImplementedError
    
    @property
    def n_modules(self) -> int:
        """
        Get the number of detector modules.
        
        Returns
        -------
        int
            Number of modules.
        """
        raise NotImplementedError
    
    def get_hits_by_module(self, module_id: int) -> list["Hit"]:
        """
        Get all hits on a specific module.
        
        Parameters
        ----------
        module_id : int
            The module identifier.
        
        Returns
        -------
        list[Hit]
            Hits on the specified module.
        """
        raise NotImplementedError
    
    def get_hits_by_track(self, track_id: int) -> list["Hit"]:
        """
        Get all hits belonging to a specific track.
        
        Parameters
        ----------
        track_id : int
            The track identifier.
        
        Returns
        -------
        list[Hit]
            Hits belonging to the specified track.
        """
        raise NotImplementedError
    
    def plot_segments(
        self,
        title: Optional[str] = None,
        show_ghosts: bool = True,
        show_modules: bool = True,
    ) -> None:
        """
        Create an interactive 3D visualization of the event.
        
        Displays hits, segments, and optionally detector modules in a
        3D matplotlib figure.
        
        Parameters
        ----------
        title : str, optional
            Title for the plot.
        show_ghosts : bool, default True
            Whether to show ghost hits (track_id == -1).
        show_modules : bool, default True
            Whether to show detector plane outlines.
        
        Notes
        -----
        Color scheme:
        - Red dots: Hits belonging to segments
        - Blue lines: Track segments
        - Green X: Ghost hits (not in any segment)
        - Gray surfaces: Detector planes
        """
        raise NotImplementedError
    
    def save_plot_segments(
        self,
        filename: str,
        params: Optional[dict] = None,
    ) -> None:
        """
        Save the event visualization to a file.
        
        Parameters
        ----------
        filename : str
            Output filename (e.g., "event.png", "event.pdf").
        params : dict, optional
            Additional matplotlib savefig parameters.
        """
        raise NotImplementedError
