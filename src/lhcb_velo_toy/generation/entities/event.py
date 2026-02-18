"""
Event dataclass representing a complete collision event.

An Event is a container holding all information about a single collision:
detector geometry, primary vertices, tracks, hits, and modules.

Events are designed to be fully JSON-serializable for storage and retrieval.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Any

import numpy as np

if TYPE_CHECKING:
    from lhcb_velo_toy.generation.entities.hit import Hit
    from lhcb_velo_toy.generation.entities.track import Track
    from lhcb_velo_toy.generation.entities.module import Module
    from lhcb_velo_toy.generation.entities.primary_vertex import PrimaryVertex
    from lhcb_velo_toy.generation.geometry.base import Geometry


@dataclass
class Event:
    """
    A complete collision event container.
    
    An Event holds all information related to a single proton-proton
    collision (or simulated event): the detector geometry, primary vertices,
    particle tracks, recorded hits, and detector modules.
    
    The Event is the top-level container with the following hierarchy:
    
    ```
    Event
    ├── Primary Vertices (list of PVs, each with track_ids)
    ├── Tracks (list of tracks, each with hit_ids and pv_id)
    ├── Hits (flat list, each with track_id back-reference)
    ├── Modules (detector layers)
    └── Geometry (detector configuration)
    ```
    
    Attributes
    ----------
    detector_geometry : Geometry
        The detector geometry configuration.
    primary_vertices : list[PrimaryVertex]
        List of primary vertices (collision points).
    tracks : list[Track]
        List of particle tracks in this event.
    hits : list[Hit]
        All hits in this event (from all tracks + ghosts).
    modules : list[Module]
        Detector modules with their recorded hits.
    metadata : dict[str, Any]
        Generation hyper-parameters needed to reproduce the event
        (geometry class, z-positions, phi/theta ranges, noise settings,
        etc.).  Auto-populated with basic geometry info when not
        supplied explicitly.
    
    Examples
    --------
    >>> event = Event(
    ...     detector_geometry=geometry,
    ...     primary_vertices=[pv],
    ...     tracks=tracks,
    ...     hits=hits,
    ...     modules=modules
    ... )
    >>> event.n_tracks
    10
    >>> event.n_hits
    50
    >>> event.to_json("event.json")  # Save to file
    >>> loaded = Event.from_json("event.json", geometry)  # Load back
    
    Notes
    -----
    Segments are NOT stored in the Event. They are computed on-demand
    from tracks using the `get_segments_from_track()` function in the
    `solvers.reconstruction` module when needed for Hamiltonian construction.
    
    All cross-references use IDs (hit_ids in Track, track_id in Hit, etc.)
    to enable clean JSON serialization.
    """
    
    detector_geometry: "Geometry"
    primary_vertices: list["PrimaryVertex"] = field(default_factory=list)
    tracks: list["Track"] = field(default_factory=list)
    hits: list["Hit"] = field(default_factory=list)
    modules: list["Module"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Internal lookup caches (not serialized)
    _hit_by_id: dict[int, "Hit"] = field(default_factory=dict, repr=False)
    _track_by_id: dict[int, "Track"] = field(default_factory=dict, repr=False)
    
    def __post_init__(self) -> None:
        """Build internal lookup caches and auto-populate metadata."""
        self._rebuild_caches()
        if not self.metadata:
            self.metadata = self._geometry_metadata()
    
    def _rebuild_caches(self) -> None:
        """Rebuild internal ID lookup caches."""
        self._hit_by_id = {h.hit_id: h for h in self.hits}
        self._track_by_id = {t.track_id: t for t in self.tracks}
    
    def _geometry_metadata(self) -> dict[str, Any]:
        """Return basic geometry metadata extracted from the detector."""
        geo = self.detector_geometry
        z_positions: list[float] = []
        module_ids: list[int] = []
        for mod_id, _lx, _ly, zpos in geo:
            module_ids.append(mod_id)
            z_positions.append(zpos)
        return {
            "geometry_class": type(geo).__name__,
            "n_modules": len(module_ids),
            "z_positions": z_positions,
            "module_ids": module_ids,
        }
    
    @property
    def n_primary_vertices(self) -> int:
        """Get the number of primary vertices."""
        return len(self.primary_vertices)
    
    @property
    def n_tracks(self) -> int:
        """Get the number of tracks in this event."""
        return len(self.tracks)
    
    @property
    def n_hits(self) -> int:
        """Get the total number of hits in this event."""
        return len(self.hits)
    
    @property
    def n_modules(self) -> int:
        """Get the number of detector modules."""
        return len(self.modules)
    
    def get_hit_by_id(self, hit_id: int) -> Optional["Hit"]:
        """
        Get a hit by its ID.
        
        Parameters
        ----------
        hit_id : int
            The hit identifier.
        
        Returns
        -------
        Hit or None
            The hit if found, None otherwise.
        """
        return self._hit_by_id.get(hit_id)
    
    def get_hits_by_ids(self, hit_ids: list[int]) -> list["Hit"]:
        """
        Get multiple hits by their IDs.
        
        Parameters
        ----------
        hit_ids : list[int]
            List of hit identifiers.
        
        Returns
        -------
        list[Hit]
            List of hits (in same order as input IDs).
            Missing hits are skipped.
        """
        return [self._hit_by_id[hid] for hid in hit_ids if hid in self._hit_by_id]
    
    def get_track_by_id(self, track_id: int) -> Optional["Track"]:
        """
        Get a track by its ID.
        
        Parameters
        ----------
        track_id : int
            The track identifier.
        
        Returns
        -------
        Track or None
            The track if found, None otherwise.
        """
        return self._track_by_id.get(track_id)
    
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
        return [h for h in self.hits if h.module_id == module_id]
    
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
        track = self.get_track_by_id(track_id)
        if track is None:
            return []
        return self.get_hits_by_ids(track.hit_ids)
    
    def get_tracks_by_pv(self, pv_id: int) -> list["Track"]:
        """
        Get all tracks originating from a primary vertex.
        
        Parameters
        ----------
        pv_id : int
            The primary vertex identifier.
        
        Returns
        -------
        list[Track]
            Tracks from the specified primary vertex.
        """
        return [t for t in self.tracks if t.pv_id == pv_id]
    
    # =========================================================================
    # JSON Serialization
    # =========================================================================
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert the event to a dictionary for JSON serialization.
        
        Note: Geometry is NOT included as it should be provided separately
        when loading.
        
        Returns
        -------
        dict
            Dictionary representation of the event.
        """
        return {
            "primary_vertices": [pv.to_dict() for pv in self.primary_vertices],
            "tracks": [t.to_dict() for t in self.tracks],
            "hits": [h.to_dict() for h in self.hits],
            "modules": [m.to_dict() for m in self.modules],
            "metadata": self.metadata.copy(),
        }
    
    def to_json(self, filepath: str, indent: int = 2) -> None:
        """
        Save the event to a JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to the output JSON file.
        indent : int, default 2
            JSON indentation level.
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)
    
    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        detector_geometry: "Geometry",
    ) -> "Event":
        """
        Create an Event from a dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary with primary_vertices, tracks, hits, modules keys.
        detector_geometry : Geometry
            The detector geometry (must be provided separately).
        
        Returns
        -------
        Event
            The reconstructed event.
        """
        from lhcb_velo_toy.generation.entities.hit import Hit
        from lhcb_velo_toy.generation.entities.track import Track
        from lhcb_velo_toy.generation.entities.module import Module
        from lhcb_velo_toy.generation.entities.primary_vertex import PrimaryVertex
        
        return cls(
            detector_geometry=detector_geometry,
            primary_vertices=[PrimaryVertex.from_dict(pv) for pv in data.get("primary_vertices", [])],
            tracks=[Track.from_dict(t) for t in data.get("tracks", [])],
            hits=[Hit.from_dict(h) for h in data.get("hits", [])],
            modules=[Module.from_dict(m) for m in data.get("modules", [])],
            metadata=data.get("metadata", {}),
        )
    
    @classmethod
    def from_json(
        cls,
        filepath: str,
        detector_geometry: "Geometry",
    ) -> "Event":
        """
        Load an event from a JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to the JSON file.
        detector_geometry : Geometry
            The detector geometry (must be provided separately).
        
        Returns
        -------
        Event
            The loaded event.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data, detector_geometry)
    
    @staticmethod
    def _build_modules_from_hits(
        hits: list["Hit"],
        detector_geometry: "Geometry",
    ) -> list["Module"]:
        """
        Derive Module objects from hits and detector geometry.
        
        Each module's dimensions (z, lx, ly) come from the geometry,
        and its hit_ids are collected from the hits that reference it.
        
        Parameters
        ----------
        hits : list[Hit]
            The hits to assign to modules.
        detector_geometry : Geometry
            The detector geometry providing module dimensions.
        
        Returns
        -------
        list[Module]
            One Module per geometry layer, populated with hit IDs.
        """
        from lhcb_velo_toy.generation.entities.module import Module
        
        modules: list[Module] = []
        for mod_id, lx, ly, zpos in detector_geometry:
            mod_hit_ids = [h.hit_id for h in hits if h.module_id == mod_id]
            modules.append(
                Module(
                    module_id=mod_id,
                    z=zpos,
                    lx=lx,
                    ly=ly,
                    hit_ids=mod_hit_ids,
                )
            )
        return modules
    
    @classmethod
    def from_tracks(
        cls,
        detector_geometry: "Geometry",
        tracks: list["Track"],
        hits: list["Hit"],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "Event":
        """
        Construct a reconstructed Event from tracks and a hit pool.
        
        This is the recommended way to build an Event from reconstruction
        output.  Only a geometry, the reconstructed tracks, and the hit
        pool are required — modules are derived automatically, and
        primary vertices are left empty (unknown after reconstruction).
        
        Hits are filtered to only those referenced by the supplied tracks
        so the resulting event is self-consistent.
        
        Parameters
        ----------
        detector_geometry : Geometry
            The detector geometry configuration.
        tracks : list[Track]
            Reconstructed tracks (each carrying ``hit_ids``).
        hits : list[Hit]
            Pool of available hits (e.g. from the original event).
        metadata : dict[str, Any], optional
            Event-level metadata.  If *None* the Event will auto-populate
            basic geometry metadata in ``__post_init__``.
        
        Returns
        -------
        Event
            A reconstructed event with auto-derived modules and
            empty primary vertices.
        
        Examples
        --------
        >>> reco_tracks = get_tracks(ham, solution, event)
        >>> reco_event = Event.from_tracks(geometry, reco_tracks, event.hits)
        """
        # Collect all hit IDs referenced by the reconstructed tracks
        referenced_ids: set[int] = set()
        for t in tracks:
            referenced_ids.update(t.hit_ids)
        
        # Filter the hit pool to only those used by these tracks
        filtered_hits = [h for h in hits if h.hit_id in referenced_ids]
        
        # Auto-derive modules from the filtered hits + geometry
        modules = cls._build_modules_from_hits(filtered_hits, detector_geometry)
        
        return cls(
            detector_geometry=detector_geometry,
            primary_vertices=[],
            tracks=tracks,
            hits=filtered_hits,
            modules=modules,
            metadata=metadata or {},
        )
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    def plot_event(
        self,
        title: Optional[str] = None,
        show_ghosts: bool = True,
        show_modules: bool = True,
    ) -> None:
        """
        Create an interactive 3D visualization of the event.
        
        Displays hits, tracks, and optionally detector modules in a
        3D matplotlib figure.
        
        Parameters
        ----------
        title : str, optional
            Title for the plot.
        show_ghosts : bool, default True
            Whether to show ghost hits (track_id == -1).
        show_modules : bool, default True
            Whether to show detector plane outlines.
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("matplotlib is required for plotting. Install with: pip install matplotlib")
            return
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color map for tracks
        import matplotlib.cm as cm
        colors = cm.rainbow(np.linspace(0, 1, max(self.n_tracks, 1)))
        
        # Plot hits by track
        track_colors = {}
        for i, track in enumerate(self.tracks):
            track_colors[track.track_id] = colors[i % len(colors)]
            
            hits = self.get_hits_by_ids(track.hit_ids)
            if hits:
                xs = [h.x for h in hits]
                ys = [h.y for h in hits]
                zs = [h.z for h in hits]
                ax.scatter(zs, xs, ys, c=[track_colors[track.track_id]], 
                          label=f'Track {track.track_id}', s=50, alpha=0.8)
                # Connect hits with lines
                ax.plot(zs, xs, ys, c=track_colors[track.track_id], alpha=0.5, linewidth=1)
        
        # Plot ghost hits
        if show_ghosts:
            ghost_hits = [h for h in self.hits if h.is_ghost]
            if ghost_hits:
                xs = [h.x for h in ghost_hits]
                ys = [h.y for h in ghost_hits]
                zs = [h.z for h in ghost_hits]
                ax.scatter(zs, xs, ys, c='gray', marker='x', 
                          label='Ghost hits', s=30, alpha=0.5)
        
        # Plot module outlines
        if show_modules and self.modules:
            for module in self.modules:
                # Draw rectangle outline at module z position
                z = module.z
                lx, ly = module.lx, module.ly
                # Four corners
                corners_x = [-lx, lx, lx, -lx, -lx]
                corners_y = [-ly, -ly, ly, ly, -ly]
                corners_z = [z, z, z, z, z]
                ax.plot(corners_z, corners_x, corners_y, 'k-', alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('Z (mm)')
        ax.set_ylabel('X (mm)')
        ax.set_zlabel('Y (mm)')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Event: {self.n_tracks} tracks, {self.n_hits} hits')
        
        ax.legend(loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.show()
