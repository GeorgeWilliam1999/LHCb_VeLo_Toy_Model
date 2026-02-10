"""
State Event Model - Core Data Structures for LHCb VELO Simulation
==================================================================

This module provides the fundamental data structures for representing particle
detector events in the LHCb VELO (Vertex Locator) detector. It defines the
building blocks for hits, segments, tracks, detector geometry, and complete
events.

The module is designed around the LHCb state vector parameterization:
(x, y, tx, ty, p/q) where:
    - x, y: transverse position coordinates
    - tx, ty: direction tangents (slopes)
    - p/q: momentum over charge

Data Structures
---------------
Hit
    A single measurement point in the detector with (x, y, z) coordinates.
    
Segment
    A connection between two hits, typically on adjacent detector layers.
    Provides vector operations for computing angular compatibility.
    
Track
    A collection of hits and segments representing a particle trajectory.
    
Module
    A detector layer/plane containing hits at a specific z position.
    
Event
    A complete collision event containing geometry, tracks, hits, and modules.

Geometry Classes
----------------
Geometry (ABC)
    Abstract base class for detector geometry specifications.
    
PlaneGeometry
    Simple planar detector geometry with rectangular active areas.
    
RectangularVoidGeometry
    Detector geometry with a rectangular beam pipe void in the center.

Visualization
-------------
The Event class provides methods for 3D visualization of tracks and hits,
including `plot_segments()` for interactive display and `save_plot_segments()`
for saving figures.

Example
-------
>>> from LHCB_Velo_Toy_Models.state_event_model import Hit, Segment, Track
>>> 
>>> # Create hits
>>> h1 = Hit(hit_id=0, x=0.0, y=0.0, z=100.0, module_id=0, track_id=0)
>>> h2 = Hit(hit_id=1, x=0.1, y=0.1, z=130.0, module_id=1, track_id=0)
>>> 
>>> # Create segment
>>> seg = Segment(hits=[h1, h2], segment_id=0)
>>> 
>>> # Compute cosine of angle between segments
>>> cos_angle = seg * other_segment

Notes
-----
The coordinate system follows LHCb conventions with z along the beam axis.
"""
from itertools import count
from abc import ABC, abstractmethod
import matplotlib.animation as animation
import dataclasses
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Abstract base class for detector geometry definitions
# -------------------------------------------------------------------------
 

@dataclasses.dataclass(frozen=False)
class Hit:
    """
    A single detector hit (measurement point).
    
    Represents a position measurement in the VELO detector, typically from
    a charged particle crossing a sensor plane.
    
    Attributes
    ----------
    hit_id : int
        Unique identifier for this hit.
    x : float
        X coordinate in mm.
    y : float
        Y coordinate in mm.
    z : float
        Z coordinate (along beam axis) in mm.
    module_id : int
        ID of the detector module/layer containing this hit.
    track_id : int
        ID of the true particle track that created this hit (-1 for ghosts).
    
    Methods
    -------
    __getitem__(index)
        Access coordinates by index: 0=x, 1=y, 2=z.
    """
    hit_id: int
    x: float
    y: float
    z: float
    module_id: int
    track_id: int

    def __getitem__(self, index):
        """Return coordinate by index (0=x, 1=y, 2=z)."""
        return (self.x, self.y, self.z)[index]
    
    def __eq__(self, __value: object) -> bool:
        """Identity comparison (same object in memory)."""
        return self is __value
        #if self.hit_id == __value.hit_id:
        #    return True
        #else:
        #    return False

@dataclasses.dataclass(frozen=False)
class Module:
    """
    A detector module (sensor plane) in the VELO.
    
    Represents a single sensor layer at a fixed z position, containing
    all hits recorded on that layer.
    
    Attributes
    ----------
    module_id : int
        Unique identifier for this module.
    z : float
        Z position of the module in mm.
    lx : float
        Half-width of the active area in x (mm).
    ly : float
        Half-width of the active area in y (mm).
    hits : list[Hit]
        List of hits on this module.
    """
    module_id: int
    z: float
    lx: float
    ly: float
    hits: list[Hit]

    def __eq__(self, __value: object) -> bool:
        """Compare modules by module_id."""
        if self.module_id == __value.module_id:
            return True
        else:
            return False
        
@dataclasses.dataclass
class Segment:
    """
    A track segment connecting two hits.
    
    Segments connect hits on adjacent detector layers and form the building
    blocks for track reconstruction. The class provides vector operations
    for computing angular compatibility between segments.
    
    Attributes
    ----------
    hits : list[Hit]
        List of two hits defining the segment endpoints [start, end].
    segment_id : int
        Unique identifier for this segment.
    
    Methods
    -------
    to_vect()
        Return the 3D direction vector (dx, dy, dz) of the segment.
    __mul__(other)
        Compute cosine of angle between this segment and another.
    
    Notes
    -----
    The multiplication operator computes the cosine of the angle between
    two segments, which is used to determine angular compatibility in
    track finding. Segments with cos(angle) close to 1 are nearly collinear.
    """
    hits: list[Hit]
    segment_id: int
    
    def __eq__(self, __value: object) -> bool:
        """Identity comparison (same object in memory)."""
        return self is __value
        #if self.segment_id == __value.segment_id:
        #    return True
        #else:
        #    return False
    
    def to_vect(self):
        """
        Compute the 3D direction vector of the segment.
        
        Returns
        -------
        tuple
            (dx, dy, dz) direction vector from start to end hit.
        """
        return (self.hits[1].x - self.hits[0].x, 
                self.hits[1].y - self.hits[0].y, 
                self.hits[1].z - self.hits[0].z)
    
    def __mul__(self, __value):
        """
        Compute cosine of angle between this segment and another.
        
        This is the dot product of normalized direction vectors, useful
        for determining angular compatibility in track finding.
        
        Parameters
        ----------
        __value : Segment
            Another segment to compare with.
        
        Returns
        -------
        float
            Cosine of the angle between segments (1 = parallel, 0 = perpendicular).
        """
        v_1 = self.to_vect()
        v_2 = __value.to_vect()
        n_1 = (v_1[0]**2 + v_1[1]**2 + v_1[2]**2)**0.5
        n_2 = (v_2[0]**2 + v_2[1]**2 + v_2[2]**2)**0.5
        
        return (v_1[0]*v_2[0] + v_1[1]*v_2[1] + v_1[2]*v_2[2])/(n_1*n_2)
        
@dataclasses.dataclass
class Track:
    """
    A particle track through the detector.
    
    Represents a reconstructed or true particle trajectory, consisting of
    a sequence of hits and the segments connecting them.
    
    Attributes
    ----------
    track_id : int
        Unique identifier for this track.
    hits : list[Hit]
        Ordered list of hits on the track (by z position).
    segments : list[Segment]
        List of segments connecting consecutive hits.
    """
    track_id    : int
    hits        : list[Hit]
    segments    : list[Segment]
    
    def __eq__(self, __value: object) -> bool:
        """Identity comparison (same object in memory)."""
        return self is __value
        #if self.track_id == __value.track_id:
        #    return True
        #else:
        #    return False
        

@dataclasses.dataclass
class module:
    """
    Legacy module class (lowercase).
    
    .. deprecated::
        Use Module (capitalized) instead.
    """
    module_id: int
    z: float
    lx: float
    ly: float
    hits: list[Hit]
    
    def __eq__(self, __value: object) -> bool:
        if self.module_id == __value.module_id:
            return True
        else:
            return False
        
# -------------------------------------------------------------------------
# Abstract geometry specification
# -------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Geometry(ABC):
    """
    Abstract base class for detector geometry definitions.
    
    Defines the interface for specifying detector geometry, including
    sensor positions and active/inactive regions.
    
    Attributes
    ----------
    module_id : list[int]
        List of module identifiers.
    
    Methods
    -------
    __getitem__(index)
        Access geometry data for a specific module.
    point_on_bulk(state)
        Check if a point is in the active detector region.
    __len__()
        Return the number of modules.
    """
    module_id: list[int]  # List of module identifiers

    @abstractmethod
    def __getitem__(self, index):
        """
        Returns geometry item data at specific index.
        
        Parameters
        ----------
        index : int
            Module index.
        
        Returns
        -------
        tuple
            Geometry data for the module (implementation-specific).
        """
        pass

    @abstractmethod
    def point_on_bulk(self, state: dict):
        """
        Check if a point is within the active detector region.
        
        Parameters
        ----------
        state : dict
            Particle state with 'x' and 'y' keys.
        
        Returns
        -------
        bool
            True if the point is in the active region.
        """
        pass

    def __len__(self):
        """Return the number of modules."""
        return len(self.module_id)


# -------------------------------------------------------------------------
# Plane geometry specification
# -------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class PlaneGeometry(Geometry):
    """
    Simple planar detector geometry with rectangular active areas.
    
    Each module is a flat sensor plane at a specific z position with
    rectangular active area defined by half-widths lx and ly.
    
    Attributes
    ----------
    module_id : list[int]
        List of module identifiers.
    lx : list[float]
        Half-widths of active areas in x (mm).
    ly : list[float]
        Half-widths of active areas in y (mm).
    z : list[float]
        Z positions of the planes (mm).
    
    Example
    -------
    >>> geometry = PlaneGeometry(
    ...     module_id=[0, 1, 2],
    ...     lx=[50.0, 50.0, 50.0],
    ...     ly=[50.0, 50.0, 50.0],
    ...     z=[100.0, 130.0, 160.0]
    ... )
    """
    lx: list[float]  # Half-sizes in the x-direction
    ly: list[float]  # Half-sizes in the y-direction
    z: list[float]   # z positions of planes

    def __getitem__(self, index):
        """
        Get geometry data for a specific module.
        
        Returns
        -------
        tuple
            (module_id, lx, ly, z) for the specified index.
        """
        return (self.module_id[index], 
                self.lx[index], 
                self.ly[index], 
                self.z[index])

    def point_on_bulk(self, state: dict):
        """
        Check if point is within any module's active area.
        
        Parameters
        ----------
        state : dict
            State with 'x' and 'y' coordinates.
        
        Returns
        -------
        bool
            True if point is within at least one module's active area.
        """
        x, y = state['x'], state['y']  # Extract x, y from particle state
        for i in range(len(self.module_id)):
            # Check if x, y are within the lx, ly boundaries
            if (x < self.lx[i] and x > -self.lx[i] and
                y < self.ly[i] and y > -self.ly[i]):
                return True
        return False


# -------------------------------------------------------------------------
# Detector geometry with a rectangular void in the middle
# -------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class RectangularVoidGeometry(Geometry):
    """
    Detector geometry with a rectangular beam pipe void in the center.
    
    This geometry models the VELO's characteristic design where the active
    silicon sensors have a rectangular cutout around the beam pipe.
    
    Attributes
    ----------
    module_id : list[int]
        List of module identifiers.
    z : list[float]
        Z positions of the modules (mm).
    void_x_boundary : list[float]
        Half-width of the void region in x (mm).
    void_y_boundary : list[float]
        Half-width of the void region in y (mm).
    lx : list[float]
        Half-width of the total sensor area in x (mm).
    ly : list[float]
        Half-width of the total sensor area in y (mm).
    
    Notes
    -----
    A point is on the bulk (active region) if it is:
    - Within the outer boundaries (lx, ly)
    - AND outside the inner void (void_x_boundary, void_y_boundary)
    """
    z: list[float]         # z positions
    void_x_boundary: list[float]  # +/- x boundary of the void
    void_y_boundary: list[float]  # +/- y boundary of the void
    lx: list[float]       # +/- x boundary of the entire detector
    ly: list[float]       # +/- y boundary of the entire detector

    def __getitem__(self, index):
        """
        Get geometry data for a specific module.
        
        Returns
        -------
        tuple
            (module_id, lx, ly, z) for the specified index.
        """
        return (
            self.module_id[index],
            self.lx[index],
            self.ly[index],
            self.z[index]
        )

    def point_on_bulk(self, state: dict):
        """
        Check if point is in the active region (outside void, inside boundaries).
        
        Parameters
        ----------
        state : dict
            State with 'x' and 'y' coordinates.
        
        Returns
        -------
        bool
            True if point is in the active (bulk) region.
        """
        x, y = state['x'], state['y']  # Extract x, y
        if (x < self.void_x_boundary and x > -self.void_x_boundary and
            y < self.void_y_boundary and y > -self.void_y_boundary) or (x > self.lx[0] or x < -self.lx[0] or y > self.ly[0] or y < -self.ly[0]):
            return False
        else:
            return True
    



@dataclasses.dataclass
class Event:
    """
    A complete collision event.
    
    Contains all information about a particle collision event including
    detector geometry, true tracks, hits, segments, and modules.
    
    Attributes
    ----------
    detector_geometry : Geometry
        The detector geometry configuration.
    tracks : list[Track]
        List of particle tracks in the event.
    hits : list[Hit]
        List of all hits in the event.
    segments : list[Segment]
        List of all segments in the event.
    modules : list[Module]
        List of detector modules with their hits.
    
    Methods
    -------
    plot_segments()
        Display interactive 3D visualization of tracks and hits.
    save_plot_segments(filename, params=None)
        Save 3D visualization to a file.
    """
    detector_geometry: Geometry
    tracks: list[Track]
    hits: list[Hit]
    segments: list[Segment]
    modules: list[Module]
    
    def __eq__(self, __value: object) -> bool:
        """Identity comparison (same object in memory)."""
        return self is __value

    def plot_segments(self):
        """
        Display interactive 3D visualization of the event.
        
        Shows:
        - Red dots: hits that are part of segments
        - Blue lines: track segments
        - Green X marks: ghost hits (not part of any segment)
        - Gray surfaces: detector planes
        
        Note: Axes are remapped for better visualization (Z->horizontal).
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Gather all hits
        hits = []
        for segment in self.segments:
            hits.extend(segment.hits)

        # Re-map: X-axis <- Z, Y-axis <- Y, Z-axis <- X
        X = [h.z for h in hits]
        Y = [h.y for h in hits]
        Z = [h.x for h in hits]
        ax.scatter(X, Y, Z, c='r', marker='o')

        # Plot lines
        for segment in self.segments:
            x = [h.z for h in segment.hits]
            y = [h.y for h in segment.hits]
            z = [h.x for h in segment.hits]
            ax.plot(x, y, z, c='b')

        # Draw planes from geometry, but only show regions that are in the bulk
        resolution = 25  # Increase for finer mesh
        # print(self.detector_geometry)
        for mod_id, lx, ly, zpos in self.detector_geometry:
            xs = np.linspace(-lx, lx, resolution)
            ys = np.linspace(-ly, ly, resolution)
            X, Y = np.meshgrid(xs, ys)
            Z = np.full_like(X, zpos, dtype=float)

            for idx in np.ndindex(X.shape):
                x_val = X[idx]
                y_val = Y[idx]
                # If not in the bulk (e.g., inside a void), mask out
                if not self.detector_geometry.point_on_bulk({'x': x_val, 'y': y_val, 'z': zpos}):
                    X[idx], Y[idx], Z[idx] = np.nan, np.nan, np.nan

            # Plot, using (Z, Y, X) to match the existing axis mappings
            ax.plot_surface(Z, Y, X, alpha=0.3, color='gray')

        # plot ghost_hits (hits that are not part of a segment)
        ghost_hits = [h for h in self.hits if not any(h in s.hits for s in self.segments)]
        X = [h.z for h in ghost_hits]
        Y = [h.y for h in ghost_hits]
        Z = [h.x for h in ghost_hits]
        ax.scatter(X, Y, Z, c='g', marker='x')

        ax.set_xlabel('Z (horizontal)')
        ax.set_ylabel('Y')
        ax.set_zlabel('X')
        plt.tight_layout()
        plt.show()

    def save_plot_segments(self, filename : str, params: dict = None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Gather all hits
        hits = []
        for segment in self.segments:
            hits.extend(segment.hits)

        # Re-map: X-axis <- Z, Y-axis <- Y, Z-axis <- X
        X = [h.z for h in hits]
        Y = [h.y for h in hits]
        Z = [h.x for h in hits]
        ax.scatter(X, Y, Z, c='r', marker='o')

        # Plot lines
        for segment in self.segments:
            x = [h.z for h in segment.hits]
            y = [h.y for h in segment.hits]
            z = [h.x for h in segment.hits]
            ax.plot(x, y, z, c='b')

        # Draw planes from geometry, but only show regions that are in the bulk
        resolution = 25
         # print(self.detector_geometry)
        for mod_id, lx, ly, zpos in self.detector_geometry:
            xs = np.linspace(-lx, lx, resolution)
            ys = np.linspace(-ly, ly, resolution)
            X, Y = np.meshgrid(xs, ys)
            Z = np.full_like(X, zpos, dtype=float)

            for idx in np.ndindex(X.shape):
                x_val = X[idx]
                y_val = Y[idx]
                # If not in the bulk (e.g., inside a void), mask out
                if not self.detector_geometry.point_on_bulk({'x': x_val, 'y': y_val, 'z': zpos}):
                    X[idx], Y[idx], Z[idx] = np.nan, np.nan, np.nan

            # Plot, using (Z, Y, X) to match the existing axis mappings
            ax.plot_surface(Z, Y, X, alpha=0.3, color='gray')

        # plot ghost_hits (hits that are not part of a segment)
        ghost_hits = [h for h in self.hits if not any(h in s.hits for s in self.segments)]
        X = [h.z for h in ghost_hits]
        Y = [h.y for h in ghost_hits]
        Z = [h.x for h in ghost_hits]
        ax.scatter(X, Y, Z, c='g', marker='x')

        ax.set_xlabel('Z (horizontal)')
        ax.set_ylabel('Y')
        ax.set_zlabel('X')
        if params:
            plt.title(f"Event Parameters: {params}")

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
