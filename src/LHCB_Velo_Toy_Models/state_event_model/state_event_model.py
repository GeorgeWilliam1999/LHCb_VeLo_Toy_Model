'''
The goal of this module is to provide a state event model for for tracks parameterised by the standard LHCb state (x,y,tx,ty,p/q)
'''

import dataclasses
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from LHCB_Velo_Toy_Models.detector_geometries.geometries import *

@dataclasses.dataclass(frozen=False)
class Hit:
    hit_id: int
    x: float
    y: float
    z: float
    module_id: int
    track_id: int

    def __getitem__(self, index):
        return (self.x, self.y, self.z)[index]
    
    def __eq__(self, __value: object) -> bool:
        return self is __value
        #if self.hit_id == __value.hit_id:
        #    return True
        #else:
        #    return False

@dataclasses.dataclass(frozen=False)
class Module:
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
        
@dataclasses.dataclass
class Segment:
    segment_id: int
    hits: list[Hit]
    
    def __eq__(self, __value: object) -> bool:
        return self is __value
        #if self.segment_id == __value.segment_id:
        #    return True
        #else:
        #    return False
        
@dataclasses.dataclass
class Track:
    track_id    : int
    hits        : list[Hit]
    segments    : list[Segment]
    
    def __eq__(self, __value: object) -> bool:
        return self is __value
        #if self.track_id == __value.track_id:
        #    return True
        #else:
        #    return False

@dataclasses.dataclass
class Event:
    detector_geometry: Geometry
    tracks: list[Track]
    hits: list[Hit]
    segments: list[Segment]
    
    def __eq__(self, __value: object) -> bool:
        return self is __value
        #if self.event_id == __value.event_id:
        #    return True
        #else:
        #    return False

    def plot_segments(self):
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
        print(self.detector_geometry)
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

        ax.set_xlabel('Z (horizontal)')
        ax.set_ylabel('Y')
        ax.set_zlabel('X')
        plt.tight_layout()
        plt.show()
