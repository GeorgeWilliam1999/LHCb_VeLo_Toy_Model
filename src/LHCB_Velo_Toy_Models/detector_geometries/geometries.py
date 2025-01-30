
import numpy as np
import LHCB_Velo_Toy_Models.state_event_model.state_event_model as em
import dataclasses
from itertools import count
from abc import ABC, abstractmethod
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Abstract base class for detector geometry definitions
# -------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Geometry(ABC):
    module_id: list[int]  # List of module identifiers

    @abstractmethod
    def __getitem__(self, index):
        """
        Returns geometry item data at specific index.
        """
        pass

    @abstractmethod
    def point_on_bulk(self, state: dict):
        """
        Checks if the (x, y) point from a particle state is within the geometry.
        """
        pass

    def __len__(self):
        """
        Returns the number of modules.
        """
        return len(self.module_id)


# -------------------------------------------------------------------------
# Plane geometry specification
# -------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class PlaneGeometry(Geometry):
    lx: list[float]  # Half-sizes in the x-direction
    ly: list[float]  # Half-sizes in the y-direction
    z: list[float]   # z positions of planes

    def __getitem__(self, index):
        """
        Returns tuple (module_id, lx, ly, z) for a specific index.
        """
        return (self.module_id[index], 
                self.lx[index], 
                self.ly[index], 
                self.z[index])

    def point_on_bulk(self, state: dict):
        """
        Checks if a given state (x, y) is within plane boundaries.
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
    Detector geometry that contains a rectangular void region in the center.
    """
    z: list[float]         # z positions
    void_x_boundary: list[float]  # +/- x boundary of the void
    void_y_boundary: list[float]  # +/- y boundary of the void
    lx: list[float]       # +/- x boundary of the entire detector
    ly: list[float]       # +/- y boundary of the entire detector

    def __getitem__(self, index):
        """
        Returns tuple with module_id, void, and boundary definitions.
        """
        return (
            self.module_id[index],
            # self.void_x_boundary,
            # self.void_y_boundary,
            self.lx[index],
            self.ly[index],
            self.z[index]
        )

    def point_on_bulk(self, state: dict):
        """
        Checks if (x, y) point is outside the void region, indicating it is on the bulk material.
        """
        x, y = state['x'], state['y']  # Extract x, y
        if (x < self.void_x_boundary and x > -self.void_x_boundary and
            y < self.void_y_boundary and y > -self.void_y_boundary):
            return False
        else:
            return True