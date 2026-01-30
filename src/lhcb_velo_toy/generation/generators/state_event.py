"""
StateEventGenerator: Primary event generator using LHCb state vectors.

This generator creates collision events by propagating particles through
the detector using LHCb-style state vectors (x, y, tx, ty, p/q).
"""

from __future__ import annotations
from typing import Optional

import numpy as np

from lhcb_velo_toy.core.types import StateVector, Position
from lhcb_velo_toy.generation.geometry.base import Geometry
from lhcb_velo_toy.generation.models.event import Event


class StateEventGenerator:
    """
    Event generator using LHCb state vectors.
    
    This class generates simulated collision events by creating particles
    at primary vertices and propagating them through the detector geometry.
    Particles are represented by LHCb-style state vectors containing position
    and slope information.
    
    Parameters
    ----------
    detector_geometry : Geometry
        The detector geometry configuration.
    primary_vertices : list[tuple[float, float, float]], optional
        Pre-defined primary vertex positions as (x, y, z) tuples.
    phi_min : float, default -0.2
        Minimum angle for tx slope (radians).
    phi_max : float, default 0.2
        Maximum angle for tx slope (radians).
    theta_min : float, default -0.2
        Minimum angle for ty slope (radians).
    theta_max : float, default 0.2
        Maximum angle for ty slope (radians).
    events : int, default 3
        Number of collision events to generate.
    n_particles : list[int], optional
        Number of particles per event.
    measurement_error : float, default 0.0
        Gaussian σ for hit position smearing (mm).
    collision_noise : float, default 1e-4
        Gaussian σ for multiple scattering angle (radians).
    
    Attributes
    ----------
    detector_geometry : Geometry
        The detector geometry.
    primary_vertices : list[tuple]
        Primary vertex positions.
    particles : list[dict]
        Generated particle state vectors.
    modules : list[Module]
        Detector modules with hits.
    
    Examples
    --------
    >>> geometry = PlaneGeometry(
    ...     module_id=[1, 2, 3, 4, 5],
    ...     lx=[33.0] * 5, ly=[33.0] * 5,
    ...     z=[20.0, 40.0, 60.0, 80.0, 100.0]
    ... )
    >>> gen = StateEventGenerator(
    ...     detector_geometry=geometry,
    ...     events=1,
    ...     n_particles=[10],
    ...     measurement_error=0.005
    ... )
    >>> gen.generate_random_primary_vertices({"x": 0.1, "y": 0.1, "z": 1.0})
    >>> gen.generate_particles([[{"type": "MIP"} for _ in range(10)]])
    >>> event = gen.generate_complete_events()
    
    Notes
    -----
    The state vector uses LHCb conventions:
    - x, y, z: position in mm
    - tx = dx/dz, ty = dy/dz: slopes
    - p/q: momentum divided by charge (MeV/c)
    """
    
    def __init__(
        self,
        detector_geometry: Geometry,
        primary_vertices: Optional[list[Position]] = None,
        phi_min: float = -0.2,
        phi_max: float = 0.2,
        theta_min: float = -0.2,
        theta_max: float = 0.2,
        events: int = 3,
        n_particles: Optional[list[int]] = None,
        particles: Optional[list[dict]] = None,
        measurement_error: float = 0.0,
        collision_noise: float = 1e-4,
    ) -> None:
        """Initialize the event generator."""
        raise NotImplementedError
    
    def generate_random_primary_vertices(
        self,
        variance: dict[str, float],
    ) -> list[Position]:
        """
        Generate Gaussian-distributed primary vertices.
        
        Creates primary vertex positions for each event, sampled from
        Gaussian distributions centered at the origin.
        
        Parameters
        ----------
        variance : dict[str, float]
            Dictionary with 'x', 'y', 'z' keys specifying the variance
            (σ²) for each coordinate.
        
        Returns
        -------
        list[tuple[float, float, float]]
            List of (x, y, z) vertex positions, one per event.
        
        Examples
        --------
        >>> gen.generate_random_primary_vertices({"x": 0.1, "y": 0.1, "z": 1.0})
        [(0.05, -0.03, 0.8), ...]
        """
        raise NotImplementedError
    
    def set_primary_vertices(
        self,
        vertices: list[Position],
    ) -> None:
        """
        Set explicit primary vertex positions.
        
        Parameters
        ----------
        vertices : list[tuple[float, float, float]]
            List of (x, y, z) vertex positions. Must have length equal
            to the number of events.
        
        Raises
        ------
        ValueError
            If the number of vertices doesn't match the number of events.
        """
        raise NotImplementedError
    
    def generate_particles(
        self,
        particles: list[list[dict]],
    ) -> list[dict]:
        """
        Generate particle state vectors.
        
        Creates initial state vectors for particles based on the provided
        particle configurations. Slopes (tx, ty) are sampled uniformly
        from the configured angular ranges.
        
        Parameters
        ----------
        particles : list[list[dict]]
            Nested list of particle configurations. Outer list is per-event,
            inner list is per-particle. Each particle dict may contain:
            - 'type': Particle type (e.g., 'MIP')
            - 'mass': Particle mass (MeV/c²)
            - 'q': Particle charge
        
        Returns
        -------
        list[dict]
            Flattened list of particle state dictionaries.
        
        Examples
        --------
        >>> particles = [[{"type": "MIP", "mass": 0.511, "q": 1} for _ in range(5)]]
        >>> gen.generate_particles(particles)
        """
        raise NotImplementedError
    
    def propagate(
        self,
        state: StateVector,
        z_target: float,
    ) -> StateVector:
        """
        Propagate a particle state to a target z position.
        
        Uses linear extrapolation based on the particle slopes.
        
        Parameters
        ----------
        state : dict[str, float]
            Current particle state with 'x', 'y', 'z', 'tx', 'ty'.
        z_target : float
            Target z position (mm).
        
        Returns
        -------
        dict[str, float]
            Updated state at the target z position.
        
        Notes
        -----
        Propagation equations:
        - x_new = x + tx * (z_target - z)
        - y_new = y + ty * (z_target - z)
        """
        raise NotImplementedError
    
    def collision_update(
        self,
        state: StateVector,
    ) -> StateVector:
        """
        Apply multiple scattering to a particle state.
        
        Adds Gaussian noise to the particle slopes to simulate multiple
        Coulomb scattering in detector material.
        
        Parameters
        ----------
        state : dict[str, float]
            Current particle state.
        
        Returns
        -------
        dict[str, float]
            Updated state with scattered slopes.
        """
        raise NotImplementedError
    
    def measurement_error_update(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        """
        Apply measurement error to hit positions.
        
        Adds Gaussian noise to simulate detector position resolution.
        
        Parameters
        ----------
        x : float
            True x coordinate (mm).
        y : float
            True y coordinate (mm).
        
        Returns
        -------
        tuple[float, float]
            Smeared (x, y) coordinates.
        """
        raise NotImplementedError
    
    def generate_complete_events(self) -> Event:
        """
        Generate complete events with tracks and hits.
        
        Propagates all particles through the detector, recording hits
        at each module crossing, and constructs the full event structure.
        
        Returns
        -------
        Event
            Complete event with geometry, tracks, hits, segments, modules.
        
        Notes
        -----
        This method:
        1. Iterates through all particles
        2. Propagates each particle to each detector module
        3. Checks if the particle is within the active region
        4. Records hits with optional measurement error
        5. Applies multiple scattering between modules
        6. Constructs tracks and segments from hits
        """
        raise NotImplementedError
    
    def make_noisy_event(
        self,
        drop_rate: float = 0.0,
        ghost_rate: float = 0.0,
    ) -> Event:
        """
        Create a noisy version of the generated event.
        
        Simulates detector inefficiencies by randomly removing hits and
        adding ghost (fake) hits.
        
        Parameters
        ----------
        drop_rate : float, default 0.0
            Probability of dropping each hit (0 to 1).
        ghost_rate : float, default 0.0
            Probability of adding a ghost hit per module (0 to 1).
        
        Returns
        -------
        Event
            Event with dropped and ghost hits applied.
        
        Examples
        --------
        >>> noisy_event = gen.make_noisy_event(drop_rate=0.1, ghost_rate=0.05)
        
        Notes
        -----
        Ghost hits are assigned track_id = -1 to distinguish them from
        true hits.
        """
        raise NotImplementedError
    
    def _rebuild_modules(self) -> None:
        """Rebuild module hit lists after modifications."""
        raise NotImplementedError
