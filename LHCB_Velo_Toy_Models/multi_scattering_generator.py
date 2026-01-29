"""
Multiple Scattering Event Generator
====================================

This module provides a legacy event generator that simulates particle tracks
with multiple scattering effects. It generates collision events where particles
undergo angular deflections as they traverse detector material.

This generator is an alternative to StateEventGenerator and provides a simpler
model focused on multiple scattering physics.

Features
--------
- Random primary vertex generation with Gaussian spread
- Particle propagation through planar detector geometry
- Multiple scattering simulation via angular deflection at each layer
- Hit position recording with measurement noise

Physics Model
-------------
Particles are generated at a primary vertex and propagated through the detector.
At each detector layer:
1. The particle position is updated based on its direction
2. Angular scattering is applied (Gaussian distributed)
3. A hit is recorded if the particle is within the sensor acceptance

The scattering angles follow a simplified model where both polar and azimuthal
angles receive independent Gaussian deflections.

Example
-------
>>> from LHCB_Velo_Toy_Models.multi_scattering_generator import (
...     SimpleDetectorGeometry, MultiScatteringGenerator
... )
>>> 
>>> # Define detector
>>> geometry = SimpleDetectorGeometry(
...     module_id=list(range(10)),
...     lx=[50.0] * 10,
...     ly=[50.0] * 10,
...     z=[100 + i * 30 for i in range(10)]
... )
>>> 
>>> # Create generator
>>> gen = MultiScatteringGenerator(detector_geometry=geometry)
>>> 
>>> # Generate events
>>> events = gen.generate_event(n_particles=5, n_events=10)

Notes
-----
For more realistic simulations with LHCb state vectors, use StateEventGenerator.
"""

import numpy as np
import LHCB_Velo_Toy_Models.state_event_model as em
import dataclasses
from itertools import count


@dataclasses.dataclass(frozen=True)
class SimpleDetectorGeometry:
    """
    Simple planar detector geometry for the multiple scattering generator.
    
    Attributes
    ----------
    module_id : list[int]
        Unique identifiers for each detector module.
    lx : list[float]
        Half-widths of active areas in x (mm).
    ly : list[float]
        Half-widths of active areas in y (mm).
    z : list[float]
        Z positions of detector planes (mm).
    """
    module_id   : list[int]
    lx          : list[float]
    ly          : list[float]
    z           : list[float]
    
    def __getitem__(self, index):
        """Return (module_id, lx, ly, z) for a given index."""
        return (self.module_id[index], self.lx[index], self.ly[index], self.z[index])
    
    def __len__(self):
        """Return number of modules."""
        return len(self.module_id)


@dataclasses.dataclass()
class MultiScatteringGenerator:
    """
    Event generator with multiple scattering simulation.
    
    Generates particle tracks from a primary vertex through a detector,
    applying random angular deflections at each layer to simulate
    multiple scattering in detector material.
    
    Parameters
    ----------
    detector_geometry : SimpleDetectorGeometry
        The detector geometry specification.
    primary_vertices : list, optional
        Pre-defined primary vertex positions.
    phi_min, phi_max : float
        Range for initial azimuthal angle (default: 0 to 2π).
    theta_min, theta_max : float
        Range for initial polar angle (default: 0 to π/10).
    rng : numpy.random.Generator
        Random number generator instance.
    
    Attributes
    ----------
    theta_divergence : float
        RMS of polar angle scattering (radians).
    phi_divergence : float
        RMS of azimuthal angle scattering (radians).
    """
    detector_geometry   : SimpleDetectorGeometry
    primary_vertices    : list = dataclasses.field(default_factory=list)
    phi_min             : float = 0.0
    phi_max             : float = 2*np.pi
    theta_min           : float = 0.0
    theta_max           : float = np.pi/10
    rng                 : np.random.Generator = np.random.default_rng()
    #ToDo : Fix the divergence angles
    theta_divergence = np.pi/20
    phi_divergence = np.pi/20

    def generate_random_primary_vertices(self, n_events, sigma):
        """
        Generate random primary vertices with Gaussian spread.
        
        Parameters
        ----------
        n_events : int
            Number of vertices to generate.
        sigma : tuple[float, float, float]
            Standard deviations for (x, y, z) coordinates.
        
        Returns
        -------
        list[tuple]
            List of (x, y, z) vertex positions.
        """
        primary_vertices = []
        for _ in range(n_events):
            x = self.rng.normal(0, sigma[0])
            y = self.rng.normal(0, sigma[1])
            z = self.rng.normal(0, sigma[2])
            primary_vertices.append((x, y, z))
        return primary_vertices

    def find_vs(self, theta, phi):
        """
        Convert spherical angles to Cartesian direction vector.
        
        Parameters
        ----------
        theta : float
            Polar angle (from z-axis).
        phi : float
            Azimuthal angle (in x-y plane).
        
        Returns
        -------
        tuple
            (vx, vy, vz) unit direction vector.
        """
        return np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
    
    #ToDo: Fix events naming
    def generate_event(self, n_particles, n_events=1, sigma=(0,0,0), defined_primary_vertex=None):
        """
        Generate collision events with multiple scattering.
        
        Creates particle tracks from primary vertices through the detector,
        applying multiple scattering at each layer crossing.
        
        Parameters
        ----------
        n_particles : int
            Number of particles per event.
        n_events : int, optional
            Number of events to generate (default: 1).
        sigma : tuple, optional
            Vertex spread (x, y, z) if generating random vertices.
        defined_primary_vertex : list[tuple], optional
            Pre-defined primary vertices for each event.
        
        Returns
        -------
        Event or list[Event]
            Single Event if n_events=1, otherwise list of Events.
        
        Notes
        -----
        Multiple scattering is simulated by adding Gaussian-distributed
        angular deflections at each detector layer.
        """
        hit_id_counter = count()
        all_events = []

        for event_index in range(n_events):
            
            if defined_primary_vertex is not None:
                primary_vertex = defined_primary_vertex[event_index]
            elif defined_primary_vertex is None:
                primary_vertex = (0,0,0)
            else:
                primary_vertex = self.generate_random_primary_vertices(1,sigma)[0]

            mc_info = []

            hits_per_module = [[] for _ in range(len(self.detector_geometry.module_id))]
            hits_per_track = []

            pvx, pvy, pvz = primary_vertex
            self.primary_vertices.append((pvx, pvy, pvz))

            for track_id in range(n_particles):
                phi = self.rng.uniform(self.phi_min, self.phi_max)
                cos_theta = self.rng.uniform(np.cos(self.theta_max), np.cos(self.theta_min))
                theta = np.arccos(cos_theta)
                sin_theta = np.sin(theta)

                vx, vy, vz = self.find_vs(theta, phi)

                track_hits = []
                zs = [pvz]
                x_hits = [pvx]
                y_hits = [pvy]
                vxs = [vx]
                vys = [vy]
                vzs = [vz]
                ts = [0]
                thetas = [theta]
                phis = [phi]
                ts = []

                for idx, (module_id, zm, lx, ly) in enumerate(
                        zip(self.detector_geometry.module_id, self.detector_geometry.z, self.detector_geometry.lx,
                            self.detector_geometry.ly)):
                    
                    t = (zm - zs[idx]) / vz
                    ts.append(t)
                    zs.append(zm)

                    x_hits.append(x_hits[idx] + vxs[idx] * t)
                    y_hits.append(y_hits[idx] + vys[idx] * t)
                    # ToDo: Name resolution noise a parameter of order 10 microns

                    #ToDo: Impliment as x,y scattering - Marcel to send specification
                    additional_cos_phi = self.rng.normal(0, self.phi_divergence)
                    additional_cos_theta = self.rng.normal(0, self.theta_divergence)
                    additional_phi = np.arccos(additional_cos_phi)
                    additional_theta = np.arccos(additional_cos_theta)

                    phis.append(phi + additional_phi)
                    thetas.append(theta + additional_theta)

                    vx, vy, vz = self.find_vs(thetas[idx+1], phis[idx+1])
                    vxs.append(vx)
                    vys.append(vy)
                    vzs.append(vz)

            
                   
                    if np.abs(x_hits[idx+1]) < lx / 2 and np.abs(y_hits[idx+1]) < ly / 2: # ToDo: Need to account for the rectangular void in the buld!
                        hit = em.Hit(next(hit_id_counter), x_hits[idx+1] + self.rng.normal(0, 1e-5), y_hits[idx+1] + self.rng.normal(0, 1e-5), zm, module_id, track_id)
                        hits_per_module[idx].append(hit)
                        # track_hits.append(hit)
                        track_hits = [em.Hit(next(hit_id_counter),x + self.rng.normal(0, 1e-5), y + self.rng.normal(0, 1e-5), z, module_id, track_id) for x, y, z in zip(x_hits[1:], y_hits[1:], zs[1:])]
                hits_per_track.append(track_hits)

            mc_info.append((track_id, em.MCInfo(
                    primary_vertex,
                    phis,
                    thetas,
                    ts)))

            modules = [em.Module(module_id, z, lx, ly, hits_per_module[idx]) for idx, (module_id, z, lx, ly) in
                       enumerate(
                           zip(self.detector_geometry.module_id, self.detector_geometry.z, self.detector_geometry.lx,
                                self.detector_geometry.ly))]
            tracks = []

            for idx, (track_id, mc_info) in enumerate(mc_info):
                tracks.append(em.Track(track_id, mc_info, hits_per_track[idx]))
            global_hits = [hit for sublist in hits_per_module for hit in sublist]

            all_events.append(em.Event(modules, tracks, global_hits))
        if n_events == 1:
            all_events = all_events[0]
        return all_events
                