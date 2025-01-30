"""
This file contains the class StateEventGenerator, which is used to generate one or multiple
collision events parameterized by the LHCb state vector (x, y, tx, ty, p/q).
"""

import numpy as np
import LHCB_Velo_Toy_Models.state_event_model.state_event_model as em
import dataclasses
from itertools import count
from abc import ABC, abstractmethod
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from LHCB_Velo_Toy_Models.detector_geometries.geometries import *

# -------------------------------------------------------------------------
# StateEventGenerator class
# -------------------------------------------------------------------------
class StateEventGenerator:
    """
    A class to generate state events for a particle detector simulation.
    """
    def __init__(
        self,
        detector_geometry: Geometry,
        primary_vertices: list[tuple[float, float, float]] = None,
        phi_min: float = - np.pi/20,
        phi_max: float =   np.pi/20,
        theta_min: float = -np.pi / 20,
        theta_max: float = np.pi / 20,
        events: int = 3,
        n_particles: list[int] = None,
        particles: list[dict] = None
    ) -> None:
        """
        Initializes the StateEventGenerator with geometry, angle limits, event counts, etc.
        """
        self.detector_geometry = detector_geometry  # Geometry of the detector
        self.primary_vertices = primary_vertices if primary_vertices is not None else []
        self.phi_min = phi_min                      # Minimum azimuthal angle
        self.phi_max = phi_max                      # Maximum azimuthal angle
        self.theta_min = theta_min                  # Minimum polar angle
        self.theta_max = theta_max                  # Maximum polar angle
        self.events = events                        # Number of events to generate
        self.n_particles = n_particles if n_particles is not None else []
        self.particles = particles if particles is not None else []
        self.rng = np.random.default_rng()          # Random number generator
        self.measurment_error_flag = True           # Flag for measurment error

    def generate_random_primary_vertices(
        self,
        n_events: int,
        phsyical_variance: dict[str, float]
    ) -> list[tuple[float, float, float]]:
        """
        Generates random primary vertices (x, y, z) for events based on provided variances.
        """
        primary_vertices = []  # Accumulate generated vertices
        # Ensure variances are small
        assert all(v <= 1e-3 for v in phsyical_variance.values()), 'Physical variances must be small'
        for _ in range(self.events):
            # Generate each primary vertex with normal distribution
            x = self.rng.normal(0, phsyical_variance['x'])
            y = self.rng.normal(0, phsyical_variance['y'])
            z = self.rng.normal(0, phsyical_variance['z'])
            primary_vertices.append((x, y, z))
        # Store back in the instance
        self.primary_vertices = primary_vertices
        return primary_vertices

    def set_primary_vertices(self, primary_vertices: list[tuple[float, float, float]]) -> None:
        """
        Sets the list of primary vertices for the events.
        """
        self.primary_vertices = primary_vertices
        # Validate length
        assert len(primary_vertices) == self.events, (
            'Number of primary vertices must be equal to the number of events'
        )
        # Validate structure
        assert all([len(vertex) == 3 for vertex in primary_vertices]), (
            'Primary vertices must be a list of 3-tuples'
        )
        # Validate closeness to origin
        assert all([np.linalg.norm(np.array(vertex)) < 1e3 for vertex in primary_vertices]), (
            'Primary vertices must be close to the origin'
        )

    def generate_particles(self, particles: list[list[dict]] = None) -> list[dict]:
        """
        Generates particle state dictionaries for each event based on the primary vertices.
        """
        init_particles = []  # Overwrite local particles list
        for event in range(self.events):
            # For each event, retrieve the stored primary vertex
            x, y, z = self.primary_vertices[event]
            event_particles = []  # Accumulate particles for this event
            for n in range(self.n_particles[event]):
                # Sample azimuthal angle (phi) in [phi_min, phi_max]
                phi = self.rng.uniform(self.phi_min, self.phi_max)
                # Sample cos(theta) to get uniform distribution in cos(theta)
                theta = self.rng.uniform(self.theta_min,self.theta_max)

                # Compute direction components from angles
                

                # Transverse slopes
                tx = np.sin(phi)
                ty = np.sin(theta)

                # Retrieve charge from input (example usage)
                # If "q" or other keys are needed, adapt code accordingly
                # Assumes 'q' is in the input dictionary per particle specification
                q = 1 if not particles else particles[event][n].get('q', 1)

                # Compute momentum (dummy usage, must adapt to actual model)
                p = np.linalg.norm([tx, ty, 1]) * particles[event][n]['mass']*0.89

                # Append new particle state
                event_particles.append(
                    {
                        'type' : particles[event][n]['type'],
                        'x': x,
                        'y': y,
                        'z': z,
                        'tx': tx,
                        'ty': ty,
                        'p/q': p/q  
                    }
                )
                # Store final list in the instance
            init_particles.append(event_particles)
            self.particles = init_particles
        return init_particles

    def collision_update(self, particle: dict) -> dict:
        """
        Updates a particle's direction to simulate a collision.
        """
        # Small random rotation in the tx, ty plane
        theta_hat = 0#self.rng.normal(0, 0.000)

        phi,theta = np.random.normal(np.arcsin(particle['tx']), 0.01), np.random.normal(np.arcsin(particle['ty']), 0.01)


    
        rotation_matrix = np.array([
            [np.cos(theta_hat), -np.sin(theta_hat)],
            [np.sin(theta_hat),  np.cos(theta_hat)]
        ])
        tx_ty = np.array([particle['tx'], particle['ty']])
        tx_ty_rotated = rotation_matrix @ tx_ty
        # Update slopes
        particle['tx'], particle['ty'] = np.sin(phi), np.sin(theta)
        return particle
    
    def measurment_error(self, particle: dict) -> dict:
        """
        Updates a particle's position to simlate a measurenemnt error
        """
        # Random slight shifts in x, y
        particle['x'] += np.random.normal(0, 1)
        particle['y'] += np.random.normal(0, 1)
        
        return particle

    def propagate(self, particle: dict, dz: float) -> dict:
        """
        Moves a particle forward by dz in the z-direction, updating x and y.
        """
        # Position updates based on slopes
        particle['x'] += particle['tx'] * dz
        particle['y'] += particle['ty'] * dz
        return particle

    def generate_complete_events(self):
        """
        Generates fully propagated events, from the primary vertices through each detector layer,
        recording hits and segments along the way.
        Returns a list of lists (one list per event), where each sublist contains tracks.
        """
        #init hit counter
        hit_counter = count()
        #init track counter
        track_counter = count()
        # Prepare container for all events
        all_event_tracks = []
        # Loop over the number of events
        for evt_idx in range(self.events):
            # Container for tracks in this event
            event_tracks = []
            # Retrieve the primary vertex for this event
            vx, vy, vz = self.primary_vertices[evt_idx]
            for p_idx in range(self.n_particles[evt_idx]):
                track_id = next(track_counter)
                # Create a new track with empty collections of hits and segments
                track = em.Track(track_id, hits=[], segments=[])
                # Initialize the particle's state at the primary vertex
                state = self.particles[evt_idx][p_idx]
                # Propagate through each layer of the detector geometry
                for i in range(len(self.detector_geometry)):
                    # Retrieve module data
                    mod_id, lx, ly, zpos = self.detector_geometry[i]
                    # Calculate distance to the next layer along z
                    dz = zpos - state['z']
                    # Update particle state by propagating in z
                    state = self.propagate(state, dz)
                    # if not self.detector_geometry.point_on_bulk(state):
                    #     continue
                    # Create and record a new hit at this layer
                    # if 
                    if self.measurment_error_flag:
                        errot_state = self.measurment_error(state)
                        hit = em.Hit(
                            hit_id = next(hit_counter),
                            track_id = track_id,         
                            x=errot_state['x'],
                            y=errot_state['y'],
                            z=zpos,
                            module_id=mod_id
                        )
                    else:
                        hit = em.Hit(
                            hit_id = next(hit_counter),
                            track_id = track_id,  
                            x=state['x'],
                            y=state['y'],
                            z=zpos,
                            module_id=mod_id
                        )
                    state = self.collision_update(state)
                    track.hits.append(hit)
                #find the segments
                for i in range(len(track.hits)-1):
                    seg = em.Segment(
                        segment_id = i,
                        hits = [track.hits[i], track.hits[i+1]]
                    )
                    track.segments.append(seg)
                # Store this track in the event collection
                event_tracks.append(track)
            # Store all tracks for this event
            all_event_tracks.append(event_tracks)
            
        # Return all events, each containing its tracks
        self.tracks = all_event_tracks
        self.hits = [hit for sublist in [track.hits for sublist in all_event_tracks for track in sublist] for h in sublist]
        self.segments = [s for sublist in [track.segments for sublist in all_event_tracks for track in sublist] for s in sublist]
        
        return em.Event(self.detector_geometry, self.tracks, self.hits, self.segments)
        # return all_event_tracks
    
    