import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from LHCB_Velo_Toy_Models.state_event_generator import state_event_generator as seg 
from LHCB_Velo_Toy_Models import state_event_model 

import numpy as np
import matplotlib.pyplot as plt

Detector = seg.PlaneGeometry(module_id=[1,2,3],lx = [10,10,10],ly = [10,10,10],z = [10,20,30])
# Detector = seg.RectangularVoidGeometry(module_id=[1,2,3],lx = [10,10,10],ly = [10,10,10],z=[10,20,30], void_x_boundary=4, void_y_boundary=4)

state_event_gen = seg.StateEventGenerator(Detector, events = 5, n_particles=[5,5,2,2,6])
primary_vertices = state_event_gen.generate_random_primary_vertices(10, {'x': 1e-3, 'y': 1e-3, 'z': 1e-3})

events = 5
n_particles = 20

event_particles = []
for event in range(events):
    particles_list = []
    for particle in range(n_particles):
        particle_dict = {
            'type' : 'electron',
            'mass': 0.511,
            'q': 1
        }
        particles_list.append(particle_dict)
    event_particles.append(particles_list)

state_event_gen.generate_particles(event_particles)

event_tracks = state_event_gen.generate_complete_events()

event_tracks.plot_segments()