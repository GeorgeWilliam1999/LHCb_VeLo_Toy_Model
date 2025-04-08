from LHCB_Velo_Toy_Models.state_event_generator import *
from LHCB_Velo_Toy_Models import state_event_model 

import numpy as np
import matplotlib.pyplot as plt
dz = 20 #mm

n_particles=[5,5,5,5]
events = len(n_particles)
layers = 10
n = np.sum(n_particles)

module_id = [l for l in range(1, layers+1)]
lx = [33 for x in range(1, layers+1)]
ly = [33 for x in range(1, layers+1)]
zs = [dz*l for l in range(1, layers+1)]


# Detector = state_event_model.PlaneGeometry(module_id=[1,2,3],lx = [10,10,10],ly = [10,10,10],z = [10,20,30])
Detector = state_event_model.RectangularVoidGeometry(module_id=module_id,lx = lx,ly = ly,z=zs, void_x_boundary=5, void_y_boundary=5)

state_event_gen = StateEventGenerator(Detector, events = len(n_particles), n_particles=n_particles)
state_event_gen.generate_random_primary_vertices(events, {'x': 0, 'y': 0, 'z': 50})

event_particles = []
for event in range(events):
    particles_list = []
    for particle in range(n):
        particle_dict = {
            'type' : 'MIP',
            'mass': 0.511,
            'q': 1
        }
        particles_list.append(particle_dict)
    event_particles.append(particles_list)

state_event_gen.generate_particles(event_particles)

event_tracks = state_event_gen.generate_complete_events()

for m in event_tracks.modules:
    print(m.module_id)
    print(m)

event_tracks.plot_segments()