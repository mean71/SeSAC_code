from scipy.spatial.distance import pdist, squareform
from particles import Particle

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation


class ParticleBox:      #experiment.py animaion.py 용 클래스 : 입자박스
    """Orbits class
    
    init_state is an [N x 4] array, where N is the number of particles:
       [[x1, y1, vx1, vy1],
        [x2, y2, vx2, vy2],
        ...               ]

    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    """
    def __init__(self,
                 init_state = [[1, 0, 0, -1],
                               [-0.5, 0.5, 0.5, 0.5],
                               [-0.5, -0.5, -0.5, 0.5]],
                 bounds = [-2, 2, -2, 4],
                 size = 0.04,
                 M = 0.05,
                 G = 0.0, 
                 interaction = None,):
        self.init_state = np.asarray(init_state, dtype=float)
        if isinstance(M, float) or isinstance(M, int):
            self.M = M * np.ones(self.init_state.shape[0])
        elif isinstance(M, list) and len(M) == len(init_state):
            self.M = np.array(M)

        self.size = size
        self.state = self.init_state.copy()
        self.particles = []
        for idx, elem in enumerate(init_state):
            if isinstance(M, float) or isinstance(M, int):
                self.particles.append(Particle(particle_id = idx, 
                                position = elem[:2], 
                                velocity = elem[2:4], 
                                properties = {'mass' : M, 
                                              'size' : size,}))
            else:
                self.particles.append(Particle(particle_id = idx, 
                                position = elem[:2], 
                                velocity = elem[2:4], 
                                properties = {'mass' : M[idx], 
                                              'size' : size,}))
        self.time_elapsed = 0
        self.bounds = bounds
        self.G = G
        self.pressure_record = []
        self.interaction = interaction
        self.no_collision = 0
        
    def get_pressure(self):
        if self.time_elapsed < 10:
            #print(self.time_elapsed)
            return 'wait!' 
            
        else:
            s = self.pressure_record[-1]
            return s

    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt
        
        # update positions
        self.state[:, :2] += dt * self.state[:, 2:]

        # find pairs of particles undergoing a collision
        D = squareform(pdist(self.state[:, :2]))
        ind1, ind2 = np.where(D < 4 * self.size)
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        # update velocities of colliding pairs
        for i1, i2 in zip(ind1, ind2):
            self.no_collision += 1 
            # mass
            m1 = self.M[i1]
            m2 = self.M[i2]

            # location vector
            r1 = self.state[i1, :2]
            r2 = self.state[i2, :2]

            # velocity vector
            v1 = self.state[i1, 2:]
            v2 = self.state[i2, 2:]

            # relative location & velocity vectors
            r_rel = r1 - r2
            v_rel = v1 - v2

            # momentum vector of the center of mass
            v_cm = (m1 * v1 + m2 * v2) / (m1 + m2)

            # collisions of spheres reflect v_rel over r_rel
            rr_rel = np.dot(r_rel, r_rel)
            vr_rel = np.dot(v_rel, r_rel)
            v_rel = 2 * r_rel * vr_rel / rr_rel - v_rel

            # assign new velocities
            self.state[i1, 2:] = v_cm + v_rel * m2 / (m1 + m2)
            self.state[i2, 2:] = v_cm - v_rel * m1 / (m1 + m2) 

        # check for crossing boundary
        crossed_x1 = (self.state[:, 0] < self.bounds[0] + self.size)
        crossed_x2 = (self.state[:, 0] > self.bounds[1] - self.size)
        crossed_y1 = (self.state[:, 1] < self.bounds[2] + self.size)
        crossed_y2 = (self.state[:, 1] > self.bounds[3] - self.size)
        
        

        self.state[crossed_x1, 0] = self.bounds[0] + self.size
        self.state[crossed_x2, 0] = self.bounds[1] - self.size

        self.state[crossed_y1, 1] = self.bounds[2] + self.size
        self.state[crossed_y2, 1] = self.bounds[3] - self.size

        self.state[crossed_x1 | crossed_x2, 2] *= -1
        self.state[crossed_y1 | crossed_y2, 3] *= -1
        
            
        # calculate pressure
        def ct(lst):
            return len([a for a in lst if a])
        
        self.pressure_record.append((ct(crossed_x1) + ct(crossed_x1) + ct(crossed_x1) + ct(crossed_x1)))
        
        # calculate forces for each particles
        
        # add gravity
        self.state[:, 3] -= self.M * self.G * dt
        
        # if there are interactions btw particles, add them
        if self.interaction is not None:
            for target_idx in range(len(self.state)):
                force = [0,0]
                for source_idx in range(len(self.state)):
                    
                    if source_idx == target_idx:
                        pass
                    else:
                        f = self.interaction(self.particles[source_idx], 
                                             self.particles[target_idx],)
                        force[0] += f[0]
                        force[1] += f[1]
                self.state[target_idx, 2] += dt * force[0]/self.M[0]
                self.state[target_idx, 3] += dt * force[1]/self.M[0]
                
        # update particle position, velocity
        for idx, elem in enumerate(self.state):
            self.particles[idx].position = elem[:2]
            self.particles[idx].velocity = elem[2:4]