from particlebox import ParticleBox
from util import *

import matplotlib.pyplot as plt
import numpy as np 

def observe_pressure(particle_number = 50, 
                     particle_velocity = 1, 
                     box_size = 2,
                     particle_mass = 0.05,
                     particle_size = 0.04,
                     gravity = 0.0,
                     dt = 1./30,
                     wait_time = 3000, 
                     observe_time = 300,):
    np.random.seed(0)

    NUMBER_OF_PARTICLES = particle_number
    MEAN_VELOCITY = particle_velocity
    init_state = []

    for i in range(NUMBER_OF_PARTICLES):
        # initial position 
        x_init, y_init = np.random.random(), np.random.random()
        x_init = 2 * box_size * (x_init - 0.5)
        y_init = 2 * box_size * (y_init - 0.5)
        
        # initial velocity
        v_dir_init = np.random.random()
        v_dir_init = 2 * np.pi * v_dir_init
        v_x_init = MEAN_VELOCITY * np.cos(v_dir_init)
        v_y_init = MEAN_VELOCITY * np.sin(v_dir_init)
        init_state.append([x_init, y_init, v_x_init, v_y_init])

    #init_state = -0.5 + np.random.random((NUMBER_OF_PARTICLES, 4))
    #init_state[:, :2] *= 3.9

    bounds = [-box_size, box_size, -box_size, box_size]

    box = ParticleBox(init_state, bounds, particle_size, particle_mass, gravity)
    dt = dt
    res = []
    
    for i in range(wait_time):
        box.step(dt)
    
    for i in range(observe_time):
        box.step(dt)
        res.append(box.get_pressure())
    
    return res
    
def observe_dist(particle_number = 50, 
         particle_velocity = None, 
         box_size = 2,
         particle_mass = 0.05,
         particle_size = 0.04,
         gravity = 0.0,
         dt = 1./30,):
    init_state = -0.5 + np.random.random((NUMBER_OF_PARTICLES, 4))
    init_state[:, :2] *= 3.9
    
    init_state = 
    
if __name__ == '__main__':

    # example run 
    res = observe_pressure()
    print(sum(res)/len(res))
    
    # plot p-v 
    print('start p-v plotting')
    pv_experiment = []
    
    v = 1
    dv = 0.1 # increment box size by 0.1
    for _ in range(10):
        p = observe_pressure(box_size = v)
        print(mean(p), v**3)                # util.py 함수호출
        pv_experiment.append((mean(p), v**3))
        v = v + dv
        
    
    p_list = [e[0] for e in pv_experiment]
    v_list = [e[1] for e in pv_experiment]
    
    plt.scatter(p_list, v_list)
    plt.savefig('pv_plot.png')
    plt.clf()
    
    # plot p-energy
    print('start p-E plotting')
    pe_experiment = []
    e = 1
    de = 0.1
    
    for _ in range(10):
        p = observe_pressure(particle_velocity = e)
        pe_experiment.append((mean(p), e**2))       #util.py 함수 호출
        e = e + de
        print(mean(p), e**2)
        
    p_list = [e[0] for e in pe_experiment]
    e_list = [e[1] for e in pe_experiment]
    
    plt.scatter(p_list, e_list)
    plt.savefig('pe_plot.png')
    
    
    
    # plot p-n 
    print('start p-n plotting')
    pe_experiment = []
    n = 100
    dn = 10
    
    for _ in range(100):
        p = observe_pressure(particle_number = n)
        pe_experiment.append((mean(p), n))              # util.py 함수 호출
        n = n + dn
        print(mean(p), n)
        
    p_list = [e[0] for e in pe_experiment]
    e_list = [e[1] for e in pe_experiment]
    
    plt.scatter(p_list, e_list)
    plt.savefig('pn_plot.png')    