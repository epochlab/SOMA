#!/usr/bin/env python3

import numpy as np

class ParticleField():
    def __init__(self, N, w, h, dt):
        self.state = np.column_stack((np.random.uniform(0, w, N),   # x
                                      np.random.uniform(0, h, N),   # y
                                      np.random.uniform(-1, 1, N),  # vx
                                      np.random.uniform(-1, 1, N),  # vy
                                      np.random.uniform(1, 1, N)) # life
                                    )
        
        self.bounds = np.array([w, h])
        self.dt = dt
        self.halflife = 2.0 # secs

    def dynamics(self, state, t):
        x, y, vx, vy, life = state[:,0], state[:,1], state[:,2], state[:,3], state[:,4]

        vx, vy = self.boundary_collision(x, y, vx, vy)

        factor = self.exp_decay()
        life = np.maximum(0, life * factor)

        return np.column_stack((vx, vy, np.zeros_like(vx), np.zeros_like(vy), np.zeros_like(life)))
    
    def linear_decay(self): return 1 * self.dt
    def exp_decay(self): return np.exp(-(np.log(2)/self.halflife) * self.dt)

    def boundary_collision(self, x, y, vx, vy):
        vx[(x < 0) | (x > self.bounds[0])] *= -1
        vy[(y < 0) | (y > self.bounds[1])] *= -1
        return vx, vy