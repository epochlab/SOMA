#!/usr/bin/env python3

import numpy as np

class ParticleField:
    def __init__(self, N, width, height, dt):
        self.N = N
        self.width = width
        self.height = height
        self.dt = dt

        self.pos = np.random.rand(N, 2) * [width, height]
        self.vel = (np.random.rand(N, 2) - 0.5) * 1000
        self.life = np.ones(N)
        
        self.state = np.hstack((self.pos, self.vel, self.life[:, None]))

    def dynamics(self, state, t):
        pos, vel, life = state[:, :2], state[:, 2:4], state[:, 4]

        vel = self.boundary_collisions(vel, pos)
        life = self.linear_decay(life)

        # Derivatives
        dv_dt = vel # vel
        da_dt = np.zeros_like(vel) # accel
        dl_dt = life

        return np.hstack((dv_dt, da_dt, dl_dt[:, None]))
    
    def linear_decay(self, life): return -np.ones_like(life) * self.dt
        
    def boundary_collisions(self, vel, pos):
        vel[(pos[:, 0] <= 0) | (pos[:, 0] >= self.width), 0] *= -1
        vel[(pos[:, 1] <= 0) | (pos[:, 1] >= self.height), 1] *= -1
        return vel