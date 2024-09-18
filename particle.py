#!/usr/bin/env python3

import numpy as np

class ParticleField:
    def __init__(self, n_particles, width, height, dt):
        self.n_particles = n_particles
        self.width = width
        self.height = height
        self.dt = dt

        self.pos = np.random.rand(n_particles, 2) * [width, height]
        self.vel = (np.random.rand(n_particles, 2) - 0.5) * 10
        self.life = np.ones(n_particles)
        
        self.state = np.hstack((self.pos, self.vel, self.life[:, None]))

    def dynamics(self, state, t):
        pos, vel, life = state[:, :2], state[:, 2:4], state[:, 4]

        # Boundary conditions
        vel[(pos[:, 0] <= 0) | (pos[:, 0] >= self.width), 0] *= -1
        vel[(pos[:, 1] <= 0) | (pos[:, 1] >= self.height), 1] *= -1
        
        # Derivatives
        dv_dt = vel # vel
        da_dt = np.zeros_like(vel) # accel
        dl_dt = -np.ones_like(life) * self.dt * 20 # linear decay

        return np.hstack((dv_dt, da_dt, dl_dt[:, None]))