#!/usr/bin/env python3

import numpy as np

class ODESolver:
    def __init__(self, particles):
        self.particles = particles
        self.ndim = len(particles) * 4

    def state_vector(self):
        return np.concatenate([p.state_vector() for p in self.particles])

    def set_state_vector(self, state):
        for i, p in enumerate(self.particles):
            p.set_state_vector(state[i*4:(i+1)*4])

    def derivatives(self):
        return np.concatenate([p.derivatives() for p in self.particles])

    def euler(self, dt):
        next_state = self.state_vector() + self.derivatives() * dt
        self.set_state_vector(next_state)
        for p in self.particles: p.reset_acceleration()

    def midpoint(self, dt):
        current_state = self.state_vector()
        k1 = self.derivatives()
        x_mp = current_state + k1 * (dt/2)
        self.set_state_vector(x_mp)
        k2 = self.derivatives()
        next_state = current_state + k2 * dt
        self.set_state_vector(next_state)
        for p in self.particles: p.reset_acceleration()
