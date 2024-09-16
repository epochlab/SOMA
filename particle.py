#!/usr/bin/env python3

import random
import numpy as np

class Particle:
    _next_idx = 0

    def __init__(self, x, y, vx, vy, ax=0, ay=0):
        self.idx = Particle._next_idx
        Particle._next_idx += 1

        self.active = True
        self.state = np.array([x, y, vx, vy, ax, ay], dtype=float)
        self.mass = self.set_value(0.8, 1)
        self.life = self.set_value(0, 1000)

    @classmethod
    def set_value(self, min=0, max=1): return random.uniform(min, max)

    @property
    def x(self): return self.state[0]

    @property
    def y(self): return self.state[1]

    def state_vector(self): return self.state[:4]
    def set_state_vector(self, state): self.state[:4] = state
    def derivatives(self): return self.state[2:]

    def halflife(self, dt): 
        self.life -= dt
        if self.life <= 0: self.active = False

    def boundary_collision(self, bounds):
        if not (0 <= self.x <= bounds[0]): self.state[2] *= -1
        if not (0 <= self.y <= bounds[1]): self.state[3] *= -1

    def apply_force(self, fx, fy):
        self.state[4:] += np.array([fx, fy]) / self.mass # a = F/m

    def apply_drag(self, coeff):
        self.state[4] -= coeff * self.state[2]  # ax (drag in x)
        self.state[5] -= coeff * self.state[3]  # ay (drag in y)

    def reset_acceleration(self): self.state[4:] = 0

def initialise(N, w, h):
    return [Particle(random.uniform(0, w), random.uniform(0, h), random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(N)]