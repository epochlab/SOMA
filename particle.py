#!/usr/bin/env python3

import random
import numpy as np

class Particle:
    def __init__(self, x, y, vx, vy, ax=0, ay=0):
        self.state = np.array([x, y, vx, vy, ax, ay], dtype=float)

    @property
    def x(self): return self.state[0]

    @property
    def y(self): return self.state[1]

    def state_vector(self): return self.state[:4]

    def set_state_vector(self, state): self.state[:4] = state
    
    def derivatives(self): return self.state[2:]

    def update_collision(self, bounds):
        if self.x < 0 or self.x > bounds[0]: self.state[2] *= -1
        if self.y < 0 or self.y > bounds[1]: self.state[3] *= -1

    def dist(self, target):
        relative_pos = [self.x, self.y] - target
        return np.sum(relative_pos**2)

def initialise(N, w, h):
    return [Particle(random.uniform(0, w), random.uniform(0, h), random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(N)]