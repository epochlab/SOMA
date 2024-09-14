#!/usr/bin/env python3

import random
import itertools
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

    def boundary_collision(self, bounds):
        if self.x < 0 or self.x > bounds[0]: self.state[2] *= -1
        if self.y < 0 or self.y > bounds[1]: self.state[3] *= -1

    def measure(self, particles, r):
        pos = np.array([[p.x, p.y] for p in particles])
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        i, j = np.triu_indices(len(particles), k=1)
        valid = dist[i, j] < r
        return [(particles[i[k]], particles[j[k]]) for k in np.where(valid)[0]]

def initialise(N, w, h):
    return [Particle(random.uniform(0, w), random.uniform(0, h), random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(N)]