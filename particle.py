#!/usr/bin/env python3

import random
import numpy as np

class Particle:
    _next_idx = 0

    def __init__(self, x, y, vx, vy, ax=0, ay=0):
        self.idx = Particle._next_idx
        Particle._next_idx += 1

        self.mass = self.set_value(0.5, 1)
        self.life = self.set_value(0, 1000)
        self.state = np.array([x, y, vx, vy, ax, ay], dtype=float)

    @property
    def x(self): return self.state[0]

    @property
    def y(self): return self.state[1]

    def set_value(self, min=0, max=1): return random.uniform(min, max)

    def state_vector(self): return self.state[:4]

    def set_state_vector(self, state): self.state[:4] = state

    def derivatives(self): return self.state[2:]

    def halflife(self, dt): self.life -= dt

    def boundary_collision(self, bounds):
        if not (0 <= self.x <= bounds[0]): self.state[2] *= -1
        if not (0 <= self.y <= bounds[1]): self.state[3] *= -1

    def apply_force(self, fx, fy):
        self.state[4:] += np.array([fx, fy]) / self.mass

    def apply_drag(self, drag_coefficient):
        self.state[4] -= drag_coefficient * self.state[2]  # ax (drag in x)
        self.state[5] -= drag_coefficient * self.state[3]  # ay (drag in y)

    def update(self, dt, drag_coefficient=0.01):
        self.apply_drag(drag_coefficient)
        self.state[2] += self.state[4] * dt  # vx
        self.state[3] += self.state[5] * dt  # vy
        self.state[0] += self.state[2] * dt  # pos x
        self.state[1] += self.state[3] * dt  # pos y
        self.state[4:] = 0 # Reset acceleration

    def interact(self, particles, r, k=1, strength=1):
        pos = np.array([p.state[:2] for p in particles])
        if pos.ndim == 1: pos = np.expand_dims(pos, axis=0)

        dist_matrix = np.sqrt(np.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=-1))
        np.fill_diagonal(dist_matrix, r + 1) # Fix zero divide
        indices = np.column_stack(np.where(dist_matrix < r))
        distances = dist_matrix[indices[:, 0], indices[:, 1]]
        forces = k / distances ** 2

        for (i, j), force, dist in zip(indices, forces, distances):
            if dist > 0:
                direction = (strength * (pos[i] - pos[j])) / dist_matrix[i, j]
                particles[i].apply_force(force * direction[0] * particles[j].mass, force * direction[1] * particles[j].mass)
                particles[j].apply_force(-force * direction[0] * particles[i].mass, -force * direction[1] * particles[i].mass)

def initialise(N, w, h):
    return [Particle(random.uniform(0, w), random.uniform(0, h), random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(N)]