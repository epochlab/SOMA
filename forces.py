#!/usr/bin/env python3

import numpy as np

def attenuation(dist, r, falloff='sqr', k=1, alpha=1, eps=1):
    if falloff == 'linear': return (r - (dist + eps)) / r
    if falloff == 'exp': return np.exp(-alpha * (dist + eps) / r)
    if falloff == 'sqr': return k / ((dist**2 + eps) / r**2)

def neighbour(particles, r):
    pos = np.array([p.state[:2] for p in particles if p.active])
    if pos.size == 0: return None, None, None
    dist_matrix = np.sqrt(np.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(dist_matrix, r + 1) # Fix zero divide
    indices = np.column_stack(np.where(dist_matrix < r))
    distances = dist_matrix[indices[:, 0], indices[:, 1]]
    return pos, indices, distances

def interact(particles, r, strength=1, falloff='sqr'):
    pos, indices, distances = neighbour(particles, r)
    if indices is None: return
    forces = np.clip(attenuation(distances, r, falloff), 0, 1)
    for (i, j), force, dist in zip(indices, forces, distances):
        if dist > 0:
            direction = (pos[j] - pos[i]) / dist
            magnitude = force * strength
            particles[i].apply_force(magnitude * direction[0] * particles[j].mass, magnitude * direction[1] * particles[j].mass)
            particles[j].apply_force(-magnitude * direction[0] * particles[i].mass, -magnitude * direction[1] * particles[i].mass)