#!/usr/bin/env python3

import torch

class ParticleField:
    def __init__(self, N, profile, width, height, dt, device):
        self.N = N
        self.profile = profile

        self.width = width
        self.height = height
        self.dt = dt
        self.device = device

        self.state = self._initialize()

    def _initialize(self):
        pI = (torch.rand(self.N, 2) * torch.tensor([self.width, self.height])).to(self.device)
        vI = (torch.rand(self.N, 2) - 0.5).to(self.device) * 1000
        return torch.cat((pI, vI), dim=1)

    def dynamics(self, state, t):
        vel = self.state[:, 2:]
        vel = self.boundary_collisions()
        vel -= self.interact(2 * self.profile['atomic_radius'])  # -:attract | +:repel
        return torch.cat((vel, torch.zeros_like(vel)), dim=1) # dx_dt | p:vel, v:accel
    
    def boundary_collisions(self):
        pos, vel = self.state[:, :2], self.state[:, 2:]
        mask_x_low = pos[:, 0] <= 0
        mask_x_high = pos[:, 0] >= self.width
        mask_y_low = pos[:, 1] <= 0
        mask_y_high = pos[:, 1] >= self.height
        vel[:, 0] = torch.where(mask_x_low | mask_x_high, -vel[:, 0], vel[:, 0])
        vel[:, 1] = torch.where(mask_y_low | mask_y_high, -vel[:, 1], vel[:, 1])
        return vel

    def interact(self, r):
        pos = self.state[:, :2]
        dir = pos[:, None] - pos[None, :]
        dist = self._distance()
        dist[dist == 0] = float('inf')
        mag = self._attenuation(dist, r)
        norm_dir = dir / dist[..., None]
        force = (mag[..., None] * norm_dir) * self.profile['atomic_weight'] # F = ma
        return force.sum(dim=1)

    def _attenuation(self, dist, r):
        return torch.clamp(1 - (dist + 1e-06) / r, min=0) # f(d) = max(0, 1 - d/r)

    def _distance(self):
        pos = self.state[:, :2]
        diff = pos[:, None] - pos[None, :]
        return torch.sqrt((diff ** 2).sum(dim=-1))