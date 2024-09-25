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
        vel = self.boundary_collisions()
        dist = self.distance()
        return torch.cat((vel, torch.zeros_like(vel)), dim=1) # dx_dt | p:vel, v:accel
        
    def boundary_collisions(self):
        pos, vel = self.state[:, :2], self.state[:, 2:]
        mask_x = (pos[:, 0] <= 0) | (pos[:, 0] >= self.width)
        mask_y = (pos[:, 1] <= 0) | (pos[:, 1] >= self.height)
        vel[mask_x, 0] *= -1
        vel[mask_y, 1] *= -1
        return vel
    
    def distance(self):
        pnts = self.state[:, :2]
        diff = pnts[:, None, :] - pnts[None, :, :]
        return torch.sqrt((diff ** 2).sum(dim=-1))