#!/usr/bin/env python3

import torch

class ParticleField:
    def __init__(self, N, profile, width, height, dt, device):
        self.N = N
        self.width = width
        self.height = height
        self.dt = dt
        self.device = device

        self.pos = (torch.rand(N, 2) * torch.tensor([width, height])).to(device)
        self.vel = (torch.rand(N, 2) - 0.5).to(device) * 1000

        self.name = profile['name']
        self.symbol = profile['symbol']
        self.Z = profile['atomic_number']
        self.u = profile['atomic_weight']
        self.p = profile['protons']
        self.e = profile['electrons']
        self.A = profile['density']
        self.X = profile['electronegativity']
        self.a0 = profile['atomic_radius']
        
        self.state = torch.cat((self.pos, self.vel), dim=1)

    def dynamics(self, state, t):
        pos, vel = state[:, :2], state[:, 2:4]
        vel = self.boundary_collisions(vel, pos)
        return torch.cat((vel, torch.zeros_like(vel)), dim=1) # dx_dt | p:vel, v:accel
        
    def boundary_collisions(self, vel, pos):
        mask_x = (pos[:, 0] <= 0) | (pos[:, 0] >= self.width)
        mask_y = (pos[:, 1] <= 0) | (pos[:, 1] >= self.height)
        vel[mask_x, 0] *= -1
        vel[mask_y, 1] *= -1
        return vel