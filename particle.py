#!/usr/bin/env python3

import torch

class ParticleField:
    def __init__(self, N, width, height, dt, device):
        self.N = N
        self.width = width
        self.height = height
        self.dt = dt

        self.pos = (torch.rand(N, 2) * torch.tensor([width, height])).to(device)
        self.vel = (torch.rand(N, 2) - 0.5).to(device) * 1000
        self.life = torch.ones(N).to(device)
        
        self.state = torch.cat((self.pos, self.vel, self.life[:, None]), dim=1)

    def dynamics(self, state, t):
        pos, vel, life = state[:, :2], state[:, 2:4], state[:, 4]

        vel = self.boundary_collisions(vel, pos)
        life -= self.dt

        return torch.cat((vel, torch.zeros_like(vel), life[:, None]), dim=1) # dx_dt | vel, accel, life
        
    def boundary_collisions(self, vel, pos):
        mask_x = (pos[:, 0] <= 0) | (pos[:, 0] >= self.width)
        mask_y = (pos[:, 1] <= 0) | (pos[:, 1] >= self.height)
        vel[mask_x, 0] *= -1
        vel[mask_y, 1] *= -1
        pos[mask_x, 0] = torch.clamp(pos[mask_x, 0], 0, self.width)
        pos[mask_y, 1] = torch.clamp(pos[mask_y, 1], 0, self.height)
        return vel