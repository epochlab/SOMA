#!/usr/bin/env python3

import os
import pygame
import torch
from matplotlib import cm

class Display(object):
    def __init__(self, W, H):
        pygame.init()
        self.width = W
        self.height = H
        self.surface = pygame.display.set_mode((W, H))
        pygame.display.set_caption("SOMA")

    def draw(self, state):
        self.surface.fill((0, 0, 0))

        vel = 1- torch.norm(state[:,2:], dim=1) / 1e3 # Fix static mapping!!
        col = attrib_Cd(vel, 'Blues')

        pts = state[:,:2].cpu().numpy()
        for i, p in enumerate(pts):
            x, y = int(p[0]), int(p[1])
            # pygame.draw.circle(self.surface, (255,255,255), (x, y), 2)
            pygame.draw.circle(self.surface, tuple(col[i]), (x, y), 2)
        pygame.display.flip()

def terminal_feedback(pf, i, diag):
    N = pf.state.shape[0]
    pos = pf.state[:, :2].cpu().numpy()
    vel = pf.state[:, 2:4].cpu().numpy()

    os.system('cls' if os.name == 'nt' else 'clear')

    print("SOMA | Particle Sim")
    print("-" * 20)
    print(f"Device: {str(pf.device).upper()}")
    print(f"Resolution: {pf.width} x {pf.height}")
    print(f"FPS: {int(1/pf.dt)}")
    print(f"Delta (dt): {pf.dt:.4f}")
    print(f"Element: {(pf.profile['name']).capitalize()}")
    print(f"N Particles: {pf.N}\n")

    print(f"Frame: {i}\n")
    print(f"Render time: {(diag[1] - diag[0]) * 1e3:.2f}ms")
    print(f"Simulation time: {(diag[2] - diag[1]) * 1e3:.2f}ms\n")

    print(f"{'ID':<6} | {'Pos (x,y)':<20} | {'Vel (vx,vy)':<20}")
    print("-" * 60)

    for i in range(len(pf.state)):
        print(f"{i:<6} | ({pos[i][0]:<8.3f}, {pos[i][1]:<8.3f}) | "
              f"({vel[i][0]:<8.3f}, {vel[i][1]:<8.3f})")
        
def attrib_Cd(attrib, col):
    cmap = cm.get_cmap(col)
    colors = cmap(attrib.cpu().numpy())[:, :3]
    return (colors * 255).astype('uint8')