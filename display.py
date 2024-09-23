#!/usr/bin/env python3

import os
import pygame

class Display(object):
    def __init__(self, W, H):
        pygame.init()
        self.width = W
        self.height = H
        self.surface = pygame.display.set_mode((W, H))
        pygame.display.set_caption("SOMA")

    def draw(self, pts, col):
        self.surface.fill((0, 0, 0))
        for p in pts[:, :2]:
            x, y = int(p[0]), int(p[1])
            pygame.draw.circle(self.surface, col, (x, y), 2)
        pygame.display.flip()

def terminal_feedback(state, config, i):
    N = state.shape[0]
    pos = state[:, :2].cpu().numpy()
    vel = state[:, 2:4].cpu().numpy()

    os.system('cls' if os.name == 'nt' else 'clear')

    print("SOMA | Particle Sim")
    print("-" * 20)
    print(f"Device: {str(config.DEVICE).upper()}")
    print(f"Resolution: {config.width} x {config.height}")
    print(f"FPS: {int(1/config.dt)}")
    print(f"Delta (dt): {config.dt:.4f}")
    print(f"N Particles: {N}")
    print(f"Frame: {i}\n")

    print(f"{'ID':<6} | {'Pos (x,y)':<20} | {'Vel (vx,vy)':<20}")
    print("-" * 60)

    for i in range(N):
        print(f"{i:<6} | ({pos[i][0]:<8.3f}, {pos[i][1]:<8.3f}) | "
                f"({vel[i][0]:<8.3f}, {vel[i][1]:<8.3f})")