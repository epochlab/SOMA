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
        for p in pts: 
            x, y = int(p[0]), int(p[1])
            pygame.draw.circle(self.surface, col, (x, y), 2)
        pygame.display.flip()

    def terminal_feedback(self, state):
        N = state.shape[0]
        pos = state[:, :2].cpu().numpy()
        vel = state[:, 2:4].cpu().numpy()
        life = state[:, 4].cpu().numpy()

        os.system('cls' if os.name == 'nt' else 'clear')

        print("SOMA | Particle Sim\n")
        print(f"Resolution: {self.width} x {self.height}")
        print(f"N Particles: {N}\n")

        print(f"{'ID':<6} | {'Pos (x,y)':<20} | {'Vel (vx,vy)':<20} | {'Life':<10}")
        print("-" * 70)

        for i in range(N):
            print(f"{i:<6} | ({pos[i][0]:<8.3f}, {pos[i][1]:<8.3f}) | "
                f"({vel[i][0]:<8.3f}, {vel[i][1]:<8.3f}) | "
                f"{life[i]:.4f}")