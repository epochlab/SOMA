#!/usr/bin/env python3

import pygame

class Display(object):
    def __init__(self, W, H):
        pygame.init()
        self.surface = pygame.display.set_mode((W, H))

    def draw(self, particles, col):
        self.surface.fill((0, 0, 0))
        for p in particles[:,:2]: 
            pygame.draw.circle(self.surface, col, (int(p[0]), int(p[1])), 2) # Render x y positions
        pygame.display.flip()