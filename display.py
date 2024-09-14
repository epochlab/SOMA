#!/usr/bin/env python3

import pygame

class Display(object):
    def __init__(self, W, H):
        pygame.init()
        self.surface = pygame.display.set_mode((W, H))

    def draw(self, particles, col):
        self.surface.fill((0, 0, 0))
        for p in particles:
            pygame.draw.circle(self.surface, col, (int(p.x), int(p.y)), 2)
        pygame.display.flip()