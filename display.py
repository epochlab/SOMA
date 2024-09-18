#!/usr/bin/env python3

import pygame

class Display(object):
    def __init__(self, W, H):
        pygame.init()
        self.surface = pygame.display.set_mode((W, H))
        pygame.display.set_caption("SOMA")

    def draw(self, pts, col):
        self.surface.fill((0, 0, 0))
        for p in pts: 
            x, y = int(p[0]), int(p[1])
            pygame.draw.circle(self.surface, col, (x, y), 2)
        pygame.display.flip()