#!/usr/bin/env python3

import pygame

from display import Display
from particle import initialise
from engine import ODESolver
import forces

def main():
    WIDTH, HEIGHT = 1024, 576
    render = Display(WIDTH, HEIGHT)

    particles = initialise(20, WIDTH, HEIGHT)
    solver = ODESolver(particles)

    dt = 1/24 # Time-step
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        # Cull dead particles
        particles = [p for p in particles if p.life >= 0]
        if not particles: break

        for p in particles:
            p.halflife(dt)
            forces.interact(particles, r=20, strength=-1) # Strength | Pos (Attract), Neg (Repel)
            p.update(dt, drag_coeff=0.001)
            p.boundary_collision([WIDTH, HEIGHT])

        solver.euler(dt)
        render.draw([p for p in particles if p.active], (255, 255, 255))

if __name__ == "__main__":
    main()