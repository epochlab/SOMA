#!/usr/bin/env python3

import pygame

from display import Display
from particle import initialise
from engine import ODESolver

def main():
    WIDTH, HEIGHT = 1024, 576
    render = Display(WIDTH, HEIGHT)

    particles = initialise(20, WIDTH, HEIGHT)
    solver = ODESolver(particles)

    dt = 1/24
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        for p in particles:
            p.halflife(dt)
            if p.life <= 0: particles.remove(p)

            p.interact(particles, r=100, strength=1)
            p.update(dt, drag_coefficient=0.001)
            p.boundary_collision([WIDTH, HEIGHT])

        if len(particles) == 0: break

        solver.euler(dt)
        render.draw(particles, (255, 255, 255))

if __name__ == "__main__":
    main()