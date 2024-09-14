#!/usr/bin/env python3

import numpy as np

import pygame
from display import Display
from particle import initialise
from engine import ODESolver

def main():
    WIDTH, HEIGHT = 1024, 576
    render = Display(WIDTH, HEIGHT)

    particles = initialise(50, WIDTH, HEIGHT)
    solver = ODESolver(particles)

    dt = 1/24
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        for p in particles:
            p.boundary_collision([WIDTH, HEIGHT])
            neighbours = p.measure(particles, 10)

        solver.euler(dt)
        render.draw(particles, (255, 255, 255))

if __name__ == "__main__":
    main()