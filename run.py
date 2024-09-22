#!/usr/bin/env python3

import numpy as np
import pygame

from display import Display
from engine import ODESolver
from particle import ParticleField

np.set_printoptions(precision=2, suppress=True)

def main():
    WIDTH, HEIGHT, FPS = 1024, 576, 120
    render = Display(WIDTH, HEIGHT)
    clock = pygame.time.Clock()
    dt = 1/FPS

    P = ParticleField(10, WIDTH, HEIGHT, dt)
    solver = ODESolver(f=P.dynamics)
    solver.reset(P.state, t_start=0.0)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        next_state, _ = solver.compute(dt, "euler")
        render.draw(next_state[:, :2], (255, 255, 255))
        P.state[:] = next_state
        print(P.state)

        clock.tick(FPS)

if __name__ == "__main__":
    main()