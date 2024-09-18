#!/usr/bin/env python3

import numpy as np
import pygame

from display import Display
from engine import ODESolver
from particle import ParticleField

np.set_printoptions(precision=2, suppress=True)

def main():
    WIDTH, HEIGHT = 1024, 576
    render = Display(WIDTH, HEIGHT)
    dt = 1/24

    P = ParticleField(10, WIDTH, HEIGHT, dt)
    solver = ODESolver(f=P.dynamics)
    solver.reset(P.state, t_start=0.0)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        next_state, _ = solver.compute(dt, "euler")  # Simulate
        render.draw(next_state, (255, 255, 255))     # Render
        P.state = next_state                         # Update article field

        print(P.state)

if __name__ == "__main__":
    main()