#!/usr/bin/env python3

import pygame

from display import Display
from engine import ODESolver
from particle import ParticleField

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

        state, _ = solver.compute(dt, "euler") # Simulate
        render.draw(state, (255, 255, 255))
        P.state = state

if __name__ == "__main__":
    main()