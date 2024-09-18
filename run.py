#!/usr/bin/env python3

import pygame

from display import Display
from engine import ODESolver
from particle import ParticleField

def main():
    WIDTH, HEIGHT = 1024, 576
    render = Display(WIDTH, HEIGHT)
    dt = 1/24

    P = ParticleField(5, WIDTH, HEIGHT, dt)
    solver = ODESolver(f=P.dynamics)
    solver.reset(P.state, t_start=0.0)

    i = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        next_state, _ = solver.compute(dt, "euler") # Simulate
        render.draw(next_state, (255, 255, 255))
        P.state = next_state

        i += 1
        if i>100: break

if __name__ == "__main__":
    main()