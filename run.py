#!/usr/bin/env python3

import sys, time
import pygame
import torch

from particle import ParticleField
from engine import ODESolver
from display import Display, terminal_feedback
from libtools import device_mapper, load_profile

torch.manual_seed(123)
torch.set_printoptions(precision=10, sci_mode=False, linewidth=sys.maxsize)

DEVICE = device_mapper()
WIDTH, HEIGHT, FPS = 1024, 576, 120
dt = 1/FPS

def main():
    render = Display(WIDTH, HEIGHT)
    clock = pygame.time.Clock()

    P = ParticleField(64, load_profile('carbon'), WIDTH, HEIGHT, dt, DEVICE)
    solver = ODESolver(f=P.dynamics, device=DEVICE)
    solver.reset(P.state, t_start=0.0)

    i = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            
        t0 = time.time()
        render.draw(P.state)

        t1 = time.time()
        with torch.no_grad():
            P.state[:], _ = solver.compute(dt, 'euler')

        t2 = time.time()
        terminal_feedback(P, i)
        print(f"Render time: {(t1 - t0) * 1e3:.2f}ms")
        print(f"Simulation time: {(t2 - t1) * 1e3:.2f}ms")

        clock.tick(FPS)
        i += 1

        if i > 100: break

if __name__ == "__main__":
    main()