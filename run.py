#!/usr/bin/env python3

import pygame
import torch

from display import Display
from engine import ODESolver
from particle import ParticleField
from libtools import device_mapper

torch.set_printoptions(precision=4, sci_mode=False) 

DEVICE = device_mapper()
print(f"Device: {str(DEVICE).upper()}")

def main():
    WIDTH, HEIGHT, FPS = 1024, 576, 120
    render = Display(WIDTH, HEIGHT)
    clock = pygame.time.Clock()
    dt = 1/FPS

    P = ParticleField(100, WIDTH, HEIGHT, dt, DEVICE)
    solver = ODESolver(f=P.dynamics, device=DEVICE)
    solver.reset(P.state, t_start=0.0)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        with torch.no_grad():
            next_state, _ = solver.compute(dt, "euler")
            
        render.draw(next_state[:, :2].cpu().numpy(), (255, 255, 255))
        P.state[:] = next_state
        print(P.state.cpu())

        clock.tick(FPS)

if __name__ == "__main__":
    main()