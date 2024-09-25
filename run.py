#!/usr/bin/env python3

import sys
import pygame
import torch

from libtools import device_mapper, load_profile
from display import Display, terminal_feedback
from engine import ODESolver
from particle import ParticleField

torch.manual_seed(123)
torch.set_printoptions(linewidth=sys.maxsize)

DEVICE = device_mapper()
WIDTH, HEIGHT, FPS = 1024, 576, 120
dt = 1/FPS

def main():
    render = Display(WIDTH, HEIGHT)
    clock = pygame.time.Clock()

    P = ParticleField(10, load_profile('helium'), WIDTH, HEIGHT, dt, DEVICE)
    solver = ODESolver(f=P.dynamics, device=DEVICE)
    solver.reset(P.state, t_start=0.0)

    i = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        terminal_feedback(P, i)
        render.draw(P.state[:, :2].cpu().numpy(), (255, 255, 255))

        with torch.no_grad():
            P.state[:], _ = solver.compute(dt, "euler")

        clock.tick(FPS)
        i += 1

if __name__ == "__main__":
    main()