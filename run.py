#!/usr/bin/env python3

from dataclasses import dataclass

import pygame
import torch

from libtools import device_mapper, load_profile
from display import Display, terminal_feedback
from engine import ODESolver
from particle import ParticleField

@dataclass
class HyperConfig:
    DEVICE = device_mapper()
    width, height, FPS = 1024, 576, 120
    dt = 1/FPS

def main():
    config = HyperConfig()

    render = Display(config.width, config.height)
    clock = pygame.time.Clock()

    P = ParticleField(10, load_profile('helium'), config.width, config.height, config.dt, config.DEVICE)
    solver = ODESolver(f=P.dynamics, device=config.DEVICE)
    solver.reset(P.state, t_start=0.0)

    i = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        terminal_feedback(P.state, config, i)
        render.draw(P.state[:, :2].cpu().numpy(), (255, 255, 255))

        with torch.no_grad():
            P.state[:], _ = solver.compute(config.dt, "euler")

        clock.tick(config.FPS)
        i += 1

if __name__ == "__main__":
    main()