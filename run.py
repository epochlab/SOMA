#!/usr/bin/env python3

from dataclasses import dataclass

import pygame
import torch

from libtools import device_mapper
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

    P = ParticleField(10, config.width, config.height, config.dt, config.DEVICE)
    solver = ODESolver(f=P.dynamics, device=config.DEVICE)
    solver.reset(P.state, t_start=0.0)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        with torch.no_grad():
            next_state, _ = solver.compute(config.dt, "euler")
        
        P.state[:] = next_state

        terminal_feedback(P.state, config)
        render.draw(P.state[:, :2].cpu().numpy(), (255, 255, 255))
        clock.tick(config.FPS)

if __name__ == "__main__":
    main()