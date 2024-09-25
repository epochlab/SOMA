#!/usr/bin/env python3

class ODESolver:
    def __init__(self, f, device):
        self.f = f
        self.device = device
       
    def reset(self, x_start, t_start=0.0):
        self.x = x_start.to(self.device)
        self.t = t_start

    def compute(self, dt, solver="euler"):
        if solver == "euler":
            self.x += self.f(self.x, self.t) * dt
        elif solver == "mdp":
            x_mp = self.x + self.f(self.x, self.t) * dt/2
            self.x += self.f(x_mp, self.t + dt/2) * dt
        elif solver == "rk2":
            k1 = self.f(self.x, self.t)
            k2 = self.f(self.x + k1 * dt, self.t + dt)
            self.x += 0.5 * (k1 + k2) * dt
        elif solver == "rk4":
            k1 = self.f(self.x, self.t)
            k2 = self.f(self.x + k1 * dt/2, self.t + dt/2)
            k3 = self.f(self.x + k2 * dt/2, self.t + dt/2)
            k4 = self.f(self.x + k3 * dt, self.t + dt)
            self.x += 1/6 * (k1 + 2*k2 + 2*k3 + k4) * dt
        else:
            raise ValueError("Invalid solver method")

        self.t += dt
        return self.x, self.t