import numpy as np

class Solver:
    def __init__(self, z0, zombieGrowth, zombieLoss, h0, humanGrowth, humanLoss, maxPopulation, minPopulation=0, tolerance=1e-6):
        # Initial conditions and parameters
        self.z0 = z0
        self.h0 = h0
        self.zombieGrowth = zombieGrowth
        self.zombieLoss = zombieLoss
        self.humanGrowth = humanGrowth
        self.humanLoss = humanLoss
        self.maxPop = maxPopulation
        self.minPop = minPopulation
        self.tolerance = tolerance  # Change threshold for stopping the simulation

        # Initial population values
        self.t = 0
        self.z = z0
        self.h = h0
        self.dt = 0.001  # Small time step for the Runge-Kutta method

    def equations(self, z, h):
        """ Defines the system of differential equations """
        dzdt = z * self.zombieGrowth * - h * z * self.zombieLoss
        dhdt = h * self.humanGrowth * - h * z * self.humanLoss
        return dzdt, dhdt

    def runge_kutta_step(self, z, h, dt):
        """ Perform a single Runge-Kutta 4th order step """
        k1z, k1h = self.equations(z, h)
        k2z, k2h = self.equations(z + 0.5 * dt * k1z, h + 0.5 * dt * k1h)
        k3z, k3h = self.equations(z + 0.5 * dt * k2z, h + 0.5 * dt * k2h)
        k4z, k4h = self.equations(z + dt * k3z, h + dt * k3h)

        # Update z and h using the weighted average of the slopes
        new_z = z + (dt / 6) * (k1z + 2 * k2z + 2 * k3z + k4z)
        new_h = h + (dt / 6) * (k1h + 2 * k2h + 2 * k3h + k4h)

        # Enforce population limits
        new_z = min(max(new_z, self.minPop), self.maxPop)
        new_h = min(max(new_h, self.minPop), self.maxPop)

        return new_z, new_h

    def update_time(self, t_new):
        """ Update the solver with a new time t_new """
        delta_t = t_new - self.t  # Calculate the time step

        if abs(delta_t) < self.tolerance:
            return self.z, self.h  # No change if the time step is negligible

        # Adjust time step for larger intervals, to avoid looping too many times
        steps = int(abs(delta_t) / self.dt)  # Number of small steps needed

        # Run the simulation in small steps until reaching t_new
        for _ in range(steps):
            self.z, self.h = self.runge_kutta_step(self.z, self.h, self.dt)
            self.t += self.dt

        return self.z, self.h

    def getZombiePrediction(self, t_new):
        """ Get zombie population prediction for a given new time t_new """
        new_z, _ = self.update_time(t_new)
        return new_z

    def getHumanPrediction(self, t_new):
        """ Get human population prediction for a given new time t_new """
        _, new_h = self.update_time(t_new)
        return new_h
    
