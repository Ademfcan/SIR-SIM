import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def __init__(self, populationSize, z0, infectionGrowth, zombieLoss, humanLoss, block_size=2):
        self.h0 = populationSize-z0  # Initial human population
        self.z0 = z0  # Initial zombie population
        self.infectionGrowth = infectionGrowth    # Infection rate
        self.humanLoss = humanLoss    # Additional loss term for humans
        self.zombieLoss = zombieLoss    # Recovery rate
        self.block_size = block_size  # Time block size
        self.total_population = populationSize  # Assume constant total

        self.blocks = {}  # Cache for computed time blocks
        self.current_max_t = -1  # Track how far we've computed

    def _model(self, t, y):
        """Differential equations for H and Z."""
        H, Z = y
        dHdt = -self.infectionGrowth * H * Z - self.humanLoss * Z
        dZdt = self.infectionGrowth * H * Z - self.zombieLoss * H
        return [dHdt, dZdt]

    def _solve_block(self, start_t):
        """Solve the system from start_t to start_t + block_size and store it."""
        previous_end_values = [self.h0, self.z0]

        # If there is a previous block, use its last values as initial conditions
        if self.blocks:
            last_block_start = max(self.blocks.keys())
            previous_end_values = self.blocks[last_block_start].y[:, -1]

        t_span = (start_t, start_t + self.block_size)
        t_eval = np.linspace(*t_span, num=2*self.block_size)  # 100 points in the block
        sol = solve_ivp(self._model, t_span, previous_end_values, t_eval=t_eval, method="RK45")
        
        # Store the block in the cache
        self.blocks[start_t] = sol
        self.current_max_t = max(self.current_max_t, start_t + self.block_size)

    def _get_nearest_value(self, sol, t):
        """Find the nearest computed value in a solved block."""
        idx = np.abs(sol.t - t).argmin()
        return sol.y[:, idx]

    def getHumanPopulation(self, t):
        """Return H(t), computing new blocks if needed."""
        if t > self.current_max_t:
            while t > self.current_max_t:
                self._solve_block(self.current_max_t)

        for start_t, sol in self.blocks.items():
            if start_t <= t <= start_t + self.block_size:
                return self._get_nearest_value(sol, t)[0]  # H value
        print(f"Could not get block!: {self.blocks} {self.current_max_t} {t}")
        return None

    def getZombiePopulation(self, t):
        """Return Z(t), computing new blocks if needed."""
        if t > self.current_max_t:
            while t > self.current_max_t:
                self._solve_block(self.current_max_t)

        for start_t, sol in self.blocks.items():
            if start_t <= t <= start_t + self.block_size:
                return self._get_nearest_value(sol, t)[1]  # Z value
        print(f"Could not get block!: {self.blocks} {self.current_max_t} {t}")
        return None

    def getRecoveredPopulation(self, t):
        """Return recovered population R(t)."""
        return self.total_population - self.getHumanPopulation(t) - self.getZombiePopulation(t)

if __name__ == "__main__":
    solver = Solver(1000, 1, 0.3, 0.2, 0)
    print("Solving ivp....")
    h = solver.getHumanPopulation(1000)
    print("Solved ivps....")
    z = solver.getZombiePopulation(1000)
    r = solver.getRecoveredPopulation(1000)

    print(f"{h=} {z=} {r=}")