import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def __init__(self, populationSize, z0, infectionGrowth, zombieLoss, humanLoss, block_size=2, t_scalar=5):
        self.h0 = populationSize - z0  # Initial human population
        self.z0 = z0  # Initial zombie population
        self.infectionGrowth = infectionGrowth  # Infection rate
        self.humanLoss = humanLoss  # Additional loss term for humans
        self.zombieLoss = zombieLoss  # Recovery rate (should be applied to Z, not H)
        self.block_size = block_size  # Time block size
        self.total_population = populationSize  # Assume constant total
        self.interactionScale = 1 / self.total_population
        self.t_scalar = t_scalar

        self.blocks = {}  # Cache for computed time blocks
        self.last_block_start = None  # Track the start time of the last computed block

    def _model(self, t, y):
        """Differential equations for H and Z."""
        H, Z = y
        H, Z = y
        scaled_interaction = self.interactionScale * H * Z
        dHdt = np.clip(-self.infectionGrowth * scaled_interaction - self.humanLoss * Z * self.interactionScale,-H,Z)
        dZdt = np.clip(self.infectionGrowth * scaled_interaction - self.zombieLoss * H * self.interactionScale,-Z,H)

        return [dHdt, dZdt]

    def _solve_block(self, start_t):
        """Solve the system from start_t to start_t + block_size and store it."""
        if start_t in self.blocks:
            return  # Block already solved

        # Use correct initial values
        if start_t == 0:
            initial_values = [self.h0, self.z0]
        else:
            previous_block = start_t - self.block_size
            if previous_block in self.blocks:
                initial_values = self.blocks[previous_block].y[:, -1]
            else:
                print(f"Warning: Missing previous block at {previous_block}, using initial values.")
                initial_values = [self.h0, self.z0]  # Fallback

        t_span = (start_t, start_t + self.block_size)
        t_eval = np.linspace(*t_span, num=20)

        sol = solve_ivp(self._model, t_span, initial_values, t_eval=t_eval, method="RK45")
        self.blocks[start_t] = sol
        self.last_block_start = start_t

    def _get_nearest_value(self, sol, t):
        """Find the nearest computed value in a solved block."""
        idx = np.abs(sol.t - t).argmin()
        return sol.y[:, idx]

    def _ensure_block(self, t):
        """Ensure that the block containing time t is solved."""
        # Align t to the block start time
        blocked_t = (t // self.block_size) * self.block_size
        # If no blocks computed yet, start with block 0
        if self.last_block_start is None:
            self._solve_block(0)
        # Compute blocks sequentially until we have the block that covers t
        while blocked_t not in self.blocks:
            # Determine the start time for the next block
            if self.last_block_start is None:
                next_start = 0
            else:
                next_start = self.last_block_start + self.block_size
            self._solve_block(next_start)
        return blocked_t

    def getHumanPopulation(self, t):
        """Return H(t), computing new blocks if needed."""
        t = t/self.t_scalar
        blocked_t = self._ensure_block(t)
        return self._get_nearest_value(self.blocks[blocked_t], t)[0]
    

    def getZombiePopulation(self, t):
        """Return Z(t), computing new blocks if needed."""
        t = t/self.t_scalar
        blocked_t = self._ensure_block(t)
        return self._get_nearest_value(self.blocks[blocked_t], t)[1]

    def getRecoveredPopulation(self, t):
        """Return recovered population R(t)."""
        h = self.getHumanPopulation(t)
        z = self.getZombiePopulation(t)
        if h is not None and z is not None:
            return self.total_population - h - z
        return None

    def isApocalypse(self, t, atol=1e-3):
        """Check if humans or zombies are extinct at time t."""
        h = self.getHumanPopulation(t)
        z = self.getZombiePopulation(t)
        if h is None or z is None:
            return False
        return np.isclose(h, 0, atol=atol) or np.isclose(z, 0, atol=atol)

if __name__ == "__main__":
    solver = Solver(1000, 1, 0.3, 0, 0)
    print("Solving IVP...")
    print("Solved IVPs...")

    for i in range(0, 1):
        h = solver.getHumanPopulation(i)
        z = solver.getZombiePopulation(i)
        r = solver.getRecoveredPopulation(i)

        print(f"{h=:.2f} {z=:.2f} {r=:.2f}")
