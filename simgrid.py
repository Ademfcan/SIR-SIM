import numpy as np
import math
import simNjits

class SimGrid:
    def __init__(self, z0, zombieGrowth, zombieLoss, h0, humanGrowth, humanLoss, gridCellCount=1000, MAX=1e7, moveProb=0.9):
        self.MAX = MAX
        self.moveProb = moveProb
        
        self.z0 = z0  # initial amount of zombies
        self.zombieGrowth = zombieGrowth  # growth percentage per day
        self.zombieLoss = zombieLoss  # death percentage per day
        self.zombieDir = -1
        
        self.h0 = h0  # initial amount of humans
        self.humanGrowth = humanGrowth  # growth percentage per day
        self.humanLoss = humanLoss  # death percentage per day
        self.humanDir = 1

        self.squareSize = math.ceil(math.sqrt(gridCellCount))
        self.gridCellCount = self.squareSize * self.squareSize  # nearest "resolution of grid"
        self.MAXCELL = self.MAX / self.gridCellCount

        self.grid = np.zeros(shape=(self.squareSize, self.squareSize), dtype=np.float64)

        # Initialize with initial conditions
        self._initialize_grid(z0, h0)

        self.timePassed = 0

    def setHumanGrowth(self, growth):
        self.humanGrowth = growth

    def setHumanLoss(self, loss):
        self.getHumanCount = loss

    def setZombieGrowth(self, growth):
        self.zombieGrowth = growth

    def setZombieLoss(self, loss):
        self.zombieLoss = loss

    def _initialize_grid(self, z0, h0):
        self._initialize_population(z0, self.zombieDir)
        self._initialize_population(h0, self.humanDir)

    def _initialize_population(self, population_count, direction):
        # Randomly assign population to grid cells
        indices = np.random.randint(0, self.gridCellCount, population_count)
        rows, cols = np.unravel_index(indices, (self.squareSize, self.squareSize))
        for row, col in zip(rows, cols):
            self.grid[row, col] += direction
    
    def propagate(self, timeStep=1):
        """ Given a timestep goes over every cell, and applies the growth and loss equations for either humans or zombies """
        self.timePassed += timeStep
        self.grid = simNjits.propagate(self.grid, timeStep, self.zombieGrowth, self.zombieLoss, self.zombieDir,
                                        self.humanGrowth, self.humanLoss, self.humanDir, self.MAXCELL, self.moveProb)

    
    # Population counts and utility methods
    def getZombiePopulation(self):
        return self.__getPopulation(self.zombieDir)

    def getHumanPopulation(self):
        return self.__getPopulation(self.humanDir)

    def getEmptyCount(self):
        return self.__getFilled(0)

    def getHumanCount(self):
        return self.__getFilled(self.humanDir)

    def getZombieCount(self):
        return self.__getFilled(self.zombieDir)

    def __getFilled(self, countDir):
        if countDir < 0:
            return np.sum(self.grid < 0)
        if countDir > 0:
            return np.sum(self.grid > 0)
        return np.sum(self.grid == 0)

    def __getPopulation(self, countDir):
        if countDir < 0:
            return abs(np.sum(self.grid[self.grid < 0]))
        else:
            return abs(np.sum(self.grid[self.grid > 0]))

    def isApocalypse(self):
        return self.getHumanPopulation() == 0 or self.getZombiePopulation() == 0

if __name__ == "__main__":
    grid = SimGrid(1000, 0.5, 0.1, 1000, 0.2, 0, 1000)
    print(f"Humans: {grid.getHumanPopulation()} Zombies: {grid.getZombiePopulation()} Empty: {grid.getEmptyCount()}")
    grid.propagate(1)
    print(f"Humans: {grid.getHumanPopulation()} Zombies: {grid.getZombiePopulation()} Empty: {grid.getEmptyCount()}")
    print(grid.grid.size)
