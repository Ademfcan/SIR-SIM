import numpy as np
import math
import simNjits

class SimGrid:
    @staticmethod
    def getNearestSquareCellCount(gridCellCount : int):
        squareSize = math.ceil(math.sqrt(gridCellCount))
        gridCellCount = squareSize * squareSize  # nearest "resolution of grid"
        return gridCellCount, squareSize

    def __init__(self, populationSize, z0, infectionGrowth, zombieLoss, humanLoss, gridCellCount=1000, moveProb=0.6):
        self.popSize = populationSize
        self.moveProb = moveProb

        self.infectionGrowth = infectionGrowth  # growth percentage per day

        self.z0 = z0  # initial amount of zombies
        self.zombieLoss = zombieLoss  # death percentage per day
        self.zombieDir = -1
        
        self.h0 = populationSize-z0  # initial amount of humans
        self.humanLoss = humanLoss
        self.humanDir = 1

        self.totalRecovered = 0

        self.gridCellCount, self.squareSize = SimGrid.getNearestSquareCellCount(gridCellCount)

        

        self.grid = np.zeros(shape=(self.squareSize, self.squareSize), dtype=np.float64)

        # Initialize with initial conditions
        self._initialize_grid(self.z0, self.h0)

        self.timePassed = 0

    def setinfectionGrowth(self, growth):
        self.infectionGrowth = growth

    def setZombieLoss(self, loss):
        self.zombieLoss = loss
    
    def setHumanLoss(self, loss):
        self.humanLoss = loss

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
        self.grid, recovered = simNjits.propagate(self.grid, timeStep, self.infectionGrowth, self.zombieLoss, self.humanLoss,
                                       self.zombieDir, self.humanDir, self.moveProb, self.popSize)
        self.totalRecovered += recovered
    
    # Population counts and utility methods
    def getZombiePopulation(self):
        return self.__getPopulation(self.zombieDir)

    def getHumanPopulation(self):
        return self.__getPopulation(self.humanDir)

    def getRecoveredPopulation(self):
        return min(self.totalRecovered,self.popSize-self.getHumanPopulation()-self.getZombiePopulation())

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
        return False
        # return np.isclose(self.getHumanPopulation(),0,atol=1e-5) or np.isclose(self.getZombiePopulation(),0,atol=1e-5)

if __name__ == "__main__":
    grid = SimGrid(1000, 10,0.1, 0.5, 0.1)
    print(f"Humans: {grid.getHumanPopulation()} Zombies: {grid.getZombiePopulation()} Empty: {grid.getRecoveredPopulation()}")
    grid.propagate(1)
    print(f"Humans: {grid.getHumanPopulation()} Zombies: {grid.getZombiePopulation()} Empty: {grid.getRecoveredPopulation()}")
    print(grid.grid.size)
