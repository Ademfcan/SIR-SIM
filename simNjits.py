import numpy as np
import numba
from numba import njit, prange
from numba.typed import Dict
from numba.core import types

@njit
def shuffle_array(arr):
    """ Fisher-Yates shuffle implementation for Numba """
    n = len(arr)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr

@njit
def getNeighbors(grid, row, col):
    dxs = shuffle_array(np.array([-1, 0, 1]))
    dys = shuffle_array(np.array([-1, 0, 1]))
    neighbors = []
    for dx in dxs:
        for dy in dys:
            if dx == 0 and dy == 0:
                continue
            newRow, newCol = row + dx, col + dy
            if 0 <= newRow < grid.shape[0] and 0 <= newCol < grid.shape[1]:
                neighbors.append((newRow, newCol))
    return neighbors

@njit
def propagateInteractionsCELL(grid, row, col, changeGrid, interactionMap, infectionGrowth, zombieLoss, humanLoss, zombieDir, humanDir) -> int:
    pop = grid[row, col]
    if np.isclose(pop, 0):
        return 0 # Skip empty cells
    
    cellPopAbs = abs(pop)
    cellIsHumanDominated = pop > 0
    cellLoss = humanLoss if cellIsHumanDominated else zombieLoss
    cellGrowthDir = humanDir if cellIsHumanDominated else zombieDir

    neighbors = getNeighbors(grid, row, col)

    recovered = 0

    for neighborRow, neighborCol in neighbors:    
        neighborPop = grid[neighborRow, neighborCol]
        neighborHumanDominated = neighborPop > 0
        isInteraction = cellIsHumanDominated != neighborHumanDominated  # Must be opposite types

        if np.isclose(neighborPop, 0) or not isInteraction:
            continue
        
        if interactionMap[row, col, neighborRow, neighborCol]:
            continue  # Skip already processed interaction

        neighborPopAbs = abs(neighborPop)

        interactionAbs = cellPopAbs * neighborPopAbs         

        neighborLoss = humanLoss if neighborHumanDominated else zombieLoss
        neighborGrowthDir = humanDir if neighborHumanDominated else zombieDir

        change = interactionAbs * infectionGrowth * zombieDir
        
        cellChangeLoss = neighborPopAbs * cellLoss * cellGrowthDir
        cellChange = change - cellChangeLoss 

        
        neighborChangeLoss = cellPopAbs * neighborLoss * neighborGrowthDir
        neighborChange = change - neighborChangeLoss

        if cellIsHumanDominated:
            recovered += abs(neighborChangeLoss)
        else:
            recovered += abs(cellChangeLoss)

        changeGrid[row, col] += cellChange
        changeGrid[neighborRow, neighborCol] += neighborChange

        # **Mark This Interaction As Processed**
        interactionMap[row, col, neighborRow, neighborCol] = True
        interactionMap[neighborRow, neighborCol, row, col] = True  # Ensure symmetry
    
    return recovered

@njit
def propagateMovementCELL(grid, row, col, movementGrid, zombieDir, humanDir, moveProb):
    cellPop = grid[row, col]
    if np.isclose(cellPop,0,atol=1e-3):
        return

    cellHumanDominated = cellPop > 0
    cellGrowthDir = humanDir if cellHumanDominated else zombieDir
    cellPopAbs = abs(cellPop)
    neighbors = getNeighbors(grid, row, col)

    for neighborRow, neighborCol in neighbors:
        if cellPopAbs <= 0:
            break  # Used all growth already

        if np.random.random() > moveProb:
            continue  # Skip movement if not lucky

        neighborPop = grid[neighborRow, neighborCol]
        neigborHumanDominated = neighborPop > 0
        isInteraction = cellHumanDominated != neigborHumanDominated

        if np.isclose(neighborPop,0,atol=1e-3) or not isInteraction:
            amountLeaveAbs = np.random.uniform(0, cellPopAbs)
            cellPopAbs -= amountLeaveAbs

            movementGrid[neighborRow, neighborCol] += amountLeaveAbs * cellGrowthDir

    movementGrid[row, col] += cellPopAbs * cellGrowthDir

@njit
def propagate(grid, timeStep, infectionGrowth, zombieLoss, humanLoss, zombieDir, humanDir, moveProb, totalPop):
    changeGrid = np.zeros_like(grid, dtype=np.float64)
    movementGrid = np.zeros_like(grid, dtype=np.float64)
    
    # **Optimized: Use Boolean NumPy Array Instead of Set**
    interactionMap = np.zeros((grid.shape[0], grid.shape[1], grid.shape[0], grid.shape[1]), dtype=np.bool_)
    totalRecovered = 0
    # **Parallelize Interaction Phase**
    for row in prange(grid.shape[0]):
        for col in range(grid.shape[1]):
            totalRecovered += propagateInteractionsCELL(grid, row, col, changeGrid, interactionMap, infectionGrowth, zombieLoss, humanLoss, zombieDir, humanDir)
    
    # **Update Grid with Changes**
    changeTimeScaled = changeGrid * timeStep
    updatedGrid = np.add(grid, changeTimeScaled)

    # **Parallelize Movement Phase**
    for row in prange(grid.shape[0]):
        for col in range(grid.shape[1]):
            propagateMovementCELL(updatedGrid, row, col, movementGrid, zombieDir, humanDir, moveProb)
    
    # maxPerCell = totalPop/updatedGrid.size
    maxPerCell = totalPop
    return np.clip(movementGrid,-maxPerCell,maxPerCell), totalRecovered
