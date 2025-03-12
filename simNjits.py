from numba import njit
import numpy as np

@njit
def shuffle_array(arr):
    # Create a random number generator
    
    # Shuffle the array using the Fisher-Yates algorithm
    n = len(arr)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr

@njit
def getNeighbors(grid, row, col):
    dxs = shuffle_array([-1, 0, 1])
    dys = shuffle_array([-1, 0, 1])
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
def propagateCELL(grid, row, col, changeGrid, zombieGrowth, zombieLoss, zombieDir, humanGrowth, humanLoss, humanDir, MAXCELL, moveProb):
    pop = grid[row, col]
    if np.isclose(pop,0):
        return  # nothing to do if the cell is empty
    
    popAbs = abs(pop)

    isHumanDominated = pop > 0
    growthConst = humanGrowth if isHumanDominated else zombieGrowth
    lossConst = humanLoss if isHumanDominated else zombieLoss
    growthDir = humanDir if isHumanDominated else zombieDir
    
    newPopGrowthAbs = popAbs * growthConst
    newPopLossAbs = 0

    neighbors = getNeighbors(grid, row, col)

    # Go over each neighbor and calculate interactions (loss phase)
    for neighborRow, neighborCol in neighbors:
        neighborPop = grid[neighborRow, neighborCol]
        neighborPopAbs = abs(neighborPop)
        
        if neighborPop == 0:
            continue
        
        neighborHumanDominated = neighborPop > 0
        isInteraction = isHumanDominated != neighborHumanDominated

        if isInteraction:
            neighborLoss = humanLoss if neighborHumanDominated else zombieLoss
            neighborDir = humanDir if neighborHumanDominated else zombieDir

            gridLossAbs = neighborPopAbs * lossConst
            neighborLossAbs = popAbs * neighborLoss

            newPopLossAbs += gridLossAbs

            changeGrid[neighborRow, neighborCol] -= neighborLossAbs * neighborDir

    newPopAbs = popAbs + newPopGrowthAbs - newPopLossAbs

    minLeavePerCell = 0
    if newPopAbs > MAXCELL:
        minLeavePerCell = (newPopAbs - MAXCELL) / 8

    # Go over each neighbor and calculate growth from this cell
    for neighborRow, neighborCol in neighbors:
        if np.random.random() > moveProb:
            continue  # not 'lucky' enough to move
        
        if newPopGrowthAbs <= 0:
            break  # used all growth already

        amountLeaveAbs = np.random.uniform(minLeavePerCell, newPopGrowthAbs)
        newPopGrowthAbs -= amountLeaveAbs

        changeGrid[neighborRow, neighborCol] += amountLeaveAbs * growthDir

    finalDelta = newPopGrowthAbs - newPopLossAbs
    changeGrid[row, col] += finalDelta * growthDir


@njit
def propagate(grid, timeStep, zombieGrowth, zombieLoss, zombieDir, humanGrowth, humanLoss, humanDir, MAXCELL, moveProb):
    changeGrid = np.zeros_like(grid, dtype=np.float64)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            propagateCELL(grid, row, col, changeGrid, zombieGrowth, zombieLoss, zombieDir, humanGrowth, humanLoss, humanDir, MAXCELL, moveProb)
    
    changeTimeScaled = changeGrid * timeStep
    updatedGrid = np.add(grid, changeTimeScaled)

    return np.clip(updatedGrid, -MAXCELL, MAXCELL)

