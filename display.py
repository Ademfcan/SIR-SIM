import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from simgrid import SimGrid

# Initialize the grid
grid = SimGrid(10, 0.4, 0.3, 10, 0.2, 0.1, 100,MAX=10000)

# Define a custom colormap
colors = [
    (0, "darkgreen"),   # Dark green for negative values
    (0.5, "white"),     # White for zero
    (1, "darkblue")     # Dark blue for positive values
]

# Create a custom colormap using LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

# Set up the plot for the grid
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Left plot: the grid
img = ax1.imshow(grid.grid, cmap=cmap, interpolation='nearest')
ax1.set_title("2D Grid: Dark Green (Zombies) -> White (Zero) -> Dark Blue (Humans)")
plt.colorbar(img, ax=ax1)

# Right plot: population trends (Zombies, Humans, Empty)
ax2.set_title("Population Trends Over Time")
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Population Count")

# Initialize population history lists
zombie_populations = []
human_populations = []

# Number of iterations for propagation
num_iterations = 1000

# Run the propagation loop
for i in range(num_iterations):
    # Propagate the grid by one time step
    grid.propagate(0.1)

    # Update the grid plot
    img.set_data(grid.grid)
    
    # Record the populations for plotting
    zombie_populations.append(grid.getZombiePopulation())
    human_populations.append(grid.getHumanPopulation())

    # Update the population trend plot
    ax2.plot(zombie_populations, label="Zombies", color='green')
    ax2.plot(human_populations, label="Humans", color='blue')

    # Pause to update the plot and make it visible
    plt.pause(0.0001)  # Adjust the pause time as needed

# Show the legend for the population trends
ax2.legend()

# Display the final plot
plt.show()
