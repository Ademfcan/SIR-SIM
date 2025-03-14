import math
import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from simgrid import SimGrid

USE_SIMPLE = True
if USE_SIMPLE:
    from solve_simple import Solver
else:
    from solve_rk import Solver

# Initial conditions
INIT_H0 = 10
INIT_HUMAN_GROWTH = 0.3
INIT_HUMAN_LOSS = 0.2
INIT_Z0 = 10
INIT_ZOMBIE_GROWTH = 0.1
INIT_ZOMBIE_LOSS = 0.1
INITGRIDSIZE = 1e4
INITMAXELEM = 1e6

# Simulation speed settings
INIT_SPEEDS = [1, 2, 5, 0.5, 0.1]
speed_idx = 0
currentSpeedFactor = 1
paused = False


preset_names = [
    "Default",
    "Savanna Equilibrium",
    "Wolf Overrun",
    "Endless Rivalry",
    "Alien Invasion",
    "Herbivore Revolution",
    "Mutual Destruction",
    "Silent Infection",
    "Rainforest Harmony"
]

preset_values = {
    "Default": {"a": INIT_ZOMBIE_GROWTH, "b": INIT_ZOMBIE_LOSS, "c": INIT_HUMAN_GROWTH, "d": INIT_HUMAN_LOSS},
    "Savanna Equilibrium": {"a": 0.8, "b": 0.6, "c": 0.5, "d": 0.4},
    "Wolf Overrun": {"a": 1.2, "b": 1.0, "c": 0.6, "d": 0.9},
    "Endless Rivalry": {"a": 1.0, "b": 0.7, "c": 1.0, "d": 0.7},
    "Alien Invasion": {"a": 1.3, "b": 0.5, "c": 0.8, "d": 0.2},
    "Herbivore Revolution": {"a": 0.5, "b": 0.3, "c": 1.2, "d": 0.7},
    "Mutual Destruction": {"a": 0.9, "b": 1.2, "c": 0.9, "d": 1.2},
    "Silent Infection": {"a": 0.6, "b": 0.2, "c": 0.4, "d": 0.9},
    "Rainforest Harmony": {"a": 0.9, "b": 0.5, "c": 0.9, "d": 0.5}
}


def changePresets(preset_name):
    global zombie_growth, zombie_loss, human_growth, human_loss
    values = preset_values[preset_name]
    zombie_growth_input.setText(str(values["a"]))
    zombie_loss_input.setText(str(values["b"]))

    human_growth_input.setText(str(values["c"]))
    human_loss_input.setText(str(values["d"]))




# Create the initial grid
grid = SimGrid(INIT_Z0, INIT_ZOMBIE_GROWTH, INIT_ZOMBIE_GROWTH,
               INIT_H0, INIT_HUMAN_GROWTH, INIT_HUMAN_LOSS,
               INITGRIDSIZE, MAX=INITMAXELEM)

solver = Solver(INIT_Z0, INIT_ZOMBIE_GROWTH, INIT_ZOMBIE_LOSS, INIT_H0, INIT_HUMAN_GROWTH, INIT_HUMAN_LOSS, INITMAXELEM)

# PyQtGraph setup
app = QtWidgets.QApplication(sys.argv)
win = QtWidgets.QMainWindow()
central_widget = QtWidgets.QWidget()
layout = QtWidgets.QVBoxLayout()
central_widget.setLayout(layout)
win.setCentralWidget(central_widget)

# Grid visualization using ImageView
grid_view = pg.ImageView()

# Define custom colormap:
# - At normalized position 0.0 (lowest value) we use dark green.
# - At 0.5 (the midpoint, corresponding to 0) we use white.
# - At 1.0 (highest value) we use dark blue.
custom_cmap = pg.ColorMap(pos=[0.0, 0.5, 1.0],
                          color=[(0, 100, 0), (255, 255, 255), (0, 0, 139)])
grid_view.setColorMap(custom_cmap)
grid_view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

layout.addWidget(grid_view, stretch=2)  # Increase stretch factor to make it the dominant widget


# Layout for two population trend plots
plot_layout = QtWidgets.QHBoxLayout()

# First plot widget
plot_widget_sim = pg.PlotWidget()
plot_layout.addWidget(plot_widget_sim)
plot_widget_sim.setTitle("Simulated Population Over Time")
plot_widget_sim.setLabel('left', "Population Count")
plot_widget_sim.setLabel('bottom', "Days Passed")
zombie_curve_sim = plot_widget_sim.plot(pen='g', name="Zombies",width=6)
human_curve_sim = plot_widget_sim.plot(pen='b', name="Humans",width=3)

# Second plot widget
plot_widget_solver = pg.PlotWidget()
plot_layout.addWidget(plot_widget_solver)
plot_widget_solver.setTitle(f"Differential equation population estimate | MODE: {"simple" if USE_SIMPLE else "rk"}")
plot_widget_solver.setLabel('left', "Population Count")
plot_widget_solver.setLabel('bottom', "Days Passed")
zombie_curve_solver = plot_widget_solver.plot(pen='g', name="Zombies", width=6)
human_curve_solver = plot_widget_solver.plot(pen='b', name="Humans", width=3) # if the functions are identical, have some way of differentiating

# Add plot layout to main layout
layout.addLayout(plot_layout, stretch=1)

# Control buttons
button_layout = QtWidgets.QHBoxLayout()
layout.addLayout(button_layout, stretch=0)
pause_button = QtWidgets.QPushButton("Pause")
reset_button = QtWidgets.QPushButton("Reset")
speed_button = QtWidgets.QPushButton("Speed x1")
constantPreset = QtWidgets.QComboBox()
constantPreset.addItems(preset_names)
constantPreset.currentIndexChanged.connect(lambda event : changePresets(constantPreset.currentText()))
button_layout.addWidget(pause_button)
button_layout.addWidget(reset_button)
button_layout.addWidget(speed_button)
button_layout.addWidget(constantPreset)

# Number inputs to modify grid parameters
param_layout = QtWidgets.QFormLayout()

human_growth_input = QtWidgets.QLineEdit(str(INIT_HUMAN_GROWTH))
human_growth = INIT_HUMAN_GROWTH
human_loss_input = QtWidgets.QLineEdit(str(INIT_HUMAN_LOSS))
human_loss = INIT_HUMAN_LOSS
zombie_growth_input = QtWidgets.QLineEdit(str(INIT_ZOMBIE_GROWTH))
zombie_growth = INIT_ZOMBIE_GROWTH
zombie_loss_input = QtWidgets.QLineEdit(str(INIT_ZOMBIE_LOSS))
zombie_loss = INIT_ZOMBIE_LOSS
grid_size_input = QtWidgets.QLineEdit(str(INITGRIDSIZE))
grid_size = INITGRIDSIZE

speed_input = QtWidgets.QLineEdit(",".join(map(str,INIT_SPEEDS)))
speeds = INIT_SPEEDS

param_layout.addRow("H Growth", human_growth_input)
param_layout.addRow("H Loss", human_loss_input)
param_layout.addRow("Z Growth", zombie_growth_input)
param_layout.addRow("Z Loss", zombie_loss_input)
param_layout.addRow("Grid Size", grid_size_input)
param_layout.addRow("Speeds", speed_input)

layout.addLayout(param_layout)

# Simulation state variables
time_stamps = []
zombie_populations_sim = []
human_populations_sim = []
zombie_populations_solver = []
human_populations_solver = []

def update_grid():
    """Update the grid image with a dynamic colormap range so that 0 maps to white."""
    current_grid = grid.grid
    grid_view.setLevels(min=-20, max=20)
    # Transpose to match the original orientation.
    grid_view.setImage(current_grid.T, autoLevels=False)

def update_sim_plot():
    zombie_curve_sim.setData(time_stamps, zombie_populations_sim)
    human_curve_sim.setData(time_stamps, human_populations_sim)

def update_solver_plot():
    # print(f"{zombie_populations_solver=} {human_populations_solver=}")
    zombie_curve_solver.setData(time_stamps,zombie_populations_solver)
    human_curve_solver.setData(time_stamps, human_populations_solver)

def toggle_pause():
    global paused
    paused = not paused
    pause_button.setText("Play" if paused else "Pause")

def reset_simulation():
    global grid, solver, time_stamps, zombie_populations_sim, human_populations_sim, zombie_populations_solver, human_populations_solver
    global human_growth, human_loss, zombie_growth, zombie_loss, grid_size

    reset_button.setStyleSheet("")


    # Reinitialize the grid with new parameters
    grid = SimGrid(INIT_Z0, zombie_growth, zombie_loss,
               INIT_H0, human_growth, human_loss,
               grid_size, MAX=INITMAXELEM)
    
    real_grid_size = math.pow(math.ceil(math.sqrt(grid_size)),2)
    grid_size_input.setText(str(real_grid_size))
    grid_size = real_grid_size

    solver = Solver(INIT_Z0, zombie_growth, zombie_loss, INIT_H0, human_growth, human_loss, INITMAXELEM)


    time_stamps, zombie_populations_sim, human_populations_sim, zombie_populations_solver, human_populations_solver = [], [], [], [], []
    update_grid()
    update_sim_plot()
    update_solver_plot()
    win.setWindowTitle("Zombie Simulation")

def toggle_speed():
    global speed_idx, currentSpeedFactor
    speed_idx = (speed_idx + 1) % len(speeds)
    currentSpeedFactor = speeds[speed_idx]
    speed_button.setText(f"Speed x{currentSpeedFactor}")

pause_button.clicked.connect(toggle_pause)
reset_button.clicked.connect(reset_simulation)
speed_button.clicked.connect(toggle_speed)

SLEEPTIME = 10  # milliseconds

def updateInputs() -> bool:
    global human_growth, human_loss, zombie_growth, zombie_loss, grid_size
    changed = False
    try:

        new_human_growth = float(human_growth_input.text())
        if human_growth != new_human_growth:
            changed = True
            human_growth = new_human_growth
            # grid.setHumanGrowth(human_growth)
        
        new_human_loss = float(human_loss_input.text())
        if human_loss != new_human_loss:
            changed = True
            human_loss = new_human_loss
            # grid.setHumanLoss(human_loss)

        new_zombie_growth = float(zombie_growth_input.text())
        if zombie_growth != new_zombie_growth:
            changed = True
            zombie_growth = new_zombie_growth
            # grid.setZombieGrowth(zombie_growth)
        
        new_zombie_loss = float(zombie_loss_input.text())
        if zombie_loss != new_zombie_loss:
            changed = True
            zombie_loss = new_zombie_loss
            # grid.setZombieLoss(zombie_loss)
        new_grid_size = float(grid_size_input.text())
        if grid_size != new_grid_size:
            changed = True
            grid_size = new_grid_size 
        
        
        return changed
    except ValueError as valueError:
        return False
hitApoc = False

def simulation_step():
    global hitApoc
    changed = updateInputs()
    if changed:
        reset_button.setStyleSheet("border: 1px solid; border-color:green;")
        reset_button.setText("Restart To Apply Changes")
        


    try:
        inputSpeeds = speed_input.text().split(",")
        global speeds
        speeds = [float(speed.strip()) for speed in inputSpeeds]
    except ValueError as valueError:
        pass
        

    if not paused and not grid.isApocalypse():
        hitApoc = False
        grid.propagate(0.1 * currentSpeedFactor)
        time_stamps.append(grid.timePassed)
        zombie_populations_sim.append(grid.getZombiePopulation())
        human_populations_sim.append(grid.getHumanPopulation())
        z_pred, h_pred = solver.getZombiePrediction(grid.timePassed), solver.getHumanPrediction(grid.timePassed)
        zombie_populations_solver.append(z_pred)
        human_populations_solver.append(h_pred)
        update_grid()
        update_sim_plot()
        update_solver_plot()
    elif grid.isApocalypse():
        if not hitApoc:
            # Update window title with winner info and stop updating
            finText = "Winner: Humans!" if grid.getHumanPopulation() > 0 else "Winner: Zombies :("
            win.setWindowTitle(finText)
            reset_button.setStyleSheet("border: 1px solid; border-color:blue;")
            reset_button.setText(f"{finText} | Restart Simulation")
            hitApoc = True
        # return
    QtCore.QTimer.singleShot(SLEEPTIME, simulation_step)

win.show()
simulation_step()
sys.exit(app.exec_())
