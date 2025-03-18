import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QCoreApplication, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton, QComboBox, QLineEdit, QSizePolicy
from simgrid import SimGrid
from solve_rk import Solver

# Initial conditions
INIT_POPSIZE = 10
INIT_Z0 = 1
INIT_INFECTION_GROWTH = 0.3
INIT_HUMAN_LOSS = 0
INIT_ZOMBIE_LOSS = 0
INITGRIDSIZE = 40

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
    "Default": {"a": INIT_INFECTION_GROWTH, "b": INIT_ZOMBIE_LOSS, "c": INIT_HUMAN_LOSS},
    "Savanna Equilibrium": {"a": 0.8, "b": 0.6, "c": 0.5},
    "Wolf Overrun": {"a": 1.2, "b": 1.0, "c": 0.6},
    "Endless Rivalry": {"a": 1.0, "b": 0.7, "c": 1.0},
    "Alien Invasion": {"a": 1.3, "b": 0.5, "c": 0.8},
    "Herbivore Revolution": {"a": 0.5, "b": 0.3, "c": 1.2},
    "Mutual Destruction": {"a": 0.9, "b": 1.2, "c": 0.9},
    "Silent Infection": {"a": 0.6, "b": 0.2, "c": 0.4},
    "Rainforest Harmony": {"a": 0.9, "b": 0.5, "c": 0.9}
}

def changePresets(preset_name):
    global infection_growth, zombie_loss, human_loss
    values = preset_values[preset_name]
    infection_growth_input.setText(str(values["a"]))
    zombie_loss_input.setText(str(values["b"]))
    human_loss_input.setText(str(values["c"]))

# Create the initial helpers
grid = SimGrid(INIT_POPSIZE, INIT_Z0, INIT_INFECTION_GROWTH, INIT_ZOMBIE_LOSS, INIT_HUMAN_LOSS, INITGRIDSIZE)
solver = Solver(INIT_POPSIZE, INIT_Z0, INIT_INFECTION_GROWTH, INIT_ZOMBIE_LOSS, INIT_HUMAN_LOSS)

# PyQtGraph setup
app = QApplication(sys.argv)
win = QMainWindow()
central_widget = QWidget()
layout = QVBoxLayout()
central_widget.setLayout(layout)
win.setCentralWidget(central_widget)

# Grid visualization using ImageView
grid_view = pg.ImageView()
grid_view.ui.roiBtn.hide()  # Hides the ROI button
grid_view.ui.menuBtn.hide()  # Hides the menu button
grid_view.ui.histogram.hide()

# Define custom colormap
custom_cmap = pg.ColorMap(pos=[0.0, 0.5, 1.0],
                          color=[(0, 100, 0), (255, 0, 0), (0, 0, 139)])
grid_view.setColorMap(custom_cmap)
grid_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
layout.addWidget(grid_view, stretch=2)

# Layout for two population trend plots
plot_layout = QHBoxLayout()

# First plot widget
plot_widget_sim = pg.PlotWidget()
plot_layout.addWidget(plot_widget_sim)
plot_widget_sim.setTitle("Simulated Population Over Time")
plot_widget_sim.setLabel('left', "Population Count")
plot_widget_sim.setLabel('bottom', "Days Passed")
zombie_curve_sim = plot_widget_sim.plot(pen='g', name="Zombies", width=6)
human_curve_sim = plot_widget_sim.plot(pen='b', name="Humans", width=3)
recovered_curve_sim = plot_widget_sim.plot(pen="red", name="Recovered", width=3)

# Second plot widget
plot_widget_solver = pg.PlotWidget()
plot_layout.addWidget(plot_widget_solver)
plot_widget_solver.setTitle(f"Differential equation population estimate")
plot_widget_solver.setLabel('left', "Population Count")
plot_widget_solver.setLabel('bottom', "Days Passed")
zombie_curve_solver = plot_widget_solver.plot(pen='g', name="Zombies", width=6)
human_curve_solver = plot_widget_solver.plot(pen='b', name="Humans", width=3)
recovered_curve_solver = plot_widget_solver.plot(pen='red', name="Recovered", width=3)

# Add plot layout to main layout
layout.addLayout(plot_layout, stretch=1)

# Control buttons
button_layout = QHBoxLayout()
layout.addLayout(button_layout, stretch=0)

pause_button = QPushButton("Pause")
reset_button = QPushButton("Reset")
speed_button = QPushButton("Speed x1")
constantPreset = QComboBox()
constantPreset.addItems(preset_names)
constantPreset.currentIndexChanged.connect(lambda event: changePresets(constantPreset.currentText()))
button_layout.addWidget(pause_button)
button_layout.addWidget(reset_button)
button_layout.addWidget(speed_button)
button_layout.addWidget(constantPreset)

# Number inputs to modify grid parameters
param_layout = QFormLayout()
total_pop_input = QLineEdit(str(INIT_POPSIZE))
total_pop = INIT_POPSIZE
init_z0_input = QLineEdit(str(INIT_Z0))
init_z0 = INIT_Z0
infection_growth_input = QLineEdit(str(INIT_INFECTION_GROWTH))
infection_growth = INIT_INFECTION_GROWTH
human_loss_input = QLineEdit(str(INIT_HUMAN_LOSS))
human_loss = INIT_HUMAN_LOSS
zombie_loss_input = QLineEdit(str(INIT_ZOMBIE_LOSS))
zombie_loss = INIT_ZOMBIE_LOSS
grid_size_input = QLineEdit(str(INITGRIDSIZE))
grid_size = INITGRIDSIZE
speed_input = QLineEdit(",".join(map(str, INIT_SPEEDS)))
speeds = INIT_SPEEDS

param_layout.addRow("Total Population", total_pop_input)
param_layout.addRow("Inital Zombie Amount", init_z0_input)
param_layout.addRow("Infection Growth", infection_growth_input)
param_layout.addRow("Human Loss (b term in -bZ)", human_loss_input)
param_layout.addRow("Zombie Loss (c term in -cH)", zombie_loss_input)
param_layout.addRow("Grid Size", grid_size_input)
param_layout.addRow("Simulation Speeds", speed_input)
layout.addLayout(param_layout)

# Simulation state variables
time_stamps = []
zombie_populations_sim = []
human_populations_sim = []
recovered_populations_sim = []
zombie_populations_solver = []
human_populations_solver = []
recovered_populations_solver = []

def update_grid():
    """Update the grid image with a dynamic colormap range so that 0 maps to white."""
    current_grid = grid.grid
    grid_view.setLevels(min=-0.1, max=0.1)
    grid_view.setImage(current_grid.T, autoLevels=False)

def update_sim_plot():
    zombie_curve_sim.setData(time_stamps, zombie_populations_sim)
    human_curve_sim.setData(time_stamps, human_populations_sim)
    recovered_curve_sim.setData(time_stamps, recovered_populations_sim)

def update_solver_plot():
    zombie_curve_solver.setData(time_stamps, zombie_populations_solver)
    human_curve_solver.setData(time_stamps, human_populations_solver)
    recovered_curve_solver.setData(time_stamps, recovered_populations_solver)

def toggle_pause():
    global paused
    paused = not paused
    pause_button.setText("Play" if paused else "Pause")

def reset_simulation():
    global grid, solver, time_stamps, zombie_populations_sim, human_populations_sim, zombie_populations_solver, human_populations_solver, recovered_populations_sim, recovered_populations_solver
    global human_growth, human_loss, zombie_growth, zombie_loss, grid_size
    reset_button.setStyleSheet("")
    reset_button.setText("Reset")
    # Reinitialize the grid with new parameters
    real_grid_size, _ = SimGrid.getNearestSquareCellCount(grid_size)
    grid_size_input.setText(str(real_grid_size))
    grid = SimGrid(total_pop, init_z0, infection_growth, zombie_loss, human_loss, real_grid_size)
    solver = Solver(total_pop, init_z0, infection_growth, zombie_loss, human_loss)
    time_stamps.clear()
    zombie_populations_sim.clear()
    human_populations_sim.clear()
    recovered_populations_sim.clear()
    zombie_populations_solver.clear()
    human_populations_solver.clear()
    recovered_populations_solver.clear()
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

SLEEPTIME = 10  # milliseconds for UI responsiveness

def updateInputs() -> bool:
    global human_growth, human_loss, zombie_growth, zombie_loss, grid_size, total_pop, init_z0, infection_growth
    changed = False
    try:
        new_total_pop = int(total_pop_input.text())
        if total_pop != new_total_pop:
            changed = True
            total_pop = new_total_pop
        new_init_z0 = int(init_z0_input.text())
        if init_z0 != new_init_z0:
            changed = True
            init_z0 = new_init_z0
        new_infection_growth = float(infection_growth_input.text())
        if infection_growth != new_infection_growth:
            changed = True
            infection_growth = new_infection_growth
        new_human_loss = float(human_loss_input.text())
        if human_loss != new_human_loss:
            changed = True
            human_loss = new_human_loss
        new_zombie_loss = float(zombie_loss_input.text())
        if zombie_loss != new_zombie_loss:
            changed = True
            zombie_loss = new_zombie_loss
        new_grid_size = int(grid_size_input.text())
        if grid_size != new_grid_size:
            changed = True
            grid_size = new_grid_size
        return changed
    except ValueError:
        return False

class SimulationThread(QThread):
    dataChanged = pyqtSignal()

    def run(self):
        global hitApoc
        hitApoc = False
        while True:
            updateInputs()
            if paused:
                continue
            if grid.isApocalypse():
                if not hitApoc:
                    finText = "Winner: Humans!" if grid.getHumanPopulation() > 0 else "Winner: Zombies :("
                    self.setWindowTitle.emit(finText)
                    hitApoc = True
            else:
                if hitApoc:
                    hitApoc = False
                grid.propagate(0.1 * currentSpeedFactor)
                time_stamps.append(grid.timePassed)
                z_sim, h_sim, r_sim = grid.getZombiePopulation(), grid.getHumanPopulation(), grid.getRecoveredPopulation()
                zombie_populations_sim.append(z_sim)
                human_populations_sim.append(h_sim)
                recovered_populations_sim.append(r_sim)
                
                # Solver computations if necessary
                z_sol, h_sol, r_sol = solver.getZombiePopulation(grid.timePassed), solver.getHumanPopulation(grid.timePassed), solver.getRecoveredPopulation(grid.timePassed) # Replace with real solver output
                zombie_populations_solver.append(z_sol)
                human_populations_solver.append(h_sol)
                recovered_populations_solver.append(r_sol)
                
                self.dataChanged.emit()

            self.msleep(SLEEPTIME)

sim_thread = SimulationThread()
sim_thread.dataChanged.connect(update_grid)
sim_thread.dataChanged.connect(update_sim_plot)
sim_thread.dataChanged.connect(update_solver_plot)
sim_thread.start()

win.show()
sys.exit(app.exec_())