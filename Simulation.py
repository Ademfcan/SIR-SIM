import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QCoreApplication, QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton, QComboBox, QLineEdit, QSizePolicy, QLabel
from simgrid import SimGrid
from solve_rk import Solver

# Initial conditions
INIT_POPSIZE = int(1e3)
INIT_Z0 = int(10)
INIT_INFECTION_GROWTH = 0.1
INIT_HUMAN_LOSS = 0
INIT_ZOMBIE_LOSS = 0.05
INITGRIDSIZE = int(1e3)

# Simulation speed settings
speeds = [1, 2, 5, 10, 0.5, 0.1]
speed_idx = 0
currentSpeedFactor = 1
paused = False


preset_values_eq = {
    "Classic Apocalypse": {"a": INIT_INFECTION_GROWTH, "b": INIT_ZOMBIE_LOSS, "c": INIT_HUMAN_LOSS},
    "Raging Outbreak": {"a": 0.2, "b": 0.02, "c": 0},
    "Human Resistance": {"a": 0.02, "b": 0.1, "c": 0},
    "Human Uprising": {"a": 0.02, "b": 0.5, "c": 0},
    "Doomsday Virus": {"a": 1, "b": 0, "c": 0},
}

preset_values_init_pop = {
    "Small Infection": {"pop": 1010, "z0": 10},
    "Progressed Infection": {"pop": 1100, "z0": 100},
    "One in a thousand": {"pop": 1000, "z0": 1},
}

def changePresetsEq(preset_name):
    values = preset_values_eq[preset_name]
    infection_growth_input.setText(str(values["a"]))
    zombie_loss_input.setText(str(values["b"]))
    human_loss_input.setText(str(values["c"]))

def changePresetsInitPop(preset_name):
    values = preset_values_init_pop[preset_name]
    total_pop_input.setText(str(values["pop"]))
    init_z0_input.setText(str(values["z0"]))


# Create the initial helpers
grid = SimGrid(INIT_POPSIZE, INIT_Z0, INIT_INFECTION_GROWTH, INIT_ZOMBIE_LOSS, INIT_HUMAN_LOSS, INITGRIDSIZE)
solver = Solver(INIT_POPSIZE, INIT_Z0, INIT_INFECTION_GROWTH, INIT_ZOMBIE_LOSS, INIT_HUMAN_LOSS)

# PyQtGraph setup
app = QApplication(sys.argv)
win = QMainWindow()
central_widget = QWidget()
layout = QVBoxLayout()
legend_label = QLabel("Green: Zombies | Blue: Humans")
legend_label.setAlignment(Qt.AlignCenter)  # Center the text
legend_label.setStyleSheet("font-size: 20px; font-weight: bold;")  # Make it more visible
layout.addWidget(legend_label)  # Add it at the top of the layout
central_widget.setLayout(layout)
win.setCentralWidget(central_widget)

# Grid visualization using ImageView
grid_view = pg.ImageView()
grid_view.ui.roiBtn.hide()  # Hides the ROI button
grid_view.ui.menuBtn.hide()  # Hides the menu button
grid_view.ui.histogram.hide()

# Define custom colormap
custom_cmap = pg.ColorMap(pos=[0.0, 1.0],
                          color=[(0, 100, 0), (0, 0, 139)])
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

# Second plot widget
plot_widget_solver = pg.PlotWidget()
plot_layout.addWidget(plot_widget_solver)
plot_widget_solver.setTitle(f"Differential equation population estimate")
plot_widget_solver.setLabel('left', "Population Count")
plot_widget_solver.setLabel('bottom', "Days Passed")
zombie_curve_solver = plot_widget_solver.plot(pen='g', name="Zombies", width=6)
human_curve_solver = plot_widget_solver.plot(pen='b', name="Humans", width=3)

# Add plot layout to main layout
layout.addLayout(plot_layout, stretch=1)

# Control buttons
button_layout = QHBoxLayout()
layout.addLayout(button_layout, stretch=0)

pause_button = QPushButton("Pause")
reset_button = QPushButton("Reset")
speed_button = QPushButton("Speed x1")
constantPresetEq = QComboBox()
constantPresetEq.addItems(preset_values_eq.keys())
constantPresetEq.currentIndexChanged.connect(lambda event: changePresetsEq(constantPresetEq.currentText()))
constantPresetInitPop = QComboBox()
constantPresetInitPop.addItems(preset_values_init_pop.keys())
constantPresetInitPop.currentIndexChanged.connect(lambda event: changePresetsInitPop(constantPresetInitPop.currentText()))
button_layout.addWidget(pause_button)
button_layout.addWidget(reset_button)
button_layout.addWidget(speed_button)
button_layout.addWidget(constantPresetEq)
button_layout.addWidget(constantPresetInitPop)

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

fullEquationHumans = QLabel()
fullEquationZombies = QLabel()


param_layout.addRow("dH/dt", fullEquationHumans)
param_layout.addRow("dZ/dt", fullEquationZombies)
param_layout.addRow("Total Population", total_pop_input)
param_layout.addRow("Inital Zombie Amount", init_z0_input)
param_layout.addRow("Infection Growth [-1,1]", infection_growth_input)
param_layout.addRow("Human Loss (b term in -bZ)", human_loss_input)
param_layout.addRow("Zombie Loss (c term in -cH)", zombie_loss_input)
param_layout.addRow("Grid Size", grid_size_input)
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
    range_val = grid.popSize/(grid.squareSize*grid.squareSize)/5

    grid_view.setLevels(min=-range_val, max=range_val)
    grid_view.setImage(current_grid.T, autoLevels=False)

def update_sim_plot():
    min_z_sim = min(len(time_stamps),len(zombie_populations_sim))
    zombie_curve_sim.setData(time_stamps[:min_z_sim], zombie_populations_sim[:min_z_sim])
    min_h_sim = min(len(time_stamps),len(human_populations_sim))
    human_curve_sim.setData(time_stamps[:min_h_sim], human_populations_sim[:min_h_sim])

def update_solver_plot():
    min_z_sol = min(len(time_stamps),len(zombie_populations_solver))
    zombie_curve_solver.setData(time_stamps[:min_z_sol], zombie_populations_solver[:min_z_sol])
    min_h_sol = min(len(time_stamps),len(human_populations_solver))
    human_curve_solver.setData(time_stamps[:min_h_sol], human_populations_solver[:min_h_sol])

def toggle_pause():
    global paused
    paused = not paused
    pause_button.setText("Play" if paused else "Pause")

def reset_simulation():
    global grid, solver, time_stamps, zombie_populations_sim, human_populations_sim, zombie_populations_solver, human_populations_solver, recovered_populations_sim, recovered_populations_solver
    global human_growth, human_loss, zombie_growth, zombie_loss, grid_size
    reset_button.setStyleSheet("")
    reset_button.setText("Reset")
    fullEquationHumans.setText(f"dH/dt = -{infection_growth}H*Z - {human_loss}*Z")
    fullEquationZombies.setText(f"dZ/dt = {infection_growth}H*Z - {zombie_loss}*H")

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

def saveAndMove():
    # Capture the current window and save it
    frame = win.grab()
    title = f"{constantPresetEq.currentText()}_{constantPresetInitPop.currentText()}.png"
    frame.save(f"Tests/{title}", "PNG")
    
    # Move to the next init pop preset; if at the end, move to next eq preset and reset init pop to first.
    current_init_index = constantPresetInitPop.currentIndex()
    if current_init_index < constantPresetInitPop.count()-1:
        constantPresetInitPop.setCurrentIndex(current_init_index + 1)
    else:
        constantPresetInitPop.setCurrentIndex(0)
        current_eq_index = constantPresetEq.currentIndex()
        if current_eq_index < constantPresetEq.count()-1:
            constantPresetEq.setCurrentIndex(current_eq_index + 1)
        else:
            # If we've reached the last eq preset, loop back to the beginning.
            constantPresetEq.setCurrentIndex(0)
    updateInputs()
    
    # Reset the simulation with the new preset options.
    reset_simulation()

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
        new_infection_growth = np.clip(float(infection_growth_input.text()),-1,1)
        if infection_growth != new_infection_growth:
            infection_growth = new_infection_growth
            infection_growth_input.setText(str(new_infection_growth))
            changed = True


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

def updateEquations():
    fullEquationHumans.setText(f"dH/dt = -{infection_growth}H*Z - {human_loss}*Z")
    fullEquationZombies.setText(f"dZ/dt = {infection_growth}H*Z - {zombie_loss}*H")

def generalguiupdate():
    updateEquations()

class SimulationThread(QThread):
    triggerSaveAndMove = pyqtSignal()
    dataChanged = pyqtSignal()
    generalupdate = pyqtSignal()

    def run(self):
        global hitApoc
        hitApoc = False
        self.generalupdate.emit()
        while True:
            if updateInputs():
                reset_button.setText("Restart Simulation to apply Changes")
                self.generalupdate.emit()

            if paused:
                continue
            atoi = 0.3
            if grid.isApocalypse(atoi) and solver.isApocalypse(grid.timePassed,atoi):
                if not hitApoc:
                    sim_humans_dead = np.isclose(grid.getHumanPopulation(),0,atol=atoi)
                    solver_humans_dead = np.isclose(solver.getHumanPopulation(grid.timePassed),0,atol=atoi)
                    if not sim_humans_dead and not solver_humans_dead:
                        finText = "Uninamous Winner: Humans!"
                    elif not sim_humans_dead:
                        finText = "Grid thinks Humans, Solver thinks Zombies!"
                    elif not solver_humans_dead:
                        finText = "Grid thinks Zombies, Solver thinks Humans!"
                    else:
                        finText = "Uninamous Winner: Zombies :("

                    reset_button.setText(f"{finText} | Restart")

                    # uncomment this line if you want the simulation to screenshot this frame (pyqt window only!) and then move on to next preset config
                    # self.triggerSaveAndMove.emit()
                    
                    hitApoc = True
            elif grid.timePassed > 50000:
                finText = "Stagnated!"
                reset_button.setText(f"{finText} | Restart")
                self.triggerSaveAndMove.emit()
                hitApoc = True
            else:
                if hitApoc:
                    hitApoc = False
                grid.propagate(currentSpeedFactor)
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
sim_thread.triggerSaveAndMove.connect(saveAndMove)
sim_thread.generalupdate.connect(generalguiupdate)
sim_thread.start()

win.show()
sys.exit(app.exec_())