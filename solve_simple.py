from sympy import Function, dsolve, Eq, Derivative, symbols, solve
from sympy.abc import t

class Solver:
    def __init__(self, z0, zombieGrowth, zombieLoss, h0, humanGrowth, humanLoss, maxPopulation, minPopulation = 0):
        z = Function('zombies')(t)
        h = Function('humans')(t)

        # Define the system of ODEs
        dzdt = Eq(Derivative(z, t), z*zombieGrowth-h*zombieLoss)
        dhdt = Eq(Derivative(h, t), h*humanGrowth-z*humanLoss)

        sol = dsolve([dzdt, dhdt])
        sol_z, sol_h = sol

        initial_conditions = {z.subs(t, 0): z0, h.subs(t, 0): h0}

        constants = solve([sol_z.rhs.subs(t, 0) - initial_conditions[z.subs(t, 0)],
                   sol_h.rhs.subs(t, 0) - initial_conditions[h.subs(t, 0)]], dict=True)

        # Substitute the constants back into the solution
        self.z_t = sol_z.rhs.subs(constants[0])
        self.h_t = sol_h.rhs.subs(constants[0])
        self.maxPop = maxPopulation
        self.minPop = minPopulation
    
    def getZombiePrediction(self, in_t):
        return min(max(float(self.z_t.subs(t, in_t).evalf()),self.minPop),self.maxPop)

    def getHumanPrediction(self, in_t):
        return min(max(float(self.h_t.subs(t, in_t).evalf()),self.minPop),self.maxPop)




