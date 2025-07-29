# /***********************************************************
# *                                                         *
# * Copyright (c) 2022                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu, Brett T. Lopez                *
# * Contact: {aaronjs, btlopez}@ucla.edu                    *
# *                                                         *
# ***********************************************************/

# PACKAGE IMPORTS
import sys
import math
import time
import casadi
import numpy as np
from copy import copy, deepcopy
from casadi import *
from functools import partial

import params as p

# from adaptor     import adaptor, adaptor_plot
# from commons     import is_key_pressed, pyramidalConstraint, genSkewSymMat, tait_bryan_to_rotation_matrix, calcCurrentPos
# from dynamics    import orbital_ellp_undrag
# from orbitsim    import mpc_plot
# from deflection  import targetDeflect, deflection_plot
# from spacecraft  import Target
# from dynamictube import ancillary_controller, calcDelta, calcD

# Simulation parameters
dt = p.dt

# Implement MPC using CasADi
# r_act = MX.sym('r_act', 6)
# r_nom = MX.sym('r_nom', 6)
# u_act = MX.sym('u_act', 3)
# u_nom = MX.sym('u_nom', 3)

# A = MX.sym('A', 6, 6)
# B = MX.sym('B', 6, 3)
# Q = MX.sym('Q', 6, 6)
# R = MX.sym('R', 3, 3)

# # Define the cost function
# cost = 0
# cost += mtimes(mtimes((r_act - r_nom).T, Q), (r_act - r_nom))
# cost += mtimes(mtimes((u_act - u_nom).T, R), (u_act - u_nom))

# # Define the constraints
# g = []
# # g.append(r_act[0] == r_nom[0])
# # g.append(r_act[1] == r_nom[1])
# # g.append(r_act[2] == r_nom[2])
# # g.append(r_act[3] == r_nom[3])
# # g.append(r_act[4] == r_nom[4])
# # g.append(r_act[5] == r_nom[5])

# # Define the dynamics
# # p = MX.sym('p')
# # dae = {'x': vertcat(r_act, r_nom), 'z': r_nom, 'p': p, 'ode': r_nom + r_nom, 'alg': r_nom*cos(p) - r_act}
# # F = integrator('F', 'idas', dae)
# # print(F)

# # Define the solver
# nlp = {'x': vertcat(r_act, u_act),
#        'p': r_nom,
#        'f': mtimes((r_act - r_nom).T, (r_act - r_nom))
#           + mtimes((u_act        ).T, (u_act)),
#        'g': vertcat(*g)}
# solver = nlpsol('solver', 'ipopt', nlp)
# print(solver)

# # p_0 = np.array([0.1])
# # r = F(x0 = r_act_0, z0 = r_nom_0, p=p_0)
# # print(r['xf'])

# # Define the parameters
A_0 = np.array(
    [
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)
B_0 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

Q_0 = np.array(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)
R_0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

from commons import discretize

A_0, B_0 = discretize(1, A_0, B_0)

# # Define the parameters
# # p = vertcat(A, B, Q, R)
# # p_0 = vertcat(A_0, B_0, Q_0, R_0)

# # Define the bounds
# # lbx = np.array([-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf])
# # ubx = np.array([inf, inf, inf, inf, inf, inf, inf, inf, inf])
# # lbg = np.array([0, 0, 0, 0, 0, 0])
# # ubg = np.array([0, 0, 0, 0, 0, 0])

# # Define the initial guess
# # x0 = vertcat(r_act_0, u_act_0)

# # Solve the NLP
# sol = solver(x0 = vertcat(r_act_0, u_act_0), lbg=0, ubg=0)
# print(sol['x'].full().flatten())
# print(sol['f'].full().flatten())
# print(sol['g'].full().flatten())

opti = casadi.Opti()

timeSteps = 10

r_act_0 = np.array([1, 0, 0, 0, 0, 0])
r_nom_0 = np.array([1, 0, 0, 0, 0, 0])
u_act_0 = np.array([0, 0, 0])
u_nom_0 = np.array([0, 0, 0])


r_act = opti.variable(6, timeSteps)
r_nom = opti.variable(6, timeSteps)
u_act = opti.variable(3, timeSteps)
u_nom = opti.variable(3, timeSteps)

A = opti.parameter(6, 6)
B = opti.parameter(6, 3)
Q = opti.parameter(6, 6)
R = opti.parameter(3, 3)

# Define the cost function
cost = 0
for time in range(timeSteps):
    opti.set_value(A, A_0)
    opti.set_value(B, B_0)
    opti.set_value(Q, Q_0)
    opti.set_value(R, R_0)
    cost += mtimes(mtimes((r_act[:, time]).T, Q), (r_act[:, time]))
    cost += mtimes(mtimes((u_act[:, time]).T, R), (u_act[:, time]))

# Define the constraints
g = []
# g.append(r_act[0] == r_nom[0])


opti.minimize(cost)
for time in range(timeSteps - 1):
    opti.subject_to(
        r_act[:, time + 1]
        == mtimes(A, r_act[:, time - 1]) + mtimes(B, u_act[:, time - 1])
    )
    opti.subject_to(
        r_nom[:, time + 1]
        == mtimes(A, r_nom[:, time - 1]) + mtimes(B, u_nom[:, time - 1])
    )
opti.subject_to(r_act[:, 0] == r_act_0)

opti.solver("ipopt")
sol = opti.solve()

print(sol.value(r_act))
print(sol.value(u_act))
