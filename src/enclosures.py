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
import numpy  as     np
from   gekko  import GEKKO
from   params import *

# FUNCTION DEFINITIONS
## Calculate the Minimum Enclosing Ellipsoid of a Set of Points
def minEnclosingEllipsoid(X):
  m = GEKKO(remote = False)
  
  R = [m.Var(value = 1) for dim in range(3)]
  for dim in range(3):
    m.Equation(R[dim] >= 0.0)
  for point in range(len(X)):
    m.Equation(np.sum([(X[point][dim] / R[dim])**2 for dim in range(3)]) <= 1.0)
  m.Minimize(4.0*np.pi*R[0]*R[1]*R[2]/3.0)

  m.options.IMODE  = 3
  m.options.SOLVER = 3
  m.solve(disp = False)

  R      = np.array([R[dim].value[-1] for dim in range(3)])
  volume = m.options.OBJFCNVAL

  m.cleanup()
  del m
  return R, volume

## Calculate the Maximum Inscribed Ellipsoid of a Set of Points
def maxInscribedEllipsoid(X):
  m = GEKKO(remote = False)
  
  R = [m.Var(value = 1) for dim in range(3)]
  for dim in range(3):
    m.Equation(R[dim] >= 0.0)
  for point in range(len(X)):
    m.Equation(np.sum([(X[point][dim] / R[dim])**2 for dim in range(3)]) >= 1.0)
  m.Maximize(4.0*np.pi*R[0]*R[1]*R[2]/3.0)

  m.options.IMODE  = 3
  m.options.SOLVER = 3
  m.solve(disp = False)

  R      =   np.array([R[dim].value[-1] for dim in range(3)])
  volume = - m.options.OBJFCNVAL

  m.cleanup()
  del m
  return R, volume

# MAIN EXECUTION
if __name__ == "__main__":
  X = np.array([[ 0.0,  0.0,  0.6,],
                [ 0.8,  0.0,  0.6,],
                [ 0.0,  0.8,  0.6,],
                [-0.8,  0.0,  0.6,],
                [ 0.0, -0.8,  0.6,],
                [ 0.8,  0.0,  0.0,],
                [ 0.0,  0.8,  0.0,],
                [-0.8,  0.0,  0.0,],
                [ 0.0, -0.8,  0.0,],
                [ 0.8,  0.0, -0.6,],
                [ 0.0,  0.8, -0.6,],
                [-0.8,  0.0, -0.6,],
                [ 0.0, -0.8, -0.6,],
                [ 0.0,  0.0, -0.6,],])
  R1, V1 = minEnclosingEllipsoid(X)
  R2, V2 = maxInscribedEllipsoid(X)
  print('Minimum Enclosing Ellipsoid Radii  : ', R1)
  print('Minimum Enclosing Ellipsoid Volume : ', V1)
  print('Maximum Inscribed Ellipsoid Radii  : ', R2)
  print('Maximum Inscribed Ellipsoid Volume : ', V2)

  # Plot the Ellipsoid and the points
  from mpl_toolkits.mplot3d import Axes3D
  import matplotlib.pyplot as plt
  for R in [R1, R2]:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], marker = 'o', color = 'b')
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = R[0] * np.cos(u)*np.sin(v)
    y = R[1] * np.sin(u)*np.sin(v)
    z = R[2] * np.cos(v)
    ax.plot_wireframe(x, y, z, color = "r", alpha = 0.2)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.show()