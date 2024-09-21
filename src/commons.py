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
import os
import sys
import json
import scipy
import datetime
import keyboard
import scipy.signal
import numpy             as     np
import matplotlib.pyplot as     plt
from   params            import *
np.random.seed(0)

# FUNCTION DEFINITIONS
def is_key_pressed(key):
  return keyboard.is_pressed(key)

def thread_worker(result_queue, func, *args, **kwargs):
  result = func(*args, **kwargs)
  result_queue.put(result)

def genInitState(numChasers, rX, rV, *args, **kwargs):
  x_0 = np.zeros((numChasers*6))
  for chaser in range(numChasers):
    phi, theta = np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi)
    x_0[chaser*6 + 0] = rX * np.cos(phi) * np.sin(theta)
    x_0[chaser*6 + 1] = rX * np.sin(phi) * np.sin(theta)
    x_0[chaser*6 + 2] = rX * np.cos(theta)
    x_0[chaser*6 + 3] = rV * np.cos(phi) * np.sin(theta)
    x_0[chaser*6 + 4] = rV * np.sin(phi) * np.sin(theta)
    x_0[chaser*6 + 5] = rV * np.cos(theta)
  return x_0

## Generate Skew Symmetric Matrix from Vector
def genSkewSymMat(val):
  try:
    return np.array([[    0.0, -val[2],  val[1]],
                      [ val[2],     0.0, -val[0]],
                      [-val[1],  val[0],     0.0],])
  except:
    return [[    0.0, -val[2],  val[1]],
            [ val[2],     0.0, -val[0]],
            [-val[1],  val[0],     0.0],]

## Generate Rotation Matrix from Tait-Bryan Angles
def tait_bryan_to_rotation_matrix(angles, *args, **kwargs):
    ### Extract individual angles ###
    alpha, beta, gamma = angles

    ### Compute sine and cosine values ###
    try:
      ca = np.cos(alpha)
      sa = np.sin(alpha)
      cb = np.cos(beta)
      sb = np.sin(beta)
      cg = np.cos(gamma)
      sg = np.sin(gamma)
    except:
      m = kwargs['m']
      ca = m.cos(alpha)
      sa = m.sin(alpha)
      cb = m.cos(beta)
      sb = m.sin(beta)
      cg = m.cos(gamma)
      sg = m.sin(gamma)

    ### Compute the rotation matrix ###
    rotation_matrix = [[               cb * cg,                -cb * sg,       sb],
                       [ca * sg + sa * sb * cg,  ca * cg - sa * sb * sg, -sa * cb],
                       [sa * sg - ca * sb * cg,  sa * cg + ca * sb * sg,  ca * cb],]
    try:
      return np.array(rotation_matrix)
    except:
      return rotation_matrix

## Calculate the Current Position of a Point on the Target based on its Initial Position, Time, and Inertial Parameters of the Target
def calcCurrentPos(target, x_i, t):
  rotMatrix = tait_bryan_to_rotation_matrix(target.getObservedState(t))
  x_t = np.dot(rotMatrix, x_i)
  x_t = np.append(x_t, [0.00, 0.00, 0.00])
  return x_t

## Convert Continuous State Space Model to Discrete State Space Model
def discretize(dt, A, B, *args):
  if len(args) == 0:
    try:
      C = np.zeros((1, A.shape[0]))
    except:
      A_val = A(0, 0)
      C = np.zeros((1, A_val.shape[0]))
    C[0][0] = 1
    D = np.zeros((B.shape[1], B.shape[1]))
  else:
    C = args[0]
    D = args[1]
  
  sys = scipy.signal.cont2discrete((A, B, C, D), dt)
  
  A_d = sys[0]
  B_d = sys[1]
  C_d = sys[2]
  D_d = sys[3]
  
  if len(args) == 0:
    return A_d, B_d
  else:
    return A_d, B_d, C_d, D_d

def calcLocalOcclusion(x, w, v, X):
  """
    Calculate the occlusion of the target as perceived by ego agent
    given a set of neighbor weights w and a set of neighbor states X.
  """
  obs = 0
  # Declustering
  for j in range(len(X)):
    obs += w[j] * np.linalg.norm(np.subtract(x, X[j]))
  # Bounding
  if   np.linalg.norm(x) < 9:
    obs -= v * (9 - np.linalg.norm(x))**2
  elif np.linalg.norm(x) > 11:
    obs -= v * (np.linalg.norm(x) - 11)**2
  # Normalization
  obs = - obs / (2*(len(X)**2))
  return obs

def calcGlobalOcclusion(X, W, V, X0, B):
  """
    Calculate the occlusion of the target as perceived by ego agent
    given a set of neighbor weights w and a set of neighbor states X.
  """
  obs = 0
  numAgents = int(len(X0)/3)
  for w in range(1, numAgents+1):
    x_w  = X[3*w-3:3*w]
    # Declustering
    for i in range(1, numAgents+1):
      if i != w:
        x_i  = X[3*i-3:3*i]
        obs -= W[i-1] * np.linalg.norm(np.subtract(x_w, x_i))
    # Travel Minimization
    x0_w = X0[3*w-3:3*w]
    obs += V[0] * np.linalg.norm(np.subtract(x_w, x0_w))
    # Bounding
    if   np.linalg.norm(x_w) < B[0]:
      obs += V[1] * (B[0] - np.linalg.norm(x_w))**2
    elif np.linalg.norm(x_w) > B[1]:
      obs += V[2] * (np.linalg.norm(x_w) - B[1])**2
  # Normalization
  obs = obs / (2*(numAgents**2))
  return obs

## Send Text to Standard Output
def streamprinter(text):
  sys.stdout.write(text)
  sys.stdout.flush()

## Calculate Distance between Two Points
def calcDistance(p1, p2):
  return np.linalg.norm(np.add(p1, -p2))

## Data Save
def saveData(fName, data):
  try:
    with open(fName, 'w', encoding='utf-8') as f:
      json.dump(data, f, ensure_ascii=False, indent=2)
    return 1
  except:
    return 0

## Data Load
def loadData(fName):
  data = None
  with open(fName, 'r', encoding='utf-8') as f:
    data = json.load(f)
  return data

def latestDataFile(folder):
  files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
  return folder+str(max(files))

def dataDeNumpyer(data):
  try:
    newData = []
    for datum in data:
      if str(type(datum)) == "<class 'dict'>":
        newData.append(datum)
      elif str(type(datum)) == "<class 'numpy.ndarray'>":
        newData.append(datum.tolist())
      else:
        newData.append(dataDeNumpyer(datum))
    return newData
  except:
    return data

def filenameCreator(folder, filetype):
  dtvar    = datetime.datetime.now()
  year     = str(dtvar.year)
  month    = str(dtvar.month)
  if len(month) == 1:
    month  = '0' + month
  day      = str(dtvar.day)
  if len(day) == 1:
    day    = '0' + day
  hour     = str(dtvar.hour)
  if len(hour) == 1:
    hour   = '0' + hour
  minute   = str(dtvar.minute)
  if len(minute) == 1:
    minute = '0' + minute
  second   = str(dtvar.second)
  if len(second) == 1:
    second = '0' + second
  date     = year + month  + day
  time     = hour + minute + second
  return folder + str(date) + '_' + str(time) + filetype

## Target Shape Constraints
def cylinderRadialUpperConstraint(r):
  return   np.sum([r[j]**2 for j in range(0, 2)]) - (targetLimit['r_T']*(1.00+0.001))**2

def cylinderRadialLowerConstraint(r):
  return - np.sum([r[j]**2 for j in range(0, 2)]) + (targetLimit['r_T']*(1.00-0.001))**2

def cylinderAxialUpperConstraint (r):
  return   np.sum([r[j]**2 for j in range(2, 3)]) - targetLimit['l_T']**2

def cylinderAxialLowerConstraint (r):
  return - np.sum([r[j]**2 for j in range(2, 3)]) - targetLimit['l_T']**2

## Return X, Y, Z grid data for a shape
def genShapeData(shape, shapeParams, numPoints = 100):
  if shape == 'cylinder':
    center, radius, height = shapeParams
    z_data = np.linspace(center[2] - height/2.0, center[2] + height/2.0, numPoints)
    theta_data = np.linspace(0, 2*np.pi, numPoints)
    theta, z = np.meshgrid(theta_data, z_data)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return x, y, z
  elif shape == 'sphere':
    center, radius = shapeParams
    return genShapeData('ellipsoid', (center, (radius, radius, radius)), numPoints = 100)
  elif shape == 'ellipsoid':
    center, radii = shapeParams
    u = np.linspace(0, 2 * np.pi, numPoints)
    v = np.linspace(0, np.pi, numPoints)
    x = center[0] + radii[0] * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radii[1] * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radii[2] * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z
  else:
    raise ValueError('Shape not recognized')

def genFibLattice(sphereParams, numPoints, **kwargs):
  sphere_center, sphere_radius = sphereParams
  if 'theta_0' not in kwargs.keys() and 'phi_0' not in kwargs.keys():
    theta_0, phi_0 = np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi)
  else:
    theta_0, phi_0 = kwargs['theta_0'], kwargs['phi_0']
  # Generate Fibonacci Lattice
  fib_lattice = np.zeros((numPoints, 3))
  goldenRatio = (1 + np.sqrt(5))/2

  for i in range(numPoints):
    theta = 2*np.pi*i/goldenRatio + theta_0
    phi = np.arccos(1 - (2*i + 1)/numPoints) + phi_0
    fib_lattice[i,0] = sphere_center[0] + sphere_radius*np.cos(theta)*np.sin(phi)
    fib_lattice[i,1] = sphere_center[1] + sphere_radius*np.sin(theta)*np.sin(phi)
    fib_lattice[i,2] = sphere_center[2] + sphere_radius*np.cos(phi)
  return fib_lattice

def pyramidalConstraint(x_0, x_f, mu):
  x_0 = np.array(x_0)
  x_f = np.array(x_f)
  mu_x, mu_y = mu['mu_x'], mu['mu_y']
  mu  = np.array([mu_x, mu_y, 0.0])
  mu__norm = np.linalg.norm(mu)
  x_0_norm = np.linalg.norm(x_0)
  x_f_norm = np.linalg.norm(x_f)      # should be 1.0 if the target location is on the unit sphere
  
  if x_f_norm == 0:
    raise ValueError('Final state must be non-zero')
  
  k   = (np.dot(x_0 - (x_f + mu), x_f)/x_f_norm**2) + 1
  x_m = k * x_f

  A      = x_0 - k * x_f
  B      = np.cross(k * x_f, A)
  A_norm = np.linalg.norm(A)
  B_norm = np.linalg.norm(B)
  X_vec  = (-A/A_norm + B/B_norm) * A_norm
  Y_vec  = (-A/A_norm - B/B_norm) * A_norm
  
  p = [x_0,
       x_0 + X_vec,
       x_0 + X_vec + Y_vec,
       x_0         + Y_vec]
  
  if mu__norm == 0:
    x_c = x_f.copy()
  else:
    x_c = x_f * (A - k * mu__norm) / (A - mu__norm)

  for i, p_i in enumerate(p):
    if not(all(p_i == 0.0)):
      p[i] = p_i * np.linalg.norm(p[0])/np.linalg.norm(p_i)
  
  def calcPlane(p_1, p_2, p_3):
    v_1 = p_2 - p_1
    v_2 = p_3 - p_1
    n   = np.cross(v_1, v_2)
    n   = n/np.linalg.norm(n)
    return np.array([n[0], n[1], n[2]]), np.dot(n, p_1)

  A_mat, B_mat = [], []
  for i in range(len(p)):
    A_i, B_i = calcPlane(x_c, p[i], p[(i+1)%len(p)])
    A_mat.append(A_i)
    B_mat.append(B_i)
  A_mat = np.vstack(A_mat)
  B_mat = np.array (B_mat)

  polarity = np.sign(np.dot(A_mat, x_m) - B_mat)
  for i, pol_i in enumerate(polarity):
    if pol_i == -1:
      pol_i = 1
      A_mat[i,:] = -A_mat[i,:]
      B_mat[i]   = -B_mat[i]
  return A_mat, B_mat, polarity

def trajopt_target(timeParams, orbitParams, solverParams, *args, **kwargs):
  ## Unpack Parameters ##
  t_s                      = timeParams["t_s"]
  timeSeq                  = timeParams["timeSeq"]
  numMPCSteps              = timeParams["numMPCSteps"]
  numActSteps              = timeParams["numActSteps"]
  eccentricity             = orbitParams['eccentricity']
  
  from gekko import GEKKO
  import params as p
  ## Initialize MPC ##
  m         = GEKKO(remote = solverParams["remote"])
  m.time    = timeSeq
  w         = np.ones (numMPCSteps)
  final     = np.zeros(numMPCSteps)
  final[-1] = 1
  target_thetas          = []

  ## Start Time Anomaly ##
  t_s = timeSeq[0]
  ## Final Time Anomaly ##
  t_f = timeSeq[-1]
  
  ## Initialize Variables ##
  if True:
    t     =  m.Var   (value =      0)
    q     =  m.Var   (value =      0, fixed_initial = False)
    W     =  m.Param (value =      w)
    final =  m.Param (value =  final)
  
  ## Constraint Equations ##
  eqs = []
  ### Time and Anomaly Update ###
  if True:
    eqs.append(t.dt() == 1)
    E = m.Intermediate (2*m.atan(np.sqrt((1-eccentricity)/(1+eccentricity))*m.tan(t/2)))
    M = m.Intermediate (E - eccentricity*m.sin(E))
    eqs.append(q      == p.t_p + t_s + M/p.n)
  ### Target Dynamics ###
  targetParams   = solverParams["targetParams"]
  target_theta_0 = targetParams["theta_0"]
  target_omega_0 = targetParams["omega_0"]
  momInertia     = targetParams["momInertia"]
  target_theta   = [m.Var(value = target_theta_0[i], fixed_initial = True) for i in range(len(target_theta_0))]
  target_omega   = [m.Var(value = target_omega_0[i], fixed_initial = True) for i in range(len(target_omega_0))]
  for i in range(len(target_theta)):
    eqs.append(target_theta[i].dt() == target_omega[i])
                                      #  (n*np.sqrt((1+eccentricity)/(1-eccentricity))*((m.cos(t/2)/m.cos(E/2))**2)/(1-eccentricity*m.cos(E))))
    eqs.append(target_omega[i].dt() == (np.matmul(np.matmul(np.linalg.inv(momInertia), genSkewSymMat(target_omega)),
                                                  np.matmul(              momInertia ,               target_omega))[i]))
                                      #  (n*np.sqrt((1+eccentricity)/(1-eccentricity))*((m.cos(t/2)/m.cos(E/2))**2)/(1-eccentricity*m.cos(E))))
  rotMatrix = m.Array(m.Var, (3,3), fixed_initial = False)
  for i in range(3):
    for j in range(3):
      eqs.append(rotMatrix[i][j] == tait_bryan_to_rotation_matrix(target_theta, m = m)[i][j])
  
  eqs = m.Equations(eqs)
  m.options.IMODE      =    6
  m.options.REDUCE     =    3
  m.options.SOLVER     =    3
  m.options.MAX_ITER   = 3000
  m.options.MAX_MEMORY =  512

  m.solve(disp = solverParams["disp"], debug = 2)

  ## Extract Solution ##
  rotMatrices = [[None for j in range(3)] for i in range(3)]
  if True:
    t = np.array(t.value)[:numActSteps]
    q = np.array(q.value)[:numActSteps]
    for i in range(3):
      for j in range(3):
        rotMatrices[i][j] = np.array(rotMatrix[i][j].value)[:numActSteps]
    rotMatrices = np.transpose(rotMatrices)

  return t, q, rotMatrices

# MAIN FUNCTIONS
## Main Function for Testing the genFibLattice Function
def main_genFibLattice():
  sphere_radius = 1.0
  sphere_center = (0.0, 0.0, 0.0)
  sphere_params = (sphere_center, sphere_radius)
  fib_lattice = genFibLattice(sphere_params, 8)
  
  # Plot the Fibonacci Lattice in three dimensions
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(fib_lattice[:,0], fib_lattice[:,1], fib_lattice[:,2])
  plt.show()

## Main Function for Testing the pyramidalConstraint Function
def main_pyramidalConstraint():
  x_0 = (1.0, 0.0, 2.0)
  x_f = (0.0, 0.0, 1.0)
  mu  = (0.0, 0.0)

  A, B, p = pyramidalConstraint(x_0, x_f, mu)
  print('A: ', A)
  print('B: ', B)
  print('p: ', p)

if __name__ == "__main__":
  main_pyramidalConstraint()