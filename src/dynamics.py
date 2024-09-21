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
import numpy as np
from commons import discretize, genSkewSymMat
from gekko import GEKKO

## Return Common Parameters for Orbital Dynamics
def orbitalParams(dt, A, B, *args, **kwargs):
  [numStates, numInputs] = B.shape
  
  if 'discretize' not in kwargs.keys():
    A, B = discretize(dt, A, B)
  elif kwargs['discretize']:
    A, B = discretize(dt, A, B)
  
  ### Objective Function
  if 'Q' not in kwargs.keys():
    Q = np.zeros((6,6))
  else:
    Q = kwargs['Q']
  if 'R' not in kwargs.keys():
    R = np.array([[1.00, 0.00, 0.00],
                  [0.00, 1.00, 0.00],
                  [0.00, 0.00, 1.00]])
  else:
    R = kwargs['R']
  
  matrices = (A, B, Q, R)
  if 'C' in kwargs.keys() and 'D' in kwargs.keys():
    C = kwargs['C']
    D = kwargs['D']
    matrices = (A, B, C, D, Q, R)
  
  ### Initial and Final States
  if 'constraints' not in kwargs.keys():
    x_0 = np.array([ 1.000, 0.000, 0.000,
                     0.000, 0.000, 0.000])
    x_f = np.array([ 0.000, 0.000, 0.000,
                     0.000, 0.000, 0.000])
    constraints = (x_0, x_f)
  else:
    constraints = kwargs['constraints']
  
  ### Bounds
  if 'bounds' not in kwargs.keys():
    stateBounds  = [{"lower": "-Inf", "upper": "+Inf"}
                             for i in range(numStates)]
    inputBounds  = [{"lower": "-Inf", "upper": "+Inf"}
                             for i in range(numInputs)]
    bounds = (stateBounds, inputBounds)
  else:
    bounds = kwargs['bounds']
  
  return matrices, constraints, bounds

## Return Dynamics for a chaser in circular orbit with unequal drag
def orbital_ellp_undrag(dt, n = 0.00113, eccentricity = 0.00, alpha = 0.00, beta = 0.00, h = 51999.48422922679, *args, **kwargs):   # 51999.48422922679 = (398600.4418**2/0.00113)**(1.0/3.0)
  mu = 3.986004418e+14
  # mu = 398600.4418
  gamma = beta - alpha
  R_0 = (6378.137 + 500.00) * 1000.00
  try:
    h = np.sqrt(mu*R_0*(1 + eccentricity)/(1 + 4*(alpha**2)))
  except:
    m = kwargs['m']
    h =  m.sqrt(mu*R_0*(1 + eccentricity)/(1 + 4*(alpha**2)))
  ## System Matrices ##
  ### State Matrix A ###
  def A(th, th_p = 0.0, *args, **kwargs):
    try:
      exp_theta = np.exp(alpha*th)
      cos_theta = np.cos(th - th_p)
      sin_theta = np.sin(th - th_p)
    except:
      m = kwargs['m']
      exp_theta = m.exp(alpha*th)
      cos_theta = m.cos(th - th_p)
      sin_theta = m.sin(th - th_p)
    R     = ((h**2.00) * (1 + 4.00*(alpha**2.00)) / mu) *                                                  1.00 /  (exp_theta**2 + eccentricity*cos_theta)
    R_dot = ((h**2.00) * (1 + 4.00*(alpha**2.00)) / mu) * (-2.00*alpha*(exp_theta**2) + eccentricity*sin_theta) / ((exp_theta**2 + eccentricity*cos_theta)**2.00)
    A = [[ 0.00,  0.00,  0.00,  1.00,  0.00,  0.00],
         [ 0.00,  0.00,  0.00,  0.00,  1.00,  0.00],
         [ 0.00,  0.00,  0.00,  0.00,  0.00,  1.00],
         [ 0.00,  0.00,  0.00,  0.00,  2.00,  0.00],
         [ 0.00,  0.00,  0.00, -2.00,  0.00,  0.00],
         [ 0.00,  0.00, -1.00,  0.00,  0.00,  0.00],]
    gamma = beta - alpha
    A[3][0]  = -gamma*R_dot/R
    A[3][1]  =  gamma
    A[3][3]  = -gamma
    A[4][0]  = -gamma
    A[4][1]  = -gamma*R_dot/R + 3.00*R*(exp_theta**2)*mu/(h**2)
    A[4][4]  = -gamma
    A[5][2] += -gamma*R_dot/R
    A[5][5]  = -gamma
    return A
  ### Input Matrix B ###
  B = np.array([[0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00],
                [1.00, 0.00, 0.00],
                [0.00, 1.00, 0.00],
                [0.00, 0.00, 1.00],])
  ### Disturbance Matrix d ###
  def d(th, th_p = 0, *args, **kwargs):
    try:
      exp_theta = np.exp(alpha*th)
      cos_theta = np.cos(th - th_p)
      sin_theta = np.sin(th - th_p)
    except:
      m = kwargs['m']
      exp_theta = m.exp(alpha*th)
      cos_theta = m.cos(th - th_p)
      sin_theta = m.sin(th - th_p)
    try:
      h_sqrt = np.sqrt(h)
    except:
      m = kwargs['m']
      h_sqrt =  m.sqrt(h)
    R     = ((h**2.00) * (1 + 4.00*(alpha**2.00)) / mu) *                                                  1.00 /  (exp_theta**2 + eccentricity*cos_theta)
    R_dot = ((h**2.00) * (1 + 4.00*(alpha**2.00)) / mu) * (-2.00*alpha*(exp_theta**2) + eccentricity*sin_theta) / ((exp_theta**2 + eccentricity*cos_theta)**2.00)
    d  = np.array([0.00, 0.00, 0.00, exp_theta, exp_theta, 0.00,])
    d *= gamma * R / h_sqrt
    # d  = d * exp_theta
    d[3] *=  R
    d[4] *= -R_dot
    return d
  
  kwargs['discretize'] = False
  matrices, constraints, bounds = orbitalParams(dt, A, B, *args, **kwargs)
  matrices += (d,)
  return matrices, constraints, bounds

## Return Dynamics for a chaser in elliptical orbit with equal drag
def orbital_ellp_eqdrag(dt, n = 0.00113, eccentricity = 0.00, alpha = 0.00, *args, **kwargs):
  return orbital_ellp_undrag(dt, n, eccentricity, alpha, alpha, *args, **kwargs)

## Return Dynamics for a chaser in circular orbit with unequal drag
def orbital_circ_undrag(dt, n = 0.00113, alpha = 0.00, beta = 0.00, h = 51999.48422922679, *args, **kwargs):   # 51999.48422922679 = (398600.4418**2/0.00113)**(1.0/3.0)
  return orbital_ellp_undrag(dt, n, 0.00, alpha, beta, h, *args, **kwargs)

## Return Dynamics for a chaser in circular orbit with equal drag
def orbital_circ_eqdrag(dt, n = 0.00113, alpha = 0.00, *args, **kwargs):
  return orbital_circ_undrag(dt, n, alpha, alpha, *args, **kwargs)

## Return Dynamics for a chaser in circular dragless orbit
def cwhEquations(dt, n = 0.00113, *args, **kwargs):
  return orbital_circ_eqdrag(dt, n, 0.00, *args, **kwargs)

## Return Dynamics for a system that is rotating according to the Newton-Euler Equations
def newtonEulerAngSys(dt, n = 0.00113, *args, **kwargs):
  rLen = kwargs["rLen"]
  fLen = kwargs["fLen"]
  numChasers = kwargs["numChasers"] 
  ### System Functions and Matrices
  A = lambda x, J: np.matmul(np.matmul(np.linalg.inv(J), genSkewSymMat(x)),
                             np.matmul(              J,                x))
  B = lambda r, f, J:        np.matmul(np.linalg.inv(J), np.cross(r, f))
  if 'Q' not in kwargs.keys():
    Q = np.array([[ 1.00,  0.00,  0.00,  0.00,  0.00,  0.00],
                  [ 0.00,  1.00,  0.00,  0.00,  0.00,  0.00],
                  [ 0.00,  0.00,  1.00,  0.00,  0.00,  0.00],
                  [ 0.00,  0.00,  0.00,  1.00,  0.00,  0.00],
                  [ 0.00,  0.00,  0.00,  0.00,  1.00,  0.00],
                  [ 0.00,  0.00,  0.00,  0.00,  0.00,  1.00]])
  else:
    Q = kwargs['Q']
  if 'R' not in kwargs.keys():
    R = np.array([[ 1.00,  0.00,  0.00],
                  [ 0.00,  1.00,  0.00],
                  [ 0.00,  0.00,  1.00]])
  else:
    R = kwargs['R']
  
  funcMatrs = (A, B, Q, R)
  if 'C' in kwargs.keys() and 'D' in kwargs.keys():
    C = kwargs['C']
    D = kwargs['D']
    funcMatrs = (A, B, C, D, Q, R)
  
  ### Initial and Final States
  if 'constraints' not in kwargs.keys():
    x_0 = np.array([1.00, 0.00, 0.00, 0.00, 0.00, 0.00])
    x_f = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
    constraints = (x_0, x_f)
  else:
    constraints = kwargs['constraints']

  ### Bounds
  if 'bounds' not in kwargs.keys():
    forceBounds = [{"lower": "-Inf",   "upper": "+Inf"}
                    for i in range(fLen*numChasers)]
    bounds = (forceBounds)
  else:
    bounds = kwargs['bounds']
  
  return funcMatrs, constraints, bounds

## Return Dynamics for a Triple Integrator system
def tripleIntegrator(dt, *args, **kwargs):
  ### System Matrices
  A = np.array([[0.00, 1.00, 0.00],
                [0.00, 0.00, 1.00],
                [0.00, 0.00, 0.00],])
  B = np.array([[0.00,],
                [0.00,],
                [1.00,],])
  if 'discretize' not in kwargs.keys():
    A, B = discretize(dt, A, B)
  else:
    if kwargs['discretize']:
      A, B = discretize(dt, A, B)
  ### Objective Function
  if 'Q' not in kwargs.keys():
    Q = np.array([[1.00, 0.00, 0.00],
                  [0.00, 1.00, 0.00],
                  [0.00, 0.00, 1.00],])
  else:
    Q = kwargs['Q']
  if 'R' not in kwargs.keys():
    R = np.array([[1]])
  else:
    R = kwargs['R']
  
  matrices = (A, B, Q, R)
  if 'C' in kwargs.keys() and 'D' in kwargs.keys():
    C = kwargs['C']
    D = kwargs['D']
    matrices = (A, B, C, D, Q, R)
  
  [numStates, numInputs] = B.shape
  
  ### Initial and Final States
  if 'constraints' not in kwargs.keys():
    x_0 = np.array([10.0, 0.0, 0.0])
    x_f = np.array([ 0.0, 0.0, 0.0])
    constraints = (x_0, x_f)
  else:
    constraints = kwargs['constraints']
  
  ### Bounds
  if 'bounds' not in kwargs.keys():
    stateBounds  = [{"lower": "-Inf", "upper": "+Inf"}
                             for i in range(numStates)]
    inputBounds  = [{"lower": "-Inf", "upper": "+Inf"}
                             for i in range(numInputs)]
    bounds = (stateBounds, inputBounds)
  else:
    bounds = kwargs['bounds']
  
  return matrices, constraints, bounds