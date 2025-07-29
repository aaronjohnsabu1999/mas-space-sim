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
import numpy    as     np
from   copy     import copy
from   gekko    import GEKKO

import orbexa.params   as     p
from   orbexa.dynamics import orbital_ellp_undrag

def ancillary_controller(t_p, t_f,
                         nom_state, act_state,
                         A_nom_val, Lambda, alpha, phi,
                         *args, **kwargs):
  m = kwargs["m"]
  
  r_tilde = np.array([(act_state[i] - nom_state[i]) for i in range(len((nom_state)))])
  x_tilde = r_tilde[:3]
  v_tilde = r_tilde[3:]
  
  s = [(v_tilde[i] + Lambda[i]*x_tilde[i]) for i in range(len(x_tilde))]
  min_s_phi = [(s[i]/m.max2(s[i], phi[i])) for i in range(len(s))]
  
  K  = calcDelta(t_f, t_p, r_tilde, m = m,
                 eRange = kwargs["eRange"],
                 aRange = kwargs["aRange"],
                 bRange = kwargs["bRange"],)
  K += np.array([alpha[i]*phi[i] for i in range(3)])
  
  state_mod = np.array([A_nom_val[i+3] @ r_tilde    -
                           Lambda[i]   * v_tilde[i] - 
                        min_s_phi[i]   *       K[i]
                        for i in range(3)])
  return state_mod

def calcDelta(t, t_p, x, *args, **kwargs):
  m = kwargs["m"]
  dt = p.dt
  if 'eRange' in kwargs:
    minEccentricity, maxEccentricity = kwargs["eRange"]
  else:
    minEccentricity, maxEccentricity = p.minEccentricity, p.maxEccentricity
  if 'aRange' in kwargs:
    minDragAlpha,    maxDragAlpha    = kwargs["aRange"]
  else:
    minDragAlpha,    maxDragAlpha    = p.minDragAlpha,    p.maxDragAlpha
  if 'bRange' in kwargs:
    minDragBeta,     maxDragBeta     = kwargs["bRange"]
  else:
    minDragBeta,     maxDragBeta     = p.minDragBeta,     p.maxDragBeta
  A_list, Delta_list, Delta_norm = [], [], []
  for eccentricity in [minEccentricity, maxEccentricity]:
    for dragAlpha  in [minDragAlpha,    maxDragAlpha]:
      for dragBeta in [minDragBeta,     maxDragBeta]:
        matrices, _, _ = orbital_ellp_undrag(dt,
                                             eccentricity = eccentricity,
                                             alpha = dragAlpha,
                                             beta = dragBeta)
        A, _, _, _, _  = matrices
        A_list.append(np.array(A(t, t_p, m = m)))
  for i, A_i in enumerate(A_list):
    for A_j in A_list[i+1:]:
      Delta = A_i - A_j
      Delta_list.append(Delta)
      Delta_norm.append(np.linalg.norm(Delta))
  Delta = Delta_list[np.argmax(Delta_norm)]
  Delta = np.matmul(Delta, x)
  return Delta[3:]

def calcD(t, t_p, *args, **kwargs):
  try:
    m = kwargs["m"]
  except:
    m = []
  dt = p.dt
  if 'eRange' in kwargs:
    minEccentricity, maxEccentricity = kwargs["eRange"]
  else:
    minEccentricity, maxEccentricity = p.minEccentricity, p.maxEccentricity
  if 'aRange' in kwargs:
    minDragAlpha,    maxDragAlpha    = kwargs["aRange"]
  else:
    minDragAlpha,    maxDragAlpha    = p.minDragAlpha,    p.maxDragAlpha
  if 'bRange' in kwargs:
    minDragBeta,     maxDragBeta     = kwargs["bRange"]
  else:
    minDragBeta,     maxDragBeta     = p.minDragBeta,     p.maxDragBeta
  d_list, D_list, D_norm = [], [], []
  for eccentricity in [minEccentricity, maxEccentricity]:
    for dragAlpha  in [minDragAlpha,    maxDragAlpha]:
      for dragBeta in [minDragBeta,     maxDragBeta]:
        matrices, _, _ = orbital_ellp_undrag(dt, eccentricity = eccentricity, alpha = dragAlpha, beta = dragBeta)
        _, _, _, _, d  = matrices
        d_list.append(d(t, t_p, m = m))
  for i, d_i in enumerate(d_list):
    for d_j in d_list[i+1:]:
      D = d_i - d_j
      D_list.append(D)
      D_norm.append(np.linalg.norm(D))
  D = D_list[np.argmax(D_norm)]
  return D[3:]