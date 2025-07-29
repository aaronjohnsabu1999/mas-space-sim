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
import time
import numpy      as np
import queue
import threading
from   gekko      import GEKKO

import orbexa.params     as     p
from   orbexa.commons    import thread_worker
from   orbexa.dynamics   import orbital_ellp_undrag
from   orbexa.orbitsim   import adaptor_plot

if __name__ == "__main__":
  np.random.seed(int(time.time()))
  # Define Parameters
  ## Feasible Set Error Bound
  D = 0.05
  ## Data Range Parameters
  if True:
    data_range       = 181
    adaptation_range =  60
    rangeParams      = {'dt'               : p.dt,
                        'data_range'       : data_range,
                        'adaptation_range' : adaptation_range,}
  ## Orbital Parameters
  if True:
    eccentricity     = np.random.random()*0.60
    drag_alpha       = np.random.random()*5.00e-7
    drag_beta        = 2.600e-7
    x_0 =  np.array([100,   50, -80,
                      0,    0,   0])
    u_t = [np.array([ 90, - 40,   60])]
    for j in range(data_range-1):
      u_t.append(np.array([u_t[-1][i] + (np.random.random()*2.0-1.0)
                                        *np.exp((np.random.random()*2.0-1.0)
                                                *np.power(j, 1/4.0))
                          for i in range(3)]))
    u_t = np.transpose(u_t)
    if True:
      import matplotlib.pyplot as plt
      plt.figure()
      plt.plot(u_t[0], label = 'u_t[0]')
      plt.plot(u_t[1], label = 'u_t[1]')
      plt.plot(u_t[2], label = 'u_t[2]')
      plt.legend()
      plt.show()
    orbitParams      = {'eccentricity'     : eccentricity,
                        'drag_alpha'       : drag_alpha,
                        'drag_beta'        : drag_beta,
                        'x_0'              : x_0,
                        'u_t'              : u_t,}

# FUNCTION DEFINITIONS
def genAdaptorData(rangeParams, orbitParams, *args, **kwargs):
  W = []
  matrices, _, _ = orbital_ellp_undrag(dt           = rangeParams['dt'],
                                       n            = p.n,
                                       eccentricity = orbitParams['eccentricity'],
                                       alpha        = orbitParams['drag_alpha'],
                                       beta         = orbitParams['drag_beta'],)
  A_act, B_act, _, _, d_act = matrices
  
  m = GEKKO(remote = True)
  m.time = np.linspace(0, rangeParams['dt']*(rangeParams["data_range"]-1), rangeParams["data_range"])
  t =  m.Var(value = 0.0)
  x_act = [m.Var  (value = orbitParams['x_0'][i], fixed_initial =  True) for i in range(6)]
  u_act = [m.Param(value = orbitParams['u_t'][i])                        for i in range(3)]
  
  eqs = []
  eqs.append(t.dt() == 1.0)
  for i in range(3):
    eqs.append(x_act[i+0].dt() == x_act[i+3])
    eqs.append(x_act[i+3].dt() == np.matmul(A_act(t, p.t_p, m = m)[i+3], x_act)
                                          + u_act[i]
                                          + d_act(t, p.t_p, m = m)[i+3])
  eqs = m.Equations(eqs)
  m.options.IMODE      =   6
  m.options.SOLVER     =   1
  m.options.MAX_MEMORY = 512
  m.solve(disp = True, debug = 2)
  
  W = np.array([x_act[i].value for i in range(6)])
  W = np.transpose(W)

  m.cleanup()
  del m

  return W

def runAdaptorOp(operation, operIter, dt, t_s, u_t, W, D, pRange, *args, **kwargs):
  w_0 = W[ 0]
  w_f = W[-1]
  
  flag = False
  ## Initialize GEKKO Model
  m         = GEKKO(remote = True)
  m.time    = np.linspace(0, dt*(len(W)-1), len(W))
  ## Define Final Time Parameter
  final     = np.zeros(len(m.time))
  final[-1] = 1
  final     = m.Param (value = final)
  ## Define Time, State, Input Variables
  t     =  m.Var  (value = 0.0)
  x_est = [m.Var  (value = w_0[i], fixed_initial = True) for i in range(6)]
  u_act = [m.Param(value = u_t[i])                       for i in range(3)]
  ## Define Estimation Parameters
  p_est = []
  for paramIter in range(len(pRange)):
    p_est.append(m.FV(value = np.mean(pRange[paramIter]),
                  lb    = pRange[paramIter][0],
                  ub    = pRange[paramIter][1],))
    p_est[paramIter].STATUS = 1
  p_est[-1].STATUS = 0                     ## beta should be known beforehand
  ## Generate Dynamics Matrices
  matrices, _, _ = orbital_ellp_undrag(dt           = dt,
                                       n            = p.n,
                                       eccentricity = p_est[0],
                                       alpha        = p_est[1],
                                       beta         = p_est[2],
                                       m            = m)
  A_est, B_est, _, _, d_est = matrices
  ## Define Equations
  eqs = []
  eqs.append(t.dt() == 1.0)
  for i in range(3):
    eqs.append(x_est[i+0].dt() == x_est[i+3])
    eqs.append(x_est[i+3].dt() == np.matmul(A_est(t + t_s, p.t_p, m = m)[i+3], x_est)
                                          + u_act[i]
                                          + d_est(t + t_s, p.t_p, m = m)[i+3])
  if operation == 'FSS':
    eqs.append(final*(np.sum([(w_f[i] - x_est[i])**2 for i in range(6)]) - D**2) < 0)
    eqs.append(final*(np.sum([(w_f[i] - x_est[i])**2 for i in range(6)]) + D**2) > 0)
  elif operation == 'Optimal':
    eqs.append(final*(np.sum([(w_f[i] - x_est[i])**2 for i in range(6)])) < 4.0e-5)
  eqs = m.Equations(eqs)
  ## Define Optimization Parameters
  m.options.SOLVER   =   3
  m.options.IMODE    =   5
  m.options.MAX_TIME = 600
  ## Define Objective Function
  if   operation == 'FSS':
    m.options.MAX_ITER = 250
    p_est[operIter//2].value = pRange[operIter//2][operIter%2]
    if operIter % 2 == 0:
      m.Minimize(p_est[operIter//2]*final)
    else:
      m.Maximize(p_est[operIter//2]*final)
  elif operation == 'Optimal':
    m.options.MAX_ITER = 500
    m.Minimize(final*np.sum([(w_f[i] - x_est[i])**2 for i in range(6)]))
  ## Solve Optimization Problem
  try:
    m.solve(disp = kwargs['disp'], debug = 2)
  except Exception as e:
    print(e)
    if operation == 'Optimal':
      flag = True
  
  if operation == 'FSS':
    output = operIter, p_est[operIter//2].value[-1]
  elif operation == 'Optimal':
    if flag == True:
      output = flag, [0 for paramIter in range(len(pRange))]
    else:
      output = flag, [p_est[paramIter].value[-1] for paramIter in range(len(pRange))]
  
  m.cleanup()
  del m

  return output

def adaptor(initParams, rangeParams, u_t, D, W, *args, **kwargs):
  dt               = rangeParams['dt']
  data_range       = rangeParams['data_range']
  adaptation_range = rangeParams['adaptation_range']
  
  numParams   = len(initParams)
  estim_lists = [ []      for i in range(numParams)]
  range_lists = [[[], []] for i in range(numParams)]
  
  for paramIter in range(numParams):
    range_lists[paramIter][0] = [initParams[list(initParams.keys())[paramIter]][0]]
    range_lists[paramIter][1] = [initParams[list(initParams.keys())[paramIter]][1]]
  if 'estimates' in kwargs:
    for paramIter in range(numParams):
      estim_lists[paramIter] = [kwargs['estimates'][paramIter]]
  else:
    for paramIter in range(numParams):
      estim_lists[paramIter] = [np.mean(range_lists[paramIter])]
  pRange = [[range_lists[paramIter][0][-1],
             range_lists[paramIter][1][-1],] for paramIter in range(numParams)]
  
  threader = False
  if 'threader' in kwargs and kwargs['threader'] == True:
    threader = True
  
  iter = 0
  for dataIter in range(1, data_range):
    pEstim = [ estim_lists[paramIter]   [-1]   for paramIter in range(numParams)]
    pRange = [[range_lists[paramIter][0][-1],
               range_lists[paramIter][1][-1],] for paramIter in range(numParams)]
    if dataIter % adaptation_range == 0 and dataIter != 0:
      iter += 1
      # adaptation_range = min(int(adaptation_range*1.2), dataIter)
      operIter = 0
      est_results = []
      
      if threader:
        result_queue = queue.Queue()
        threads = []
      
      while operIter < 2*(len(pRange)-1):
        adaptorArgs = {'operation' :     'FSS',
                       'operIter'  :  operIter,
                       'dt'        :        dt,
                       't_s'       :        (dataIter - adaptation_range) * dt,
                       'u_t'       : [u_t[i][dataIter - adaptation_range : dataIter] for i in range(3)],
                       'W'         :    W   [dataIter - adaptation_range : dataIter],
                       'D'         :    D,
                       'pRange'    : pRange,
                       'disp'      :  kwargs['disp']}
        if threader:
          thread = threading.Thread(target = thread_worker,
                                    args   = (result_queue, runAdaptorOp),
                                    kwargs = adaptorArgs)
          thread.start()
          threads.append(thread)
        else:
          result = runAdaptorOp(**adaptorArgs)
          est_results.append(result)
        operIter += 1
      
      if threader:
        for thread in threads:
          thread.join()
        while not result_queue.empty():
          result = result_queue.get()
          est_results.append(result)

      pXi = [[0, 0] for paramIter in range(len(pRange))]
      pXi[len(pRange)-1] = pRange[len(pRange)-1].copy()
      for result in est_results:
        operIter, pXi_val = result
        pXi[operIter//2][operIter%2] = pXi_val
      for paramIter in range(len(pRange)):
        pRange[paramIter] = [max(min(pXi[paramIter]), pRange[paramIter][0]),
                             min(max(pXi[paramIter]), pRange[paramIter][1])]
        estimates = [np.mean(pRange[paramIter]) for paramIter in range(len(pRange))]
      
      flag, estimates = runAdaptorOp(operation = 'Optimal',
                                     operIter  =     1,
                                     dt        =    dt,
                                     t_s       =        (dataIter - adaptation_range) * dt,
                                     u_t       = [u_t[i][dataIter - adaptation_range : dataIter] for i in range(3)],
                                     W         =    W   [dataIter - adaptation_range : dataIter],
                                     D         =    D,
                                     pRange    =  pRange,
                                     disp      =  kwargs['disp'],)
      print("~~  Parameter Estimation : Iteration ", iter, "  ~~")
      print('  Adaptation Range : ', adaptation_range)
      if flag == False and estimates[0] not in pRange[0] and estimates[1] not in pRange[1]:
        pEstim = estimates
        print("!!! Parameter Estimation : Success !!!")
        for paramIter in range(numParams):
          print("  Working  pRange[", paramIter, "] : ", pRange[paramIter])
        for paramIter in range(numParams):
          print("  Working  pEstim[", paramIter, "] : ", pEstim[paramIter])
        print()
      else:
        print("!!! Parameter Estimation : Failure !!!")
        for paramIter in range(numParams):
          print("  Failed   pRange[", paramIter, "] : ", pRange[paramIter])
        print()
        pEstim = [ estim_lists[i]   [-1]  for i in range(numParams)]
        pRange = [[range_lists[i][0][-1],
                   range_lists[i][1][-1]] for i in range(numParams)]
        for paramIter in range(numParams):
          print("  Original pRange[", paramIter, "] : ", pRange[paramIter])
        for paramIter in range(numParams):
          print('  Original pEstim[', paramIter, '] : ', pEstim[paramIter])
        print()
    
    for paramIter in range(numParams):
      estim_lists[paramIter]   .append(pEstim[paramIter])
      range_lists[paramIter][0].append(pRange[paramIter][0])
      range_lists[paramIter][1].append(pRange[paramIter][1])
  
  FSS = {}
  for paramIter in range(numParams):
    FSS[list(initParams.keys())[paramIter]] = pRange[paramIter]
  return FSS, estim_lists, range_lists

# MAIN FUNCTION
if __name__ == "__main__":
  ## Generate Noisy Data
  W = genAdaptorData(rangeParams, orbitParams)
  ## Generate Adaptor
  startTime = time.time()
  FSS, estim_lists, range_lists = adaptor(p.initAdaptParams, rangeParams, orbitParams['u_t'], D, W,
                                          disp = True, threader = True)
  stopTime = time.time()
  ## Print Final Estimates
  if True:
    print('Estimation Time                  : ', stopTime - startTime)
    print('Total Number of Data Points      : ', rangeParams['data_range'])
    print('Number of Adaptation Data Points : ', rangeParams['adaptation_range'])
    print('Error Allowance                  : ', D)
    print()
    print("Actual Value   of Eccentricity   : ", orbitParams['eccentricity'])
    print("Final Estimate of Eccentricity   : ", estim_lists[0][-1])
    print("Actual Value   of Alpha          : ", orbitParams['drag_alpha'])
    print("Final Estimate of Alpha          : ", estim_lists[1][-1])
    print("Actual Value   of Beta           : ", orbitParams['drag_beta'])
    print("Final Estimate of Beta           : ", estim_lists[2][-1])
    print()
    print("Final Range of Eccentricity      : ", FSS['eccentricity'])
    print("Final Range of Alpha             : ", FSS['drag_alpha'])
    print("Final Range of Beta              : ", FSS['drag_beta'])
  
  adaptor_plot(estim_lists, range_lists, orbitParams, rangeParams, W,)