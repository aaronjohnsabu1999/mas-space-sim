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

import numpy as np
import scipy
from gekko import GEKKO
import matplotlib.pyplot as plt
from IPython.display import clear_output

np.random.seed(1)

# GLOBAL CONSTANTS
dt = 0.1
n = 0.00113
numStates = 6
numInputs = 3
numMPCSteps = 40
numActSteps = 8


## Convert Continuous State Space Model to Discrete State Space Model
def discretize(dt, A, B, *args):
    if len(args) == 0:
        C = np.zeros((1, A.shape[0]))
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


## Generate Cost as C=intg(x_iQx_i+u_iRu_i-(x_i-x_j)S(x_i-x_j))
def generateCost(x, u, costMatrices, dt):
    Q, R, S = costMatrices
    costVec = []
    cmlCVec = [0.0]
    numAgents = int(len(x) / 6)
    x = np.transpose(x)
    u = np.transpose(u)
    for iter in range(x):
        cost_iter = 0
        for agent_i in range(numAgents):
            cost_iter += np.multiply(
                np.add(
                    np.atleast_1d(
                        np.matmul(
                            np.atleast_1d(
                                np.matmul(x[iter][agent_i * 6 : (agent_i + 1) * 6], Q)
                            ),
                            x[iter][agent_i * 6 : (agent_i + 1) * 6],
                        )
                    ),
                    np.atleast_1d(
                        np.matmul(
                            np.atleast_1d(
                                np.matmul(u[iter][agent_i * 6 : (agent_i + 1) * 6], R)
                            ),
                            u[iter][agent_i * 6 : (agent_i + 1) * 6],
                        )
                    ),
                ),
                dt,
            )
            for agent_j in range(agent_i, numAgents):
                cost_iter -= np.multiply(
                    np.atleast_1d(
                        np.matmul(
                            np.atleast_1d(
                                np.matmul(
                                    x[iter][agent_i * 6 : (agent_i + 1) * 6]
                                    - x[iter][agent_j * 6 : (agent_j + 1) * 6],
                                    S,
                                )
                            ),
                            x[iter][agent_i * 6 : (agent_i + 1) * 6]
                            - x[iter][agent_j * 6 : (agent_j + 1) * 6],
                        )
                    ),
                    dt,
                )
        costVec.append(cost_iter)
        cmlCVec.append(float(cmlCVec[-1] + cost_iter))
    cmlCVec = cmlCVec[1:]
    totCost = np.sum(costVec)
    return totCost, cmlCVec


def mpc(system, solverParams, *args, **kwargs):
    dt = solverParams["dt"]
    t_0 = solverParams["t_0"]
    u_0 = solverParams["u_0"]
    x_0 = solverParams["x_0"]
    x_f = solverParams["x_f"]
    rLen = solverParams["rLen"]
    fLen = solverParams["fLen"]
    bounds = solverParams["bounds"]
    x_f_rad = solverParams["x_f_rad"]
    numAgents = solverParams["numAgents"]
    mpcIterMax = solverParams["mpcIterMax"]
    numChasers = solverParams["numChasers"]
    numActSteps = solverParams["numActSteps"]
    numMPCSteps = solverParams["numMPCSteps"]

    t_f = t_0 + numMPCSteps * dt

    Q = np.zeros((6, 6))
    # Q[3,3] = 1.00
    # Q[4,4] = 1.00
    # Q[5,5] = 1.00
    R = np.eye(3) * 0.00
    S = np.zeros((6, 6))
    S[0, 0] = 10.00
    S[1, 1] = 10.00
    S[2, 2] = 10.00
    ipData = system(dt=dt, Q=Q, R=R, constraints=(x_0, x_f), discretize=False)
    matrices, constraints, _ = ipData
    A, B, Q, R, d = matrices
    x_0, x_f = constraints

    stateBounds, inputBounds = bounds

    timeSeq = np.linspace(0, numMPCSteps * dt, numMPCSteps)
    w = np.ones(numMPCSteps)
    states = []
    inputs = []
    X_f = []

    initDist = np.linalg.norm(np.subtract(x_0, x_f))
    mpcIter = 0
    sigma_FS = 100.0

    tryIter = 0
    tryIterMax = 3
    init_x_f_rad = x_f_rad
    #### MPC LOOP ####
    while True:
        ##### Exit Conditions #####
        print(
            "Distance from start state to target      : ",
            np.linalg.norm((x_0 - x_f)[:3]),
        )
        if np.linalg.norm((x_0 - x_f)[:3]) < init_x_f_rad:
            print("Target Reached at Iteration ", mpcIter)
            break

        if is_key_pressed("q"):
            print("MPC Manually Terminated at Iteration ", mpcIter)
            break

        mpcIter += 1
        if mpcIter > mpcIterMax:
            print("MPC Iteration Limit reached at Iteration ", mpcIter)
            break

        ##### Update Target #####
        t_f = t_0 + mpcIter * numMPCSteps * dt
        if "x_f_callable" in solverParams.keys():
            x_f = f_x_f(numCompletedSteps=t_f / dt, t_0=t_0, t=t_f)
            if mpcIter % 4 == 0:
                x_f_rad /= 2.0
                print("Updating epsilon to ", x_f_rad)
        X_f.append(x_f)

        ##### Print Iteration Info #####
        print()
        print("MPC Iteration    : ", mpcIter)
        print("Initial  Time    : ", t_0)
        print("Start    Time    : ", t_0 + (mpcIter - 1) * numMPCSteps * dt)
        print("Final    Time    : ", t_f)
        print("Initial Position : ", x_0)
        print("Final   Position : ", x_f)

        ##### Initialize MPC #####
        m = GEKKO(remote=solverParams["remote"])
        m.time = timeSeq
        final = np.zeros(len(m.time))
        for i in range(len(m.time)):
            if m.time[i] < (numMPCSteps - 1) * dt:
                final[i] = 0
            else:
                final[i] = 1

        ##### Initialize Variables #####
        t = m.Var(value=t_0 + (mpcIter - 1) * numMPCSteps * dt)
        x = [m.Var(value=x_0[i], fixed_initial=True) for i in range(len(x_0))]
        u = [m.Var(value=u_0[i], fixed_initial=False) for i in range(len(u_0))]
        W = m.Param(value=w)
        final = m.Param(value=final)

        ##### Initialize Bounds #####
        for i in range(len(x)):
            if stateBounds[i]["lower"] != "-Inf":
                x[i].lower = stateBounds[i]["lower"]
            if stateBounds[i]["upper"] != "+Inf":
                x[i].upper = stateBounds[i]["upper"]
        for i in range(len(u)):
            if inputBounds[i]["lower"] != "-Inf":
                u[i].lower = inputBounds[i]["lower"]
            if inputBounds[i]["upper"] != "+Inf":
                u[i].upper = inputBounds[i]["upper"]

        ##### Constraint Equations #####
        eqs = []
        ###### Time Update ######
        eqs.append(t.dt() == dt)
        ###### System Dynamics ######
        for i in range(len(x)):
            eqs.append(
                x[i].dt()
                == np.matmul((A(t.value))[i], x) + np.matmul(np.atleast_1d(B[i]), u)
            )
        ###### Radial Limit Constraint ######
        if "radialLimit" in solverParams:
            radialLimit = solverParams["radialLimit"]
            eqs.append(np.sum([x[i] ** 2 for i in range(3)]) > radialLimit**2)
        ###### Target Limit Constraint ######
        if "targetLimit" in solverParams:
            target = solverParams["target"]
            targetLimit = solverParams["targetLimit"]
            ## TODO: Need to verify if time change happens for these lines of code
            ##########
            rotMatrix = scipy.spatial.transform.Rotation.from_euler(
                "xyz", target.getObservedState(t=t.value), degrees=False
            ).as_matrix()
            x_derot = np.matmul(rotMatrix.T, x[:3])
            eqs.append(
                m.max2(
                    x_derot[0] ** 2 + x_derot[1] ** 2 - targetLimit["r_T"] ** 2,
                    x_derot[2] ** 2 - targetLimit["l_T"] ** 2,
                )
                >= 0
            )
            ##########
        ###### Pyramidal Limit Constraint ######
        if "pyramidalLimit" in solverParams:
            mu = solverParams["pyramidalLimit"]
            if mu != None:
                A_PL, B_PL, pol_PL = pyramidalConstraint(x_0[:3], x_f[:3], mu)
                if not (all(pol_PL)):
                    raise ValueError("Pyramidal limit polarity should be all True")
                for i in range(len(A_PL)):
                    eqs.append(np.matmul(A_PL[i], x[:3]) - B_PL[i] > 0)

        ##### Tube MPC Implementation #####
        if "tubeMPC" in solverParams:
            tubeMPC = solverParams["tubeMPC"]
            alpha_0 = tubeMPC["alpha_0"]
            omega_0 = tubeMPC["omega_0"]
            phi_0 = tubeMPC["phi_0"]
            eta = tubeMPC["eta"]
            Lambda = tubeMPC["Lambda"]
            D = tubeMPC["D"]
            v = tubeMPC["v"]
            n_omega = 2
            A_C = np.array(
                [
                    [0 for j in range(i + 1)]
                    + [
                        1,
                    ]
                    + [0 for j in range(n_omega - i - 3)]
                    for i in range(n_omega - 2)
                ]
            )
            try:
                A_C = np.append(
                    A_C,
                    np.array(
                        [
                            [
                                -math.comb(n_omega - 1, i)
                                * (Lambda ** (n_omega - i - 1))
                                for i in range(n_omega - 1)
                            ]
                        ]
                    ),
                    axis=0,
                )
            except:
                A_C = np.append(
                    A_C,
                    np.array(
                        [
                            [
                                -math.comb(n_omega - 1, i)
                                * (Lambda ** (n_omega - i - 1))
                                for i in range(n_omega - 1)
                            ]
                        ]
                    ),
                )
            B_C = np.array(
                [
                    [0 for i in range(n_omega - 2)]
                    + [
                        1,
                    ]
                ]
            ).T
            ###### Initialize Variables ######
            alpha = m.Var(value=alpha_0, lb=0.0)
            omega = [
                m.Var(value=omega_0[i], fixed_initial=True) for i in range(len(omega_0))
            ]
            phi = [m.Var(value=phi_0[i], fixed_initial=True) for i in range(len(phi_0))]
            Delta = m.Var(value=0.0)
            ###### Initialize Bounds ######
            alpha.lower = tubeMPC["alpha_range"][0]
            alpha.upper = tubeMPC["alpha_range"][1]
            ###### Tube Equations ######
            eqs.append(Delta == 0)
            eqs.append(alpha.dt() == v)
            for i in range(len(phi)):
                eqs.append(phi[i].dt() == -alpha * phi[i] + Delta + D + eta)
            for i in range(len(omega)):
                eqs.append(omega[i].dt() == np.matmul(A_C, omega) + np.matmul(B_C, phi))

        ##### Objective Function Definition #####
        errorArr = [np.matmul(x - x_f, Q @ (x - x_f)), np.matmul(u, R @ u)]
        errorArr.append(
            sigma_FS
            * m.max2(0, (np.sum([(x[i] - x_f[i]) ** 2 for i in range(3)]) - x_f_rad**2))
        )
        error = np.sum(errorArr)

        ##### Solver Parameters #####
        eqs = m.Equations(eqs)
        m.Minimize(error * W)
        m.options.IMODE = 8
        m.options.SOLVER = 3
        m.options.MAX_ITER = 3000
        m.options.COLDSTART = 0
        m.options.MAX_MEMORY = 512

        ##### Solve MPC #####
        startTime = time.time()
        try:
            m.solve(disp=solverParams["disp"])
        except:
            print("MPC Failed - Try ", tryIter)
            X_f.pop()
            if tryIter < tryIterMax:
                print("Retrying")
                mpcIter -= 1
                tryIter += 1
                continue
            else:
                break
        stopTime = time.time()

        ##### Print MPC Info #####
        print("MPC Time         : ", stopTime - startTime)
        print("MPC Objective    : ", m.options.objfcnval)
        print("MPC Status       : ", m.options.APPSTATUS)
        print("MPC Solver Time  : ", m.options.SOLVETIME)
        print()

        ##### Result Formatting and Truncation #####
        x_p = [x[i].value for i in range(len(x))]
        u_p = [u[i].value for i in range(len(u))]
        x_p = np.transpose(x_p)
        u_p = np.transpose(u_p)
        for i in range(numActSteps - 1):
            states.append(x_p[i])
            inputs.append(u_p[i])

        ##### Check for Progress #####
        print(
            "Distance from start state to final state : ",
            np.linalg.norm(x_0 - x_p[numActSteps - 1]),
        )
        if np.linalg.norm(x_0 - x_p[numActSteps - 1]) < initDist * 1e-3:
            print("No Progress in State - Try ", tryIter)
            if tryIter < tryIterMax:
                print("Retrying")
                mpcIter -= 1
                tryIter += 1
            else:
                break

        ##### Update Initial Conditions #####
        x_0 = x_p[numActSteps - 1]
        u_0 = u_p[numActSteps - 1]

        ##### Clear Loop #####
        print()
        m.clear()
        del m

    return states, inputs, X_f


def estimator(*args):
    return args[0]


if __name__ == "__main__":
    # System Matrices
    A = np.array(
        [
            [0.00, 0.00, 0.00, 1.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
            [3 * (n**2), 0.00, 0.00, 0.00, 2 * n, 0.00],
            [0.00, 0.00, 0.00, -2 * n, 0.00, 0.00],
            [0.00, 0.00, -(n**2), 0.00, 0.00, 0.00],
        ]
    )
    B = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [1.00, 0.00, 0.00],
            [0.00, 1.00, 0.00],
            [0.00, 0.00, 1.00],
        ]
    )
    A, B = discretize(1, A, B)

    # Optimization Matrices
    Q = np.array(
        [
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 1.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
        ]
    )
    R = np.array([[10.00, 0.00, 0.00], [0.00, 10.00, 0.00], [0.00, 0.00, 10.00]])

    # State Constraints
    x_0 = np.array([20.00, 10.00, -10.00, 15.00, 25.00, 50.00])
    x_f = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00])

    print(A)
    print(B)
    print(Q)
    print(R)
    print(x_0)
    print(x_f)

    x_0_step = x_0.copy()
    states = [[x_0[i]] for i in range(numStates)]
    inputs = [[] for i in range(numInputs)]
    epsilon = 0.001
    while np.linalg.norm(np.subtract(x_0_step, x_f)) > epsilon:
        x_f = estimator(x_f)
        step_states, step_inputs = trajopt(
            numStates, numInputs, numMPCSteps, A, B, Q, R, x_0_step, x_f
        )
        # step_states, step_inputs = trajopt(numStates, numInputs, numMPCSteps, A, B, Q, R, x_0_step, x_f, radialLimit = 1.0, sigma_RL = 5000)
        clear_output(wait=False)
        for i in range(len(step_states)):
            step_states[i] = step_states[i][:numActSteps]
            states[i] = np.append(states[i], step_states[i][1:])
        for i in range(len(step_inputs)):
            step_inputs[i] = step_inputs[i][:numActSteps]
            inputs[i] = np.append(inputs[i], step_inputs[i][1:])
        x_0_step = np.array([states[i][-1] for i in range(numStates)])

    # Plot results.
    fig, ax = plt.subplots(4, 3)
    plt.rcParams["figure.figsize"] = (18, 12)

    for i in range(3):
        plt.subplot(4, 3, i + 1)
        u_i = inputs[i]
        plt.plot(u_i)
        label = "$(u_t)_{}$".format(i)
        plt.ylabel(label, fontsize=16)
        plt.yticks(
            np.linspace(
                int(np.min(u_i)),
                int(np.max(u_i)),
                min(12, max(3, int((np.max(u_i) - np.min(u_i) + 1) / 5))),
            )
        )
        # plt.xticks([])

    for i in range(6):
        plt.subplot(4, 3, i + 4)
        x_i = states[i]
        plt.plot(x_i)
        label = "$(x_t)_{}$".format(i)
        plt.ylabel(label, fontsize=16)
        plt.yticks(
            np.linspace(
                int(np.min(x_i)),
                int(np.max(x_i)),
                min(12, max(3, int((np.max(x_i) - np.min(x_i) + 1) / 5))),
            )
        )
        # plt.xticks([])

    plt.subplot(4, 3, 10)
    x_n = [
        np.linalg.norm([states[j][i] for j in range(numStates)])
        for i in range(len(states[0]))
    ]
    plt.plot(x_n)
    label = "$||x_t||$"
    plt.ylabel(label, fontsize=16)
    plt.yticks(
        np.linspace(
            int(np.min(x_n)),
            int(np.max(x_n)),
            min(12, max(3, int((np.max(x_n) - np.min(x_n) + 1) / 5))),
        )
    )

    plt.tight_layout()
    plt.show()

    print([states[i][-1] for i in range(3)])
