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
import numpy as np
from copy import copy, deepcopy
from gekko import GEKKO
from functools import partial

import orbexa.params as p
from orbexa.adaptor import adaptor, adaptor_plot
from orbexa.utils import (
    is_key_pressed,
    pyramidalConstraint,
    genSkewSymMat,
    tait_bryan_to_rotation_matrix,
    calcCurrentPos,
    get_next_test_folder,
    save_test_result,
)
from orbexa.dynamics import orbital_ellp_undrag
from orbexa.orbitsim import mpc_plot
from orbexa.deflection import targetDeflect, deflection_plot
from orbexa.spacecraft import Target
from orbexa.dynamictube import ancillary_controller, calcDelta, calcD


def trajopt_mpc(
    timeParams,
    nom_matrices,
    act_matrices,
    bounds,
    solverParams,
    numChasers=1,
    *args,
    **kwargs,
):
    ## Unpack Parameters ##
    t_s = timeParams["t_s"]
    timeSeq = timeParams["timeSeq"]
    numMPCSteps = timeParams["numMPCSteps"]
    numActSteps = timeParams["numActSteps"]
    X_0 = solverParams["X_0"]
    X_f = solverParams["X_f"]
    U_0 = solverParams["U_0"]
    stateBounds, inputBounds = bounds
    eccentricity = nomOrbitParams["eccentricity"]
    A_nom, B_nom, Q_nom, R_nom, d_nom = nom_matrices
    A_act, B_act, d_act = act_matrices

    ## Initialize MPC ##
    m = GEKKO(remote=solverParams["remote"])
    m.time = timeSeq
    w = np.ones(numMPCSteps)
    final = np.zeros(numMPCSteps)
    final[-1] = 1
    nom_states, nom_inputs = [], []
    act_states, act_inputs = [], []
    target_thetas = []

    ## Start Time Anomaly ##
    t_s = timeSeq[0]
    ## Final Time Anomaly ##
    t_f = timeSeq[-1]

    ## Initialize Variables ##
    t = m.Var(value=0)
    q = m.Var(value=0, fixed_initial=False)
    X_nom = [m.Var(value=X_0[i], fixed_initial=True) for i in range(len(X_0))]
    X_act = [m.Var(value=X_0[i], fixed_initial=True) for i in range(len(X_0))]
    U_nom = [m.Var(value=U_0[i], fixed_initial=False) for i in range(len(U_0))]
    U_act = [m.Var(value=U_0[i], fixed_initial=False) for i in range(len(U_0))]
    W = m.Param(value=w)
    final = m.Param(value=final)

    A_nom_val = A_nom(t + t_s, p.t_p, m=m)
    d_nom_val = d_nom(t + t_s, p.t_p, m=m)

    ## Constraint Equations Initialization ##
    eqs = []
    ## Objective Function Initialization ##
    intErrorArr = []
    finErrorArr = []

    ### Time and Anomaly Update ###
    eqs.append(t.dt() == 1)
    E = m.Intermediate(
        2 * m.atan(np.sqrt((1 - eccentricity) / (1 + eccentricity)) * m.tan(t / 2))
    )
    M = m.Intermediate(E - eccentricity * m.sin(E))
    eqs.append(q == p.t_p + t_s + M / p.n)

    ### Target Dynamics ###
    if "targetParams" in solverParams:
        targetParams = solverParams["targetParams"]
        target_theta_0 = targetParams["theta_0"]
        target_omega_0 = targetParams["omega_0"]
        momInertia = targetParams["momInertia"]
        target_theta = [
            m.Var(value=target_theta_0[i], fixed_initial=True)
            for i in range(len(target_theta_0))
        ]
        target_omega = [
            m.Var(value=target_omega_0[i], fixed_initial=True)
            for i in range(len(target_omega_0))
        ]
        for i in range(len(target_theta)):
            eqs.append(target_theta[i].dt() == target_omega[i])
            #  (n*np.sqrt((1+eccentricity)/(1-eccentricity))*((m.cos(t/2)/m.cos(E/2))**2)/(1-eccentricity*m.cos(E))))
            eqs.append(
                target_omega[i].dt()
                == (
                    np.matmul(
                        np.matmul(
                            np.linalg.inv(momInertia), genSkewSymMat(target_omega)
                        ),
                        np.matmul(momInertia, target_omega),
                    )[i]
                )
            )
            #  (n*np.sqrt((1+eccentricity)/(1-eccentricity))*((m.cos(t/2)/m.cos(E/2))**2)/(1-eccentricity*m.cos(E))))

    ### Individual Chaser Equations ###
    for agent in range(numChasers):
        #### Individual Chaser Variables ####
        x_nom = X_nom[agent * 6 : (agent + 1) * 6]
        x_act = X_act[agent * 6 : (agent + 1) * 6]
        u_nom = U_nom[agent * 3 : (agent + 1) * 3]
        u_act = U_act[agent * 3 : (agent + 1) * 3]
        x_0 = X_0[agent * 6 : (agent + 1) * 6]
        x_f = X_f[agent * 6 : (agent + 1) * 6]
        u_0 = U_0[agent * 3 : (agent + 1) * 3]
        #### Nominal System Dynamics ####
        for i in range(0, 3):
            eqs.append(x_nom[i + 0].dt() == x_nom[i + 3])
            eqs.append(
                x_nom[i + 3].dt()
                == np.matmul(A_nom_val[i + 3], x_nom) + u_nom[i] + d_nom_val[i + 3]
            )
        ### Actual System Dynamics ###
        for i in range(0, 3):
            eqs.append(x_act[i + 0].dt() == x_act[i + 3])
            eqs.append(
                x_act[i + 3].dt()
                == np.matmul((A_act(t + t_s, p.t_p, m=m))[i + 3], x_act)
                + u_act[i]
                + d_act(t + t_s, p.t_p, m=m)[i + 3]
            )
        #### System Error ####
        if "tubeMPC" in solverParams and solverParams["tubeMPC"]["runTube"]:
            r_tilde = [1.10 * (x_act[i] - x_nom[i]) for i in range(6)]
            r_tilde_bound = [m.abs2(r_tilde[i]) for i in range(6)]
            p_tilde_norm = m.Intermediate(
                m.sqrt(np.sum([r_tilde[i] ** 2 for i in range(0, 3)]))
            )
            xy_tilde_norm = m.Intermediate(
                m.sqrt(np.sum([r_tilde[i] ** 2 for i in range(0, 2)]))
            )
            z_tilde_norm = m.Intermediate(
                m.sqrt(np.sum([r_tilde[i] ** 2 for i in range(2, 3)]))
            )
            v_tilde_norm = m.Intermediate(
                m.sqrt(np.sum([r_tilde[i] ** 2 for i in range(3, 6)]))
            )
            r_tilde_norm = m.Intermediate(
                m.sqrt(np.sum([r_tilde[i] ** 2 for i in range(0, 6)]))
            )
        else:
            r_tilde = [0.00 for i in range(6)]
            r_tilde_bound = [0.00 for i in range(6)]
            p_tilde_norm = 0.00
            xy_tilde_norm = 0.00
            z_tilde_norm = 0.00
            v_tilde_norm = 0.00
            r_tilde_norm = 0.00
        ### Scaling of Final State ###
        if np.linalg.norm(x_f) > 0.00:
            x_f_scaled = [m.Var() for i in range(len(x_f))]
            for i in range(len(x_f)):
                eqs.append(x_f_scaled[i] == x_f[i] * (1 + 2.0 * p_tilde_norm))
        else:
            x_f_scaled = x_f.copy()
        ## Tube MPC Implementation ##
        if "tubeMPC" in solverParams:
            tubeMPC = solverParams["tubeMPC"]
            Lambda = tubeMPC["Lambda"]
            alpha_0 = tubeMPC["alpha_0"]
            omega_0 = tubeMPC["omega_0"]
            phi_0 = tubeMPC["phi_0"]
            v_0 = tubeMPC["v_0"]
            eta = tubeMPC["eta"]
            ### Initialize Variables ###  # len(v_0) = len(alpha_0) = len(omega_0) = len(phi_0) = 3
            v = [m.Var(value=v_0[i], fixed_initial=False) for i in range(3)]
            alpha = [
                m.Var(
                    value=alpha_0[i],
                    fixed_initial=False,
                    lb=tubeMPC["alpha_range"]["lower"],
                    ub=tubeMPC["alpha_range"]["upper"],
                )
                for i in range(3)
            ]
            omega = [m.Var(value=omega_0[i], fixed_initial=True) for i in range(3)]
            phi = [m.Var(value=phi_0[i], fixed_initial=True) for i in range(3)]
            ### Tube Equations ###
            Delta_nom = calcDelta(
                t_f + t_s,
                p.t_p,
                x_nom,
                m=m,
                eRange=kwargs["eRange"],
                aRange=kwargs["aRange"],
                bRange=kwargs["bRange"],
            )
            D_nom = calcD(
                t_f + t_s,
                p.t_p,
                m=m,
                eRange=kwargs["eRange"],
                aRange=kwargs["aRange"],
                bRange=kwargs["bRange"],
            )
            for i in range(3):
                eqs.append(alpha[i].dt() == v[i])
                eqs.append(omega[i].dt() == Lambda[i] * omega[i] + phi[i])
                eqs.append(
                    phi[i].dt() == -alpha[i] * phi[i] + Delta_nom[i] + D_nom[i] + eta[i]
                )
        ### Radial Limit Constraint (RENDEZVOUS ONLY) ###
        if "radialLimit" in solverParams:
            radialLimit = solverParams["radialLimit"]
            eqs.append(
                np.sum([x_nom[i] ** 2 for i in range(3)])
                >= (radialLimit + p_tilde_norm) ** 2
            )
        ### Target Limit Constraint (DOCKING ONLY) ###
        if "targetLimit" in solverParams:
            targetLimit = solverParams["targetLimit"]
            rotMatrix = m.Array(m.Var, (3, 3), fixed_initial=False)
            x_derot = m.Array(m.Var, 3, fixed_initial=False)
            for i in range(3):
                for j in range(3):
                    eqs.append(
                        rotMatrix[i][j]
                        == tait_bryan_to_rotation_matrix(target_theta, m=m)[i][j]
                    )
            for i in range(3):
                eqs.append(x_derot[i] == np.matmul(rotMatrix.T, x_nom[:3])[i])
            targetLimit1 = m.Intermediate(
                x_derot[0] ** 2
                + x_derot[1] ** 2
                - (targetLimit["r_T"] + p_tilde_norm) ** 2
            )
            targetLimit2 = m.Intermediate(
                x_derot[2] ** 2 - (targetLimit["l_T"] + p_tilde_norm) ** 2
            )
            targetLimitMultiplier = 1.00e3
            targetLimit12 = m.Var(fixed_initial=False)
            orFunctionOption = 5
            if orFunctionOption == 1:
                targetLimit12 = m.Intermediate(
                    targetLimit1
                    * targetLimit2
                    * (
                        (targetLimit1 + targetLimit2)
                        - (m.abs2(targetLimit1) + m.abs2(targetLimit2))
                    )
                )
                eqs.append(targetLimit12 >= 0.00)
            elif orFunctionOption == 2:
                targetLimit12 = m.Intermediate(
                    m.max2(
                        targetLimitMultiplier * targetLimit1,
                        targetLimitMultiplier * targetLimit2,
                    )
                )
                eqs.append(targetLimit12 > 1.00e-2)
            elif orFunctionOption == 3:
                targetLimit12 = m.Intermediate(
                    m.if2(
                        targetLimitMultiplier * targetLimit1 * targetLimit2,
                        1,
                        targetLimitMultiplier * (targetLimit1 + targetLimit2),
                    )
                )
                eqs.append(targetLimit12 > 1.00e-2)
            elif orFunctionOption == 4:
                targetLimit12 = m.Intermediate(
                    m.if2(
                        targetLimitMultiplier * targetLimit1,
                        targetLimitMultiplier * targetLimit2,
                        1,
                    )
                )
                eqs.append(targetLimit12 > 1.00e-2)
            elif orFunctionOption == 5:
                targetLimit12 = m.Intermediate(
                    m.sign2(targetLimitMultiplier * targetLimit1 - 1.00e-7)
                    + m.sign2(targetLimitMultiplier * targetLimit2 - 1.00e-7)
                )
                eqs.append(targetLimit12 >= 0.00)
            elif orFunctionOption == 6:
                targetLimit12 = m.Intermediate(
                    m.exp(
                        targetLimitMultiplier
                        * targetLimit1
                        * targetLimit2
                        * (
                            (targetLimit1 + targetLimit2)
                            - (m.abs2(targetLimit1) + m.abs2(targetLimit2))
                        )
                    )
                )
                eqs.append(targetLimit12 >= 1.00)
            elif orFunctionOption == 7:
                targetLimit12 = m.Intermediate(
                    m.max2(
                        targetLimitMultiplier * targetLimit1,
                        targetLimitMultiplier * targetLimit2,
                    )
                    - m.abs2(
                        m.max2(
                            targetLimitMultiplier * targetLimit1,
                            targetLimitMultiplier * targetLimit2,
                        )
                    )
                )
                eqs.append(targetLimit12 == 0.00)
            elif orFunctionOption == 8:
                # targetLimit12 = m.Intermediate(m.max2(targetLimitMultiplier * targetLimit1, targetLimitMultiplier * targetLimit2))
                eqs.append(
                    targetLimit12
                    == m.max2(
                        targetLimitMultiplier * targetLimit1,
                        targetLimitMultiplier * targetLimit2,
                    )
                )
                eqs.append(m.abs2(targetLimit12 - m.abs2(targetLimit12)) < 1.00e-9)
            # eqs.append(targetLimit12**2 >= 1.00e-5 * np.sum([(x_nom[i] - x_f_scaled[i])**2 for i in range(0,3)]))
            # eqs.append(targetLimit12.dt() <= 0.00)
        ### Pyramidal Limit Constraint (DOCKING ONLY) ###
        if "pyramidalLimit" in solverParams:
            mu_PL = solverParams["pyramidalLimit"]
            if mu_PL != None:
                A_PL, B_PL, pol_PL = pyramidalConstraint(x_0[:3], x_f[:3], mu_PL)
                if not (all(pol_PL)):
                    raise ValueError("Pyramidal limit polarity should be all True")
                for i in range(len(A_PL)):
                    eqs.append(np.matmul(A_PL[i], x_nom[:3]) - B_PL[i] > 0)
        ### Actual Input Modification based on Tube Dynamics ###
        if "tubeMPC" in solverParams:
            a_ctrl = ancillary_controller(
                p.t_p,
                t_f + t_s,
                x_nom,
                x_act,
                A_nom_val,
                Lambda,
                alpha,
                phi,
                m=m,
                eRange=kwargs["eRange"],
                aRange=kwargs["aRange"],
                bRange=kwargs["bRange"],
            )
            if solverParams["tubeMPC"]["runTube"]:
                for i in range(len(u_nom)):
                    eqs.append(u_act[i] == u_nom[i] + a_ctrl[i])
            else:
                for i in range(len(u_nom)):
                    eqs.append(u_act[i] == u_nom[i])
        else:
            for i in range(len(u_nom)):
                eqs.append(u_act[i] == u_nom[i])

        ### State and Input Bounds ###
        #### TubeMPC Bound Tightening ####
        if "tubeMPC" in solverParams:
            r_bar = [x_nom[i] - r_tilde_bound[i] for i in range(6)]
            r_Bar = [x_nom[i] + r_tilde_bound[i] for i in range(6)]
            F_Bar = [
                m.max2(
                    m.abs2(np.matmul(A_nom_val[i + 3], r_bar)),
                    m.abs2(np.matmul(A_nom_val[i + 3], r_Bar)),
                )
                for i in range(3)
            ]
            Delta_Bar = calcDelta(
                t_f + t_s,
                p.t_p,
                r_Bar,
                m=m,
                eRange=kwargs["eRange"],
                aRange=kwargs["aRange"],
                bRange=kwargs["bRange"],
            )
            Delta_bar = calcDelta(
                t_f + t_s,
                p.t_p,
                r_bar,
                m=m,
                eRange=kwargs["eRange"],
                aRange=kwargs["aRange"],
                bRange=kwargs["bRange"],
            )
            K_Bar = [
                alpha[i] * phi[i] + m.max2(Delta_Bar[i], Delta_bar[i]) for i in range(3)
            ]
            u_tilde_bound = [
                m.abs2(F_Bar[i] + K_Bar[i] + Lambda[i] * r_tilde_bound[i + 3])
                for i in range(3)
            ]
        else:
            u_tilde_bound = [0.00 for i in range(3)]
        #### Bound Equations ####
        for i in range(0, 3):
            if stateBounds[0]["upper"] != "+Inf":
                eqs.append(
                    x_nom[i + 0] > stateBounds[i + 0]["lower"] + r_tilde_bound[i + 0]
                )
                eqs.append(
                    x_nom[i + 0] < stateBounds[i + 0]["upper"] - r_tilde_bound[i + 0]
                )
            if stateBounds[3]["upper"] != "+Inf":
                eqs.append(
                    x_nom[i + 3] > stateBounds[i + 3]["lower"] + r_tilde_bound[i + 3]
                )
                eqs.append(
                    x_nom[i + 3] < stateBounds[i + 3]["upper"] - r_tilde_bound[i + 3]
                )
            if inputBounds[0]["upper"] != "+Inf":
                eqs.append(
                    u_nom[i + 0] > inputBounds[i + 0]["lower"] + u_tilde_bound[i + 0]
                )
                eqs.append(
                    u_nom[i + 0] < inputBounds[i + 0]["upper"] - u_tilde_bound[i + 0]
                )

        #### Objective Function Definition ####
        intErrorArr.append(np.sum([R_nom[i][i] * (u_nom[i] ** 2) for i in range(0, 3)]))
        intErrorArr.append(
            np.sum(
                [Q_nom[i][i] * ((x_nom[i] - x_f_scaled[i]) ** 2) for i in range(0, 6)]
            )
        )
        try:
            intErrorArr.append(
                -(
                    m.min2(
                        0,
                        targetLimit12**2
                        - 1.00e-3
                        * np.sum(
                            [(x_nom[i] - x_f_scaled[i]) ** 2 for i in range(0, 3)]
                        ),
                    )
                )
            )
            # intErrorArr.append(-1.00e-3*targetLimit12)
        except:
            pass
        if solverParams["sigma_xFS"] > 0.00:
            finErrorArr.append(
                solverParams["sigma_xFS"]
                * (
                    m.max2(
                        0,
                        (
                            np.sum(
                                [(x_nom[i] - x_f_scaled[i]) ** 2 for i in range(0, 3)]
                            )
                            - solverParams["x_f_rad"] ** 2
                        ),
                    )
                )
            )
        if solverParams["sigma_vFS"] > 0.00:
            finErrorArr.append(
                solverParams["sigma_vFS"]
                * (
                    m.max2(
                        0,
                        (
                            np.sum([(x_nom[i] - x_f[i]) ** 2 for i in range(3, 6)])
                            - solverParams["v_f_rad"] ** 2
                        ),
                    )
                )
            )

    ### Objective Function Accumulation ###
    intError = np.sum(intErrorArr)
    finError = np.sum(finErrorArr)

    ## Solver Parameters ##
    eqs = m.Equations(eqs)
    m.Minimize(W * intError + final * finError)
    # m.options.OTOL       =    1e-7
    # m.options.RTOL       =    1e-7
    m.options.IMODE = 6
    # m.options.REDUCE     =    3
    m.options.SOLVER = 3
    m.options.MAX_ITER = 4000
    # m.options.DIAGLEVEL  =    0
    m.options.MAX_MEMORY = 768
    # m.options.COLDSTART  =    0
    # m.options.TIME_SHIFT =    0

    ## Solve MPC ##
    startTime = time.time()
    try:
        m.solve(disp=solverParams["disp"], debug=2)
    except:
        m.cleanup()
        del m
        return 1, [], [], [], [], []
    stopTime = time.time()

    ## Print MPC Info ##
    if True:
        print("MPC Time         : ", stopTime - startTime)
        print("MPC Objective    : ", m.options.objfcnval)
        print("MPC Status       : ", m.options.APPSTATUS)
        print("MPC Solver Time  : ", m.options.SOLVETIME)
        print()

    ## Result Formatting and Truncation ##
    if True:
        xn_t = np.transpose([x_nom[i].value for i in range(len(x_nom))])
        xa_t = np.transpose([x_act[i].value for i in range(len(x_act))])
        un_t = np.transpose([u_nom[i].value for i in range(len(u_nom))])
        ua_t = np.transpose([u_act[i].value for i in range(len(u_act))])
        if "targetParams" in solverParams:
            th_t = np.transpose(
                [target_theta[i].value for i in range(len(target_theta))]
            )
            target_thetas.extend(th_t[1 : numActSteps + 1])
        else:
            target_thetas.extend([np.array([0, 0, 0]) for i in range(numActSteps)])
        nom_states.extend(xn_t[1 : numActSteps + 1])
        act_states.extend(xa_t[1 : numActSteps + 1])
        nom_inputs.extend(un_t[1 : numActSteps + 1])
        act_inputs.extend(ua_t[1 : numActSteps + 1])

        #### Print Final Position ####
        if True:
            print("Target  State                : ", x_f)
            print("Nominal State at numMPCSteps : ", xn_t[-1])
            print("Nominal State at numActSteps : ", nom_states[-1])
            print("Actual  State at numActSteps : ", act_states[-1])
            print(
                "Closest State is at Iteration ",
                np.argmin(
                    [
                        np.linalg.norm(x_f[:3] - act_states[i][:3])
                        for i in range(len(act_states))
                    ]
                ),
            )
            print(
                "Distance from Supposed Final State to Target = ",
                np.linalg.norm(x_f[:3] - xn_t[-1][:3]),
            )
            print(
                "Distance from Nominal  Final State to Target = ",
                np.linalg.norm(x_f[:3] - nom_states[-1][:3]),
            )
            print(
                "Distance from Actual   Final State to Target = ",
                np.linalg.norm(x_f[:3] - act_states[-1][:3]),
            )
            print(
                "Distance from Closest  Final State to Target = ",
                np.min(
                    [
                        np.linalg.norm(x_f[:3] - act_states[i][:3])
                        for i in range(len(act_states))
                    ]
                ),
            )
            try:
                print(
                    "Maximum Error in Positional State            = ",
                    max(p_tilde_norm.value),
                )
                print(
                    "Maximum Error in Radial     State            = ",
                    max(xy_tilde_norm.value),
                )
                print(
                    "Maximum Error in Altitude   State            = ",
                    max(z_tilde_norm.value),
                )
                print(
                    "Maximum Error in Velocity   State            = ",
                    max(v_tilde_norm.value),
                )
                print(
                    "Maximum Error in Total      State            = ",
                    max(r_tilde_norm.value),
                )
            except:
                pass
            if "targetLimit" in solverParams:
                for i in range(1, numActSteps + 1):
                    try:
                        p_tilde_norm_t = p_tilde_norm.value[i]
                    except:
                        p_tilde_norm_t = p_tilde_norm
                    targetLimit1_t_solv = targetLimit1.value[i]
                    targetLimit2_t_solv = targetLimit2.value[i]
                    targetLimit1_t_calc = (
                        x_derot[0].value[i] ** 2
                        + x_derot[1].value[i] ** 2
                        - (targetLimit["r_T"] + p_tilde_norm_t) ** 2
                    )
                    targetLimit2_t_calc = (
                        x_derot[2].value[i] ** 2
                        - (targetLimit["l_T"] + p_tilde_norm_t) ** 2
                    )
                    targetLimit1_t_true = (
                        x_derot[0].value[i] ** 2
                        + x_derot[1].value[i] ** 2
                        - (targetLimit["r_T"]) ** 2
                    )
                    targetLimit2_t_true = (
                        x_derot[2].value[i] ** 2 - (targetLimit["l_T"]) ** 2
                    )
                    targetLimit12_t_solv = targetLimit12.value[i]
                    targetLimit12_t_calc = max(
                        targetLimitMultiplier * targetLimit1_t_calc,
                        targetLimitMultiplier * targetLimit2_t_calc,
                    )
                    targetLimit12_t_true = max(
                        targetLimitMultiplier * targetLimit1_t_true,
                        targetLimitMultiplier * targetLimit2_t_true,
                    )
                    distFromTarget_t = np.linalg.norm(x_f[:3] - xa_t[i][:3])
                    if (
                        targetLimit1_t_solv < 0
                        and targetLimit2_t_solv < 0
                        or targetLimit1_t_calc < 0
                        and targetLimit2_t_calc < 0
                        or targetLimit1_t_true < 0
                        and targetLimit2_t_true < 0
                    ):
                        print("Iteration ", i, " violates true targetLimit constraint")
                        print("p_tilde_norm_t       = ", p_tilde_norm_t)
                        print("targetLimit1_t_calc  = ", targetLimit1_t_calc)
                        print("targetLimit2_t_calc  = ", targetLimit2_t_calc)
                        print("targetLimit1_t_true  = ", targetLimit1_t_true)
                        print("targetLimit2_t_true  = ", targetLimit2_t_true)
                        print("targetLimit1_t_solv  = ", targetLimit1_t_solv)
                        print("targetLimit2_t_solv  = ", targetLimit2_t_solv)
                        print("targetLimit12_t_calc = ", targetLimit12_t_calc)
                        print("targetLimit12_t_true = ", targetLimit12_t_true)
                        print("targetLimit12_t_solv = ", targetLimit12_t_solv)
                        print("distFromTarget_t     = ", distFromTarget_t)
                        print(
                            "targetLimit1_t_solv * targetLimit2_t_solv = ",
                            targetLimit1_t_solv * targetLimit2_t_solv,
                        )
                        print(
                            "targetLimit1_t_solv + targetLimit2_t_solv = ",
                            targetLimit1_t_solv + targetLimit2_t_solv,
                        )
                        print(
                            "targetLimit1_t_calc * targetLimit2_t_calc = ",
                            targetLimit1_t_calc * targetLimit2_t_calc,
                        )
                        print(
                            "targetLimit1_t_calc + targetLimit2_t_calc = ",
                            targetLimit1_t_calc + targetLimit2_t_calc,
                        )
                        print(
                            "targetLimit1_t_true * targetLimit2_t_true = ",
                            targetLimit1_t_true * targetLimit2_t_true,
                        )
                        print(
                            "targetLimit1_t_true + targetLimit2_t_true = ",
                            targetLimit1_t_true + targetLimit2_t_true,
                        )
                        print()
            print()

    ## Clear Loop ##
    m.cleanup()
    del m

    ## Return Results ##
    return 0, nom_states, act_states, nom_inputs, act_inputs, target_thetas


def mpc(
    operation,
    dt,
    t_0,
    rLen,
    fLen,
    numChasers,
    numMPCSteps,
    numActSteps,
    X_0,
    f_X_f,
    x_f_rad,
    v_f_rad,
    U_0,
    actOrbitParams,
    nomOrbitParams,
    bounds,
    initAdaptParams,
    *args,
    **kwargs,
):
    ## Define Generic Solver Parameters ##
    if operation == "rendezvous":
        mpcMaxIter = 100
    elif operation == "docking":
        mpcMaxIter = 300
    solverParams = {
        "remote": True,
        "disp": True,
        "rLen": rLen,
        "fLen": fLen,
        "dt": dt,
        "X_0": X_0,
        "x_f_rad": x_f_rad,
        "v_f_rad": v_f_rad,
        "U_0": U_0,
        "bounds": bounds,
        "numChasers": numChasers,
        "numMPCSteps": numMPCSteps,
        "numActSteps": numActSteps,
        "mpcMaxIter": mpcMaxIter,
    }

    ## Define Mission-Specific Solver Parameters ##
    if "targetParams" in kwargs:
        solverParams["targetParams"] = kwargs["targetParams"]
    if "radialLimit" in kwargs:
        solverParams["radialLimit"] = kwargs["radialLimit"]
    if "targetLimit" in kwargs:
        solverParams["targetLimit"] = kwargs["targetLimit"]
    if "pyramidalLimit" in kwargs:
        solverParams["pyramidalLimit"] = kwargs["pyramidalLimit"]

    ## Define Cost Matrices ##
    Q = np.zeros((6, 6))
    R = np.zeros((3, 3))
    if operation == "rendezvous":
        sigma_xFS = 1.00e9
        sigma_vFS = 0.00e0
        for i in range(3):
            Q[i + 0][i + 0] = 1.00e4
            Q[i + 3][i + 3] = 6.00e3
            R[i + 0][i + 0] = 1.00e0
    elif operation == "docking":
        sigma_xFS = 0.00e0
        sigma_vFS = 0.00e0
        for i in range(3):
            Q[i + 0][i + 0] = 8.00e3
            Q[i + 3][i + 3] = 1.00e1
            R[i + 0][i + 0] = 1.00e-3
    solverParams["sigma_xFS"] = sigma_xFS
    solverParams["sigma_vFS"] = sigma_vFS

    ## Define Tube MPC Solver Parameters ##
    if "tubeMPC" in kwargs:
        solverParams["tubeMPC"] = kwargs["tubeMPC"]

    ## Define Final State for First Action Iteration ##
    t_s = t_0
    t_f = t_0 + (numActSteps - 1) * dt
    X_f = []
    if callable(f_X_f) or "list" in str(type(f_X_f)) and callable(f_X_f[0]):
        try:
            for agent in range(numChasers):
                X_f.extend(f_X_f[agent](t=t_f))
        except:
            X_f = f_X_f
    else:
        X_f = f_X_f
    X_f = np.array(X_f)
    print(X_f)

    ## Define Nominal System ##
    nom_system = partial(
        orbital_ellp_undrag,
        eccentricity=nomOrbitParams["eccentricity"],
        alpha=nomOrbitParams["drag_alpha"],
        beta=nomOrbitParams["drag_beta"],
    )
    nom_matrices, _, _ = nom_system(
        dt=dt, Q=Q, R=R, constraints=(X_0, X_f), discretize=False
    )
    A_nom, B_nom, Q_nom, R_nom, d_nom = nom_matrices

    ## Define Actual System ##
    act_system = partial(
        orbital_ellp_undrag,
        eccentricity=actOrbitParams["eccentricity"],
        alpha=actOrbitParams["drag_alpha"],
        beta=actOrbitParams["drag_beta"],
    )
    act_matrices, _, _ = act_system(dt=dt, discretize=False)
    A_act, B_act, _, _, d_act = act_matrices

    ## Define Time and Iteration Parameters ##
    mpcIter = 0
    tryIter = 0
    tryIterMax = 2
    init_x_f_rad = x_f_rad

    ## Initialize Return Variables ##
    mpc_x_f_list = []
    mpc_nom_states, mpc_nom_inputs = [], []
    mpc_act_states, mpc_act_inputs = [], []
    mpc_target_thetas = []
    mpc_FSS = initAdaptParams.copy()
    mpc_estim_lists = [[] for i in range(3)]
    mpc_range_lists = [[[], []] for i in range(3)]

    ## BEGIN MPC Loop ##
    while True:
        ##### Print Existing Parameter Estimates #####
        if True:
            print()
            print("Nominal Value of Eccentricity = ", nomOrbitParams["eccentricity"])
            print("Nominal Value of Alpha        = ", nomOrbitParams["drag_alpha"])
            print("Nominal Value of Beta         = ", nomOrbitParams["drag_beta"])
            print()

        ##### Exit Conditions #####
        print([(X_0[6 * agent : 6 * agent + 3]) for agent in range(numChasers)])
        print([(X_f[6 * agent : 6 * agent + 3]) for agent in range(numChasers)])
        print(
            "Distances from Start States to Targets   : ",
            [
                np.linalg.norm(
                    (X_0[6 * agent : 6 * agent + 3] - X_f[6 * agent : 6 * agent + 3])
                )
                for agent in range(numChasers)
            ],
        )
        if all(
            [
                np.linalg.norm(
                    (X_0[6 * agent : 6 * agent + 3] - X_f[6 * agent : 6 * agent + 3])
                )
                < init_x_f_rad
                for agent in range(numChasers)
            ]
        ):
            print("Target Reached at Iteration ", mpcIter)
            break

        if is_key_pressed("q"):
            print("MPC Manually Terminated at Iteration ", mpcIter)
            break

        mpcIter += 1
        if mpcIter > mpcMaxIter:
            print("MPC Iteration Limit reached at Iteration ", mpcIter)
            break

        ##### Update Target #####
        t_s = t_0 + ((mpcIter - 1) * numActSteps) * dt
        t_f = t_0 + ((mpcIter - 1) * numActSteps + numActSteps - 1) * dt
        timeSeq = np.linspace(0, 0 + (numMPCSteps - 1) * dt, numMPCSteps)
        X_f = []
        if callable(f_X_f):
            try:
                for agent in range(numChasers):
                    X_f.extend(f_X_f[agent](t=t_f))
            except:
                X_f = f_X_f
        else:
            X_f = f_X_f
        X_f = np.array(X_f)

        ##### Print Iteration Info #####
        if True:
            print()
            print("MPC       Iteration    : ", mpcIter)
            print("Periapsis True Anomaly : ", p.t_p)
            print("Initial   True Anomaly : ", t_0)
            print("Start     True Anomaly : ", t_s)
            print("Final     True Anomaly : ", t_f + t_s)
            print("Initial   Position     : ", x_0)
            print("Final     Position     : ", x_f)
            print()

        ##### Update Initial Tube Geometry #####
        if "tubeMPC" in kwargs:
            try:
                omega_0 = mpc_act_states[-1] - mpc_nom_states[-1]
                omega_0 = omega_0[:3]
                solverParams["tubeMPC"]["omega_0"] = omega_0
            except:
                pass

        flag = 1
        #### BEGIN Generate Trajectory ####
        if True:
            nom_states, nom_inputs = [], []
            act_states, act_inputs = [], []
            ##### Run Nominal Trajectory Optimization and Actual Trajectory Generation #####
            solverParams["x_0"] = x_0
            solverParams["x_f"] = x_f
            output = trajopt_mpc(
                timeParams={
                    "timeSeq": timeSeq,
                    "numMPCSteps": numMPCSteps,
                    "numActSteps": numActSteps,
                    "t_s": t_s,
                },
                nom_matrices=(
                    A_nom,
                    B_nom,
                    Q_nom,
                    R_nom,
                    d_nom,
                ),
                act_matrices=(
                    A_act,
                    B_act,
                    d_act,
                ),
                bounds=bounds,
                solverParams=solverParams,
                eRange=kwargs["eRange"],
                aRange=kwargs["aRange"],
                bRange=kwargs["bRange"],
            )
            flag, nom_states, act_states, nom_inputs, act_inputs, target_thetas = output
            if flag:
                ##### Deal with Failure #####
                print("MPC Failed - Try ", tryIter)
                if tryIter < tryIterMax:
                    print("Retrying")
                    mpcIter -= 1
                    tryIter += 1
                    try:
                        time.sleep(15)
                    except:
                        break
                    continue
                else:
                    break
            else:
                ##### Update States and Inputs #####
                tryIter = 0
                mpc_x_f_list.append(x_f)
                mpc_nom_states.extend(nom_states)
                mpc_act_states.extend(act_states)
                mpc_nom_inputs.extend(nom_inputs)
                mpc_act_inputs.extend(act_inputs)
                mpc_target_thetas.extend(target_thetas)
        #### END Generate Trajectory ####

        #### Update Initial Conditions ####
        x_0 = mpc_act_states[-1]
        u_0 = mpc_act_inputs[-1]

        #### Estimate Parameters ####
        if kwargs["adaptorFlag"] and operation == "rendezvous":
            # if kwargs['adaptorFlag']:
            # if mpcIter <= 100:
            if mpcIter <= 10:
                data_range = len(act_inputs) * min(1, mpcIter)
                # data_range = len(act_inputs)
                D = 0.015 * np.exp(-0.08 * mpcIter)
                # D = 0.12
                mpc_FSS, estim_lists, range_lists = adaptor(
                    initParams=mpc_FSS,
                    rangeParams={
                        "dt": dt,
                        "adaptation_range": data_range - 1,
                        "data_range": data_range,
                    },
                    u_t=np.transpose(
                        mpc_act_inputs[len(mpc_act_inputs) - data_range :]
                    ),
                    D=D,
                    W=mpc_act_states[len(mpc_act_states) - data_range :],
                    estimates=(
                        nomOrbitParams["eccentricity"],
                        nomOrbitParams["drag_alpha"],
                        nomOrbitParams["drag_beta"],
                    ),
                    disp=True,
                    threader=False,
                )
            else:
                estim_lists = [
                    [nomOrbitParams["eccentricity"] for i in range(len(act_inputs))],
                    [nomOrbitParams["drag_alpha"] for i in range(len(act_inputs))],
                    [nomOrbitParams["drag_beta"] for i in range(len(act_inputs))],
                ]
                range_lists = [
                    [
                        [mpc_FSS["eccentricity"][0] for i in range(len(act_inputs))],
                        [mpc_FSS["eccentricity"][1] for i in range(len(act_inputs))],
                    ],
                    [
                        [mpc_FSS["drag_alpha"][0] for i in range(len(act_inputs))],
                        [mpc_FSS["drag_alpha"][1] for i in range(len(act_inputs))],
                    ],
                    [
                        [mpc_FSS["drag_beta"][0] for i in range(len(act_inputs))],
                        [mpc_FSS["drag_beta"][1] for i in range(len(act_inputs))],
                    ],
                ]
            nomOrbitParams = {
                "eccentricity": estim_lists[0][-1],
                "drag_alpha": estim_lists[1][-1],
                "drag_beta": estim_lists[2][-1],
            }
            for i in range(3):
                mpc_estim_lists[i].extend(estim_lists[i])
                mpc_range_lists[i][0].extend(range_lists[i][0])
                mpc_range_lists[i][1].extend(range_lists[i][1])
    ## END MPC Loop ##

    mpc_nominal = [mpc_nom_states, mpc_nom_inputs]
    mpc_actual = [mpc_act_states, mpc_act_inputs]
    mpc_adaptation = [mpc_FSS, mpc_estim_lists, mpc_range_lists]
    return mpc_nominal, mpc_actual, mpc_x_f_list, mpc_adaptation, mpc_target_thetas


def parse_arguments():
    """Parse command-line arguments to configure the operation."""
    parser = argparse.ArgumentParser(
        description="Run MPC rendezvous and docking simulation."
    )
    parser.add_argument(
        "opIter",
        type=int,
        nargs="?",
        default=0,
        help="Operation iteration (e.g., 1-4 for predefined tests).",
    )
    parser.add_argument(
        "opType",
        type=int,
        nargs="?",
        default=0,
        help="Operation type: 0=both, 1=rendezvous, 2=docking.",
    )
    parser.add_argument(
        "opInit",
        type=int,
        nargs="?",
        default=0,
        help="Initial condition choice (0 for command line input).",
    )
    parser.add_argument(
        "xfCalc",
        type=int,
        nargs="?",
        default=1,
        choices=[0, 1],
        help="Enable final state calculation (0=False, 1=True).",
    )
    parser.add_argument(
        "initial_states",
        type=float,
        nargs="*",
        default=[],
        help="Initial states for each chaser (6 values per chaser).",
    )
    parser.add_argument(
        "numChasers",
        type=int,
        nargs="?",
        default=1,
        help="Number of chaser spacecrafts.",
    )

    args = parser.parse_args()

    # Configure flags based on opIter
    ad_flags = [True, False, True, False]
    dt_flags = [True, True, False, False]

    op_flags = {
        "opIter": args.opIter,
        "opType": args.opType,
        "opInit": args.opInit,
        "xfCalc": bool(args.xfCalc),
        "adFlag": (
            ad_flags[args.opIter - 1] if 0 < args.opIter <= len(ad_flags) else False
        ),
        "DTFlag": (
            dt_flags[args.opIter - 1] if 0 < args.opIter <= len(dt_flags) else False
        ),
    }

    if args.opInit == 0 and not args.initial_states:
        # Default initial states for a single chaser if not provided
        if args.numChasers == 1:
            # Example default state; replace with a meaningful default if needed
            args.initial_states = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            raise ValueError("Initial states must be provided for multiple chasers.")

    return op_flags, args.numChasers, args.initial_states


def get_initial_conditions(num_chasers, initial_states_raw):
    """Set initial conditions from parsed arguments."""
    if initial_states_raw:
        if len(initial_states_raw) != num_chasers * 6:
            raise ValueError(
                f"Expected {num_chasers * 6} initial states, but got {len(initial_states_raw)}."
            )
        x_0 = np.array(initial_states_raw)
    else:
        # Fallback for when opInit is not 0 and initial_states is not provided
        # This part of the original logic seems incomplete, so we provide a placeholder.
        x_0 = np.zeros(num_chasers * 6)  # Or some other default logic

    u_0 = np.zeros(num_chasers * 3)  # Assuming 3 inputs per chaser
    return x_0, u_0


def setup_simulation_components(op_flags, num_chasers, x_0):
    """Initialize core simulation objects and parameters."""
    target = Target(
        {"name": "Target", "numStates": 3, "dt": p.dt, "initState": p.th_T0},
        {"angularVelocity": p.w_T0, "momInertia": p.I_T0},
    )

    bounds = (p.stateBounds, p.inputBounds)
    p.tubeMPC["runTube"] = op_flags["DTFlag"]

    # Final state calculation
    x_f_body = np.array([[0.00, -0.80, 0.60], [0.00, 0.80, -0.60]])[
        :num_chasers
    ]  # Use only as many as needed

    if op_flags["xfCalc"]:
        print("~ Final State Calculation ~")
        output = targetDeflect(
            target=deepcopy(target),
            dt=p.dt,
            x_f=np.zeros(
                6
            ),  # This seems to be a placeholder based on the original code
            bounds=(p.forceBounds),
            numSteps=200,
            numChasers=num_chasers,
            rLen=3,
            fLen=3,
            chaserMinDist=p.chaserMinDist,
            shapeParams={
                "cylHeight": p.targetLimit["l_T"],
                "cylRadius": p.targetLimit["r_T"],
                "cylCenter": p.targetCenter,
            },
        )
        _, _, x_f_body, _ = (
            output  # The original code uses 'angles' and 'forces' which are not used later
        )

    # Pad x_f_body with zeros for velocity components
    x_f_body = np.pad(x_f_body, ((0, 0), (0, 3)), "constant")

    return target, bounds, x_f_body


def run_mpc_operation(
    op_type, num_chasers, target, bounds, x_0, u_0, x_f_body, op_flags, main_data
):
    """Run a single MPC operation (rendezvous or docking)."""
    operation_types = []
    if op_type == 0 or op_type == 1:
        operation_types.append("rendezvous")
    if op_type == 0 or op_type == 2:
        operation_types.append("docking")

    for op_type_str in operation_types:
        if op_type_str == "rendezvous":
            target_rads = (
                p.rendezvous_radii if hasattr(p, "rendezvous_radii") else [2.0]
            )
            v_f_rad = 0.1
        elif op_type_str == "docking":
            target_rads = p.docking_radii if hasattr(p, "docking_radii") else [0.0]
            v_f_rad = 0.0

        for x_f_rad_i in target_rads:
            print(
                f"\n~ {op_type_str.capitalize()} to Sphere of Radius {x_f_rad_i} ~"
                if op_type_str == "rendezvous"
                else "\n~ Docking to Target ~"
            )

            # Update initial conditions for subsequent operations
            t_0 = p.dt * len(main_data["actual_states"])
            if main_data["actual_states"]:
                x_0 = main_data["actual_states"][-1]
            if main_data["nominal_inputs"]:
                u_0 = main_data["nominal_inputs"][-1]
            else:
                u_0 = np.zeros(num_chasers * 3)

            mpc_params = build_mpc_parameters(
                op_type_str,
                num_chasers,
                target,
                t_0,
                x_0,
                u_0,
                x_f_body,
                x_f_rad_i,
                v_f_rad,
                op_flags,
                bounds,
            )
            output = mpc(**mpc_params)

            nominal, actual, x_f_list, adaptation, target_thetas = output
            nom_states, nom_inputs = nominal
            act_states, act_inputs = actual
            fss, estim_lists, range_lists = adaptation

            # Update nominal orbit parameters for next step
            if estim_lists and estim_lists[0]:
                main_data["nom_orbit_params"] = {
                    "eccentricity": estim_lists[0][-1],
                    "drag_alpha": estim_lists[1][-1],
                    "drag_beta": estim_lists[2][-1],
                }

            # Append results to main data lists
            append_results(
                main_data,
                act_states,
                act_inputs,
                nom_states,
                nom_inputs,
                x_f_list,
                target_thetas,
                estim_lists,
                range_lists,
                x_f_body,
                target,
                t_0,
                op_type_str,
            )

    # Handle the "docking" vs "rendezvous" split for plotting
    if "rendezvous" in operation_types and "docking" in operation_types:
        main_data["dock_index"] = len(main_data["actual_states"])


def build_mpc_parameters(
    op_type_str,
    num_chasers,
    target,
    t_0,
    x_0,
    u_0,
    x_f_body,
    x_f_rad,
    v_f_rad,
    op_flags,
    bounds,
):
    """Build the dictionary of parameters for the mpc function."""
    f_x_f = [
        partial(calcCurrentPos, target=target, x_i=x_f_body[agent][:3])
        for agent in range(num_chasers)
    ]

    mpc_params = {
        "operation": op_type_str,
        "dt": p.dt,
        "t_0": t_0,
        "rLen": 6,  # Assuming 6 states per chaser
        "fLen": 3,  # Assuming 3 inputs per chaser
        "numChasers": num_chasers,
        "numMPCSteps": p.numMPCSteps[op_type_str],
        "numActSteps": p.numActSteps[op_type_str],
        "X_0": x_0,
        "f_X_f": f_x_f,
        "x_f_rad": x_f_rad,
        "v_f_rad": v_f_rad,
        "U_0": u_0,
        "actOrbitParams": p.actOrbitParams,
        "nomOrbitParams": p.nomOrbitParams,
        "bounds": bounds,
        "initAdaptParams": p.initAdaptParams,
        "targetParams": {
            "theta_0": target.getObservedState(t=t_0),
            "omega_0": target.getObservedAngVel(t=t_0),
            "momInertia": target.momInertia,
        },
        "tubeMPC": p.tubeMPC,
        "adaptorFlag": op_flags["adFlag"],
        "eRange": p.initAdaptParams["eccentricity"],
        "aRange": p.initAdaptParams["drag_alpha"],
        "bRange": p.initAdaptParams["drag_beta"],
    }

    if op_type_str == "rendezvous":
        mpc_params["radialLimit"] = p.radialLimit
    elif op_type_str == "docking":
        mpc_params["targetLimit"] = p.targetLimit

    return mpc_params


def append_results(
    main_data,
    act_states,
    act_inputs,
    nom_states,
    nom_inputs,
    x_f_list,
    target_thetas,
    estim_lists,
    range_lists,
    x_f_body,
    target,
    t_0,
    op_type_str,
):
    """Append results from a single MPC run to the main data lists."""
    num_act_steps = p.numActSteps[op_type_str]

    main_data["actual_states"].extend(act_states)
    main_data["actual_inputs"].extend(act_inputs)
    main_data["nominal_states"].extend(nom_states)
    main_data["nominal_inputs"].extend(nom_inputs)
    main_data["target_thetas"].extend(target_thetas)

    # Calculate and append final and target states for plotting
    fin_states = [x_f_i for x_f_i in x_f_list for _ in range(num_act_steps)]
    main_data["final_states"].extend(fin_states)

    tgt_states = []
    for t_iter in range(len(act_states)):
        rot_matrix = tait_bryan_to_rotation_matrix(
            target.getObservedState(t=t_0 + t_iter * p.dt)
        )
        x_f = np.dot(rot_matrix, x_f_body)
        tgt_states.append(x_f)
    main_data["target_states"].extend(tgt_states)

    if estim_lists:
        for i in range(3):
            main_data["estim_lists"][i].extend(estim_lists[i])
            main_data["range_lists"][i][0].extend(range_lists[i][0])
            main_data["range_lists"][i][1].extend(range_lists[i][1])

    main_data["x_f_list"].extend(x_f_list)


def setup_and_run_dummy(op_flags, main_data):
    """Run a dummy operation for specific opType and opIter values."""
    print("\n~ Running Dummy Operation ~")
    dummy_steps = 10
    # Create lists of dummy data for each output type
    main_data["actual_states"].extend([np.zeros(6) for _ in range(dummy_steps)])
    main_data["actual_inputs"].extend([np.zeros(3) for _ in range(dummy_steps)])
    main_data["nominal_states"].extend([np.zeros(6) for _ in range(dummy_steps)])
    main_data["nominal_inputs"].extend([np.zeros(3) for _ in range(dummy_steps)])
    main_data["final_states"].extend([np.zeros(6) for _ in range(dummy_steps)])
    main_data["target_states"].extend([np.zeros(6) for _ in range(dummy_steps)])
    main_data["target_thetas"].extend([np.zeros(3) for _ in range(dummy_steps)])
    main_data["x_f_list"].extend([np.zeros(6) for _ in range(dummy_steps)])
    for i in range(3):
        main_data["estim_lists"][i].extend([0 for _ in range(dummy_steps)])
        main_data["range_lists"][i][0].extend([0 for _ in range(dummy_steps)])
        main_data["range_lists"][i][1].extend([0 for _ in range(dummy_steps)])


def save_and_plot_data(
    main_data, op_flags, target, x_f_body, num_chasers, target_folder=None
):
    """Save simulation data and generate plots."""
    print("~ Saving Results ~")
    if op_flags["opIter"] != 0 and op_flags["opType"] in [0, 1, 2]:
        if op_flags["xfCalc"]:
            np.savez(f"{target_folder}/deflection.npy", T=target, x_f_body=x_f_body)
        if op_flags["adFlag"]:
            np.savez(
                f"{target_folder}/adaptor.npy",
                main_estim_lists=main_data["estim_lists"],
                main_range_lists=main_data["range_lists"],
                actOrbitParams=p.actOrbitParams,
            )
        np.savez(f"{target_folder}/mpc.npy", **main_data)
        print(f"Data saved to {target_folder}")

    print("~ Plotting Results ~")
    if main_data["actual_states"]:
        plot_kwargs = {
            "act_states": main_data["actual_states"],
            "act_inputs": main_data["actual_inputs"],
            "nom_states": main_data["nominal_states"],
            "nom_inputs": main_data["nominal_inputs"],
            "fin_states": main_data["final_states"],
            "tgt_states": main_data["target_states"],
            "x_f_list": main_data["x_f_list"],
            "dt": p.dt,
            "plotFlags": {"plot_act": True, "plot_nom": True},
            "target_thetas": main_data["target_thetas"],
            "dock_index": main_data["dock_index"],
        }
        if target_folder:
            mpc_plot_kwargs["targetFolder"] = target_folder
        mpc_plot(**plot_kwargs)

    if op_flags["xfCalc"]:
        deflection_plot_kwargs = {
            "target": target,
            "x": [],
            "r": [x_f_body],
            "f": [],
            "plotFlags": {
                "plot_target": True,
                "plot_position": True,
                "plot_force": True,
            },
            "numChasers": num_chasers,
            "dt": p.dt,
            "numSteps": 200,
            "shapeParams": {
                "cylHeight": p.targetLimit["l_T"],
                "cylRadius": p.targetLimit["r_T"],
                "cylCenter": p.targetCenter,
            },
            "targetShape": p.targetShape,
            "initialTimeLapse": p.initialTimeLapse,
        }
        if target_folder:
            deflection_plot_kwargs["target_folder"] = target_folder
        deflection_plot(**deflection_plot_kwargs)

    if op_flags["adFlag"] and main_data["estim_lists"][0]:
        adaptor_plot_kwargs = {
            "estim_lists": main_data["estim_lists"],
            "range_lists": main_data["range_lists"],
            "orbitParams": p.actOrbitParams,
        }
        if target_folder:
            adaptor_plot_kwargs["targetFolder"] = target_folder
        adaptor_plot(**adaptor_plot_kwargs)


def main():
    """Main function to run the simulation pipeline."""
    try:
        op_flags, num_chasers, initial_states = parse_arguments()
        x_0, u_0 = get_initial_conditions(num_chasers, initial_states)
    except (ValueError, IndexError) as e:
        print(f"Error parsing command line arguments: {e}")
        return

    # Initialize data containers
    main_data = {
        "actual_states": [],
        "actual_inputs": [],
        "nominal_states": [],
        "nominal_inputs": [],
        "final_states": [],
        "target_states": [],
        "estim_lists": [[] for _ in range(3)],
        "range_lists": [[[], []] for _ in range(3)],
        "target_thetas": [],
        "x_f_list": [],
        "dock_index": 0,
        "nom_orbit_params": p.nomOrbitParams.copy(),
    }

    target, bounds, x_f_body = setup_simulation_components(op_flags, num_chasers, x_0)

    # Main simulation loop
    if op_flags["opType"] in [0, 1, 2]:
        run_mpc_operation(
            op_flags["opType"],
            num_chasers,
            target,
            bounds,
            x_0,
            u_0,
            x_f_body,
            op_flags,
            main_data,
        )
    else:
        setup_and_run_dummy(op_flags, main_data)

    # Prepare for saving and plotting
    target_folder = save_test_result(
        properties={
            "pipeline": {
                "rendezvous": op_flags["opType"] in [0, 1],
                "docking": op_flags["opType"] in [0, 2],
            },
            "algorithm": {
                "adaptation": op_flags["adFlag"],
                "dynamic_tubes": op_flags["DTFlag"],
                "mpc": True,
            },
            "sim_details": {"validity": True, "reason": "working model"},
        },
        files_to_copy=[
            ("config/default.yaml", "config.yaml"),
            ("results/plot.png", "plot.png"),
        ],
    )

    save_and_plot_data(
        main_data, op_flags, target, x_f_body, num_chasers, target_folder
    )


if __name__ == "__main__":
    main()
