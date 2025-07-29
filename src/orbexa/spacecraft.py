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
import random
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from gekko import GEKKO
from scipy import optimize as opt
from scipy import integrate as intg
from itertools import count

from orbexa.params import *
from orbexa.utils import genShapeData, genSkewSymMat, calcGlobalOcclusion
from orbexa.dynamics import cwhEquations

random.seed(0)


# CLASS DEFINITIONS
### Spacecraft Definition
class Spacecraft:
    _ids = count(0)  ## https://stackoverflow.com/a/8628132/6539635
    from params import dt

    dt = dt

    def __init__(self, *args):
        self.id = next(self._ids)
        self.name = ""
        self.numStates = 6
        self.initState = np.array([0 for i in range(self.numStates)])
        if len(args) == 0:
            return
        try:
            self.name = args[0]["name"]
        except:
            pass
        try:
            self.numStates = args[0]["numStates"]
        except:
            pass
        try:
            self.initState = args[0]["initState"]
        except:
            pass
        self.currState = self.initState
        self.stateHistory = [self.currState]

    def updateState(self, *args):
        raise NotImplementedError("Spacecraft.updateState() is not implemented.")

    def plotStateHistory(self, params, *args, **kwargs):
        indStateHistory = np.transpose(self.stateHistory)
        cmap = plt.cm.get_cmap(plt.cm.viridis, 256)
        plt.figure(figsize=(4, 3))
        timeSeq = [step * self.dt for step in range(len(indStateHistory[0]))]
        if params["sep_plots"] == True:
            for i in range(self.numStates):
                plt.subplot(self.numStates, 1, i + 1)
                label = "$x_" + str(i) + "$"
                plt.plot(timeSeq, indStateHistory[i], c=cmap(96 * i), label=label)
                plt.legend()
                plt.ylabel("State History")
                plt.xlabel("Time")
        else:
            plt.subplot(1, 1, 1)
            for i in range(self.numStates):
                label = "$x_" + str(i) + "$"
                plt.plot(timeSeq, indStateHistory[i], c=cmap(96 * i), label=label)
            plt.legend()
            plt.ylabel("State History")
            plt.xlabel("Time")

        if params["disp_plot"] == True and "fLoc" not in kwargs:
            plt.show()
        elif "fLoc" in kwargs:
            plt.gcf().set_size_inches(10 * plt.gcf().get_size_inches())
            plt.tight_layout()
            plt.savefig(kwargs["fLoc"] + "target_ang_pos.png")
            plt.close()


### Target Definition
class Target(Spacecraft):
    _tids = count(0)
    ObservationError_angPos = 4.0e-8
    ObservationError_angVel = 4.0e-8

    def __init__(self, *args):
        self.tid = next(self._tids) + 1
        super().__init__(*args)
        try:
            self.angularVelocity = args[1]["angularVelocity"]
            self.angularVelocityHistory = [
                self.angularVelocity,
            ]
        except:
            pass
        try:
            self.momInertia = args[1]["momInertia"]
        except:
            pass
        try:
            self.geometry = args[1]["geometry"]
        except:
            self.geometry = {"Ineqs": [], "Eqs": []}
        try:
            self.dt = args[1]["dt"]
        except:
            pass

    def updateGeometry(self, **kwargs):
        if "geometryIneqs" in kwargs:
            self.geometry["Ineqs"] = kwargs["geometryIneqs"]
        if "geometryEqs" in kwargs:
            self.geometry["Eqs"] = kwargs["geometryEqs"]
        return self.geometry

    def updateState(self, *args, **kwargs):
        if "newAngularPos" in kwargs and "newAngularVel" in kwargs:
            newAngularPos = kwargs["newAngularPos"]
            newAngularVel = kwargs["newAngularVel"]
        else:
            if "numSteps" in kwargs:
                numSteps = kwargs["numSteps"]
            else:
                numSteps = 1
            if "torqueVal" in kwargs and "torqueType" in kwargs:
                torqueVal = kwargs["torqueVal"]
                torqueType = kwargs["torqueType"]
            else:
                torqueVal = [
                    np.array([0.0 for i in range(len(self.angularVelocity))])
                    for j in range(numSteps + 1)
                ]
                torqueType = "zero"

            newState = intg.solve_ivp(
                fun=targetStateUpdateFunc,
                y0=np.array(
                    [list(self.currState), list(self.angularVelocity)]
                ).flatten(),
                t_span=[0, self.dt * numSteps],
                method="RK45",
                t_eval=[i for i in np.arange(0, self.dt * numSteps, self.dt)],
                dense_output=False,
                args=(
                    self.dt,
                    self.momInertia,
                    genSkewSymMat(self.angularVelocity),
                    torqueVal,
                    torqueType,
                ),
            ).y
            newAngularPos = list(np.transpose(newState[:3]))
            newAngularVel = list(np.transpose(newState[3:]))

        self.angularVelocityHistory.extend(newAngularVel)
        self.stateHistory.extend(newAngularPos)
        self.angularVelocity = self.angularVelocityHistory[-1]
        self.currState = self.stateHistory[-1]
        return self.stateHistory

    def getObservedState(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            state = self.currState
        elif "t" in kwargs or len(args) == 1:
            try:
                t = float(kwargs["t"])
            except:
                t = float(args[0])
            try:
                state = self.stateHistory[int(t / self.dt)]
            except:
                self.updateState(numSteps=int(t / self.dt) - len(self.stateHistory) + 1)
                state = self.stateHistory[int(t / self.dt)]
        return np.multiply(state, random.gauss(1.00, Target.ObservationError_angPos))

    def getObservedAngVel(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            angVel = self.angularVelocity
        elif "t" in kwargs:
            try:
                t = float(kwargs["t"])
            except:
                raise ValueError(
                    "Target.getObservedState() requires a numeric value for 't'."
                )
            try:
                angVel = self.angularVelocityHistory[int(t / self.dt)]
            except:
                self.updateState(
                    numSteps=int(t / self.dt) - len(self.angularVelocityHistory) + 1
                )
                angVel = self.angularVelocityHistory[int(t / self.dt)]
        return np.multiply(angVel, random.gauss(1.00, Target.ObservationError_angVel))

    def getMomInertia(self, *args, **kwargs):
        return self.momInertia

    def plotStateHistory(self, params, *args, **kwargs):
        super().plotStateHistory(params, *args, **kwargs)
        indAngularVelocityHistory = np.transpose(self.angularVelocityHistory)
        timeSeq = [i * self.dt for i in range(len(self.angularVelocityHistory))]
        plt.title("Angular Velocity History")
        plt.plot(timeSeq, indAngularVelocityHistory[0], c="r", label="$\omega_x$")
        plt.plot(timeSeq, indAngularVelocityHistory[1], c="g", label="$\omega_y$")
        plt.plot(timeSeq, indAngularVelocityHistory[2], c="b", label="$\omega_z$")
        plt.ylabel("Angular Velocity")
        plt.xlabel("Time")
        plt.legend()
        if "fLoc" in kwargs:
            plt.gcf().set_size_inches(10 * plt.gcf().get_size_inches())
            plt.tight_layout()
            plt.savefig(kwargs["fLoc"] + "target_ang_vel.png")
            plt.close()
        else:
            plt.show()


### Chaser Definition
class Chaser(Spacecraft):
    _cids = count(0)

    def __init__(self, *args, **kwargs):
        if "repeat" not in kwargs or not kwargs["repeat"]:
            self.cid = next(self._cids) + 1
            super().__init__(*args)
            from params import n, stateBounds, inputBounds, goalBounds

            self.n = n
            self.inputs = [np.zeros(3)]
            self.stateBounds, self.inputBounds = [], []
            for i in range(int(len(self.initState) / 6)):
                self.stateBounds.extend(stateBounds)
            for i in range(int(len(self.inputs[0]) / 3)):
                self.inputBounds.extend(inputBounds)
            self.goalBounds = goalBounds
        self.neighbors = []
        self.tNInfo = {}
        self.gNInfo = {}
        self.rNInfo = {"len": 0}
        self.goalState = None
        self.targetObserveConsensus = False
        self.neighborsLocnConsensus = False
        self.goalCalculateConsensus = False

    def resetInfo(self):
        self.__init__(repeat=True)
        return self.goalState

    def updateNeighbors(self, operation, agentList):
        if operation == "set":
            self.neighbors = agentList.copy()
        elif operation == "append":
            for agent in agentList:
                if agent not in self.neighbors:
                    self.neighbors.append(agent)
        elif operation == "remove":
            for agent in agentList:
                if agent in self.neighbors:
                    self.neighbors.remove(agent)
        return self.neighbors

    def commData(self, type, info, *args):
        if type == "target":
            neighbor = args[0]
            self.tNInfo[neighbor] = info
        elif type == "get_locations":
            self.rNInfo[self.id] = self.currState[:3]
            for agent in info.keys():
                if agent not in self.rNInfo.keys() and agent != "len":
                    self.rNInfo[agent] = info[agent]
                    self.neighborsLocnConsensus = False
            self.rNInfo["len"] = len(self.rNInfo.keys()) - 1
            if info["len"] == self.rNInfo["len"]:
                self.neighborsLocnConsensus = True
        elif type == "goal_list":
            if self.occlusion > info["occlusion"]:
                self.goalLocations = info["goalLocations"]
                self.occlusion = info["occlusion"]
            if self.occlusion > 400:
                self.goalCalculateConsensus = False
            else:
                self.goalCalculateConsensus = True
            return {"goalLocations": self.goalLocations, "occlusion": self.occlusion}
        elif type == "goal":
            self.gNInfo.append(info)
            pass
        else:
            raise ValueError("Invalid Type of Communication Data")

    def updateState(self, numSteps, *args):
        matrices, _, _ = cwhEquations(self.dt, n=self.n)
        A, B, _, _, _ = matrices
        self.inputs.extend(
            [np.zeros(len(self.inputs[0])) for i in range(numSteps - len(self.inputs))]
        )
        for i in range(numSteps):
            self.lastInput = self.inputs[0]
            self.currState = np.add(
                np.matmul(A(0), self.currState), np.matmul(B, self.lastInput)
            )
            self.inputs = self.inputs[1:]
            self.stateHistory.append(self.currState)
        return self.stateHistory

    def getInputs(self, *args):
        return self.inputs

    def setInputs(self, inputs):
        self.inputs = inputs
        return self.inputs

    def getObservedState(self, *args):
        return np.multiply(self.currState, random.gauss(1.00, 0.001))

    def observeTarget(self, target: Target, *args):
        epsilon = 0.01
        TState = target.getObservedState()
        TAngVel = target.getObservedAngVel()
        self.TState, self.TAngVel = TState.copy(), TAngVel.copy()
        for neighbor in self.tNInfo.keys():
            self.TState += self.tNInfo[neighbor][0]
            self.TAngVel += self.tNInfo[neighbor][1]
        self.TState /= len(self.tNInfo) + 1
        self.TAngVel /= len(self.tNInfo) + 1
        if (
            np.linalg.norm(TState - self.TState) < epsilon
            and np.linalg.norm(TAngVel - self.TAngVel) < epsilon
            and len(self.tNInfo) == len(self.neighbors)
        ):
            self.targetObserveConsensus = True
        else:
            self.targetObserveConsensus = False
        return (self.TState, self.TAngVel)

    def calculateGoals(self):
        numAgents = len(self.rNInfo.keys()) - 1
        w = np.array([1.0e3 for ego in range(numAgents)])
        v = [1.0e2, 1.0e6, 1.0e6]
        try:
            x0 = self.goalLocations
            RIDs = list(self.rNInfo.keys())
        except:
            RIDs, x0 = [], []
            for agent in self.rNInfo.keys():
                if agent != "len":
                    RIDs.append(agent)
                    x0.append(self.rNInfo[agent])

        constraints = []
        for agent in range(1, numAgents + 1):
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x: (
                        np.linalg.norm(x[3 * agent - 3 : 3 * agent])
                        - self.goalBounds[0]
                    ),
                }
            )
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x: -(
                        np.linalg.norm(x[3 * agent - 3 : 3 * agent])
                        - self.goalBounds[1]
                    ),
                }
            )
        constraints = tuple(constraints)

        self.goalLocations = opt.minimize(
            calcGlobalOcclusion,
            x0=np.array(
                [
                    x0i
                    * (sum(self.goalBounds) / len(self.goalBounds))
                    / np.linalg.norm(x0i)
                    for x0i in x0
                ]
            ).flatten(),
            args=(w, v, np.array(x0).flatten(), self.goalBounds),
            options={"disp": False, "maxiter": 100},
            constraints=constraints,
        ).x
        self.occlusion = calcGlobalOcclusion(
            self.goalLocations, w, v, np.array(x0).flatten(), self.goalBounds
        )
        return {"goalLocations": self.goalLocations, "occlusion": self.occlusion}

    def determineGoalInit(self, type, *args):
        if type == "pick_prelim_goal":
            self.goalState = self.goalLocations[3 * self.id - 3 : 3 * self.id]
            return self.goalState
        elif type == "create_agent":
            self.taskAllocAgent = args[0](
                self.id - 1, self.currState[:3], self.goalState, self.neighbors
            )
            return self.taskAllocAgent
        else:
            raise ValueError("Invalid Type of Goal Determination Command")

    def determineInputs(self, numSteps):
        f = trajopt_dynamics  # out of trajopt_dynamics() and mpc()
        self.inputs = f(
            cwhEquations,
            numSteps,
            self.dt,
            constraints=(self.currState, self.goalState),
            bounds=(self.stateBounds, self.inputBounds),
        )
        self.inputs = list(np.transpose(self.inputs))
        return self.inputs


# FUNCTION DEFINITIONS
def trajopt_dynamics(
    system,
    numSteps,
    dt,
    constraints,
    bounds,
    solverParams={
        "remote": False,
        "disp": False,
        "comp_time": False,
        "no_soln_disp": True,
    },
    returnStates=False,
    *args,
    **kwargs,
):
    ### Unpacking System Parameters
    timeSeq = np.linspace(0, numSteps * dt, numSteps)
    if "w" not in solverParams.keys():  ## Integration Weight
        w = np.ones(numSteps)
    else:
        w = solverParams["w"]
    systemParams = (
        {"dt": dt, "bounds": bounds, "constraints": constraints, "discretize": False},
    )
    matrices, constraints, bounds = system(**systemParams)
    A, B, Q, R, d = matrices
    x_0, x_f = constraints
    stateBounds, inputBounds = bounds

    eccentricity = 0
    t_s = 0

    stateConstraints = {}
    inputConstraints = {}
    for i in range(len(x_0)):
        stateConstraints[i] = [[numSteps - 1, x_f[i]]]

    ### Declaration of Gekko Model
    if True:
        m = GEKKO(remote=solverParams["remote"])
        m.time = timeSeq
        w = np.ones(numSteps)
        final = np.zeros(numSteps)
        final[-1] = 1

    ### Declaration of Gekko Variables
    if True:
        t = m.Var(value=0)
        q = m.Var(value=0, fixed_initial=False)
        x = [m.Var(value=x_0[i], fixed_initial=True) for i in range(len(x_0))]
        u = [m.Var(value=0, fixed_initial=False) for i in range(int(len(x) / 2))]
        w = m.Param(value=w)
        final = m.Param(value=final)

    ## Constraint Equations ##
    eqs = []
    ### State and Input Bounds ###
    for i in range(len(x)):
        if stateBounds[i]["lower"] != "-Inf":
            eqs.append(x[i] > stateBounds[i]["lower"])
        if stateBounds[i]["upper"] != "+Inf":
            eqs.append(x[i] < stateBounds[i]["upper"])
    for i in range(len(u)):
        if inputBounds[i]["lower"] != "-Inf":
            eqs.append(u[i] > inputBounds[i]["lower"])
        if inputBounds[i]["upper"] != "+Inf":
            eqs.append(u[i] < inputBounds[i]["upper"])

    ### State and Input Fixed Constraints ###
    for i in range(len(x)):
        if i in stateConstraints:
            for constraint in stateConstraints[i]:
                m.fix(x[i], pos=constraint[0], val=constraint[1])
    for i in range(len(u)):
        if i in inputConstraints:
            for constraint in inputConstraints[i]:
                m.fix(u[i], pos=constraint[0], val=constraint[1])

    ### Time and Anomaly Update ###
    if True:
        eqs.append(t.dt() == 1)
        E = m.Intermediate(
            2 * m.atan(np.sqrt((1 - eccentricity) / (1 + eccentricity)) * m.tan(t / 2))
        )
        M = m.Intermediate(E - eccentricity * m.sin(E))
        eqs.append(q == t_p + t_s + M / n)
    ### Nominal System Dynamics ###
    for agent in range(len(x) / 6):
        for i in range(0, 3):
            x_agent = np.array(x[agent * 6 : (agent + 1) * 6])
            u_agent = np.array(u[agent * 6 : (agent + 1) * 6])
            eqs.append(x_agent[i + 0].dt() == x_agent[i + 3])
            eqs.append(
                x_agent[i + 3].dt()
                == np.matmul(A(t + t_s, t_p, m=m), x_agent)[i + 3]
                + u_agent[i + 0]
                + d(t + t_s, t_p, m=m)[i + 3]
            )

    ## Objective Function Definition ##
    if True:
        intErrorArr = []
        for agent in range(len(x) / 6):
            x_agent = np.array(x[agent * 6 : (agent + 1) * 6])
            u_agent = np.array(u[agent * 6 : (agent + 1) * 6])
            intErrorArr.append(x_agent @ Q @ x_agent.T + u_agent @ R @ u_agent.T)
        intError = np.sum(intErrorArr)

    ## Solver Parameters ##
    if True:
        eqs = m.Equations(eqs)
        m.Minimize(w * intError)
        m.options.OTOL = 1e-7
        m.options.RTOL = 1e-7
        m.options.IMODE = 6
        m.options.SOLVER = 3
        m.options.MAX_ITER = 3000
        m.options.MAX_MEMORY = 512
        # m.options.COLDSTART  =    0
        # m.options.TIME_SHIFT =    0

    ## Solve MPC ##
    startTime = time.time()
    try:
        m.solve(disp=solverParams["disp"])
        states = [x[i].value for i in range(len(x))]
        inputs = [u[i].value for i in range(len(u))]
        timing = stopTime - startTime
    except:
        states = []
        inputs = []
        timing = 0
        if solverParams["no_soln_disp"]:
            print("Optimization Solution Not Found")
            print("Constraints: ", constraints)
            print("Bounds: ", bounds)
    stopTime = time.time()

    ## Print MPC Info ##
    if True:
        print("Solver Objective    : ", m.options.objfcnval)
        print("Solver Status       : ", m.options.APPSTATUS)
        if solverParams["comp_time"]:
            print("Solver Calc Time    : ", timing)
            print("Solver Meas Time    : ", m.options.SOLVETIME)
        print()

    if returnStates:
        return states, inputs
    return inputs


def targetStateUpdateFunc(
    t, state, dt, momInertia, skewSymMat, torqueVal, torqueType, *args, **kwargs
):
    angularPos = state[:3]
    angularVel = state[3:]
    if torqueType == "zero":
        torque = np.array([0.0 for i in range(len(angularVel))])
    elif torqueType == "given":
        torque = torqueVal[min(int(np.floor(t / dt)), len(torqueVal) - 1)]
    elif torqueType == "function":
        torqueValEval = torqueVal(state, momInertia)
        torque = torqueValEval["torque"]
    stateUpdate = list(angularVel)
    stateUpdate.extend(
        np.add(
            np.matmul(
                np.matmul(np.linalg.inv(momInertia), skewSymMat),
                np.matmul(momInertia, angularVel),
            ),
            np.matmul(np.linalg.inv(momInertia), torque),
        )
    )
    return np.array(stateUpdate)


def torqueFeedback(state, momInertia):
    angularPos = np.multiply(state[:3], random.gauss(1.00, 0.1))
    angularVel = np.multiply(state[3:], random.gauss(1.00, 0.5))
    torque = np.subtract(
        np.matmul(np.matmul(genSkewSymMat(angularVel), momInertia), angularVel),
        np.add(np.multiply(angularVel, 8.0), np.multiply(angularPos, 4.0)),
    )

    runForceOpt = False
    if runForceOpt:
        print("torque    : ", torque)
        epsilon = 0.05
        rad_T = 2.0
        len_T = 20.0
        numChasers = 1  ### TODO: Replace with actual number of chasers
        constraintsInCost = True
        forceOpt_constants = (epsilon, rad_T, len_T, constraintsInCost)

        prev_posit_forces = np.array([i for i in range(6 * numChasers)])
        posit_forces = forceOpt(
            copy(torque),
            prev_posit_forces,
            next(copy(Spacecraft._ids)) - 1,
            forceOpt_constants,
        )
        positions = np.array(
            [
                posit_forces[6 * agent - 6 : 6 * agent - 3]
                for agent in range(1, numChasers + 1)
            ]
        )
        forces = np.array(
            [
                posit_forces[6 * agent - 3 : 6 * agent - 0]
                for agent in range(1, numChasers + 1)
            ]
        )

        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        Xc, Yc, Zc = genShapeData("cylinder", ((0.0, 0.0, 0.0), rad_T, len_T))
        ax.plot_surface(Xc, Yc, Zc, alpha=0.5)

        print("positions")
        for pos_id, position in enumerate(positions):
            print("position ", pos_id, " radius = ", np.linalg.norm(position[0:2]))
            print("position ", pos_id, " length = ", np.linalg.norm(position[2]))
            print("position ", pos_id, " force  = ", np.linalg.norm(forces[pos_id]))
            ax.scatter(position[0], position[1], position[2], c="r", marker="o")
            # ax.quiver (position[0], position[1], position[2], forces[pos_id][0], forces[pos_id][1], forces[pos_id][2], length=0.5, normalize=True)
        print(
            "error of sum of torques = ",
            100
            * np.linalg.norm(
                sum(
                    [
                        np.cross(positions[agent - 1], forces[agent - 1])
                        for agent in range(1, numChasers + 1)
                    ]
                )
                - torque
            )
            / np.linalg.norm(torque),
            "%",
        )
        print()

        plt.show()
        return {"torque": torque, "positions": positions, "forces": forces}
    else:
        return {"torque": torque}


def forceOpt(torque, prev_posit_forces, numChasers, constants):
    posit_forces = prev_posit_forces.copy()

    if any(torque):
        epsilon, rad_T, len_T, constraintsInCost = constants

        constraints = []
        if not constraintsInCost:
            for agent in range(1, numChasers + 1):
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x: +(
                            np.linalg.norm(x[6 * agent - 6 : 6 * agent - 4])
                            - rad_T
                            + epsilon
                        ),
                    }
                )
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x: -(
                            np.linalg.norm(x[6 * agent - 6 : 6 * agent - 4])
                            - rad_T
                            - epsilon
                        ),
                    }
                )
                constraints.append(
                    {"type": "ineq", "fun": lambda x: +(x[6 * agent - 4] + len_T / 2)}
                )
                constraints.append(
                    {"type": "ineq", "fun": lambda x: -(x[6 * agent - 4] - len_T / 2)}
                )
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x: -(
                        np.linalg.norm(
                            sum(
                                [
                                    np.cross(
                                        x[6 * agent - 6 : 6 * agent - 3],
                                        x[6 * agent - 3 : 6 * agent - 0],
                                    )
                                    for agent in range(1, numChasers + 1)
                                ]
                            )
                            - torque
                        )
                        - epsilon
                    ),
                }
            )
        constraints = tuple(constraints)

        posit_forces = opt.minimize(
            forceMinFunc,
            x0=posit_forces,
            args=(torque, numChasers, constants),
            options={"disp": True, "maxiter": 150},
            constraints=constraints,
        ).x
        print("posit_forces: ", posit_forces)

    return posit_forces


def forceMinFunc(posit_forces, torque, numChasers, constants):
    inf = 1e7
    epsilon, rad_T, len_T, constraintsInCost = constants

    cost = sum(
        [
            np.linalg.norm(posit_forces[6 * agent - 3 : 6 * agent - 0]) ** 2
            for agent in range(1, numChasers + 1)
        ]
    )
    if constraintsInCost:
        for agent in range(1, numChasers + 1):
            if (
                np.linalg.norm(posit_forces[6 * agent - 6 : 6 * agent - 4]) - rad_T
            ) < -epsilon or (
                np.linalg.norm(posit_forces[6 * agent - 6 : 6 * agent - 4]) - rad_T
            ) > epsilon:
                cost += (
                    inf
                    * (
                        np.linalg.norm(posit_forces[6 * agent - 6 : 6 * agent - 4])
                        - rad_T
                    )
                    ** 2
                )
            if (posit_forces[6 * agent - 4] + len_T / 2) < 0:
                cost += inf * (posit_forces[6 * agent - 4] + len_T / 2)
            if (posit_forces[6 * agent - 4] - len_T / 2) > 0:
                cost += inf * (posit_forces[6 * agent - 4] - len_T / 2)
        if (
            np.linalg.norm(
                sum(
                    [
                        np.cross(
                            posit_forces[6 * agent - 6 : 6 * agent - 3],
                            posit_forces[6 * agent - 3 : 6 * agent - 0],
                        )
                        for agent in range(1, numChasers + 1)
                    ]
                )
                - torque
            )
            > epsilon
        ):
            cost += (
                inf
                * np.linalg.norm(
                    sum(
                        [
                            np.cross(
                                posit_forces[6 * agent - 6 : 6 * agent - 3],
                                posit_forces[6 * agent - 3 : 6 * agent - 0],
                            )
                            for agent in range(1, numChasers + 1)
                        ]
                    )
                    - torque
                )
                ** 2
            )
    return cost
