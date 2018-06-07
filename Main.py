#!/usr/bin/python

import sys, os
from klampt import *
from klampt.vis import GLSimulationProgram
import numpy as np
sys.path.insert(0, '/home/shihao/trajOptLib')
from trajOptLib.io import getOnOffArgs
from trajOptLib import trajOptCollocProblem
from trajOptLib.snoptWrapper import directSolve
from trajOptLib.libsnopt import snoptConfig, probFun, solver
import functools
import ipdb; ipdb.set_trace()

Inf = float("inf")
pi = 3.1415926535897932384626
Aux_Link_Ind = [1, 3, 5, 6, 7, 11, 12, 13, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 31, 33, 34, 35]  # This is the auxilliary joints
End_Effector_Ind = [11, 17, 27, 34]                     # The link index of the end effectors
Local_Extremeties = [0,0,1, 0 ,0 ,1, 1 ,0 ,0, 1 ,0 ,1];       # This 4 * 3 vector describes the local coordinate of the contact extremeties in their local coordinate
Environment = np.array([-5.0, 0.0, 5.0, 0.0])                     # The environment is defined by the edge points on the falling plane
Environment = np.append(Environment, [5.0, 0, 5.0, 3.0])          # The default are two walls: flat ground [(-5.0,0,0) to (5.0,0.0)] and the vertial wall [(5.0,0.0) to (5.0, 3.0)]
Environment_Normal = np.array([0.0, 0.0])                                       # Surface normal of the obs surface
mini = 0.05

def Environment_Normal_Cal(Environment):
    # This function will calculate the surface normal given the Environment
    global Environment_Normal
    Obs_Num = len(Environment)/4
    for i in range(0,Obs_Num):
        Environment_i = Environment[4*i:4*i+4]
        x1 = Environment_i[0:2]
        x2 = Environment_i[2:4]
        delta = x2 - x1
        tang_i = np.arctan2(delta[1], delta[0])
        Environment_Normal = np.append(Environment_Normal, [np.cos(tang_i + pi/2.0), np.sin(tang_i + pi/2.0)])
    Environment_Normal = Environment_Normal[2:]

def Dimension_Reduction(high_dim_obj):
    # The input to this function should be a np.array object
    # This function will trim down the unnecessary joints for the current research problem
    low_dim_obj = np.delete(high_dim_obj, Aux_Link_Ind)
    return low_dim_obj

class MyGLViewer(GLSimulationProgram):
    def __init__(self,world):
        #create a world from the given files
        self.world = world
        GLSimulationProgram.__init__(self,world,"My GL program")
        self.traj = model.trajectory.RobotTrajectory(world.robot(0))
        self.traj.load("/home/shihao/Klampt/data/motions/athlete_flex.path")
    def control_loop(self):
        #Put your control handler here
        sim = self.sim
        traj = self.traj
        starttime = 2.0
        if sim.getTime() > starttime:
            (q,dq) = (traj.eval(self.sim.getTime()-starttime),traj.deriv(self.sim.getTime()-starttime))
            sim.controller(0).setPIDCommand(q,dq)

    def contact_force_login(self):
        world = self.world
        terrain = TerrainModel();  # Now terrain is an instance of the TerrainModel class
        terrainid = terrain.getID()
        print terrainid
        # objectid = world.rigidObject(object_index).getID()
        # linkid = world.robot(robot_index).link(link_index).getID()
        # #equivalent to
        # linkid = world.robotlink(robot_index,link_index).getID()

    def mousefunc(self,button,state,x,y):
        #Put your mouse handler here
        #the current example prints out the list of objects clicked whenever
        #you right click
        print "mouse",button,state,x,y
        if button==2:
            if state==0:
                print [o.getName() for o in self.click_world(x,y)]
            return
        GLSimulationProgram.mousefunc(self,button,state,x,y)

    def motionfunc(self,x,y,dx,dy):
        return GLSimulationProgram.motionfunc(self,x,y,dx,dy)

def RobotInitState_Loader():
    # This function is used to load the robot_angle_init.txt and robot_angvel_init.txt into a numpy.ndarray instance
    with open("robot_angle_init.txt",'r') as robot_angle_file:
        robotstate_angle_i = robot_angle_file.readlines()
        angle_temp = [x.strip() for x in robotstate_angle_i]
    angle_temp = np.array(angle_temp, dtype = float)

    with open("robot_angvel_init.txt",'r') as robot_angvel_file:
        robotstate_angvel_i = robot_angvel_file.readlines()
        angvel_temp = [x.strip() for x in robotstate_angvel_i]
    angvel_temp = np.array(angvel_temp, dtype = float)
    return angle_temp, angvel_temp

class Initial_Robotstate_Validation_Prob(probFun):
    # This is the class type used for the initial condition validation optimization problem
    def __init__(self, world, sigma_init, robotstate_init):
        nx = len(robotstate_init)
        ObjNConstraint_Val, ObjNConstraint_Type = Robotstate_ObjNConstraint_Init(world, sigma_init, robotstate_init, robotstate_init)
        probFun.__init__(self, nx, len(ObjNConstraint_Type))
        self.grad = False
        self.world = world
        self.sigma_init = sigma_init
        self.robotstate_init = robotstate_init

    def __callf__(self, x, y):
        
        ObjNConstraint_Val, ObjNConstraint_Type = Robotstate_ObjNConstraint_Init(self.world, self.sigma_init, self.robotstate_init, robotstate_init)

    def __callg__(self, x, y, G, row, col, rec, needg):
        # This function will be used if the analytic gradient is provided
        y[0] = x[0] ** 2 + x[1] ** 2
        y[1] = x[0] + x[1]
        G[:2] = 2 * x
        G[2:] = 1.0
        if rec:
            row[:] = [0, 0, 1, 1]
            col[:] = [0, 1, 0, 1]

def End_Effector_Vel(hrp2_robot):
    # This function is used to return the global translational velocity of the origin of the end effector
    End_Effector_Vel_Array = np.array([0,0])
    End_Link_No_Index = -1
    for End_Effector_Link_Index in End_Effector_Ind:
        End_Link_No_Index = End_Link_No_Index + 1
        End_Link_i = hrp2_robot.link(End_Effector_Link_Index)
        End_Link_i_Extre_Loc = Local_Extremeties[End_Link_No_Index*3:End_Link_No_Index*3+3]
        End_Link_i_Extre_Vel = End_Link_i.getPointVelocity(End_Link_i_Extre_Loc)
        del End_Link_i_Extre_Vel[1]
        End_Effector_Vel_Array = np.append(End_Effector_Vel_Array, End_Link_i_Extre_Vel)
    return End_Effector_Vel_Array[2:]

def Initial_Condition_Validation(world, sigma_init, robotstate_init):
    # The inputs to this function is the WorldModel, initial contact mode, and the initial state
    # The output is the feasible robotstate
    hrp2_robot = world.robot(0)
    qmin, qmax = hrp2_robot.getJointLimits()
    qmin = Dimension_Reduction(qmin)
    qmax = Dimension_Reduction(qmax)

    dqmax_val = hrp2_robot.getVelocityLimits()
    dqmax_val = Dimension_Reduction(dqmax_val)

    dqmin = []
    dqmax = []
    for dqmax_val in dqmax_val:
        dqmax.append(dqmax_val)
        dqmin.append(-dqmax_val)
    xlb = np.zeros(len(robotstate_init))
    xub = np.zeros(len(robotstate_init))

    for i in range(0, len(robotstate_init)):
        if i<len(qmin):
            xlb[i] = qmin[i]
            xub[i] = qmax[i]
        else:
            xlb[i] = dqmin[i-len(qmin)]
            xub[i] = dqmax[i-len(qmin)]

    # Optimization problem setup
    Initial_Condition_Opt.xlb = xlb
    Initial_Condition_Opt.xub = xub
    ObjNConstraint_Val, ObjNConstraint_Type = Robotstate_ObjNConstraint_Init(world, sigma_init, robotstate_init, robotstate_init)
    lb = np.zeros(len(ObjNConstraint_Type))
    ub = np.zeros(len(ObjNConstraint_Type))
    for i in range(0,len(ObjNConstraint_Type)):
        lb[i] = 0.0
        if ObjNConstraint_Type[0]==0:
            ub[i] = 0.0
        else:
            ub[i] = Inf
    Initial_Condition_Opt.lb = lb
    Initial_Condition_Opt.ub = ub
    Initial_Condition_Opt = Initial_Robotstate_Validation_Prob(world, hrp2_robot, sigma_init, robotstate_init)


    cfg = snoptConfig()
    cfg.printLevel = 1
    cfg.printFile = "result.txt"
    slv = solver(Initial_Condition_Opt, cfg)
    # rst = slv.solveRand()
    rst = slv.solveGuess(x0)

    file_object  = open("Optimized_Angle.config", 'w')
    # print rst.sol
    file_object.write("36\t")
    for i in range(0,36):
        file_object.write(str(rst.sol[i]))
        file_object.write(' ')

    file_object.close()
    return rst.sol

    # clb = np.array([0, 1.0])
    #
    # cub = np.array([0, 1.0])
    # # test plain fun
    # rst = directSolve(plainFun, x0, nf=None, xlb=xlb, xub=xub, clb=clb, cub=cub)
    # print(rst)

def Robotstate_ObjNConstraint_Init(world, sigma_init, robotstate_init, robotstate_opt):
    # This function is used to generate the value of the objecttive function and the constraints
    # The inputs to this functions:
    #                             world -----------------> should be already updated by setConfig and setVelocity
    # The output of this function are two np.array objects: ObjNCon_Bds, ObjNCon_Vals

    # The constraints will be on the position and the velocity of the robot end effector extremeties

    robotstate_violation = np.subtract(robotstate_init, robotstate_opt)       # This measures how large it is from the given robotstate to the optimal robotstate
    ObjNConstraint_Val = [0]

    ObjNConstraint_Val.append(np.sum(np.square(robotstate_violation)))
    ObjNConstraint_Type = [1]                                                 # 1------------------> inequality constraint
    # Constraint 1: Distance constraint: This constraint is undertood in two folds:
    #                                    1. The "active" relative distance has to be zero
    #                                    2. The global orientations of certain end effectors have to be "flat"
    Rel_Dist, Nearest_Obs = Relative_Dist(world.robot(0))
    for i in range(len(ObjNConstraint_Type),len(Rel_Dist) + len(ObjNConstraint_Type)):
        ObjNConstraint_Val.append(Rel_Dist[i-1] * sigma_init[i-1])
        ObjNConstraint_Type.append([0])

    Right_Foot_Ori, Left_Foot_Ori = Foot_Orientation(hrp2_robot)
    ObjNConstraint_Val.append(sigma_init[0] * Right_Foot_Ori[0])
    ObjNConstraint_Type.append([0])
    ObjNConstraint_Val.append(sigma_init[1] * Left_Foot_Ori[0])
    ObjNConstraint_Type.append([0])

    # Constraint 2: The global velocity of certain body parts has to be zero
    End_Effector_Vel_Array = End_Effector_Vel(hrp2_robot) # This function return a list of 8 elements
    End_Effector_Vel_Matrix = np.diag([sigma_init[0],sigma_init[0],sigma_init[1],sigma_init[1],sigma_init[2],sigma_init[2],sigma_init[3],sigma_init[3]])
    End_Effector_Vel_Constraint = np.dot(End_Effector_Vel_Matrix, End_Effector_Vel_Array)
    for i in range(len(ObjNConstraint_Type),len(End_Effector_Vel_Constraint) + len(ObjNConstraint_Type)):
        ObjNConstraint_Val.append(End_Effector_Vel_Constraint[i])
        ObjNConstraint_Type.append([0])
    return ObjNConstraint_Val, ObjNConstraint_Type

def Relative_Dist(hrp2_robot):
    # This function is used to measure the relative distance between the robot to the nearest world feature
    # The environmental obstacles will be coded as terrians
    Relative_Dist_Array = np.zeros(len(End_Effector_Ind))
    Nearest_Obs_Array = np.zeros(len(End_Effector_Ind))
    End_Link_No_Index = -1
    for End_Effector_Link_Index in End_Effector_Ind:
        End_Link_No_Index = End_Link_No_Index + 1
        End_Link_i = hrp2_robot.link(End_Effector_Link_Index)
        End_Link_i_Extre_Loc = Local_Extremeties[End_Link_No_Index*3:End_Link_No_Index*3+3]
        End_Link_i_Extre_Pos = End_Link_i.getWorldPosition(End_Link_i_Extre_Loc)
        # Then the job is to compute the relative distance
        End_Link_i_Extre_dist, End_Link_i_Extre_obs_ind = Single_End_Effector_Obs_Dist(End_Link_i_Extre_Pos)
        Relative_Dist_Array[End_Link_No_Index] = End_Link_i_Extre_dist
        Nearest_Obs_Array[End_Link_No_Index] = End_Link_i_Extre_obs_ind
    return Relative_Dist_Array, Nearest_Obs_Array

def Single_End_Effector_Obs_Dist(rPos):
    # This function is used to calculate the relative distance between the robot end effectors and the obstacles
    # due to the simplification of the motion on the y plane here we only consider about the position of the end effector in x-z plane
    rPos = np.array([rPos[0],rPos[2]])
    Relative_Dist = [100000]           # Given a highest value for initialization
    for i in range(0,len(Environment)/4):
        Obs_Edge = Environment[4*i:4*i+2]
        Edge2rPos = np.subtract(rPos, Obs_Edge)
        Relative_Dist_i = np.dot(Edge2rPos, Environment_Normal[2*i:2*i+2])
        Relative_Dist.append(Relative_Dist_i)
    Relative_Dist = Relative_Dist[1:]
    return np.min(Relative_Dist), np.argmin(Relative_Dist)

def Foot_Orientation(hrp2_robot):
    # This function returns the orientation of the foot end effectors of the robot
    # Here we only care about the four links: right foot ,right hand ,left foot, left hand so we extract them
    Foot_Index_Array = End_Effector_Ind[0:2]

    End_Effector_Orientation_Array = np.zeros(len(Foot_Index_Array))

    Rigt_foot_link = hrp2_robot.link(Foot_Index_Array[0])
    Rigt_foot_Ori = Rigt_foot_link.getWorldDirection([1, 0, 0])
    # Since we only consider the Sagittal plane dynamics, we can make let the local axis align with the global axis

    Left_foot_link = hrp2_robot.link(Foot_Index_Array[1])
    Left_foot_Ori = Left_foot_link.getWorldDirection([1, 0, 0])
    return Rigt_foot_Ori,Left_foot_Ori

def main():
    # This funciton is used for the multi-contact humanoid push recovery
    # The default robot to be loaded is the HRP2 robot in this same folder
    print "This funciton is used for the multi-contact humanoid push recovery"
    if len(sys.argv)<=1:
        print "USAGE: The default robot to be loaded is the HRP2 robot in this same folder"
        exit()
    world = WorldModel() # WorldModel is a pre-defined class
    input_files = sys.argv[1:];  # sys.argv will automatically capture the input files' names

    for fn in input_files:
        result = world.readFile(fn) # Here result is a boolean variable indicating the result of this loading operation
        if not result:
            raise RuntimeError("Unable to load model "+fn)

    # Now the first job is to load the initial state of the robot from the Klampt config file into the
    init_state_cmd = './Config2Text'  # Config2Text is a program used to rewrite the row-wise Klampt config file into a column wise txt file
    os.system(init_state_cmd)
    # This system call will rewrite the robot_angle_init.config into the robot_angle_init.txt
    # However, the initiali angular velocities can be customerized by the user in robot_angvel_init.txt

    # # Now world has already read the world file
    # hrp2_robot = world.robot(0)  # Now hrp2_robot is an instance of RobotModel type
    # D_q = hrp2_robot.getMassMatrix()
    # C_q_qdot = hrp2_robot.getCoriolisForces()
    # G_q = hrp2_robot.getGravityForces((0,0,-9.8))
    #

    # The first step is to validate the feasibility of the given initial condition


    sigma_init = np.array([1,1,0,0])            # This is the initial contact status:  1__------> contact constraint is active,
                                            #                                      0--------> contact constraint is inactive
                                            #                                   [left foot, right foot, left hand, right hand]

    angle_init, angvel_init = RobotInitState_Loader()   # This is the initial condition: joint angle and joint angular velocities
    # The following two are used to reduce the auxilliary joints
    angle_init = Dimension_Reduction(angle_init)
    angvel_init = Dimension_Reduction(angvel_init)

    robotstate_length = len(angle_init) + len(angvel_init)
    robotstate_init = np.zeros(robotstate_length)
    for i in range(0, robotstate_length):
        if i<len(angle_init):
            robotstate_init[i] = angle_init[i]
        else:
            robotstate_init[i] = angvel_init[i-len(angle_init)]
    Environment_Normal_Cal(Environment)

    # Now it is the validation of the feasibility of the given initial condition
    Robotstate_Opt = Initial_Condition_Validation(world, sigma_init, robotstate_init)



    print Contact_ForceNTorque_Cal(world,Opt_Robotstate[0:36], Opt_Robotstate[36:])
    print "Gotcha"


    # MyGLViewer(world)

    # viewer = MyGLViewer(sys.argv[1:])
    # viewer.run()

if __name__ == "__main__":
    main()
