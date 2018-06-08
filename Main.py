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
Aux_Link_Ind = [1, 3, 5, 6, 7, 11, 12, 13, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 31, 33, 34, 35]  # This is the auxilliary joints causing lateral motion
Act_Link_Ind = [0, 2, 4, 8, 9, 10, 14, 15, 16, 22, 25, 29, 32]                                          # This is the active joints to be considered
Tot_Link_No = len(Aux_Link_Ind) + len(Act_Link_Ind)
End_Effector_Ind = [11, 11, 17, 17, 27, 34]                       # The link index of the end effectors
Local_Extremeties = [0.1, 0, -0.1, -0.15 , 0, -0.1, 0.1, 0, -0.1, -0.15 , 0, -0.1, 0, 0, -0.22, 0, 0, -0.205]         # This 6 * 3 vector describes the local coordinate of the contact extremeties in their local coordinate
Environment = np.array([-5.0, 0.0, 5.0, 0.0])                     # The environment is defined by the edge points on the falling plane
Environment = np.append(Environment, [5.0, 0, 5.0, 3.0])          # The default are two walls: flat ground [(-5.0,0,0) to (5.0,0.0)] and the vertial wall [(5.0,0.0) to (5.0, 3.0)]
Environment_Normal = np.array([0.0, 0.0])                                       # Surface normal of the obs surface
mini = 0.05
Grids = 20
class Tree_Node:
    def __init__(self, robot, sigma, robotstate):
        self.sigma = sigma
        self.robotstate = robotstate
        self.KE = KE_fn(robotstate)

        self.time = 0.0         # initialize to be zero
        self.StateNDot_Traj = []
        self.Ctrl_Traj = []
        self.Contact_Force_Traj = []

        self.self_time = 0.0
        self.self_StateNDot_Traj = []
        self.self_Ctrl_Traj = []
        self.self_Contact_Force_Traj = []

        self.End_Effector_Pos = get_End_Effector_Pos(robot)
        self.End_Effector_Vel = get_End_Effector_Vel(robot)

        self.Parent_Node = None
        self.Children_Nodes = []

    def Add_Child(self, Child_Node):
        self.Children_Nodes.append(Child_Node)

    def Add_Parent(self, Parent_Node):
        self.Parent_Node = Parent_Node
def Pop_Node(Frontier_Nodes, Frontier_Nodes_Cost):
    # This fucntion is used to pop the node with the minimum kinetic energy out of the Total nodes
    Minimum_Index = np.argmin(Frontier_Nodes_Cost)
    Node = Frontier_Nodes[Minimum_Index]
    Frontier_Nodes = np.delete(Frontier_Nodes, Minimum_Index)
    Frontier_Nodes_Cost = np.delete(Frontier_Nodes_Cost, Minimum_Index)
    return Node, Frontier_Nodes, Frontier_Nodes_Cost
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
def Dimension_Recovery(low_dim_obj):
    high_dim_obj = np.zeros(Tot_Link_No)
    for i in range(0,len(Act_Link_Ind)):
        high_dim_obj[Act_Link_Ind[i]] = low_dim_obj[i]
    return high_dim_obj
def KE_fn(robotstate):
    # Since the kinetic energy is not a built-in function, the external call will be made to get its value
    file_object  = open("robotstate4KE.txt", 'w')
    for i in range(0,26):
        file_object.write(str(robotstate[i]))
        file_object.write('\n')
    file_object.close()
    KE_cmd = './KE'  # Config2Text is a program used to rewrite the row-wise Klampt config file into a column wise txt file
    os.system(KE_cmd)
    with open("KE4robotstate.txt",'r') as KE_file:
        T = map(float, KE_file)
    return T[0]
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
        world = self.world
        robot = world.robot(0)
        Robot_ConfigNVel_Update(robot, x)

        ObjNConstraint_Val, ObjNConstraint_Type = Robotstate_ObjNConstraint_Init(world, self.sigma_init, self.robotstate_init, x)
        for i in range(0,len(ObjNConstraint_Val)):
            y[i] = ObjNConstraint_Val[i]

    def __callg__(self, x, y, G, row, col, rec, needg):
        # This function will be used if the analytic gradient is provided
        pass
class Seed_Robotstate_Optimization_Prob(probFun):
    # This is the class type used for the optimization for the seed robot state generation
    def __init__(self, world, Node_i, Node_i_child):
        nx = len(Node_i.robotstate)
        ObjNConstraint_Val, ObjNConstraint_Type = Robotstate_ObjNConstraint_Init(world, Node_i, Node_i_child)
        probFun.__init__(self, nx, len(ObjNConstraint_Type))
        self.grad = False
        self.world = world
        self.sigma_init = sigma_init
        self.robotstate_init = robotstate_init

    def __callf__(self, x, y):
        world = self.world
        robot = world.robot(0)
        Robot_ConfigNVel_Update(robot, x)

        ObjNConstraint_Val, ObjNConstraint_Type = Robotstate_ObjNConstraint_Init(world, self.sigma_init, self.robotstate_init, x)
        for i in range(0,len(ObjNConstraint_Val)):
            y[i] = ObjNConstraint_Val[i]

    def __callg__(self, x, y, G, row, col, rec, needg):
        # This function will be used if the analytic gradient is provided
        pass
def Robot_ConfigNVel_Update(robot, x):
    OptConfig_Low = x[0:len(x)/2]
    OptVelocity_Low = x[len(x)/2:]

    OptConfig_High = Dimension_Recovery(OptConfig_Low)
    OptVelocity_High = Dimension_Recovery(OptVelocity_Low)

    robot.setConfig(OptConfig_High)
    robot.setVelocity(OptVelocity_High)
def get_End_Effector_Pos(hrp2_robot):
    End_Effector_Pos_Array = np.array([0,0])
    End_Link_No_Index = -1
    for End_Effector_Link_Index in End_Effector_Ind:
        End_Link_No_Index = End_Link_No_Index + 1
        End_Link_i = hrp2_robot.link(End_Effector_Link_Index)
        End_Link_i_Extre_Loc = Local_Extremeties[End_Link_No_Index*3:End_Link_No_Index*3+3]
        End_Link_i_Extre_Pos = End_Link_i.getWorldPosition(End_Link_i_Extre_Loc)
        del End_Link_i_Extre_Pos[1]
        End_Effector_Pos_Array = np.append(End_Effector_Pos_Array, End_Link_i_Extre_Pos)
    return End_Effector_Pos_Array[2:]
def get_End_Effector_Vel(hrp2_robot):
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
    xlb, xub = Robotstate_Bounds(world, robotstate_init)

    # Optimization problem setup
    Initial_Condition_Opt = Initial_Robotstate_Validation_Prob(world, sigma_init, robotstate_init)
    Initial_Condition_Opt.xlb = xlb
    Initial_Condition_Opt.xub = xub
    ObjNConstraint_Val, ObjNConstraint_Type = Robotstate_ObjNConstraint_Init(world, sigma_init, robotstate_init, robotstate_init)
    lb, ub = ObjNCon_Bds(ObjNConstraint_Type)
    Initial_Condition_Opt.lb = lb
    Initial_Condition_Opt.ub = ub

    cfg = snoptConfig()
    cfg.printLevel = 1
    cfg.printFile = "result.txt"
    slv = solver(Initial_Condition_Opt, cfg)
    # rst = slv.solveRand()
    rst = slv.solveGuess(robotstate_init)

    # Then it is to take out the optimized robot configuration
    robot_angle_opt = np.zeros(Tot_Link_No)
    for i in range(0,len(Act_Link_Ind)):
        robot_angle_opt[Act_Link_Ind[i]] = rst.sol[i]

    file_object  = open("robot_angle_opt.config", 'w')
    file_object.write("36\t")
    for i in range(0,Tot_Link_No):
        file_object.write(str(robot_angle_opt[i]))
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
    ObjNConstraint_Val = ObjNConstraint_Val[1:]
    ObjNConstraint_Type = [1]                   # 1------------------> inequality constraint
    ObjNConstraint_Val, ObjNConstraint_Type = Distance_Velocity_Constraint(world, sigma_init, ObjNConstraint_Val, ObjNConstraint_Type)
    return ObjNConstraint_Val, ObjNConstraint_Type
def Distance_Velocity_Constraint(world, sigma_i, ObjNConstraint_Val, ObjNConstraint_Type):
    # Constraint 1: Distance constraint: This constraint is undertood in two folds:
    #                                    1. The "active" relative distance has to be zero
    #                                    2. The global orientations of certain end effectors have to be "flat"
    Rel_Dist, Nearest_Obs = Relative_Dist(world.robot(0))
    for i in range(0,len(Rel_Dist)):
        ObjNConstraint_Val.append(Rel_Dist[i] * sigma_i[i])
        ObjNConstraint_Type.append(0)
        ObjNConstraint_Val.append((not sigma_i[i]) *(Rel_Dist[i] - mini))
        ObjNConstraint_Type.append(1)

        Right_Foot_Ori, Left_Foot_Ori = Foot_Orientation(world.robot(0))
        ObjNConstraint_Val.append(sigma_i[0] * Right_Foot_Ori[2])
        ObjNConstraint_Type.append(0)
        ObjNConstraint_Val.append(sigma_i[1] * Left_Foot_Ori[2])
        ObjNConstraint_Type.append(0)

        # Constraint 2: The global velocity of certain body parts has to be zero
        End_Effector_Vel_Array = get_End_Effector_Vel(world.robot(0)) # This function return a list of 8 elements
        End_Effector_Vel_Matrix = np.diag([sigma_i[0], sigma_i[0], sigma_i[0], sigma_i[0], \
                                           sigma_i[1], sigma_i[1], sigma_i[1], sigma_i[1], \
                                           sigma_i[2], sigma_i[2], sigma_i[3], sigma_i[3]])
        End_Effector_Vel_Constraint = np.dot(End_Effector_Vel_Matrix, End_Effector_Vel_Array)
        for i in range(0,len(End_Effector_Vel_Constraint)):
            ObjNConstraint_Val.append(End_Effector_Vel_Constraint[i])
            ObjNConstraint_Type.append(0)
    return ObjNConstraint_Val, ObjNConstraint_Type
def Robotstate_ObjNConstraint_Seed(world, Node_i, Node_i_child, robotstate):
    # This function is used to generate the value of the objecttive function and the constraints
    # The constraints will be on the position and the velocity of the robot end effector extremeties
    KE_i_child = KE_fn(robotstate)
    ObjNConstraint_Val = [0]
    ObjNConstraint_Val.append(KE_i_child)
    ObjNConstraint_Val = ObjNConstraint_Val[1:]
    ObjNConstraint_Type = [1]                                                 # 1------------------> inequality constraint
    ObjNConstraint_Val, ObjNConstraint_Type = Distance_Velocity_Constraint(world, ObjNConstraint_Val, ObjNConstraint_Type)
    ObjNConstraint_Val, ObjNConstraint_Type = Contact_Maintenance(world, Node_i, Node_i_child, ObjNConstraint_Val, ObjNConstraint_Type)
    return ObjNConstraint_Val, ObjNConstraint_Type
def Contact_Maintenance(world, Node_i, Node_i_child, ObjNConstraint_Val, ObjNConstraint_Type):
    # This function is used to make sure that the previous contact should be maintained
    sigma_i = Sigma_Modi_De(Node_i.sigma)
    sigma_i_child = Sigma_Modi_De(Node_i_child.sigma)
    End_Effector_Pos = get_End_Effector_Pos(world.robot(0))         # 6 * 2 by 1
    End_Effector_Vel = get_End_Effector_Vel(world.robot(0))         # 6 * 2 by 1
    End_Effector_Pos_ref = Node_i.End_Effector_Pos
    End_Effector_Vel_ref = Node_i.End_Effector_Vel
    Maint_Matrix = np.diag([sigma_i[0] * sigma_i_child[0], sigma_i[0] * sigma_i_child[0], sigma_i[0] * sigma_i_child[0], sigma_i[0] * sigma_i_child[0], \
                             sigma_i[1] * sigma_i_child[1], sigma_i[1] * sigma_i_child[1], sigma_i[1] * sigma_i_child[1], sigma_i[1] * sigma_i_child[1], \
                             sigma_i[2] * sigma_i_child[2], sigma_i[2] * sigma_i_child[2], sigma_i[3] * sigma_i_child[3], sigma_i[3] * sigma_i_child[3]])
    End_Effector_Pos_Maint = np.subtract(End_Effector_Pos_ref, End_Effector_Pos)
    End_Effector_Vel_Maint = np.subtract(End_Effector_Vel_ref, End_Effector_Vel)

    End_Effector_Pos_Maint_Val = np.dot(Maint_Matrix, End_Effector_Pos_Maint)
    End_Effector_Vel_Maint_Val = np.dot(Maint_Matrix, End_Effector_Vel_Maint)
    for i in range(0, len(End_Effector_Pos_ref)):
        ObjNConstraint_Val.append(End_Effector_Pos_Maint_Val[i])
        ObjNConstraint_Type.append(0)
        ObjNConstraint_Val.append(End_Effector_Vel_Maint_Val[i])
        ObjNConstraint_Type.append(0)
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
def Nodes_Optimization_fn(world, Node_i, Node_i_child, Opt_Self_Flag):
    # // This function will optimize the joint trajectories to minimize the robot kinetic energy while maintaining a smooth transition fashion
    Opt_Seed = Seed_Guess_Gene(Node_i, Node_i_child)
def Seed_Guess_Gene(world, Node_i, Node_i_child):
    # This function is used to generate the intial guess used for the optimization

    # The first step is to generate a feasible configuration that can satisfy the contact mode in the node i child
    Seed_Guess_Gene_Robotstate(world, Node_i, Node_i_child)
def Seed_Guess_Gene_Robotstate(world, Node_i, Node_i_child):
    # This function is used to generate a desired configuration that satisfies the goal contact mode
    # There are two possibilities: one is self_opt, the other is connectivity_opt
    xlb, xub = Robotstate_Bounds(world, Node_i.robotstate)
    Seed_Robotstate_Optimization = Seed_Robotstate_Optimization_Prob(world, Node_i, Node_i_child)
    Seed_Robotstate_Optimization.xlb = xlb
    Seed_Robotstate_Optimization.xub = xub
    ObjNConstraint_Val, ObjNConstraint_Type = Robotstate_ObjNConstraint_Seed(world, Node_i, Node_i_child, robotstate)
    lb, ub = ObjNCon_Bds(ObjNConstraint_Type)
    Initial_Condition_Opt.lb = lb
    Initial_Condition_Opt.ub = ub

    cfg = snoptConfig()
    cfg.printLevel = 1
    cfg.printFile = "result.txt"
    slv = solver(Initial_Condition_Opt, cfg)
    # rst = slv.solveRand()
    rst = slv.solveGuess(Node_i.robotstate)

    # Then it is to take out the optimized robot configuration
    robot_angle_opt = np.zeros(Tot_Link_No)
    for i in range(0,len(Act_Link_Ind)):
        robot_angle_opt[Act_Link_Ind[i]] = rst.sol[i]

        file_object  = open("robot_angle_seed_opt.config", 'w')
        file_object.write("36\t")
        for i in range(0,Tot_Link_No):
            file_object.write(str(robot_angle_opt[i]))
            file_object.write(' ')
            file_object.close()
    return rst.sol
def ObjNCon_Bds(ObjNConstraint_Type):
    lb = np.zeros(len(ObjNConstraint_Type))
    ub = np.zeros(len(ObjNConstraint_Type))
    for i in range(0,len(ObjNConstraint_Type)):
        lb[i] = 0.0
        if ObjNConstraint_Type[i]==0:
            ub[i] = 0.0
        else:
            ub[i] = Inf
    return lb, ub
def Robotstate_Bounds(world, robotstate):
    hrp2_robot = world.robot(0)
    qmin, qmax = hrp2_robot.getJointLimits()
    qmin = Dimension_Reduction(qmin)
    qmax = Dimension_Reduction(qmax)
    dqmax_val = hrp2_robot.getVelocityLimits()
    dqmax_val = Dimension_Reduction(dqmax_val)
    dqmin = []
    dqmax = []
    for dqmax_val_i in dqmax_val:
        dqmax.append(dqmax_val_i)
        dqmin.append(-dqmax_val_i)
    xlb = np.zeros(len(robotstate))
    xub = np.zeros(len(robotstate))
    for i in range(0, len(robotstate)):
        if i<len(qmin):
            xlb[i] = qmin[i]
            xub[i] = qmax[i]
        else:
            xlb[i] = dqmin[i-len(qmin)]
            xub[i] = dqmax[i-len(qmin)]
    return xlb, xub
def Sigma_Modi_In(sigma_i):
    sigma_full = np.zeros(len(End_Effector_Ind))
    sigma_full[0] = sigma_i[0]
    sigma_full[1] = sigma_i[0]
    sigma_full[2] = sigma_i[1]
    sigma_full[3] = sigma_i[1]
    sigma_full[4] = sigma_i[2]
    sigma_full[5] = sigma_i[3]
    return sigma_full
def Sigma_Modi_De(sigma_i):
    sigma_full = np.zeros(4)
    sigma_full[0] = sigma_init[0]
    sigma_full[1] = sigma_init[2]
    sigma_full[2] = sigma_init[4]
    sigma_full[3] = sigma_init[5]
    return sigma_full
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
    sigma_init = Sigma_Modi_In(sigma_init)
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
    robotstate_init_Opted = Initial_Condition_Validation(world, sigma_init, robotstate_init)
    # The output is the optimized feasible initial condition for the robot: 26 by 1 list

    Robot_ConfigNVel_Update(world.robot(0),robotstate_init_Opted)
    RootNode = Tree_Node(world.robot(0), sigma_init, robotstate_init_Opted)

    All_Nodes = np.array([RootNode])

    Frontier_Nodes = np.array([RootNode])
    Frontier_Nodes_Cost = np.array([RootNode.KE])

    while len(Frontier_Nodes)>0:
        # /**
        # * For the current node, first is the Node_Self_Opt to optimize a motion while maintain the current mode (the fly in the air case will not be considered)
        # * 						if this does not work, then expand the current node into the adjacent nodes and figure out whether there exists a path to reach that contact mode
        # * Description
        # */
        Soln_Flag = 0
        Node, Frontier_Nodes, Frontier_Nodes_Cost = Pop_Node(Frontier_Nodes, Frontier_Nodes_Cost)
        Opt_Seed = Nodes_Optimization_fn(world, Node_i, Node_i_child, Opt_Self_Flag)
        print Opt_Seed

    # MyGLViewer(world)

    # viewer = MyGLViewer(sys.argv[1:])
    # viewer.run()
if __name__ == "__main__":
    main()
