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
from math import sin, cos, sqrt
import ipdb;
# ipdb.set_trace()
Inf = float("inf")
pi = 3.1415926535897932384626
Aux_Link_Ind = [1, 3, 5, 6, 7, 11, 12, 13, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 31, 33, 34, 35]  # This is the auxilliary joints causing lateral motion
Act_Link_Ind = [0, 2, 4, 8, 9, 10, 14, 15, 16, 22, 25, 29, 32]                                          # This is the active joints to be considered
Ctrl_Link_Ind = Act_Link_Ind[3:]
Tot_Link_No = len(Aux_Link_Ind) + len(Act_Link_Ind)
End_Effector_Ind = [11, 11, 17, 17, 27, 34]                       # The link index of the end effectors
Local_Extremeties = [0.1, 0, -0.1, -0.15 , 0, -0.1, 0.1, 0, -0.1, -0.15 , 0, -0.1, 0, 0, -0.22, 0, 0, -0.205]         # This 6 * 3 vector describes the local coordinate of the contact extremeties in their local coordinate
Environment = np.array([-5.0, 0.0, 5.0, 0.0])                     # The environment is defined by the edge points on the falling plane
Environment = np.append(Environment, [5.0, 0, 5.0, 3.0])          # The default are two walls: flat ground [(-5.0,0,0) to (5.0,0.0)] and the vertial wall [(5.0,0.0) to (5.0, 3.0)]
Environment_Normal = np.array([0.0, 0.0])                         # Surface normal of the obs surface
Environment_Tange = np.array([0.0,0.0])                           # Surface tangential vector
mini = 0.05
Grids = 10              # This is the grid number for each segment
mu = 0.5                # default friction coefficients
robotstate_lb = [0]
robotstate_ub = [0]
control_lb = [0]
control_ub = [0]
class Tree_Node:
    def __init__(self, robot, sigma, robotstate):
        self.sigma = sigma
        self.robotstate = robotstate
        self.KE = KE_fn(robotstate)
        self.Node_Index = 0

        # self.time = 0.0         # initialize to be zero
        # self.StateNDot_Traj = []
        # self.Ctrl_Traj = []
        # self.Contact_Force_Traj = []
        #
        # self.self_time = 0.0
        # self.self_StateNDot_Traj = []
        # self.self_Ctrl_Traj = []
        # self.self_Contact_Force_Traj = []

        self.End_Effector_Pos = get_End_Effector_Pos(robot).copy()
        self.End_Effector_Vel = get_End_Effector_Vel(robot).copy()

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
    global Environment_Tange
    Obs_Num = len(Environment)/4
    for i in range(0,Obs_Num):
        Environment_i = Environment[4*i:4*i+4]
        x1 = Environment_i[0:2]
        x2 = Environment_i[2:4]
        delta = x2 - x1
        tang_i = np.arctan2(delta[1], delta[0])
        Environment_Normal = np.append(Environment_Normal, [np.cos(tang_i + pi/2.0), np.sin(tang_i + pi/2.0)])
        Environment_Tange = np.append(Environment_Tange, [np.cos(tang_i), np.sin(tang_i)])
    Environment_Normal = Environment_Normal[2:]
    Environment_Tange = Environment_Tange[2:]
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
def KE_fn(dataArray):
    # # The first method is to read the cpp program but it is too slow
    # # Since the kinetic energy is not a built-in function, the external call will be made to get its value
    # file_object  = open("robotstate4KE.txt", 'w')
    # for i in range(0,26):
    #     file_object.write(str(robotstate[i]))
    #     file_object.write('\n')
    # file_object.close()
    # KE_cmd = './KE'  # Config2Text is a program used to rewrite the row-wise Klampt config file into a column wise txt file
    # os.system(KE_cmd)
    # with open("KE4robotstate.txt",'r') as KE_file:
    #     T = map(float, KE_file)
    # return T[0]
    rIx = dataArray[0]
    rIy = dataArray[1]
    theta = dataArray[2]
    q1 = dataArray[3]
    q2 = dataArray[4]
    q3 = dataArray[5]
    q4 = dataArray[6]
    q5 = dataArray[7]
    q6 = dataArray[8]
    q7 = dataArray[9]
    q8 = dataArray[10]
    q9 = dataArray[11]
    q10 = dataArray[12]
    rIxdot = dataArray[13]
    rIydot = dataArray[14]
    thetadot = dataArray[15]
    q1dot = dataArray[16]
    q2dot = dataArray[17]
    q3dot = dataArray[18]
    q4dot = dataArray[19]
    q5dot = dataArray[20]
    q6dot = dataArray[21]
    q7dot = dataArray[22]
    q8dot = dataArray[23]
    q9dot = dataArray[24]
    q10dot = dataArray[25]
    # The second method is to do the substitution of symbolic expression
    T = (q1dot*q1dot)*cos(q2)*2.9575E-1+(q1dot*q1dot)*cos(q3)*(1.3E1/5.0E2)+(q2dot*q2dot)*cos(q3)*(1.3E1/5.0E2)+(q4dot*q4dot)*cos(q5)*2.9575E-1+(q4dot*q4dot)*cos(q6)*(1.3E1/5.0E2)+(q5dot*q5dot)*cos(q6)*(1.3E1/5.0E2)+(q7dot*q7dot)*cos(q8)*(6.3E1/3.2E2)+(q9dot*q9dot)*cos(q10)*(6.3E1/3.2E2)+(thetadot*thetadot)*cos(q2)*2.9575E-1+(thetadot*thetadot)*cos(q3)*(1.3E1/5.0E2)+(thetadot*thetadot)*cos(q5)*2.9575E-1+(thetadot*thetadot)*cos(q6)*(1.3E1/5.0E2)-(thetadot*thetadot)*cos(q7)*6.25625E-1+(thetadot*thetadot)*cos(q8)*(6.3E1/3.2E2)-(thetadot*thetadot)*cos(q9)*6.25625E-1+(thetadot*thetadot)*cos(q10)*(6.3E1/3.2E2)-(q1dot*q1dot)*sin(q3)*(1.3E1/5.0E2)-(q2dot*q2dot)*sin(q3)*(1.3E1/5.0E2)-(q4dot*q4dot)*sin(q6)*(1.3E1/5.0E2)-(q5dot*q5dot)*sin(q6)*(1.3E1/5.0E2)-(thetadot*thetadot)*sin(q3)*(1.3E1/5.0E2)-(thetadot*thetadot)*sin(q6)*(1.3E1/5.0E2)+q10dot*q9dot*1.954166666666667E-1+q1dot*q2dot*2.785E-1+q1dot*q3dot*(1.0/4.0E1)+q2dot*q3dot*(1.0/4.0E1)+q4dot*q5dot*2.785E-1+q4dot*q6dot*(1.0/4.0E1)+q5dot*q6dot*(1.0/4.0E1)+q7dot*q8dot*1.954166666666667E-1+q10dot*thetadot*1.954166666666667E-1+q1dot*thetadot*8.418333333333333E-1+q2dot*thetadot*2.785E-1+q3dot*thetadot*(1.0/4.0E1)+q4dot*thetadot*8.418333333333333E-1+q5dot*thetadot*2.785E-1+q6dot*thetadot*(1.0/4.0E1)+q7dot*thetadot*4.824166666666667E-1+q8dot*thetadot*1.954166666666667E-1+q9dot*thetadot*4.824166666666667E-1+(q10dot*q10dot)*9.770833333333333E-2+(q1dot*q1dot)*4.209166666666667E-1+(q2dot*q2dot)*1.3925E-1+(q3dot*q3dot)*(1.0/8.0E1)+(q4dot*q4dot)*4.209166666666667E-1+(q5dot*q5dot)*1.3925E-1+(q6dot*q6dot)*(1.0/8.0E1)+(q7dot*q7dot)*2.412083333333333E-1+(q8dot*q8dot)*9.770833333333333E-2+(q9dot*q9dot)*2.412083333333333E-1+(rIxdot*rIxdot)*(2.71E2/1.0E1)+(rIydot*rIydot)*(2.71E2/1.0E1)+(thetadot*thetadot)*4.3795+(q1dot*q1dot)*cos(q2+q3)*(1.3E1/5.0E2)+(q4dot*q4dot)*cos(q5+q6)*(1.3E1/5.0E2)+(thetadot*thetadot)*cos(q2+q3)*(1.3E1/5.0E2)+(thetadot*thetadot)*cos(q5+q6)*(1.3E1/5.0E2)-(thetadot*thetadot)*cos(q7+q8)*4.33125E-1-(thetadot*thetadot)*cos(q9+q10)*4.33125E-1-(q1dot*q1dot)*sin(q2+q3)*(1.3E1/5.0E2)-(q4dot*q4dot)*sin(q5+q6)*(1.3E1/5.0E2)-(thetadot*thetadot)*sin(q2+q3)*(1.3E1/5.0E2)-(thetadot*thetadot)*sin(q5+q6)*(1.3E1/5.0E2)+q10dot*q9dot*cos(q10)*(6.3E1/3.2E2)+q1dot*q2dot*cos(q2)*2.9575E-1+q1dot*q2dot*cos(q3)*(1.3E1/2.5E2)+q1dot*q3dot*cos(q3)*(1.3E1/5.0E2)+q2dot*q3dot*cos(q3)*(1.3E1/5.0E2)+q4dot*q5dot*cos(q5)*2.9575E-1+q4dot*q5dot*cos(q6)*(1.3E1/2.5E2)+q4dot*q6dot*cos(q6)*(1.3E1/5.0E2)+q5dot*q6dot*cos(q6)*(1.3E1/5.0E2)+q7dot*q8dot*cos(q8)*(6.3E1/3.2E2)+q10dot*thetadot*cos(q10)*(6.3E1/3.2E2)+q1dot*thetadot*cos(q2)*5.915E-1+q1dot*thetadot*cos(q3)*(1.3E1/2.5E2)+q2dot*thetadot*cos(q2)*2.9575E-1+q2dot*thetadot*cos(q3)*(1.3E1/2.5E2)+q3dot*thetadot*cos(q3)*(1.3E1/5.0E2)+q4dot*thetadot*cos(q5)*5.915E-1+q4dot*thetadot*cos(q6)*(1.3E1/2.5E2)+q5dot*thetadot*cos(q5)*2.9575E-1+q5dot*thetadot*cos(q6)*(1.3E1/2.5E2)+q6dot*thetadot*cos(q6)*(1.3E1/5.0E2)-q7dot*thetadot*cos(q7)*6.25625E-1+q7dot*thetadot*cos(q8)*(6.3E1/1.6E2)+q8dot*thetadot*cos(q8)*(6.3E1/3.2E2)-q9dot*thetadot*cos(q9)*6.25625E-1+q9dot*thetadot*cos(q10)*(6.3E1/1.6E2)+rIxdot*thetadot*cos(theta)*1.3585E1+sqrt(4.1E1)*(q1dot*q1dot)*cos(8.960553845713439E-1)*(1.0/1.0E3)+sqrt(4.1E1)*(q2dot*q2dot)*cos(8.960553845713439E-1)*(1.0/1.0E3)+sqrt(4.1E1)*(q3dot*q3dot)*cos(8.960553845713439E-1)*(1.0/1.0E3)+sqrt(4.1E1)*(q4dot*q4dot)*cos(8.960553845713439E-1)*(1.0/1.0E3)+sqrt(4.1E1)*(q5dot*q5dot)*cos(8.960553845713439E-1)*(1.0/1.0E3)+sqrt(4.1E1)*(q6dot*q6dot)*cos(8.960553845713439E-1)*(1.0/1.0E3)+sqrt(4.1E1)*(thetadot*thetadot)*cos(8.960553845713439E-1)*(1.0/5.0E2)-q1dot*q2dot*sin(q3)*(1.3E1/2.5E2)-q1dot*q3dot*sin(q3)*(1.3E1/5.0E2)-q2dot*q3dot*sin(q3)*(1.3E1/5.0E2)-q4dot*q5dot*sin(q6)*(1.3E1/2.5E2)-q4dot*q6dot*sin(q6)*(1.3E1/5.0E2)-q5dot*q6dot*sin(q6)*(1.3E1/5.0E2)-q1dot*thetadot*sin(q3)*(1.3E1/2.5E2)-q2dot*thetadot*sin(q3)*(1.3E1/2.5E2)-q3dot*thetadot*sin(q3)*(1.3E1/5.0E2)-q4dot*thetadot*sin(q6)*(1.3E1/2.5E2)-q5dot*thetadot*sin(q6)*(1.3E1/2.5E2)-q6dot*thetadot*sin(q6)*(1.3E1/5.0E2)-rIydot*thetadot*sin(theta)*1.3585E1-sqrt(4.1E1)*(q1dot*q1dot)*sin(8.960553845713439E-1)*(1.0/1.0E3)-sqrt(4.1E1)*(q2dot*q2dot)*sin(8.960553845713439E-1)*(1.0/1.0E3)-sqrt(4.1E1)*(q3dot*q3dot)*sin(8.960553845713439E-1)*(1.0/1.0E3)-sqrt(4.1E1)*(q4dot*q4dot)*sin(8.960553845713439E-1)*(1.0/1.0E3)-sqrt(4.1E1)*(q5dot*q5dot)*sin(8.960553845713439E-1)*(1.0/1.0E3)-sqrt(4.1E1)*(q6dot*q6dot)*sin(8.960553845713439E-1)*(1.0/1.0E3)-sqrt(4.1E1)*(thetadot*thetadot)*sin(8.960553845713439E-1)*(1.0/5.0E2)-q10dot*rIxdot*cos(q9+q10+theta)*(6.3E1/8.0E1)-q1dot*rIxdot*cos(q1+q2+theta)*(9.1E1/1.0E2)-q2dot*rIxdot*cos(q1+q2+theta)*(9.1E1/1.0E2)-q4dot*rIxdot*cos(q4+q5+theta)*(9.1E1/1.0E2)-q5dot*rIxdot*cos(q4+q5+theta)*(9.1E1/1.0E2)-q7dot*rIxdot*cos(q7+q8+theta)*(6.3E1/8.0E1)-q8dot*rIxdot*cos(q7+q8+theta)*(6.3E1/8.0E1)-q9dot*rIxdot*cos(q9+q10+theta)*(6.3E1/8.0E1)-rIxdot*thetadot*cos(q1+q2+theta)*(9.1E1/1.0E2)-rIxdot*thetadot*cos(q4+q5+theta)*(9.1E1/1.0E2)-rIxdot*thetadot*cos(q7+q8+theta)*(6.3E1/8.0E1)-rIxdot*thetadot*cos(q9+q10+theta)*(6.3E1/8.0E1)+sqrt(4.1E1)*(q1dot*q1dot)*cos(q2+q3-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*(q4dot*q4dot)*cos(q5+q6-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*(thetadot*thetadot)*cos(q2+q3-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*(thetadot*thetadot)*cos(q5+q6-8.960553845713439E-1)*6.5E-3+q10dot*rIydot*sin(q9+q10+theta)*(6.3E1/8.0E1)+q1dot*rIydot*sin(q1+q2+theta)*(9.1E1/1.0E2)+q2dot*rIydot*sin(q1+q2+theta)*(9.1E1/1.0E2)+q4dot*rIydot*sin(q4+q5+theta)*(9.1E1/1.0E2)+q5dot*rIydot*sin(q4+q5+theta)*(9.1E1/1.0E2)+q7dot*rIydot*sin(q7+q8+theta)*(6.3E1/8.0E1)+q8dot*rIydot*sin(q7+q8+theta)*(6.3E1/8.0E1)+q9dot*rIydot*sin(q9+q10+theta)*(6.3E1/8.0E1)+rIydot*thetadot*sin(q1+q2+theta)*(9.1E1/1.0E2)+rIydot*thetadot*sin(q4+q5+theta)*(9.1E1/1.0E2)+rIydot*thetadot*sin(q7+q8+theta)*(6.3E1/8.0E1)+rIydot*thetadot*sin(q9+q10+theta)*(6.3E1/8.0E1)-q1dot*rIxdot*cos(q1+q2+q3+theta)*(2.0/2.5E1)-q2dot*rIxdot*cos(q1+q2+q3+theta)*(2.0/2.5E1)-q3dot*rIxdot*cos(q1+q2+q3+theta)*(2.0/2.5E1)-q4dot*rIxdot*cos(q4+q5+q6+theta)*(2.0/2.5E1)-q5dot*rIxdot*cos(q4+q5+q6+theta)*(2.0/2.5E1)-q6dot*rIxdot*cos(q4+q5+q6+theta)*(2.0/2.5E1)+q1dot*rIydot*cos(q1+q2+q3+theta)*(2.0/2.5E1)+q2dot*rIydot*cos(q1+q2+q3+theta)*(2.0/2.5E1)+q3dot*rIydot*cos(q1+q2+q3+theta)*(2.0/2.5E1)+q4dot*rIydot*cos(q4+q5+q6+theta)*(2.0/2.5E1)+q5dot*rIydot*cos(q4+q5+q6+theta)*(2.0/2.5E1)+q6dot*rIydot*cos(q4+q5+q6+theta)*(2.0/2.5E1)-rIxdot*thetadot*cos(q1+q2+q3+theta)*(2.0/2.5E1)-rIxdot*thetadot*cos(q4+q5+q6+theta)*(2.0/2.5E1)+rIydot*thetadot*cos(q1+q2+q3+theta)*(2.0/2.5E1)+rIydot*thetadot*cos(q4+q5+q6+theta)*(2.0/2.5E1)+q1dot*rIxdot*sin(q1+q2+q3+theta)*(2.0/2.5E1)+q2dot*rIxdot*sin(q1+q2+q3+theta)*(2.0/2.5E1)+q3dot*rIxdot*sin(q1+q2+q3+theta)*(2.0/2.5E1)+q4dot*rIxdot*sin(q4+q5+q6+theta)*(2.0/2.5E1)+q5dot*rIxdot*sin(q4+q5+q6+theta)*(2.0/2.5E1)+q6dot*rIxdot*sin(q4+q5+q6+theta)*(2.0/2.5E1)+q1dot*rIydot*sin(q1+q2+q3+theta)*(2.0/2.5E1)+q2dot*rIydot*sin(q1+q2+q3+theta)*(2.0/2.5E1)+q3dot*rIydot*sin(q1+q2+q3+theta)*(2.0/2.5E1)+q4dot*rIydot*sin(q4+q5+q6+theta)*(2.0/2.5E1)+q5dot*rIydot*sin(q4+q5+q6+theta)*(2.0/2.5E1)+q6dot*rIydot*sin(q4+q5+q6+theta)*(2.0/2.5E1)+rIxdot*thetadot*sin(q1+q2+q3+theta)*(2.0/2.5E1)+rIxdot*thetadot*sin(q4+q5+q6+theta)*(2.0/2.5E1)+rIydot*thetadot*sin(q1+q2+q3+theta)*(2.0/2.5E1)+rIydot*thetadot*sin(q4+q5+q6+theta)*(2.0/2.5E1)+q1dot*q2dot*cos(q2+q3)*(1.3E1/5.0E2)+q1dot*q3dot*cos(q2+q3)*(1.3E1/5.0E2)+q4dot*q5dot*cos(q5+q6)*(1.3E1/5.0E2)+q4dot*q6dot*cos(q5+q6)*(1.3E1/5.0E2)-q10dot*thetadot*cos(q9+q10)*4.33125E-1+q1dot*thetadot*cos(q2+q3)*(1.3E1/2.5E2)+q2dot*thetadot*cos(q2+q3)*(1.3E1/5.0E2)+q3dot*thetadot*cos(q2+q3)*(1.3E1/5.0E2)+q4dot*thetadot*cos(q5+q6)*(1.3E1/2.5E2)+q5dot*thetadot*cos(q5+q6)*(1.3E1/5.0E2)+q6dot*thetadot*cos(q5+q6)*(1.3E1/5.0E2)-q7dot*thetadot*cos(q7+q8)*4.33125E-1-q8dot*thetadot*cos(q7+q8)*4.33125E-1-q9dot*thetadot*cos(q9+q10)*4.33125E-1-q1dot*rIxdot*cos(q1+theta)*(3.9E1/2.0E1)-q4dot*rIxdot*cos(q4+theta)*(3.9E1/2.0E1)-q7dot*rIxdot*cos(q7+theta)*(9.1E1/8.0E1)-q9dot*rIxdot*cos(q9+theta)*(9.1E1/8.0E1)-rIxdot*thetadot*cos(q1+theta)*(3.9E1/2.0E1)-rIxdot*thetadot*cos(q4+theta)*(3.9E1/2.0E1)-rIxdot*thetadot*cos(q7+theta)*(9.1E1/8.0E1)-rIxdot*thetadot*cos(q9+theta)*(9.1E1/8.0E1)+sqrt(4.1E1)*(q1dot*q1dot)*cos(q3-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*(q2dot*q2dot)*cos(q3-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*(q4dot*q4dot)*cos(q6-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*(q5dot*q5dot)*cos(q6-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*(thetadot*thetadot)*cos(q3-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*(thetadot*thetadot)*cos(q6-8.960553845713439E-1)*6.5E-3-q1dot*q2dot*sin(q2+q3)*(1.3E1/5.0E2)-q1dot*q3dot*sin(q2+q3)*(1.3E1/5.0E2)-q4dot*q5dot*sin(q5+q6)*(1.3E1/5.0E2)-q4dot*q6dot*sin(q5+q6)*(1.3E1/5.0E2)-q1dot*thetadot*sin(q2+q3)*(1.3E1/2.5E2)-q2dot*thetadot*sin(q2+q3)*(1.3E1/5.0E2)-q3dot*thetadot*sin(q2+q3)*(1.3E1/5.0E2)-q4dot*thetadot*sin(q5+q6)*(1.3E1/2.5E2)-q5dot*thetadot*sin(q5+q6)*(1.3E1/5.0E2)-q6dot*thetadot*sin(q5+q6)*(1.3E1/5.0E2)+q1dot*rIydot*sin(q1+theta)*(3.9E1/2.0E1)+q4dot*rIydot*sin(q4+theta)*(3.9E1/2.0E1)+q7dot*rIydot*sin(q7+theta)*(9.1E1/8.0E1)+q9dot*rIydot*sin(q9+theta)*(9.1E1/8.0E1)+rIydot*thetadot*sin(q1+theta)*(3.9E1/2.0E1)+rIydot*thetadot*sin(q4+theta)*(3.9E1/2.0E1)+rIydot*thetadot*sin(q7+theta)*(9.1E1/8.0E1)+rIydot*thetadot*sin(q9+theta)*(9.1E1/8.0E1)+sqrt(4.1E1)*q1dot*q2dot*cos(q3-8.960553845713439E-1)*(1.3E1/1.0E3)+sqrt(4.1E1)*q1dot*q3dot*cos(q3-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*q2dot*q3dot*cos(q3-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*q4dot*q5dot*cos(q6-8.960553845713439E-1)*(1.3E1/1.0E3)+sqrt(4.1E1)*q4dot*q6dot*cos(q6-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*q5dot*q6dot*cos(q6-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*q1dot*thetadot*cos(q3-8.960553845713439E-1)*(1.3E1/1.0E3)+sqrt(4.1E1)*q2dot*thetadot*cos(q3-8.960553845713439E-1)*(1.3E1/1.0E3)+sqrt(4.1E1)*q3dot*thetadot*cos(q3-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*q4dot*thetadot*cos(q6-8.960553845713439E-1)*(1.3E1/1.0E3)+sqrt(4.1E1)*q5dot*thetadot*cos(q6-8.960553845713439E-1)*(1.3E1/1.0E3)+sqrt(4.1E1)*q6dot*thetadot*cos(q6-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*q1dot*q2dot*cos(8.960553845713439E-1)*(1.0/5.0E2)+sqrt(4.1E1)*q1dot*q3dot*cos(8.960553845713439E-1)*(1.0/5.0E2)+sqrt(4.1E1)*q2dot*q3dot*cos(8.960553845713439E-1)*(1.0/5.0E2)+sqrt(4.1E1)*q4dot*q5dot*cos(8.960553845713439E-1)*(1.0/5.0E2)+sqrt(4.1E1)*q4dot*q6dot*cos(8.960553845713439E-1)*(1.0/5.0E2)+sqrt(4.1E1)*q5dot*q6dot*cos(8.960553845713439E-1)*(1.0/5.0E2)+sqrt(4.1E1)*q1dot*thetadot*cos(8.960553845713439E-1)*(1.0/5.0E2)+sqrt(4.1E1)*q2dot*thetadot*cos(8.960553845713439E-1)*(1.0/5.0E2)+sqrt(4.1E1)*q3dot*thetadot*cos(8.960553845713439E-1)*(1.0/5.0E2)+sqrt(4.1E1)*q4dot*thetadot*cos(8.960553845713439E-1)*(1.0/5.0E2)+sqrt(4.1E1)*q5dot*thetadot*cos(8.960553845713439E-1)*(1.0/5.0E2)+sqrt(4.1E1)*q6dot*thetadot*cos(8.960553845713439E-1)*(1.0/5.0E2)-sqrt(4.1E1)*q1dot*q2dot*sin(8.960553845713439E-1)*(1.0/5.0E2)-sqrt(4.1E1)*q1dot*q3dot*sin(8.960553845713439E-1)*(1.0/5.0E2)-sqrt(4.1E1)*q2dot*q3dot*sin(8.960553845713439E-1)*(1.0/5.0E2)-sqrt(4.1E1)*q4dot*q5dot*sin(8.960553845713439E-1)*(1.0/5.0E2)-sqrt(4.1E1)*q4dot*q6dot*sin(8.960553845713439E-1)*(1.0/5.0E2)-sqrt(4.1E1)*q5dot*q6dot*sin(8.960553845713439E-1)*(1.0/5.0E2)-sqrt(4.1E1)*q1dot*thetadot*sin(8.960553845713439E-1)*(1.0/5.0E2)-sqrt(4.1E1)*q2dot*thetadot*sin(8.960553845713439E-1)*(1.0/5.0E2)-sqrt(4.1E1)*q3dot*thetadot*sin(8.960553845713439E-1)*(1.0/5.0E2)-sqrt(4.1E1)*q4dot*thetadot*sin(8.960553845713439E-1)*(1.0/5.0E2)-sqrt(4.1E1)*q5dot*thetadot*sin(8.960553845713439E-1)*(1.0/5.0E2)-sqrt(4.1E1)*q6dot*thetadot*sin(8.960553845713439E-1)*(1.0/5.0E2)-sqrt(4.1E1)*q1dot*rIxdot*cos(q1+q2+q3+theta-8.960553845713439E-1)*(1.0/5.0E1)-sqrt(4.1E1)*q2dot*rIxdot*cos(q1+q2+q3+theta-8.960553845713439E-1)*(1.0/5.0E1)-sqrt(4.1E1)*q3dot*rIxdot*cos(q1+q2+q3+theta-8.960553845713439E-1)*(1.0/5.0E1)-sqrt(4.1E1)*q4dot*rIxdot*cos(q4+q5+q6+theta-8.960553845713439E-1)*(1.0/5.0E1)-sqrt(4.1E1)*q5dot*rIxdot*cos(q4+q5+q6+theta-8.960553845713439E-1)*(1.0/5.0E1)-sqrt(4.1E1)*q6dot*rIxdot*cos(q4+q5+q6+theta-8.960553845713439E-1)*(1.0/5.0E1)-sqrt(4.1E1)*rIxdot*thetadot*cos(q1+q2+q3+theta-8.960553845713439E-1)*(1.0/5.0E1)-sqrt(4.1E1)*rIxdot*thetadot*cos(q4+q5+q6+theta-8.960553845713439E-1)*(1.0/5.0E1)+sqrt(4.1E1)*q1dot*rIydot*sin(q1+q2+q3+theta-8.960553845713439E-1)*(1.0/5.0E1)+sqrt(4.1E1)*q2dot*rIydot*sin(q1+q2+q3+theta-8.960553845713439E-1)*(1.0/5.0E1)+sqrt(4.1E1)*q3dot*rIydot*sin(q1+q2+q3+theta-8.960553845713439E-1)*(1.0/5.0E1)+sqrt(4.1E1)*q4dot*rIydot*sin(q4+q5+q6+theta-8.960553845713439E-1)*(1.0/5.0E1)+sqrt(4.1E1)*q5dot*rIydot*sin(q4+q5+q6+theta-8.960553845713439E-1)*(1.0/5.0E1)+sqrt(4.1E1)*q6dot*rIydot*sin(q4+q5+q6+theta-8.960553845713439E-1)*(1.0/5.0E1)+sqrt(4.1E1)*rIydot*thetadot*sin(q1+q2+q3+theta-8.960553845713439E-1)*(1.0/5.0E1)+sqrt(4.1E1)*rIydot*thetadot*sin(q4+q5+q6+theta-8.960553845713439E-1)*(1.0/5.0E1)+sqrt(4.1E1)*q1dot*q2dot*cos(q2+q3-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*q1dot*q3dot*cos(q2+q3-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*q4dot*q5dot*cos(q5+q6-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*q4dot*q6dot*cos(q5+q6-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*q1dot*thetadot*cos(q2+q3-8.960553845713439E-1)*(1.3E1/1.0E3)+sqrt(4.1E1)*q2dot*thetadot*cos(q2+q3-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*q3dot*thetadot*cos(q2+q3-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*q4dot*thetadot*cos(q5+q6-8.960553845713439E-1)*(1.3E1/1.0E3)+sqrt(4.1E1)*q5dot*thetadot*cos(q5+q6-8.960553845713439E-1)*6.5E-3+sqrt(4.1E1)*q6dot*thetadot*cos(q5+q6-8.960553845713439E-1)*6.5E-3
    return T
def Spline2RobotstateNKe(Opt_Soln, Grid_Index, sigma_i, sigma_i_child, robot):
    # This function is used to calcuale robot state and kinetic energy based on the the spline coefficients
    T, StateNDot_Coeff, Ctrl_Coeff, Contact_Force_Coeff = Seed_Guess_Unzip(Opt_Soln)
    if Grid_Index == Grids:
        Pos,  Vel,  Acc,  VelfromPos  = StateNDot_Coeff2PosVelAcc(T, StateNDot_Coeff, Grids-1, 1)
    else:
        Pos,  Vel,  Acc,  VelfromPos  = StateNDot_Coeff2PosVelAcc(T, StateNDot_Coeff, Grids, 0)

    # Here we need to know whether there is impact involved in this whole process
    sigma_result = np.subtract(sigma_i_child, sigma_i)
    if np.max(sigma_result)<1:
        # No impact mapping
        robotstate = List_Append(Pos, Vel)
    else:
        # There is impact mapping
        Pos, Vel = Impact_Mapping_fn(robot, Pos, Vel, Acc)
        robotstate = List_Append(Pos, Vel)
    KE = KE_fn(robotstate)
    return robotstate, KE
def Impact_Mapping_fn(robot, Pos, Vel, Acc):
    # This function is used to take carfe of the impact maping case
    D_q, C_q_qdot, G_q, End_Effector_JacTrans = Dynamics_Matrics(robot, Pos, Vel, Acc)
    End_Effector_Jac = np.transpose(End_Effector_JacTrans)
    D_q_inv = np.linalg.inv(D_q)
    Jac_Dinv_Jtrans_inv = np.linalg.inv(np.dot(np.dot(End_Effector_Jac, D_q_inv)), End_Effector_JacTrans)
    # Impulse
    lamda = -np.dot(np.dot(Jac_Dinv_Jtrans_inv, End_Effector_Jac), Vel)
    # New robotstate
    RHS = np.dot(np.dot(D_q_inv, End_Effector_JacTrans), lamda)
    Vel_new = np.zeros(len(Vel))
    for i range(0, len(Vel)):
        Vel_new[i] = Vel[i] + RHS[i]
    return Pos, Vel
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
        ObjNConstraint_Val, ObjNConstraint_Type = Robotstate_ObjNConstraint_Seed(world, Node_i, Node_i_child, Node_i.robotstate)
        probFun.__init__(self, nx, len(ObjNConstraint_Type))
        self.grad = False
        self.world = world
        self.Node_i = Node_i
        self.Node_i_child = Node_i_child

    def __callf__(self, x, y):
        world = self.world
        robot = world.robot(0)
        Robot_ConfigNVel_Update(robot, x)
        ObjNConstraint_Val, ObjNConstraint_Type = Robotstate_ObjNConstraint_Seed(world, self.Node_i, self.Node_i_child, x)
        for i in range(0,len(ObjNConstraint_Val)):
            y[i] = ObjNConstraint_Val[i]

    def __callg__(self, x, y, G, row, col, rec, needg):
        # This function will be used if the analytic gradient is provided
        pass
class Real_Optimization_fn(probFun):
    # This is the class type used for the optimization for the seed robot state generation
    def __init__(self, world, Node_i, Node_i_child, Opt_Seed):
        nx = len(Opt_Seed)
        ObjNConstraint_Val, ObjNConstraint_Type = Real_ObjNConstraint(world, Node_i, Node_i_child, Opt_Seed)
        probFun.__init__(self, nx, len(ObjNConstraint_Type))
        self.grad = False
        self.world = world
        self.Node_i = Node_i
        self.Node_i_child = Node_i_child

    def __callf__(self, x, y):
        world = self.world
        ObjNConstraint_Val, ObjNConstraint_Type = Real_ObjNConstraint(world, self.Node_i, self.Node_i_child, x)
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
    global robotstate_lb
    global robotstate_ub
    global control_lb
    global control_ub

    robotstate_xlb, robotstate_xub = Robotstate_Bounds(world, robotstate_init)
    robotstate_lb = robotstate_xlb
    robotstate_ub = robotstate_xub

    control_xlb, control_xub = Control_Bounds(world,len(Ctrl_Link_Ind))
    control_lb = control_xlb
    control_ub = control_xub
    # Optimization problem setup
    Initial_Condition_Opt = Initial_Robotstate_Validation_Prob(world, sigma_init, robotstate_init)
    Initial_Condition_Opt.xlb = robotstate_lb
    Initial_Condition_Opt.xub = robotstate_ub
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

    # ipdb.set_trace()

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
    ObjNConstraint_Val, ObjNConstraint_Type = Distance_Velocity_Constraint(world.robot(0), sigma_init, ObjNConstraint_Val, ObjNConstraint_Type)
    return ObjNConstraint_Val, ObjNConstraint_Type
def Distance_Velocity_Constraint(robot, sigma_i, ObjNConstraint_Val, ObjNConstraint_Type):
    # Constraint 1: Distance constraint: This constraint is undertood in two folds:
    #                                    1. The "active" relative distance has to be zero
    #                                    2. The global orientations of certain end effectors have to be "flat"
    Rel_Dist, Nearest_Obs = Relative_Dist(robot)
    for i in range(0,len(Rel_Dist)):
        ObjNConstraint_Val.append(Rel_Dist[i] * sigma_i[i])
        ObjNConstraint_Type.append(0)
        ObjNConstraint_Val.append((not sigma_i[i]) *(Rel_Dist[i] - mini))
        ObjNConstraint_Type.append(1)

    Right_Foot_Ori, Left_Foot_Ori = Foot_Orientation(robot)
    ObjNConstraint_Val.append(sigma_i[0] * Right_Foot_Ori[2])
    ObjNConstraint_Type.append(0)
    ObjNConstraint_Val.append(sigma_i[1] * Left_Foot_Ori[2])
    ObjNConstraint_Type.append(0)

    # Constraint 2: The global velocity of certain body parts has to be zero
    End_Effector_Vel_Array = get_End_Effector_Vel(robot) # This function return a list of 8 elements
    End_Effector_Vel_Matrix = np.diag([sigma_i[0], sigma_i[0], sigma_i[0], sigma_i[0], \
                                       sigma_i[1], sigma_i[1], sigma_i[1], sigma_i[1], \
                                       sigma_i[2], sigma_i[2], sigma_i[3], sigma_i[3]])
    End_Effector_Vel_Constraint = np.dot(End_Effector_Vel_Matrix, End_Effector_Vel_Array)
    for j in range(0,len(End_Effector_Vel_Constraint)):
        ObjNConstraint_Val.append(End_Effector_Vel_Constraint[j])
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

    ObjNConstraint_Val, ObjNConstraint_Type = Distance_Velocity_Constraint(world.robot(0), Node_i_child.sigma, ObjNConstraint_Val, ObjNConstraint_Type)
    ObjNConstraint_Val, ObjNConstraint_Type = Contact_Maintenance(world.robot(0), Node_i, Node_i_child, ObjNConstraint_Val, ObjNConstraint_Type)
    return ObjNConstraint_Val, ObjNConstraint_Type
def Real_ObjNConstraint(world, Node_i, Node_i_child, Opt_Seed):
    # This function is used to generate the value of the objecttive function and the constraints
    # The constraints will be on the position and the velocity of the robot end effector extremeties
    T, StateNDot_Coeff, Ctrl_Coeff, Contact_Force_Coeff = Seed_Guess_Unzip(Opt_Seed)

    Pos_ref = Node_i.robotstate[0:len(Act_Link_Ind)]
    Vel_ref = Node_i.robotstate[len(Act_Link_Ind):]

    Ctrl_ref = Control_Coeff2Vec(T, Ctrl_Coeff, 0, 0)
    Contact_Force_ref = Contact_Force_Coeff2Vec(T, Contact_Force_Coeff, 0, 0)

    ObjNConstraint_Val = [0]
    ObjNConstraint_Type = [1]                                                 # 1------------------> inequality constraint

    KE_tot = [Node_i.KE]
    if np.max(np.subtract(Node_i_child.sigma, Node_i.sigma)) == 1:
        sigma_tran = Node_i.sigma
        sigma_goal = Node_i_child.sigma
    else:
        sigma_tran = Node_i_child.sigma
        sigma_goal = Node_i_child.sigma
    if np.array_equal(Node_i.sigma, Node_i_child.sigma) == True:
        KE_Const = 0.1
    else:
        KE_Const = 0.2 * Node_i.KE

    for i in range(0,Grids-1):
        # Extract the coefficients for these Grids-1 splines
        # For init and end on this segment

        Pos_init, Vel_init, Acc_init, VelfromPos_init = StateNDot_Coeff2PosVelAcc(T, StateNDot_Coeff, i, 0)
        Pos_end,  Vel_end,  Acc_end,  VelfromPos_end  = StateNDot_Coeff2PosVelAcc(T, StateNDot_Coeff, i, 1)

        robotstate_init = List_Append(Pos_init, Vel_init)
        robotstate_end = List_Append(Pos_end, Vel_end)

        KE_i = KE_fn(robotstate_end)
        KE_tot.append(KE_i)

        Ctrl_init = Control_Coeff2Vec(T, Ctrl_Coeff, i, 0)
        Ctrl_end = Control_Coeff2Vec(T, Ctrl_Coeff, i, 1)

        Contact_Force_init = Contact_Force_Coeff2Vec(T, Contact_Force_Coeff, i, 0)
        Contact_Force_end = Contact_Force_Coeff2Vec(T, Contact_Force_Coeff, i, 1)

        # 1. Constraint on smoothness at both edges
        # Robotstate, Control and Control force
        for j in range(0,len(Pos_init)):
            ObjNConstraint_Val.append(Pos_init[j] - Pos_ref[j])
            ObjNConstraint_Type.append(0)
            ObjNConstraint_Val.append(Vel_init[j] - Vel_ref[j])
            ObjNConstraint_Type.append(0)
            ObjNConstraint_Val.append(VelfromPos_init[j] - Vel_init[j])
            ObjNConstraint_Type.append(0)
            ObjNConstraint_Val.append(Pos_init[j] - robotstate_lb[j])
            ObjNConstraint_Type.append(1)
            ObjNConstraint_Val.append(robotstate_ub[j] - Pos_init[j])
            ObjNConstraint_Type.append(1)
            ObjNConstraint_Val.append(Vel_init[j] - robotstate_lb[j+len(Act_Link_Ind)])
            ObjNConstraint_Type.append(1)
            ObjNConstraint_Val.append(robotstate_ub[j+len(Act_Link_Ind)] - Vel_init[j])
            ObjNConstraint_Type.append(1)
            if j<len(Ctrl_Link_Ind):
                ObjNConstraint_Val.append(Ctrl_init[j] - Ctrl_ref[j])
                ObjNConstraint_Type.append(0)
                ObjNConstraint_Val.append(Ctrl_init[j] - control_lb[j])
                ObjNConstraint_Type.append(1)
                ObjNConstraint_Val.append(control_ub[j] - Ctrl_init[j])
                ObjNConstraint_Type.append(1)
            if j<2*len(End_Effector_Ind):
                ObjNConstraint_Val.append(Contact_Force_init[j] - Contact_Force_ref[j])
                ObjNConstraint_Type.append(1)

        # New references get updated
        Pos_ref = Pos_end
        Vel_ref = Vel_end
        Ctrl_ref = Ctrl_end
        Contact_Force_ref = Contact_Force_end

        # 2. Dynamics Constraints
        # 3. Complementarity constraints: Distance!
        # 4. Complementarity constraints: Contact Force!
        # 5. Contact force feasibility constraints: normal force should be positive and the friction cone constraint has to be satisfied
        # 6. Contact maintenance constraint: the previous unchanged active constraint have to be satisfied
        # All the above constraint will be taken care of

        Real_ObjNConstraint_Stage(world.robot(0), Node_i, Node_i_child, T, Pos_init, Vel_init, Acc_init, Ctrl_init, Contact_Force_init, sigma_tran, ObjNConstraint_Val, ObjNConstraint_Type)

        # End stage constraint
        if i==Grids-1:
            Real_ObjNConstraint_Stage(world.robot(0), Node_i, Node_i_child, T, Pos_end, Vel_end, Acc_end, Ctrl_end, Contact_Force_end, sigma_goal, ObjNConstraint_Val, ObjNConstraint_Type)

    # Final computation is to update the objective function value and add constraint
    ObjNConstraint_Val[0] = Obj_Cal(KE_tot, T)
    ObjNConstraint_Val.append(KE_Const - KE_tot[0])
    ObjNConstraint_Type.append(1)

    # ipdb.set_trace()
    return ObjNConstraint_Val, ObjNConstraint_Type
def Real_ObjNConstraint_Stage(robot, Node_i, Node_i_child, T, Pos, Vel, Acc, Ctrl, Contact_Force, sigma, ObjNConstraint_Val, ObjNConstraint_Type):
    # This function will compute the 2 to 6 constraints

    # 2. Dynamics Constraints
    temp_result = Dynamics_Constraint(robot, T, Pos, Vel, Acc, Ctrl, Contact_Force)
    for j in range(0,len(temp_result)):
        ObjNConstraint_Val.append(temp_result[j])
        ObjNConstraint_Type.append(0)

    # 3. Complementarity constraints: Distance!
    ObjNConstraint_Val, ObjNConstraint_Type = Distance_Velocity_Constraint(robot, sigma, ObjNConstraint_Val, ObjNConstraint_Type)

    # 4. Complementarity constraints: Contact Force!
    ObjNConstraint_Val, ObjNConstraint_Type = Contact_Force_Complem_Constraint(Contact_Force, sigma, ObjNConstraint_Val, ObjNConstraint_Type)

    # 5. Contact force feasibility constraints: normal force should be positive and the friction cone constraint has to be satisfied
    robotstate = List_Append(Pos, Vel)
    ObjNConstraint_Val, ObjNConstraint_Type = Contact_Force_Feasibility_Constraint(robot, robotstate, Contact_Force, ObjNConstraint_Val, ObjNConstraint_Type)

    # 6. Contact maintenance constraint: the previous unchanged active constraint have to be satisfied
    ObjNConstraint_Val, ObjNConstraint_Type = Contact_Maintenance(robot, Node_i, Node_i_child, ObjNConstraint_Val, ObjNConstraint_Type)

    return ObjNConstraint_Val, ObjNConstraint_Type
def Obj_Cal(KE_tot, T):
    # This function is used to calculate the quadratic sum of the rate of the change of the kinetic energy
    dt = T/(Grids - 1) * 1.0
    Obj = 0.0
    for i in range(0, len(KE_tot)-1):
        # Obj = Obj + (KE_tot[i+1] - KE_tot[i]) * (KE_tot[i+1] - KE_tot[i])/(dt * dt)
        Obj = Obj + (KE_tot[i+1] - KE_tot[i]) * (KE_tot[i+1] - KE_tot[i])
    return Obj
def List_Append(List_A, List_B):
    List_C = [0]
    for i in range(0,len(List_A)):
        List_C.append(List_A[i])
    for i in range(0,len(List_B)):
        List_C.append(List_B[i])
    return List_C[1:]
def Contact_Force_Feasibility_Constraint(robot, robotstate, Contact_Force, ObjNConstraint_Val, ObjNConstraint_Type):
    Norm_Force = [0]
    Tang_Force = [0]
    Normal_Force_i, Tang_Force_i = Contact_Force_Proj(robot, robotstate, Contact_Force)
    Norm_Force.append(Normal_Force_i[0] + Normal_Force_i[1]);
    Tang_Force.append(Tang_Force_i[0] + Tang_Force_i[1]);
    Norm_Force.append(Normal_Force_i[2] + Normal_Force_i[3])
    Tang_Force.append(Tang_Force_i[2] + Tang_Force_i[3])
    Norm_Force.append(Normal_Force_i[4])
    Tang_Force.append(Tang_Force_i[4])
    Norm_Force.append(Normal_Force_i[5])
    Tang_Force.append(Tang_Force_i[5])
    Norm_Force = Norm_Force[1:]
    Tang_Force = Tang_Force[1:]
    for i in range(0, len(Norm_Force)):
        ObjNConstraint_Val.append(Norm_Force[i])
        ObjNConstraint_Type.append(1)

    for j in range(0,len(Norm_Force)):
        ObjNConstraint_Val.append(Norm_Force[j] * Norm_Force[j] * mu * mu - Tang_Force[j] * Tang_Force[j])
        ObjNConstraint_Type.append(1)
    # ipdb.set_trace()
    return ObjNConstraint_Val, ObjNConstraint_Type
def Contact_Force_Proj(robot, robotstate, contact_force):
    Robot_ConfigNVel_Update(robot, robotstate)
    _, Nearest_Obs_index = Relative_Dist(robot)
    Normal_Force = [0]
    Tange_Force = [0]
    # ipdb.set_trace()
    for i in range(0,len(Nearest_Obs_index)):
        Contact_Force_i = contact_force[2*i:2*i+2]
        Nearest_Obs_index_i = Nearest_Obs_index[i].astype(int)
        Environment_Normal_i = Environment_Normal[2*Nearest_Obs_index_i:2*Nearest_Obs_index_i+2]
        Environment_Tange_i = Environment_Tange[2*Nearest_Obs_index_i:2*Nearest_Obs_index_i+2]
        Normal_Force_i = np.dot(Contact_Force_i, Environment_Normal_i)
        Tange_Force_i = np.dot(Contact_Force_i, Environment_Tange_i)
        Normal_Force.append(Normal_Force_i)
        Tange_Force.append(Tange_Force_i)
    Normal_Force = Normal_Force[1:]
    Tange_Force = Tange_Force[1:]
    return Normal_Force, Tange_Force
def Contact_Force_Complem_Constraint(Contact_Force, sigma_i, ObjNConstraint_Val, ObjNConstraint_Type):
    Contact_Force_Complem_Matrix = np.diag([not sigma_i[0], not sigma_i[0], not sigma_i[0], not sigma_i[0], \
                                            not sigma_i[1], not sigma_i[1], not sigma_i[1], not sigma_i[1], \
                                            not sigma_i[2], not sigma_i[2], not sigma_i[3], not sigma_i[3]])
    temp_result = np.dot(Contact_Force_Complem_Matrix, Contact_Force)
    for i in range(0, len(temp_result)):
        ObjNConstraint_Val.append(temp_result[i])
        ObjNConstraint_Type.append(0)
    return ObjNConstraint_Val, ObjNConstraint_Type
def Dynamics_Constraint(robot, T, Pos, Vel, Acc, Ctrl, Contact_Force):
    D_q, C_q_qdot, G_q, End_Effector_JacTrans = Dynamics_Matrics(robot, Pos, Vel, Acc)
    Acc = Dimension_Recovery(Acc)
    Ctrl = np.append(np.array([0,0,0]), Ctrl)
    Ctrl.shape = (len(Act_Link_Ind),1)
    Dynamics_LHS = np.add(np.add(np.dot(D_q, Acc), C_q_qdot), G_q)
    Contact_Force_Full = np.dot(End_Effector_JacTrans, Contact_Force)
    Contact_Sim = np.zeros(len(Act_Link_Ind))
    for i in range(0, len(Act_Link_Ind)):
        Contact_Sim[i] = Contact_Force_Full[Act_Link_Ind[i]]
    temp_result = np.zeros(len(Act_Link_Ind))
    for i in range(0, len(Act_Link_Ind)):
        temp_result[i] = Dynamics_LHS[Act_Link_Ind[i]] - Contact_Sim[i] - Ctrl[i]
    # ipdb.set_trace()
    return temp_result
def Dynamics_Matrics(robot, Pos, Vel, Acc):
    Acc = Dimension_Recovery(Acc)
    robotstate = np.append(Pos, Vel)
    Robot_ConfigNVel_Update(robot, robotstate)
    D_q = np.asarray(robot.getMassMatrix())
    C_q_qdot = np.asarray(robot.getCoriolisForces())
    G_q = np.asarray(robot.getGravityForces((0,0,-9.8)))
    End_Effector_Jac = End_Effector_Jacobian(robot)           # 12 x 36
    End_Effector_JacTrans = np.transpose(End_Effector_Jac)
    return D_q, C_q_qdot, G_q, End_Effector_JacTrans
def Contact_Maintenance(robot, Node_i, Node_i_child, ObjNConstraint_Val, ObjNConstraint_Type):
    # This function is used to make sure that the previous contact should be maintained
    sigma_i = Sigma_Modi_De(Node_i.sigma)
    sigma_i_child = Sigma_Modi_De(Node_i_child.sigma)
    End_Effector_Pos = get_End_Effector_Pos(robot)         # 6 * 2 by 1
    End_Effector_Vel = get_End_Effector_Vel(robot)         # 6 * 2 by 1
    End_Effector_Pos_ref = Node_i.End_Effector_Pos
    End_Effector_Vel_ref = Node_i.End_Effector_Vel
    Maint_Matrix = np.diag([ sigma_i[0] * sigma_i_child[0], sigma_i[0] * sigma_i_child[0], sigma_i[0] * sigma_i_child[0], sigma_i[0] * sigma_i_child[0], \
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
def Nodes_Optimization_fn(world, Node_i, Node_i_child):
    # // This function will optimize the joint trajectories to minimize the robot kinetic energy while maintaining a smooth transition fashion
    Seed_Flag, T, StateNDot_Coeff, Ctrl_Coeff, Contact_Force_Coeff = Seed_Guess_Gene(world, Node_i, Node_i_child)
    # ipdb.set_trace()
    if Seed_Flag == 1:
        Opt_Var_Num = Opt_Var_Num_fn(StateNDot_Coeff, Ctrl_Coeff, Contact_Force_Coeff) + 1
        Seed_Guess_vec = Seed_Guess_Zip(T, StateNDot_Coeff, Ctrl_Coeff, Contact_Force_Coeff)
        xlb = -Inf * np.ones(Opt_Var_Num)
        xub = Inf * np.ones(Opt_Var_Num)
        xlb[0] = mini
        xub[0] = 0.1 * Grids
        # Optimization problem setup
        Real_Optimization_Prob = Real_Optimization_fn(world, Node_i, Node_i_child, Seed_Guess_vec)
        Real_Optimization_Prob.xlb = xlb
        Real_Optimization_Prob.xub = xub
        ObjNConstraint_Val, ObjNConstraint_Type = Real_ObjNConstraint(world, Node_i, Node_i_child, Seed_Guess_vec)
        lb, ub = ObjNCon_Bds(ObjNConstraint_Type)
        Real_Optimization_Prob.lb = lb
        Real_Optimization_Prob.ub = ub
        cfg = snoptConfig()
        cfg.majorIterLimit = 10000
        cfg.minorIterLimit = 10000
        cfg.iterLimit = 500000
        cfg.feaTol = 1e-5
        cfg.optTol = 1e-5
        ipdb.set_trace()
        cfg.printLevel = 1
        cfg.printFile = "result.txt"
        slv = solver(Real_Optimization_Prob, cfg)
        rst = slv.solveGuess(Seed_Guess_vec)
        return rst.sol, rst.sol
def Seed_Guess_Zip(T, StateNDot_Coeff, Ctrl_Coeff, Contact_Force_Coeff):
    # This function is used to zip the seed guess into a single column vector
    Seed_Guess = np.array([T])
    a,b = StateNDot_Coeff.shape
    for i in range(0,b):
        Seed_Guess = np.append(Seed_Guess, StateNDot_Coeff[:,i])
    a,b = Ctrl_Coeff.shape
    for i in range(0,b):
        Seed_Guess = np.append(Seed_Guess, Ctrl_Coeff[:,i])
    a,b = Contact_Force_Coeff.shape
    for i in range(0,b):
        Seed_Guess = np.append(Seed_Guess, Contact_Force_Coeff[:,i])
    return Seed_Guess
def Seed_Guess_Unzip(Opt_Seed):
    # This function is used to unzip the Opt_Seed back into the coefficients
    T = Opt_Seed[0]
    # First is to retrieve the stateNdot coefficients from the Opt_Seed: 1 to 1 + (Grids - 1) * 26 * 4
    Var_Init = 1
    Var_Length = (Grids-1) * 26 * 4
    Var_End = Var_Init + Var_Length
    StateNDot_Coeff = Opt_Seed[Var_Init:Var_End]
    StateNDot_Coeff = np.reshape(StateNDot_Coeff, (Grids-1,26 * 4)).transpose()

    Var_Init = Var_End - 1
    Var_Length = (Grids-1) * 10 * 2
    Var_End = Var_Init + Var_Length
    Ctrl_Coeff = Opt_Seed[Var_Init:Var_End]
    Ctrl_Coeff = np.reshape(Ctrl_Coeff, (Grids-1, 10*2)).transpose()

    Var_Init = Var_End - 1
    Var_Length = (Grids-1) * 12 * 2
    Var_End = Var_Init + Var_Length
    Contact_Force_Coeff = Opt_Seed[Var_Init:Var_End]
    Contact_Force_Coeff = np.reshape(Contact_Force_Coeff, (Grids-1, 12*2)).transpose()
    return T, StateNDot_Coeff, Ctrl_Coeff, Contact_Force_Coeff
def Opt_Var_Num_fn(StateNDot_Coeff, Ctrl_Coeff, Contact_Force_Coeff):
    Opt_Var_Num = 0
    a,b = StateNDot_Coeff.shape
    Opt_Var_Num = Opt_Var_Num + a * b
    a,b = Ctrl_Coeff.shape
    Opt_Var_Num = Opt_Var_Num + a * b
    a,b = Contact_Force_Coeff.shape
    Opt_Var_Num = Opt_Var_Num + a * b
    return Opt_Var_Num
def Seed_Guess_Gene(world, Node_i, Node_i_child):
    # This function is used to generate the intial guess used for the optimization
    T = 0.5
    # The first step is to generate a feasible configuration that can satisfy the contact mode in the node i child
    Seed_Flag, Seed_Config = Seed_Guess_Gene_Robotstate(world, Node_i, Node_i_child)
    StateNDot_Traj = np.zeros(shape = (len(Seed_Config), Grids))
    Ctrl_Traj = np.zeros(shape = (10, Grids))
    Contact_Force_Traj = np.zeros(shape = (2*len(End_Effector_Ind), Grids))

    StateNDot_Coeff = np.zeros(shape = (4*len(Seed_Config), Grids-1))               # 4*26 x Grids-1
    Ctrl_Coeff = np.zeros(shape = (2*10, Grids-1))                                  # 4*10 x Grids-1
    Contact_Force_Coeff = np.zeros(shape = (4*len(End_Effector_Ind), Grids-1))      # 4*12 x Grids-1
    if Seed_Flag == 1:
        # 1. Initialization of the StateNdot
        # For each variable, conduct a linspace interpolation
        for i in range(0, len(Seed_Config)):
            StateNDot_i = np.linspace(Node_i.robotstate[i], Seed_Config[i], Grids)
            StateNDot_Traj[i] = StateNDot_i[0:]
        # The initialzation for values of stateNdot has been complete now it is time to compute the coefficients for the cubic spline y(s) = a*s^3 + b*s^2 + c*s + d over T
        for i in range(0,Grids-1):
            for j in range(0, len(Seed_Config)/2):
                x_init = StateNDot_Traj.item(j,i)
                x_end = StateNDot_Traj.item(j,i+1)
                xdot_init = StateNDot_Traj.item(j+len(Seed_Config)/2,i)
                xdot_end = StateNDot_Traj.item(j+len(Seed_Config)/2,i+1)
                pos_a, pos_b, pos_c, pos_d = CubicSpline_Coeff(T, x_init, x_end, xdot_init, xdot_end)
                _, _, xddot_init = CubicSpline_PosVelAcc4(T,pos_a,pos_b,pos_c,pos_d,0)
                _, _, xddot_end = CubicSpline_PosVelAcc4(T,pos_a,pos_b,pos_c,pos_d,1)
                vel_a, vel_b, vel_c, vel_d = CubicSpline_Coeff(T, xdot_init, xdot_end, xddot_init, xddot_end)
                StateNDot_Coeff[8*j:(8*j+8),i] = [pos_a, pos_b, pos_c, pos_d, vel_a, vel_b, vel_c, vel_d]
        #2. Initialization of the coefficients for Control and Contact Force
        hrp2_robot = world.robot(0)                         # Now hrp2_robot is an instance of RobotModel type
        for i in range(0,Grids-1):
            Pos_init, Vel_init, Acc_init, VelfromPos_init= StateNDot_Coeff2PosVelAcc(T, StateNDot_Coeff, i, 0)
            # ipdb.set_trace()
            Contact_Force_init, Control_init = Contact_ForceNControl_initialization(world.robot(0), Pos_init, Vel_init, Acc_init)
            Control_init = Control_init[3:]
            Pos_end, Vel_end, Acc_end, VelfromPos_init = StateNDot_Coeff2PosVelAcc(T, StateNDot_Coeff, i, 1)
            Contact_Force_end, Control_end = Contact_ForceNControl_initialization(world.robot(0), Pos_end, Vel_end, Acc_end)
            Control_end = Control_end[3:]
            Contact_Force_Traj[:,i] = Contact_Force_init
            Contact_Force_Traj[:,i+1] = Contact_Force_end
            Ctrl_Traj[:,i] = Control_init
            Ctrl_Traj[:,i+1] = Control_end
            for j in range(0,len(Contact_Force_end)):
                cf_a, cf_b = Linear_Coeff(Contact_Force_init[j], Contact_Force_end[j])
                Contact_Force_Coeff[2*j,i] = cf_a
                Contact_Force_Coeff[2*j+1,i] = cf_b
            for z in range(0,len(Control_end)):
                ctrl_a, ctrl_b = Linear_Coeff(Control_init[z], Control_end[z])
                Ctrl_Coeff[2*z,i] = ctrl_a
                Ctrl_Coeff[2*z+1,i] = ctrl_b
    return Seed_Flag, T, StateNDot_Coeff, Ctrl_Coeff, Contact_Force_Coeff
def Linear_Coeff(value_a, value_b):
    b = value_a
    a = value_b - value_a
    return a,b
def Contact_ForceNControl_initialization(hrp2_robot, Pos, Vel, Acc):
    D_q, C_q_qdot, G_q, End_Effector_JacTrans = Dynamics_Matrics(hrp2_robot, Pos, Vel, Acc)
    Acc = Dimension_Recovery(Acc)
    RHS = np.add(np.add(np.dot(D_q, Acc),C_q_qdot), G_q)
    temp_mat = np.identity(Tot_Link_No)
    LHS = Matrix_Stack(End_Effector_JacTrans, temp_mat)
    Contact_ForceNControl = np.dot(np.linalg.pinv(LHS), RHS)
    Contact_Force = Contact_ForceNControl[0:12]
    Control = Dimension_Reduction(Contact_ForceNControl[12:])
    return Contact_Force, Control
def Matrix_Stack(A, B):
    # This function is used to stack the matrices
    Row_No_A, Col_No_A = A.shape
    Row_No_B, Col_No_B = B.shape

    Stack_Mat = np.zeros(shape = (Row_No_A, Col_No_A + Col_No_B))
    for i in range(0,Col_No_A + Col_No_B):
        if i<Col_No_A:
            Stack_Mat[:,i] = A[:,i]
        else:
            Stack_Mat[:,i] = B[:,i-Col_No_A]
    return Stack_Mat
def End_Effector_Jacobian(hrp2_robot):
    # This function is used to get the Jaociban matrix
    End_Effector_Jac_Array = np.zeros(shape = (2*len(End_Effector_Ind), Tot_Link_No))
    End_Link_No_Index = -1
    # ipdb.set_trace()
    for End_Effector_Link_Index in End_Effector_Ind:
        End_Link_No_Index = End_Link_No_Index + 1
        End_Link_i = hrp2_robot.link(End_Effector_Link_Index)
        End_Link_i_Extre_Loc = Local_Extremeties[End_Link_No_Index*3:End_Link_No_Index*3+3]
        End_Link_i_Extre_Jac = End_Link_i.getPositionJacobian(End_Link_i_Extre_Loc)
        End_Effector_Jac_Array[2*End_Link_No_Index] = np.asarray(End_Link_i_Extre_Jac[0])
        End_Effector_Jac_Array[2*End_Link_No_Index+1] = np.asarray(End_Link_i_Extre_Jac[2])
    return End_Effector_Jac_Array
def StateNDot_Coeff2PosVelAcc(T, StateNDot_Coeff, Grid_Ind, s):
    # Here Grid_Ind denotes which Grid we are talking about and s is a value between 0 and 1
    StateNDot_Coeff_i = StateNDot_Coeff[:,Grid_Ind]
    Pos = [0]
    Vel = [0]
    Acc = [0]
    VelfromPos = [0]
    for i in range(0,len(Act_Link_Ind)):
        StateNDot_Coeff_i = StateNDot_Coeff[8*i:(8*i+8),Grid_Ind]
        Pos_i, Vel_i, Acc_i, VelfromPos_i = CubicSpline_PosVelAcc8(T, StateNDot_Coeff_i[0], StateNDot_Coeff_i[1], StateNDot_Coeff_i[2], StateNDot_Coeff_i[3], \
                                                        StateNDot_Coeff_i[4], StateNDot_Coeff_i[5], StateNDot_Coeff_i[6], StateNDot_Coeff_i[7], s)
        Pos.append(Pos_i)
        Vel.append(Vel_i)
        Acc.append(Acc_i)
        VelfromPos.append(VelfromPos_i)
    return Pos[1:], Vel[1:], Acc[1:], VelfromPos[1:]
def Control_Coeff2Vec(T, Ctrl_Coeff, Grid_Ind, s):
    # Here Grid_Ind denotes which Grid we are talking about and s is a value between 0 and 1
    # y = a * s + b
    Control_len = 10
    Ctrl_i = np.zeros(Control_len)
    # ipdb.set_trace()
    for i in range(0,Control_len):
        Ctrl_Coeff_i = Ctrl_Coeff[2*i:(2*i+2),Grid_Ind]
        Ctrl_i[i] = Ctrl_Coeff_i[0] * s + Ctrl_Coeff_i[1]
    return Ctrl_i
def Contact_Force_Coeff2Vec(T, Contact_Force_Coeff, Grid_Ind, s):
    # Here Grid_Ind denotes which Grid we are talking about and s is a value between 0 and 1
    # y = a * s + b
    Contact_Force_len = 12
    Contact_Force_i = np.zeros(Contact_Force_len)
    for i in range(0,Contact_Force_len):
        Contact_Force_Coeff_i = Contact_Force_Coeff[2*i:(2*i+2):,Grid_Ind]
        Contact_Force_i[i] = Contact_Force_Coeff_i[0] * s + Contact_Force_Coeff_i[1]
    return Contact_Force_i
def CubicSpline_PosVelAcc8(T, x_a, x_b, x_c, x_d, xdot_a, xdot_b, xdot_c, xdot_d, s):
    Pos = x_a*s*s*s + x_b*s*s + x_c*s + x_d
    Vel =  xdot_a*s*s*s + xdot_b*s*s + xdot_c*s + xdot_d
    Acc = (3*xdot_a*s*s + 2*xdot_b*s + xdot_c)/T
    VelfromPos = (3*x_a*s*s + 2*x_b*s + x_c)/T
    return Pos, Vel, Acc, VelfromPos
def CubicSpline_Coeff(T, x_init, x_end, xdot_init, xdot_end):
    # This funciton is used to calculate the coefficients of the cubic spliine given two edge values: x and xdot
    # y(s) = a*s^3 + b*s^2 + c*s + d over dt
    a = 2*x_init - 2*x_end - T*x_init + T*xdot_end + T*xdot_init
    b = 3*x_end - 3*x_init + 2*T*x_init - T*xdot_end - 2*T*xdot_init
    c = -T*(x_init - xdot_init)
    d = x_init
    return a, b, c, d
def CubicSpline_PosVelAcc4(T,a,b,c,d,s):
    # This function is used to calcualte the position, velocity and acceleration given the spline coefficients
    # Here T is the duration constant, s is the path variable
    Pos = a*s*s*s + b*s*s + c*s + d
    Vel = (3*a*s*s + 2*b*s + c)/T
    Acc = (6*a*s+2*b)/(T*T)
    return Pos, Vel, Acc
def Seed_Guess_Gene_Robotstate(world, Node_i, Node_i_child):
    # This function is used to generate a desired configuration that satisfies the goal contact mode
    # There are two possibilities: one is self_opt, the other is connectivity_opt
    # xlb, xub = Robotstate_Bounds(world, Node_i.robotstate)
    Seed_Robotstate_Optimization = Seed_Robotstate_Optimization_Prob(world, Node_i, Node_i_child)
    Seed_Robotstate_Optimization.xlb = robotstate_lb
    Seed_Robotstate_Optimization.xub = robotstate_ub
    ObjNConstraint_Val, ObjNConstraint_Type = Robotstate_ObjNConstraint_Seed(world, Node_i, Node_i_child, Node_i.robotstate)
    lb, ub = ObjNCon_Bds(ObjNConstraint_Type)
    Seed_Robotstate_Optimization.lb = lb
    Seed_Robotstate_Optimization.ub = ub

    cfg = snoptConfig()
    cfg.printLevel = 1
    cfg.printFile = "result.txt"
    slv = solver(Seed_Robotstate_Optimization, cfg)
    # rst = slv.solveRand()
    rst = slv.solveGuess(Node_i.robotstate.copy())

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
    return rst.flag, rst.sol
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
def Control_Bounds(world,Control_len):
    hrp2_robot = world.robot(0)
    TorqueLimits = hrp2_robot.getTorqueLimits()
    xlb = np.zeros(Control_len)
    xub = np.zeros(Control_len)
    for i in range(0,Control_len):
        xlb[i] = -TorqueLimits[Ctrl_Link_Ind[i]]
        xub[i] = TorqueLimits[Ctrl_Link_Ind[i]]
    return xlb, xub
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
    sigma_full[0] = sigma_i[0]
    sigma_full[1] = sigma_i[2]
    sigma_full[2] = sigma_i[4]
    sigma_full[3] = sigma_i[5]
    return sigma_full
def Sigma_Expansion(sigma_i, ):
    # This function is used to expande the current sigma to
def Node_Expansion(Node_i):
    # % This function is the main function used to expansion the given node to
    # % its adjacent nodes without any connectivity test

    # Every expansion has 4 possibilities: 2 hand X 2 foot
    sigma_i = Sigma_Modi_De(Node_i.sigma)
    sigma_i_children = np.array([sigma_i])
    # Left_Foot_sigma = sigma_i[0]
    # Rght_Foot_sigma = sigma_i[1]
    # Left_Hand_sigma = sigma_i[2]
    # Rght_Hand_sigma = sigma_i[3]
    for i range(0,len(sigma_i)):
        sigma_temp = sigma_i
        sigma_temp[i] = !sigma_temp[i]
        sigma_i_children = np.append(sigma_i_children, sigma_temp)
    sigma_i_children = sigma_i_children[len(sigma_i):]
    return sigma_i_children
def Node_Add(All_Nodes, Frontier_Nodes, Frontier_Nodes_Cost, Node_i):
    All_Nodes = np.append(All_Nodes, Node_i)
    Frontier_Nodes = np.append(Frontier_Nodes, Node_i)
    Frontier_Nodes_Cost = np.append(Frontier_Nodes_Cost, Node_i.KE)
    return All_Nodes, Frontier_Nodes, Frontier_Nodes_Cost
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

    # The first step is to validate the feasibility of the given initial condition
    sigma_init = np.array([1,0,0,0])            # This is the initial contact status:  1__------> contact constraint is active,
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
    RootNode.Parent_Node = []

    All_Nodes = np.array([RootNode])
    Frontier_Nodes = np.array([RootNode])
    Frontier_Nodes_Cost = np.array([RootNode.KE])

    while len(Frontier_Nodes)>0:
        # /**
        # * For the current node, first is the Node_Self_Opt to optimize a motion while maintain the current mode
        # * if this does not work, then expand the current node into the adjacent nodes then do the Nodes_Connectivity_Opt
        # */
        Node_i, Frontier_Nodes, Frontier_Nodes_Cost = Pop_Node(Frontier_Nodes, Frontier_Nodes_Cost)
        Self_Opt_Flag, Self_Opt_Soln = Nodes_Optimization_fn(world, Node_i, Node_i)       # Node_Self_Opt
        if Self_Opt_Flag == 1:
            file_name = "Node " + str(Node_i_child.Node_Index) + " Self_Opt_Soln.txt"
            file_object  = open(file_name, 'w')
            for i in range(0,len(Self_Opt_Soln)):
                file_object.write(str(Self_Opt_Soln[i]))
                file_object.write('\n')
            file_object.close()
            # Retrieve the parent node index
            Node_temp = Node_i
            file_name = "Node Sequence.txt"
            file_object  = open(file_name, 'w')
            while len(Node_temp.Parent_Node)>0:
                file_object.write(str(Node_temp.Parent_Node))
                file_object.write('\n')
                Node_temp = Node_temp.Parent_Node
            file_object.close()
            break
        else:
            # THe current node needs to be expanded
            sigma_i_children = Node_Expansion(Node_i) # Here sigma_i_children is a 4*4 x 1 array
            for i range(0,len(sigma_i_children)/4):
                sigma_i_child = Sigma_Modi_In(sigma_i_children[4*i:4*i+4])
                Node_i_child = Tree_Node(world.robot(0), sigma_i_child, Node_i.robotstate)
                Opt_Flag, Opt_Soln = Nodes_Optimization_fn(world, Node_i, Node_i_child)       # Nodes_Connectivity_Opt
                if Opt_Flag == 1:
                    Node_i_child_robotstate, Node_i_child_KE = Spline2RobotstateNKe(Opt_Flag, Grids, Node_i.sigma, sigma_i_child, world.robot(0))
                    Node_i_child.robotstate = Node_i_child_robotstate
                    Node_i_child.KE = Node_i_child_KE
                    Node_i_child.Node_Index = len(All_Nodes)-1
                    Node_i_child.Add_Parent(Node_i)
                    Node_i.Add_Child(Node_i_child)
                    All_Nodes, Frontier_Nodes, Frontier_Nodes_Cost = Node_Add(All_Nodes, Frontier_Nodes, Frontier_Nodes_Cost, Node_i_child)
                    file_name = "Node " + str(Node_i_child.Node_Index) + " Opt_Soln.txt"
                    file_object  = open(file_name, 'w')
                    for i in range(0,len(Opt_Soln)):
                        file_object.write(str(Opt_Soln[i]))
                        file_object.write('\n')
                    file_object.close()
                else:
                    pass
if __name__ == "__main__":
    main()
