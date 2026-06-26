import numpy as np


# number of joints
N = 7

# DH parameters (a, alpha, d, theta), and T_base->0, T_n->endeffector
dh_params = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, 0.0],
                        [0.0, 0.0, 0.1155, 0.0, 0.1278, 0.0, 0.066],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype = float)

# T_space->0 or T_base->0
Ts0 = np.array([[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.1695],
                [0.0, 0.0, 0.0, 1.0]])

# T_n->body or T_n->endeffector
Tnb = np.array([[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]])

# initialize screws
screws = {}

# space frame screws
S1 = np.array([0., 0., 1., 0., 0., 0.])
S2 = np.array([0., 1., 0., -0.1695, 0., 0.])
S3 = np.array([0., 0., 1., 0., 0., 0.])
S4 = np.array([0., -1., 0., 0.285, 0., 0.])
S5 = np.array([0., 0., 1., 0., 0., 0.])
S6 = np.array([0., -1., 0., 0.4128, 0., 0.])
S7 = np.array([0., 0., 1., 0., 0., 0.])
screws["space"] = [S1, S2, S3, S4, S5, S6, S7]

# body frame screws
B1 = np.array([0., 0., 1., 0., 0., 0.])
B2 = np.array([0., 1., 0., 0.3093, 0., 0.])
B3 = np.array([0., 0., 1., 0., 0., 0.])
B4 = np.array([0., -1., 0., -0.1938, 0., 0.])
B5 = np.array([0., 0., 1., 0., 0., 0.])
B6 = np.array([0., -1., 0., -0.066, 0., 0.])
B7 = np.array([0., 0., 1., 0., 0., 0.])
screws["body"] = [B1, B2, B3, B4, B5, B6, B7]

# T_space->body or T_base->endeffector
Tsb = np.array([[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.4788],
                [0.0, 0.0, 0.0, 1.0]])

# joints types and q, qdot limits
joints_types = ["revolute", "revolute", "revolute", "revolute", "revolute", "revolute", "revolute"]
q_lb = np.deg2rad([-160, -70, -170, -110, -170, -115, -180])  # joint position lower bounds in rad
q_ub = np.deg2rad([160, 115, 170, 75, 170, 115, 180])  # joint position upper bounds in rad
qdot_lb = np.deg2rad([-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2])  # joint speed lower bounds in rad/sec
qdot_ub = np.deg2rad([np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2])  # joint speed upper bounds in rad/sec

# T_world->space or T_world->base
z_rotation = np.deg2rad(-270)
translation = np.array([-0.08, 0.165, -0.15])
Tws = np.array([[np.cos(z_rotation), -np.sin(z_rotation), 0.0, translation[0]],
                [np.sin(z_rotation), np.cos(z_rotation), 0.0, translation[1]],
                [0.0, 0.0, 1.0, translation[2]],
                [0.0, 0.0, 0.0, 1.0]])

