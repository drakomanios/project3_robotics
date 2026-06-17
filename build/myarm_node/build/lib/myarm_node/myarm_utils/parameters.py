import numpy as np


# number of joints
N = 7

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
B1 = np.array([0., 0., 1., 0., 0.008, 0.])
B2 = np.array([0., 1., 0., 0.3843, 0., -0.008])
B3 = np.array([0., 0., 1., 0., 0.008, 0.])
B4 = np.array([0., -1., 0., -0.2688, 0., 0.008])
B5 = np.array([0., 0., 1., 0., 0.008, 0.])
B6 = np.array([0., -1., 0., -0.141, 0., 0.008])
B7 = np.array([0., 0., 1., 0., 0.008, 0.])
screws["body"] = [B1, B2, B3, B4, B5, B6, B7]

# T_space->body or T_base->endeffector
Tsb = np.array([[1.0, 0.0, 0.0, 0.008],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.5538],
                [0.0, 0.0, 0.0, 1.0]])

# joints types and q limits
q_lb = np.deg2rad([-160, -70, -170, -113, -170, -115, -180])
q_ub = np.deg2rad([160, 115, 170, 75, 170, 115, 180])

# T_world->space or T_world->base
z_rotation = np.deg2rad(0)
translation = np.array([0.0, 0.0, 0.0])
Tws = np.array([[np.cos(z_rotation), -np.sin(z_rotation), 0.0, translation[0]],
                [np.sin(z_rotation), np.cos(z_rotation), 0.0, translation[1]],
                [0.0, 0.0, 1.0, translation[2]],
                [0.0, 0.0, 0.0, 1.0]])
