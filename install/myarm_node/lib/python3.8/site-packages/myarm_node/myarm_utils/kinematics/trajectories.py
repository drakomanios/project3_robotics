import numpy as np
import math


def circular_trajectory(t, radius, omega, rotation, translation, pos_ref = "world", vel_ref = "world"):
    theta = omega * t

    # position on the xy plane (circle's plane / body frame)
    x = radius * math.cos(theta)
    y = radius * math.sin(theta)
    z = 0.0
    pb = np.array([x, y, z]).reshape((3, 1))

    # velocity on the xy plane (circle's plane / body frame)
    vx = -radius * omega * math.sin(theta)
    vy = radius * omega * math.cos(theta)
    vz = 0.0
    vb = np.array([vx, vy, vz]).reshape((3, 1))
    
    if pos_ref == "body":
        pos = np.copy(pb)
    elif pos_ref == "world":
        # tranlate and rotate circle's plane to get the 3d world position
        translation = translation.reshape((3, 1))
        pos = rotation @ pb + translation

    if vel_ref == "body":
        vel = np.copy(vb)
    elif vel_ref == "world":
        # velocity in the 3d world frame
        vel = rotation @ vb

    return pos, vel
