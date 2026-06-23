import numpy as np
from .parameters import N, screws, Tws, Tsb, q_lb, q_ub
from . import viz_utils


def robot_scene(robot_urdf):
    viz, model, data = viz_utils.add_robot(robot_urdf)
    viz_utils.set_Tws(viz, Tws)
    viz_utils.send_q(viz, model, data, np.zeros(model.nq))
    viz_utils.add_frame(viz, "bb_d", 0.10, 0.010, 0.5, 0.020)
    viz_utils.add_frame(viz, "bb", 0.05, 0.005, 1.0, 0.010)
    return viz, model, data

def box_cube_scene(robot_urdf):
    viz, model, data = viz_utils.add_robot(robot_urdf)
    viz_utils.add_box(viz, T = np.array([[1, 0, 0, 0.175], [0, 1, 0, 0.025], [0, 0, 1, 0.025], [0, 0, 0, 1]]), size_xyz = [0.05, 0.05, 0.05], color = 0x0000ff, name = "blue_cube")
    viz_utils.add_box(viz, T = np.array([[1, 0, 0, 0.1], [0, 1, 0, 0.125], [0, 0, 1, 0.1], [0, 0, 0, 1]]), size_xyz = [0.2, 0.05, 0.2], color = 0xff0000, name = "red_box")
    viz_utils.add_box(viz, T = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/2, 0, 0.1], [0, 0, 1, 0.125], [-np.sqrt(2)/2, np.sqrt(2)/2, 0, 0.1], [0, 0, 0, 1]]), size_xyz = [0.05, 0.05, 0.05], color = 0x000000, name = "black_hole")
    viz_utils.set_Tws(viz, Tws)
    viz_utils.send_q(viz, model, data, np.zeros(N))
    viz_utils.add_frame(viz, "bb_d", 0.10, 0.010, 0.5, 0.020)
    viz_utils.add_frame(viz, "bb", 0.05, 0.005, 1.0, 0.010)
    return viz, model, data

def ball_sphere_scene(robot_urdf, radius_sphere = 0.05, pos_sphere = np.array([0.1, 0.1, 0.1])):
    viz, model, data = viz_utils.add_robot(robot_urdf)
    T_sphere = np.array([[1.0, 0.0, 0.0, pos_sphere[0]], [0.0, 1.0, 0.0, pos_sphere[1]], [0.0, 0.0, 1.0, pos_sphere[2]], [0.0, 0.0, 0.0, 1.0]])
    viz_utils.add_sphere(viz, T = T_sphere, radius = radius_sphere, color = 0xFF0000, name = "sphere_obstacle")
    viz_utils.set_Tws(viz, Tws)
    viz_utils.send_q(viz, model, data, np.zeros(N))
    viz_utils.add_frame(viz, "bb_d", 0.10, 0.010, 0.5, 0.020)
    viz_utils.add_frame(viz, "bb", 0.05, 0.005, 1.0, 0.010)
    return viz, model, data
