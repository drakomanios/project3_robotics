import numpy as np
from .parameters import dh_params, Ts0, Tnb, Tws, joints_types


# calculate the transformation matrix for the specified Denavit-Hartenberg parameters
def dh_transformation_matrix(a, alpha, d, theta, joint_type, q):
    a = float(a); alpha = float(alpha); d = float(d); theta = float(theta)
    if joint_type == "revolute":
        theta += q
    elif joint_type == "prismatic":
        d += q
    return np.array([[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
                    [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
                    [0, np.sin(alpha), np.cos(alpha), d],
                    [0.0, 0.0, 0.0, 1.0]], dtype = float)

# calculate the forward kinematics of the robotic arm defined by the DH parameters
def compute_forward_kinematics(q_joints):
    a = dh_params[0, :].reshape((-1,)); alpha = dh_params[1, :].reshape((-1,)); d = dh_params[2, :].reshape((-1,)); theta = dh_params[3, :].reshape((-1,))
    T = np.array(Tws, dtype = float) @ np.array(Ts0, dtype = float)
    for k in range(len(q_joints)):
        Ak = dh_transformation_matrix(a[k], alpha[k], d[k], theta[k], joints_types[k], q_joints[k])
        T = T @ Ak
    T = T @ np.array(Tnb, dtype = float)
    return T

# calculate the partial forward kinematics of the robotic arm until the specified frame
def compute_forward_kinematics_until_frame(q_joints, until_frame = "end-effector"):
    a = dh_params[0, :].reshape((-1,)); alpha = dh_params[1, :].reshape((-1,)); d = dh_params[2, :].reshape((-1,)); theta = dh_params[3, :].reshape((-1,))
    if until_frame == "base":
        return np.array(Tws, dtype = float)
    elif "frame" in until_frame:
        T = np.array(Tws, dtype = float) @ np.array(Ts0, dtype = float)
        until_frame = int(until_frame.split(" ")[1])
        for k in range(until_frame):
            Ak = dh_transformation_matrix(a[k], alpha[k], d[k], theta[k], joints_types[k], q_joints[k])
            T = T @ Ak
        return T
    elif until_frame == "end-effector":
        T = compute_forward_kinematics(q_joints)
        return T

# calculate the partial forward kinematics of the robotic arm for all the important frames
def compute_forward_kinematics_all_frames(q_joints):
    a = dh_params[0, :].reshape((-1,)); alpha = dh_params[1, :].reshape((-1,)); d = dh_params[2, :].reshape((-1,)); theta = dh_params[3, :].reshape((-1,))
    frames_list = [np.array(Tws, dtype = float), np.array(Tws, dtype = float) @ np.array(Ts0, dtype = float)]
    T = frames_list[-1]
    for k in range(len(q_joints)):
        Ak = dh_transformation_matrix(a[k], alpha[k], d[k], theta[k], joints_types[k], q_joints[k])
        T = T @ Ak
        frames_list.append(T)
    frames_list.append(T @ np.array(Tnb, dtype = float))
    return frames_list
