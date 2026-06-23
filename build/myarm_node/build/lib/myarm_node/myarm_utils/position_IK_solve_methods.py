import numpy as np
from kinematics.poe_fkine import poe_fk
from kinematics.poe_diffkine import jacobian
from kinematics.poe_idiffkine import inverse_jacobian
from kinematics.SE3_functions import Log
from kinematics.parameters import q_lb, q_ub
from kinematics.parameters import N, screws, Tws, Tsb, q_lb, q_ub, qdot_lb, qdot_ub
from QP_solver.QP import QP

def poe_ik_newton(
        q0,
        pd,
        frame_in = "space",
        frame_ref = "world",
        max_ik_iter = 100,
        error_tol = 1e-5,
):
    q = np.copy(q0)
    count_iter = 0
    for _ in range(max_ik_iter):
        count_iter += 1
        T = poe_fk(q, frame_in, frame_ref if frame_ref in ("world", "space") else "world")
        p = T[:3, 3].reshape(3, 1)
        e_p = pd.reshape(3, 1) - p
        if np.linalg.norm(e_p) < error_tol:
            break
        J = jacobian(q, frame_in, "local_world_aligned")
        Jp = J[3:, :]  # get only translational part
        dq = np.linalg.pinv(Jp) @ e_p
        q += dq.flatten()
        q = np.clip(q, q_lb, q_ub)
    return q, count_iter

def poe_ik_dls_lm(
        q0,
        pd,
        frame_in = "space",
        frame_ref = "world",
        max_ik_iter = 100,
        error_tol = 1e-5,
        damping = 1e-3,
):
    q = np.copy(q0)
    count_iter = 0
    for _ in range(max_ik_iter):
        count_iter += 1
        T = poe_fk(q, frame_in, frame_ref if frame_ref in ("world", "space") else "world")
        p = T[:3, 3].reshape(3, 1)
        e_p = pd.reshape(3, 1) - p
        if np.linalg.norm(e_p) < error_tol:
            break
        J = jacobian(q, frame_in, "local_world_aligned")
        Jp = J[3:, :]  # get only translational part
        JpT = Jp.T
        H = JpT @ Jp + (damping ** 2) * np.eye(q.size)
        dq = np.linalg.pinv(H) @ (JpT @ e_p)
        q += dq.flatten()
        q = np.clip(q, q_lb, q_ub)
    return q, count_iter

def poe_ik_dls_qp(
        q0,
        pd,
        frame_in = "space",
        frame_ref = "world",
        twist_dt = 1.0,
        max_ik_iter = 100,
        error_tol = 1e-5,
        damping = 1e-3,
        qp_solver = "custom",
):
    q = np.copy(q0)
    count_iter = 0
    for _ in range(max_ik_iter):
        count_iter += 1
        T = poe_fk(q, frame_in, frame_ref if frame_ref in ("world", "space") else "world")
        p = T[:3, 3]
        bb_error = (pd.reshape(3) - p.reshape(3)).reshape(-1, 1) 
        Vb_linear = bb_error / twist_dt
        
        if np.linalg.norm(bb_error) < error_tol:
            break
        
        J = jacobian(q, frame_in, "local_world_aligned")
        Jp = J[3:, :]  # get only translational part
        JpT = Jp.T

        ##### ENTER CODE HERE #####
        robot_joints = Jp.shape[1]

        Q = JpT @ Jp + damping**2 * np.eye(robot_joints)
        q_lin = -JpT @ Vb_linear

        C = np.vstack([twist_dt*np.eye(robot_joints),
                     -twist_dt*np.eye(robot_joints),
                     np.eye(robot_joints),
                     -np.eye(robot_joints)
                     ])
        
        d = np.array([q_ub - q, 
                      q - q_lb,
                      qdot_ub,
                      -qdot_lb])

        qp = QP(Q,q_lin,C=C,d=d)
        ###########################

        if qp_solver == "custom":
            v_linear, _ = qp.QP_custom_solver()
        else:
            v_linear = qp.QP_lib_solver(qp_solver)

        if v_linear is None:
            break
        q += twist_dt * v_linear.flatten()
    return q, count_iter
