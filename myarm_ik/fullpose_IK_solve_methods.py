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
        Td,
        frame_in = "space",
        frame_ref = "world",
        max_ik_iter = 100,
        error_tol = 1e-5,
):
    q = np.copy(q0)
    count_iter = 0
    for _ in range(max_ik_iter):
        count_iter += 1
        if frame_ref == "world" or frame_ref == "space":
            Trefb = poe_fk(q, frame_in, frame_ref)
            bb_error = Log(Td @ np.linalg.inv(Trefb))
        elif frame_ref == "body":
            Twb = poe_fk(q, frame_in, "world")
            bb_error = Log(np.linalg.inv(Twb) @ Td)
        if np.linalg.norm(bb_error) < error_tol:
            break
        jac_pinv = inverse_jacobian(q, frame_in, frame_ref)
        dq = jac_pinv @ bb_error
        q += dq.flatten()
        q = np.clip(q, q_lb, q_ub)
    return q, count_iter

def poe_ik_dls_lm(
        q0,
        Td,
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
        if frame_ref == "world" or frame_ref == "space":
            Trefb = poe_fk(q, frame_in, frame_ref)
            bb_error = Log(Td @ np.linalg.inv(Trefb))
        elif frame_ref == "body":
            Twb = poe_fk(q, frame_in, "world")
            bb_error = Log(np.linalg.inv(Twb) @ Td)
        if np.linalg.norm(bb_error) < error_tol:
            break
        J = jacobian(q, frame_in, frame_ref)
        JT = J.T
        H = JT @ J + (damping ** 2) * np.eye(q.size)
        dq = np.linalg.pinv(H) @ (JT @ bb_error)
        q += dq.flatten()
        q = np.clip(q, q_lb, q_ub)
    return q, count_iter

def poe_ik_dls_qp(
        q0,
        Td,
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
        if frame_ref == "world" or frame_ref == "space":
            Trefb = poe_fk(q, frame_in, frame_ref)
            bb_error = Log(Td @ np.linalg.inv(Trefb))
            Vb = bb_error / twist_dt
        elif frame_ref == "body":
            Twb = poe_fk(q, frame_in, "world")
            bb_error = Log(np.linalg.inv(Twb) @ Td)
            Vb = bb_error / twist_dt
        
        if np.linalg.norm(bb_error) < error_tol:
            break
        
        J = jacobian(q, frame_in, frame_ref)
        JT = J.T

        ##### ENTER CODE HERE #####


        robot_joints = J.shape[1]

        Q = JT @ J + damping**2 * np.eye(robot_joints)
        q_qp = -JT @ Vb

        C = np.vstack([twist_dt*np.eye(robot_joints),
                     -twist_dt*np.eye(robot_joints),
                     np.eye(robot_joints),
                     -np.eye(robot_joints)
                     ])
        
        d = np.array([q_ub - q, 
                      q - q_lb,
                      qdot_ub,
                      -qdot_lb])

        qp = QP(Q,q_qp,C=C,d=d)


        ###########################

        if qp_solver == "custom":  # solve with our solver
            v,_ = qp.QP_custom_solver()
        else:  # solve with library solver
            v = qp.QP_lib_solver(qp_solver)

        if v is None:
            break
        q += twist_dt * v.flatten()
    return q, count_iter
