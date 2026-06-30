import numpy as np
from myarm_node.myarm_utils.kinematics.poe_fkine import poe_fk
from myarm_node.myarm_utils.kinematics.poe_diffkine import jacobian
from myarm_node.myarm_utils.kinematics.poe_idiffkine import inverse_jacobian
from myarm_node.myarm_utils.kinematics.SE3_functions import Log
from myarm_node.myarm_utils.kinematics.parameters import N, screws, Tws, Tsb, q_lb, q_ub, qdot_lb, qdot_ub
from myarm_node.myarm_utils.QP_solver.QP import QP


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
        else:
            raise ValueError(f"Unsupported frame_ref '{frame_ref}'. Use 'world', 'space', or 'body'.")
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
        else:
            raise ValueError(f"Unsupported frame_ref '{frame_ref}'. Use 'world', 'space', or 'body'.")
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
        else:
            raise ValueError(f"Unsupported frame_ref '{frame_ref}'. Use 'world', 'space', or 'body'.")
        if np.linalg.norm(bb_error) < error_tol:
            break
        
        J = jacobian(q, frame_in, frame_ref)
        JT = J.T

        robot_joints = J.shape[1]
        W = np.diag([
            1.0, 1.0, 3.0, 1.0, 1.0, 0.01   
        ])

        Q = JT @ W @ J + damping**2 * np.eye(robot_joints)
        q_qp = -JT @ W @ Vb
        # Q = JT @ J + damping**2 * np.eye(robot_joints)
        # q_qp = -JT @ Vb

        C = np.vstack([twist_dt*np.eye(robot_joints),
                     -twist_dt*np.eye(robot_joints),
                     np.eye(robot_joints),
                     -np.eye(robot_joints)
                     ])
        
        d = np.concatenate([q_ub - q, 
                      q - q_lb,
                      qdot_ub,
                      -qdot_lb])
        
        qp = QP(Q,q_qp,C=C,d=d)

        if qp_solver == "custom":  # solve with our solver
            v,_ = qp.QP_custom_solver()
        else:  # solve with library solver
            v = qp.QP_lib_solver(qp_solver)

        if v is None:
            break
        q += twist_dt * v.flatten()
    else:
        count_iter = 0

    return q, count_iter
