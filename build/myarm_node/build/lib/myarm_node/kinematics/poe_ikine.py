import numpy as np
from .poe_fkine import poe_fk
from .poe_diffkine import jacobian
from .poe_idiffkine import inverse_jacobian
from .SE3_functions import Log
from .parameters import q_lb, q_ub


def poe_ik(q0, Td, frame_in = "space", frame_ref = "world", max_iter = 100, error_tol = 1e-5):
    q = np.copy(q0)
    for _ in range(max_iter):
        if frame_ref == "world" or frame_ref == "space":
            Twref = poe_fk(q, frame_in, frame_ref)
            bb_error = Log(Td @ np.linalg.inv(Twref))
        elif frame_ref == "body":
            Twb = poe_fk(q, frame_in, "world")
            bb_error = Log(np.linalg.inv(Twb) @ Td)
        if np.linalg.norm(bb_error) < error_tol:
            break
        jac_pinv = inverse_jacobian(q, frame_in, frame_ref)
        dq = jac_pinv @ bb_error
        q = q + dq
        q = np.clip(q, q_lb, q_ub)
    return q

def poe_ik_dls(q0, Td, frame_in = "space", frame_ref = "world", max_iter = 100, error_tol = 1e-5, damping = 1e-3):
    q = np.copy(q0)
    for _ in range(max_iter):
        if frame_ref == "world" or frame_ref == "space":
            Twref = poe_fk(q, frame_in, frame_ref)
            bb_error = Log(Td @ np.linalg.inv(Twref))
        elif frame_ref == "body":
            Twb = poe_fk(q, frame_in, "world")
            bb_error = Log(np.linalg.inv(Twb) @ Td)
        if np.linalg.norm(bb_error) < error_tol:
            break
        J = jacobian(q, frame_in, frame_ref)
        JT = J.T
        H = JT @ J + (damping ** 2) * np.eye(q.size)
        dq = np.linalg.solve(H, JT @ bb_error)
        q = q + dq
        q = np.clip(q, q_lb, q_ub)
    return q
