import numpy as np
from .poe_fkine import poe_fk
from .SE3_functions import hat, exp, adjoint
from .parameters import screws, Tws, Tsb, q_lb, q_ub


def numerical_pos_jacobian(q, frame_in = "space", frame_ref = "world", eps = 1e-7):
    p0 = poe_fk(q, frame_in, frame_ref)[:3, 3]
    Jn = np.zeros((3, len(q)))
    for i in range(len(q)):
        dq = np.zeros_like(q)
        dq[i] = eps
        p1 = poe_fk(q + dq, frame_in, frame_ref)[:3, 3]
        Jn[:, i] = (p1 - p0) / eps
    return Jn

def jacobian(q, frame_in = "space", frame_ref = "world"):
    if frame_in == "space":
        jac_adjoints = [np.eye(6)]
        for k in range(len(screws[frame_in]) - 1):
            jac_adjoint_prev = jac_adjoints[-1]
            jac_adjoints.append(jac_adjoint_prev @ adjoint(exp(hat(screws[frame_in][k]) * q[k])))
    elif frame_in == "body":
        # jac_adjoints = [adjoint(exp(-hat(screws[frame_in][-1]) * q[-1]))]
        # for k in range(len(screws[frame_in]) - 2, -1, -1):
        #     jac_adjoint_prev = jac_adjoints[0]
        #     jac_adjoints.insert(0, jac_adjoint_prev @ adjoint(exp(-hat(screws[frame_in][k]) * q[k])))
        jac_adjoints = [np.eye(6)]
        for k in range(len(screws[frame_in]) - 1, 0, -1):
            jac_adjoint_prev = jac_adjoints[0]
            jac_adjoints.insert(0, jac_adjoint_prev @ adjoint(exp(-hat(screws[frame_in][k]) * q[k])))
    else:
        raise ValueError(f"Unknown frame_in='{frame_in}'")

    jac_screws = []
    for k in range(len(jac_adjoints)):
        jac_screws.append(jac_adjoints[k] @ screws[frame_in][k])

    jac = np.zeros((len(screws[frame_in][0]), len(q)))
    for k in range(len(jac_screws)):
        jac[:, k] = jac_screws[k]

    if frame_ref == "local_world_aligned":
        Twb = poe_fk(q, frame_in, "world")
        J_ang = jac[:3, :]
        J_lin = numerical_pos_jacobian(q, frame_in, "world", eps = 1e-7)
        return np.vstack([J_ang, J_lin])

    if frame_in == "space":
        if frame_ref == "space":
            return jac
        elif frame_ref == "body":
            Twb_q = poe_fk(q, frame_in, "world")
            Tsb_q = np.linalg.inv(Tws) @ Twb_q
            return adjoint(np.linalg.inv(Tsb_q)) @ jac
        elif frame_ref == "world":
            return adjoint(Tws) @ jac
    elif frame_in == "body":
        if frame_ref == "body":
            return jac
        elif frame_ref == "space":
            Twb_q = poe_fk(q, frame_in, "world")
            Tsb_q = np.linalg.inv(Tws) @ Twb_q
            return adjoint(Tsb_q) @ jac
        elif frame_ref == "world":
            Twb_q = poe_fk(q, frame_in, "world")
            return adjoint(Twb_q) @ jac
        
    return jac  # default is the world frame

def poe_dk(q, q_dot, frame_in = "space", frame_ref = "world"):
    jac = jacobian(q, frame_in, frame_ref)
    v = jac @ q_dot.reshape(-1, 1)
    return v  # default is the world frame
