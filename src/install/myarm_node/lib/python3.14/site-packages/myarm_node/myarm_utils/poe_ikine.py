import numpy as np
# import pinocchio as pin
from .poe_fkine import poe_fk
from .SE3_utils import hat, exp, adjoint, Log
from .parameters import screws, Tws, Tsb, q_lb, q_ub


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
    jac_screws = []
    for k in range(len(jac_adjoints)):
        jac_screws.append(jac_adjoints[k] @ screws[frame_in][k])
    jac = np.zeros((len(screws[frame_in][0]), len(q)))
    for k in range(len(jac_screws)):
        jac[:, k] = jac_screws[k]
    if frame_in == "space":
        if frame_ref == "space":
            return jac
        elif frame_ref == "body":
            Twb_q = poe_fk(q, "space", "world")
            Tsb_q = np.linalg.inv(Tws) @ Twb_q
            return adjoint(np.linalg.inv(Tsb_q)) @ jac
        elif frame_ref == "world":
            return adjoint(Tws) @ jac
    elif frame_in == "body":
        if frame_ref == "body":
            return jac
        elif frame_ref == "space":
            Twb_q = poe_fk(q, "space", "world")
            Tsb_q = np.linalg.inv(Tws) @ Twb_q
            return adjoint(Tsb_q) @ jac
        elif frame_ref == "world":
            Twb_q = poe_fk(q, "space", "world")
            return adjoint(Twb_q) @ jac
    return jac  # default is the world frame

def poe_ik(q0, Td, frame_in = "space", frame_ref = "world", max_iter = 100, error_tol = 1e-5):
    q = np.copy(q0)
    for _ in range(max_iter):
        if frame_ref == "world" or frame_ref == "space":
            Twref = poe_fk(q, frame_in, frame_ref)
            error = Log(Td @ np.linalg.inv(Twref))
            # error = (lambda e: np.block([e[3:], e[:3]]))(np.array(pin.log(Td @ np.linalg.inv(Twref))))
        elif frame_ref == "body":
            Twb = poe_fk(q, frame_in, "world")
            error = Log(np.linalg.inv(Twb) @ Td)
        if np.linalg.norm(error) < error_tol:
            break
        jac_pinv = np.linalg.pinv(jacobian(q, frame_in, frame_ref))
        dq = jac_pinv @ error
        q = q + dq
        q = np.minimum(np.maximum(q, q_lb), q_ub)
    return q
