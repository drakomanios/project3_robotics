import numpy as np
from .poe_diffkine import jacobian


def inverse_jacobian(q, frame_in = "space", frame_ref = "world"):
    jac_inv = np.linalg.pinv(jacobian(q, frame_in, frame_ref))
    return jac_inv

def poe_idk(q, v, frame_in = "space", frame_ref = "world"):
    jac_inv = inverse_jacobian(q, frame_in, frame_ref)
    q_dot = jac_inv @ v
    return q_dot
