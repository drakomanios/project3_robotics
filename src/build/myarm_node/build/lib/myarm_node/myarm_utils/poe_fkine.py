import numpy as np
from scipy.stats import qmc
from .kinematics.SE3_functions import hat, exp, Log
from .kinematics.parameters import screws, Tws, Tsb, q_lb, q_ub
from .kinematics.general_utils import generate_random_samples


def poe_fk(q, frame_in = "space", frame_ref = "world"):
    q = np.clip(q, q_lb, q_ub)
    poe = Tsb
    if frame_in == "space":
        for k in range(len(q) - 1, -1, -1):
            poe = exp(hat(screws[frame_in][k]) * q[k]) @ poe
    elif frame_in == "body":
        for k in range(0, len(q), 1):
            poe = poe @ exp(hat(screws[frame_in][k]) * q[k])
    if frame_ref == "world":
        return Tws @ poe
    elif frame_ref == "space":
        return poe
    elif frame_ref == "body":
        return np.eye(4)
    return poe  # default is the world frame

# find configurations q closed to a desired one qd, searching for poses T close to a desired one Td
# choose random initial configurations using Sobol sequence
def find_q_close_to_qd(Td, criterion = "fullpose",  frame_in = "space", frame_ref = "world", sample_size = 2**8, sampler = "sobol", sort_samples = True, seed = None):
    q_samples = generate_random_samples(q_lb, q_ub, sample_size, sampler, seed)
    if frame_ref == "body":
        frame_ref = "world"
    if sort_samples:
        errors = np.zeros(sample_size)
        for k in range(sample_size):
            T = poe_fk(q_samples[k], frame_in, frame_ref)
            if criterion == "fullpose":
                errors[k] = np.linalg.norm(Log(Td @ np.linalg.inv(T)))
            elif criterion == "position":
                errors[k] = np.linalg.norm(Td[:3, 3] - T[:3, 3])
            else:
                errors[k] = np.linalg.norm(Log(Td @ np.linalg.inv(T)))
        sorted_indices = np.argsort(errors)
        sorted_errors = errors[sorted_indices]
        sorted_q_samples = q_samples[sorted_indices]
        return sorted_q_samples, sorted_errors
    else:
        return q_samples, None
