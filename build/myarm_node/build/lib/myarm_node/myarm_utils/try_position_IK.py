import numpy as np
import time
from scipy.stats import qmc

from kinematics.poe_fkine import poe_fk, find_q_close_to_qd
from kinematics.SE3_functions import Log
from kinematics.parameters import N, screws, Tws, Tsb, q_lb, q_ub
import kinematics.viz_utils as viz_utils
import kinematics.viz_scenes as viz_scenes
from general_utils import build_random_Td_list
import position_IK_solve_methods as position_IK


# Define parameters

frame_in = "space"  # "space" or "body"
frame_ref = "space"  # "world" or "space" or "body"
error_tol = 1e-4  # error tolerance
max_ik_iter = 100  # maximum inverse kinematics (IK) iterations
damping = 1e-1  # for the damped least squares (DLS) methods
twist_dt = 1.0  # for the DLS QP with twists
ik_solver = "dls_qp"  # "newton", "dls_lm", "dls_qp"
qp_solver = "proxqp"  # "custom", "proxqp", "daqp", "cvxopt", "ecos", "scs", "osqp"

Td_number = 2**7  # number of the desired poses
warm_start = False  # if True it finds good initial configurations for the IK search, if False it sets q0 = 0
max_q0_search = 2**6  # maximum number of q0 candidates for the IK search
max_q0_tries = max(3, int(max_q0_search / 4))  # choose a number of the best q0 candidates for the IK search
Td_seed = 0  # random seed to search for the desired poses
q0_seed = None  # random seed for q0 candidates for the warm start (different than Td_seed)
verbose = False  # for printing

Td_list = build_random_Td_list(frame_in, frame_ref, Td_number, "sobol", Td_seed)
Twbd_list = Td_list.copy() if frame_ref != "space" else [Tws @ Td for Td in Td_list]
pd_list = [Td[:3, 3].reshape(3) for Td in Td_list]
pwbd_list = [Twbd[:3, 3].reshape(3) for Twbd in Twbd_list]


# Try position-only inverse kinematics (IK) solver
def try_position_ik_solver(
        q0,
        pd,
        frame_in = "space",  # "space" or "body"
        frame_ref = "world",  # "world" or "space" or "body"
        error_tol = 1e-5,
        max_ik_iter = 100,
        damping = 0.0,
        twist_dt = 1.0,
        ik_solver = "dls_lm",  # "newton", "dls_lm", "dls_qp"
        qp_solver = "proxqp",  # "custom", "proxqp", "daqp", "cvxopt", "ecos", "scs", "osqp"
):
    if ik_solver == "newton":
        qd, count_iter = position_IK.poe_ik_newton(q0, pd, frame_in, frame_ref, max_ik_iter, error_tol)
    elif ik_solver == "dls_lm":
        qd, count_iter = position_IK.poe_ik_dls_lm(q0, pd, frame_in, frame_ref, max_ik_iter, error_tol, damping)
    elif ik_solver == "dls_qp":
        qd, count_iter = position_IK.poe_ik_dls_qp(q0, pd, frame_in, frame_ref, twist_dt, max_ik_iter, error_tol, damping, qp_solver)
    else:
        qd, count_iter = position_IK.poe_ik_dls_lm(q0, pd, frame_in, frame_ref, max_ik_iter, error_tol, damping)
    # because it makes no sense to have a pose expressed in body frame for FK, IK tasks of the end-effector
    pd_verify = poe_fk(qd, frame_in, frame_ref if frame_ref in ("world", "space") else "world")[:3, 3]
    error = np.linalg.norm(pd_verify.reshape(3) - pd.reshape(3))
    return qd, error, count_iter


# Apply the inverse kinematics (IK) solver

qd_list = [np.zeros(N) for _ in range(len(Td_list))]
q0_used_list = [np.zeros(N) for _ in range(len(Td_list))]
error_list = [np.nan for _ in range(len(Td_list))]
attempt_used_list = [0 for _ in range(len(Td_list))]
ik_time_list = [np.nan for _ in range(len(Td_list))]
success_list = [False for _ in range(len(Td_list))]

t_total_start = time.perf_counter()
success_ikine = 0
for i, pd in enumerate(pd_list):
    t_start = time.perf_counter()
    # Build candidate initial guesses
    if warm_start:
        # search for random initial configurations q close to the desired one qd, searching for poses T close to the desired ones Td
        q0_candidates, _ = find_q_close_to_qd(Td_list[i], "position", frame_in, frame_ref, max_q0_search, "sobol", True, q0_seed)
        q0_candidates = q0_candidates[:max_q0_tries]
    else:
        q0_candidates = [np.zeros(N)]
    # Try the candidate initial guesses
    for j, q0 in enumerate(q0_candidates):
        qd, err, _ = try_position_ik_solver(
            q0,
            pd,
            frame_in,
            frame_ref,
            error_tol,
            max_ik_iter,
            damping,
            twist_dt,
            ik_solver,
            qp_solver
        )
        if err < error_tol and np.all(q_lb <= qd) and np.all(qd <= q_ub):
            success_ikine += 1
            success_list[i] = True
            qd_list[i] = qd
            q0_used_list[i] = q0
            error_list[i] = err
            attempt_used_list[i] = j + 1
            ik_time_list[i] = time.perf_counter() - t_start
            print(f"Position {i+1}/{len(Td_list)} success!")
            break
t_total = time.perf_counter() - t_total_start


# print the results
print(f"\nSuccessful inverse kinematics: {success_ikine}/{len(Td_list)}")
print(f"Total IK time: {t_total:.6f} sec")
print(f"Average IK time per successful position: {np.nanmean(ik_time_list):.6f} sec")
if verbose:
    for i in range(len(Td_list)):
        print("\n" + "=" * 50)
        print(f"Position {i + 1}/{len(Td_list)}")
        if success_list[i]:
            print(f"qd found = {np.round(qd_list[i], 3)}")
            print(f"error = {error_list[i]:.6e}")
            print(f"q0 (try {attempt_used_list[i]}) = {np.round(q0_used_list[i], 3)}")
            print(f"time = {ik_time_list[i]:.6f} sec")
        else:
            print("IK FAILED")
            print(f"q0 = {np.round(q0_used_list[i], 3)}")
            print(f"time = {ik_time_list[i]:.6f} sec")


# visualize the results
pose_index = 0
viz, model, data = viz_scenes.robot_scene("myarm_300_pi.urdf")
print("Press ENTER to send the next pose. Type 'q' then ENTER to quit.")
while True:
    user_input = input("> ")
    if user_input.lower() in ["q", "quit", "exit"]:
        break
    viz_utils.send_next_pose(viz, model, data, qd_list, Twbd_list, frame_in, pose_index)
    pose_index = (pose_index + 1) % len(qd_list)
