import numpy as np
import time
from .poe_fkine import poe_fk
from .poe_diffkine import jacobian
from .SE3_functions import Log
from .parameters import N, screws, Tws, Tsb, q_lb, q_ub
from .trajectories import circular_trajectory
import viz_utils
import viz_scenes


# circular trajectory parameters
t_traj = 10.0  # total time for trajectory
r = 0.03  # radius of circle (meters)
omega = 2 * np.pi / t_traj  # angular velocity (rad/s)
circle_rotation = np.array([[0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0],
                            [-1.0, 0.0, 0.0]])
circle_translation = np.array([0.2, 0.0, 0.15])

# initialize arrays for trajectory
dt = 0.01
t_total = 100
num_steps = int(t_total / dt)
q_trajectory = []
q_trajectory.append(np.zeros((N)))  # start from initial configuration

print(f"Starting circular trajectory simulation...")

# velocity control loop
t = 0
Kp = np.diag([3, 3, 3, 5, 5, 5])
Ki = np.diag([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
integral_error = np.zeros(6)
for i in range(1, num_steps):
    # compute current pose
    q_cur = q_trajectory[i - 1]
    T_cur = poe_fk(q_cur, "space", "world")
    
    # Get desired position and velocity at current time
    posw_d, velb_d = circular_trajectory(t, r, omega, circle_rotation, circle_translation, pos_ref = "world", vel_ref = "body")

    # create desired pose Td w.r.t world (4x4 matrix: [R, t; 0, 1])
    T_d = np.block([[circle_rotation, posw_d.reshape((3, 1))], [np.zeros((1, 3)), 1]])
    
    # create desired task_space twist (6x1 vector: [omega, v])
    Vb_d = np.zeros(6)
    Vb_d[3:6] = velb_d.ravel()
    
    # error skew in task_space
    Xe = Log(np.linalg.inv(T_cur) @ T_d).ravel()

    # compute integral error
    integral_error += Xe * dt

    # compute task space velocity command
    Vb = Vb_d + Kp @ Xe + Ki @ integral_error

    # compute body-frame Jacobian at current configuration
    Jb = jacobian(q_cur, "space", "body")
    
    # compute joint velocities using pseudo-inverse of Jacobian
    try:
        Jb_pinv = np.linalg.pinv(Jb)
        q_dot = (Jb_pinv @ Vb).ravel()
    except:
        q_dot = np.zeros(N)
    
    # euler integration to get the next joint positions, respecting the joint limits
    q_next = q_cur + q_dot * dt
    q_next = np.clip(q_next, q_lb, q_ub)
    q_trajectory.append(q_next)
    
    if i % 100 == 0:
        print(f"Step {i}/{num_steps}, t = {t:.2f}s")
        print(Xe.ravel())

    t += dt

print("Trajectory computation complete!")


# # Print some statistics
# print(f"\nTrajectory statistics:")
# print(f"Joint angles range:")
# for j in range(N):
#     min_q = np.min(q_trajectory[:, j])
#     max_q = np.max(q_trajectory[:, j])
#     print(f"  Joint {j+1}: [{np.degrees(min_q):.1f}°, {np.degrees(max_q):.1f}°]")

# Verify final position
T_final = poe_fk(q_trajectory[-1], "space", "world")
print(f"\nFinal end-effector position: {T_final[:3, 3]}")

# Visualize the results
print("\nStarting 3D visualization...")
viz, model, data = viz_scenes.robot_scene("myarm_300_pi.urdf")

for i in range(0, num_steps, 10):  # show every 10th frame for speed
    q_cur = q_trajectory[i]
    viz_utils.send_q(viz, model, data, q_cur)
    Twb = poe_fk(q_cur, "space", "world")
    viz_utils.update_frame(viz, "bb", Twb)
    time.sleep(0.05)

print("3D visualization complete!")
