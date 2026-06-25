import time
import numpy as np
from scipy.stats import qmc
from kinematics.poe_fkine import poe_fk
from kinematics.parameters import N, screws, Tws, Tsb, q_lb, q_ub


# generate random samples following a defined random distribution
def generate_random_samples(lb, ub, sample_size = 100, sampler = "normal", seed = None):
    rng = np.random.default_rng(seed = seed)
    lb = np.copy(lb.flatten())
    ub = np.copy(ub.flatten())
    if sampler == "normal":
        mean = (lb + ub) / 2
        std = (ub - lb) / 4
        samples = np.array([rng.normal(loc = mean, scale = std) for _ in range(sample_size)])
        samples = np.clip(samples, lb, ub)
    elif sampler == "uniform":
        samples = np.array([rng.uniform(lb, ub) for _ in range(sample_size)])
    elif sampler == "sobol":
        if not np.log2(sample_size).is_integer():
            sample_size = int(2.0 ** round(np.log2(sample_size)))
        sampler = qmc.Sobol(d = len(lb), scramble = True, seed = seed)
        unit_samples = sampler.random(sample_size)
        samples = qmc.scale(unit_samples, lb, ub)
    elif sampler == "halton":
        if not np.log2(sample_size).is_integer():
            sample_size = int(2.0 ** round(np.log2(sample_size)))
        sampler = qmc.Halton(d = len(lb), scramble = True)
        unit_samples = sampler.random(sample_size)
        samples = qmc.scale(unit_samples, lb, ub)
    return samples

# Build a list of random poses
def build_random_Td_list(
        frame_in = "space",  # "space" or "body"
        frame_ref = "world",  # "world" or "space" or "body"
        number = 2**6,
        sampler = "sobol",  # "normal", "uniform", "sobol", "halton"
        seed = None,
):
    q_samples = generate_random_samples(q_lb, q_ub, number, sampler, seed)
    Twbd = [
        poe_fk(q, frame_in = frame_in, frame_ref = "world")
        for q in q_samples
    ]
    if frame_ref in ("world", "body"):
        # because it makes no sense to have a pose expressed in body frame for FK, IK tasks of the end-effector
        return Twbd
    elif frame_ref == "space":
        Tsw = np.linalg.inv(Tws)
        return [Tsw @ Td for Td in Twbd]

# find the exact solution of a linear system
# A should be square (n x n) and nonsingular / invertible
def exact_solution(A, b):
    return np.linalg.solve(A, b).reshape(-1)

# solve a linear system using gradient descent method
# A should be square (n x n), symmetric, and positive definite (SPD)
def gradient_descent(x, A, b, tol = 1e-10, max_iter = 1000):
    n = A.shape[0]
    x = np.asarray(x, dtype = float).reshape(n,)
    b = np.asarray(b, dtype = float).reshape(n,)
    count_iter = 0
    r = A @ x - b
    while np.linalg.norm(r) > tol and count_iter < max_iter:
        Ar = A @ r
        gamma = np.dot(r, r) / np.dot(r, Ar)
        x = x - gamma * r
        r = r - gamma * Ar
        count_iter += 1
    return x, count_iter

# solve a linear system using conjugate gradient method
# A should be square (n x n), symmetric, and positive definite (SPD)
def conjugate_gradient(x, A, b, tol = 1e-10, max_iter = 1000):
    n = A.shape[0]
    x = np.asarray(x, dtype = float).reshape(n,)
    b = np.asarray(b, dtype = float).reshape(n,)
    count_iter = 0
    r = b - A @ x
    p = r
    while np.linalg.norm(r) > tol and count_iter < max_iter:
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        beta = np.dot(r, r) / np.dot(r + alpha * Ap, r + alpha * Ap)
        p = r + beta * p
        count_iter += 1
    return x, count_iter


if __name__ == "__main__":
    n = 7
    rng = np.random.default_rng(seed = None)
    A = rng.standard_normal((n, n))
    A = (A + A.T) / 2.0  # ensure symmetric A
    A = A + (-np.min(np.linalg.eigvals(A)) + 1.0) * np.eye(n)  # ensure symmetric SPD A
    b = rng.standard_normal((n, 1))
    x0 = np.zeros((n, 1))

    t0 = time.perf_counter()
    x_exact = exact_solution(A, b)
    time_exact = time.perf_counter() - t0

    t0 = time.perf_counter()
    x_gd, iter_gd = gradient_descent(x0, A, b)
    time_gd = time.perf_counter() - t0

    t0 = time.perf_counter()
    P = np.diag(np.diag(A))  # Jacobi preconditioner
    P_inv = np.linalg.inv(P)
    x_gd_prec, iter_gd_prec = gradient_descent(x0, A @ P_inv, b)
    x_gd_prec = P_inv @ x_gd_prec
    time_gd_prec = time.perf_counter() - t0

    t0 = time.perf_counter()
    x_cg, iter_cg = conjugate_gradient(x0, A, b)
    time_cg = time.perf_counter() - t0

    print(f"Matrix A:")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            print(f"{A[i, j]:.2f}", end = "\t")
        print()
    print(f"Vector b: {b.reshape(-1)}")
    print("\nComparing Linear System Solvers:")
    print(f"Exact Solution                  : x = {x_exact} \t| Iterations: - \t| Time: {time_exact:.6f} sec")
    print(f"Gradient descent                : x = {x_gd} \t| Iterations: {iter_gd} \t| Time: {time_gd:.6f} sec")
    print(f"Preconditioned Gradient Descent : x = {x_gd_prec} \t| Iterations: {iter_gd_prec} \t| Time: {time_gd_prec:.6f} sec")
    print(f"Conjugate Gradient              : x = {x_cg} \t| Iterations: {iter_cg} \t| Time: {time_cg:.6f} sec")
