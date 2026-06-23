import time
import sys
import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse import csc_matrix
from . import general_utils
from qpsolvers import solve_qp


class QP:
    # Solver parameters
    qp_iter_max = 30
    qp_tol = 1e-5
    qp_penalty_factor = 10.0
    newton_iter_max = 20
    newton_tol = 1e-3
    newton_reg_par = 1e-6
    newton_reg_factor = 10.0

    def __init__(self, Q, q = None, A = None, b = None, C = None, d = None, lb = None, ub = None):
        if Q is None:
            raise ValueError("Q must be provided.")

        # QP problem parameters
        self.Q = np.asarray(Q, dtype = float)
        if self.Q.ndim != 2 or self.Q.shape[0] != self.Q.shape[1]:
            raise ValueError("Q must be a square matrix.")

        self.x_dim = self.Q.shape[0]

        self.q = np.zeros((self.x_dim, 1), dtype = float) if q is None else np.asarray(q, dtype = float).reshape(-1, 1)
        if self.q.shape[0] != self.x_dim:
            raise ValueError("q must have length equal to the dimension of x.")

        self.A = None if A is None else np.asarray(A, dtype = float)
        self.b = None if b is None else np.asarray(b, dtype = float).reshape(-1, 1)

        self.C = None if C is None else np.asarray(C, dtype = float)
        self.d = None if d is None else np.asarray(d, dtype = float).reshape(-1, 1)

        self.lb = None if lb is None else np.asarray(lb, dtype = float).reshape(-1, 1)
        self.ub = None if ub is None else np.asarray(ub, dtype = float).reshape(-1, 1)

        if self.A is not None and self.A.shape[1] != self.x_dim:
            raise ValueError("A must have the same number of columns as the dimension of x.")
        if self.C is not None and self.C.shape[1] != self.x_dim:
            raise ValueError("C must have the same number of columns as the dimension of x.")
        if self.lb is not None and self.lb.shape[0] != self.x_dim:
            raise ValueError("lb must have length equal to the dimension of x.")
        if self.ub is not None and self.ub.shape[0] != self.x_dim:
            raise ValueError("ub must have length equal to the dimension of x.")

        if self.A is not None and self.b is None:
            self.b = np.zeros((self.A.shape[0], 1), dtype=float)
        if self.C is not None and self.d is None:
            self.d = np.zeros((self.C.shape[0], 1), dtype=float)

        self.eq_dim = 0 if self.A is None else self.A.shape[0]
        self.explicit_ineq_dim = 0 if self.C is None else self.C.shape[0]

        self.lb_mask = np.zeros(self.x_dim, dtype = bool) if self.lb is None else np.isfinite(self.lb[:, 0])
        self.ub_mask = np.zeros(self.x_dim, dtype = bool) if self.ub is None else np.isfinite(self.ub[:, 0])

        self.lb_dim = int(np.sum(self.lb_mask))
        self.ub_dim = int(np.sum(self.ub_mask))
        self.ineq_dim = self.explicit_ineq_dim + self.lb_dim + self.ub_dim

        self.eq_exist = (self.eq_dim > 0)
        self.ineq_exist = (self.ineq_dim > 0)

    # Objective
    def f(self, x):
        x = np.asarray(x, dtype = float).reshape((self.x_dim, 1))
        val = 0.5 * (x.T @ self.Q @ x).item() + (self.q.T @ x).item()
        return np.array([[val]])

    # Derivative of the objective / Gradient transpose
    def df_dx(self, x):
        x = np.asarray(x, dtype = float).reshape((self.x_dim, 1))
        return (self.Q @ x + self.q).T

    # Hessian
    def ddf_ddx(self, x):
        return self.Q

    # Equality constraints
    def h(self, x):
        x = np.asarray(x, dtype = float).reshape((self.x_dim, 1))
        if self.A is None:
            return np.zeros((0, 1))
        return self.A @ x - self.b

    # Derivative of the equality constraints
    def dh_dx(self, x):
        if self.A is None:
            return np.zeros((0, self.x_dim))
        return self.A

    # Inequality constraints
    def g(self, x):
        x = np.asarray(x, dtype = float).reshape((self.x_dim, 1))
        parts = []
        if self.C is not None:
            parts.append(self.C @ x - self.d)
        if self.lb is not None and np.any(self.lb_mask):
            parts.append(self.lb[self.lb_mask] - x[self.lb_mask])
        if self.ub is not None and np.any(self.ub_mask):
            parts.append(x[self.ub_mask] - self.ub[self.ub_mask])
        if len(parts) == 0:
            return np.zeros((0, 1))
        return np.vstack(parts)

    # Derivative of the inequality constraints
    def dg_dx(self, x):
        parts = []
        if self.C is not None:
            parts.append(self.C)
        if self.lb is not None and np.any(self.lb_mask):
            parts.append(-np.eye(self.x_dim)[self.lb_mask, :])
        if self.ub is not None and np.any(self.ub_mask):
            parts.append(np.eye(self.x_dim)[self.ub_mask, :])
        if len(parts) == 0:
            return np.zeros((0, self.x_dim))
        return np.vstack(parts)

    # Augmented Lagrangian using max(0, g(x)), vectorized
    def aug_lag(self, x, l = None, m = None, rho_l = None, rho_m = None):
        x = np.asarray(x, dtype = float).reshape((self.x_dim, 1))
        l = np.zeros((self.eq_dim, 1)) if l is None else np.asarray(l, dtype = float).reshape((self.eq_dim, 1))
        m = np.zeros((self.ineq_dim, 1)) if m is None else np.asarray(m, dtype = float).reshape((self.ineq_dim, 1))
        rho_l = np.ones((self.eq_dim, 1)) if rho_l is None else np.asarray(rho_l, dtype = float).reshape((self.eq_dim, 1))
        rho_m = np.ones((self.ineq_dim, 1)) if rho_m is None else np.asarray(rho_m, dtype = float).reshape((self.ineq_dim, 1))
        L = np.copy(self.f(x))
        if self.eq_exist:
            h_val = self.h(x)
            L = np.copy(L + l.T @ h_val + 0.5 * np.sum(rho_l * h_val**2))
        if self.ineq_exist:
            g_val = self.g(x)
            g_max = np.maximum(g_val, 0)
            L = np.copy(L + m.T @ g_val + 0.5 * np.sum(rho_m * g_max**2))
        return L

    # Gradient w.r.t x (column vector), vectorized
    def aug_lag_grad(self, x, l = None, m = None, rho_l = None, rho_m = None):
        x = np.asarray(x, dtype = float).reshape((self.x_dim, 1))
        l = np.zeros((self.eq_dim, 1)) if l is None else np.asarray(l, dtype = float).reshape((self.eq_dim, 1))
        m = np.zeros((self.ineq_dim, 1)) if m is None else np.asarray(m, dtype = float).reshape((self.ineq_dim, 1))
        rho_l = np.ones((self.eq_dim, 1)) if rho_l is None else np.asarray(rho_l, dtype = float).reshape((self.eq_dim, 1))
        rho_m = np.ones((self.ineq_dim, 1)) if rho_m is None else np.asarray(rho_m, dtype = float).reshape((self.ineq_dim, 1))
        grad = np.copy(self.df_dx(x).T)
        if self.eq_exist:
            grad = np.copy(grad + self.dh_dx(x).T @ (l + rho_l * self.h(x)))
        if self.ineq_exist:
            g_val = self.g(x)
            grad = np.copy(grad + self.dg_dx(x).T @ (m + rho_m * np.maximum(g_val, 0.0)))
        return grad

    # Derivative w.r.t x (row vector), vectorized
    def aug_lag_der(self, x, l = None, m = None, rho_l = None, rho_m = None):
        return self.aug_lag_grad(x, l, m, rho_l, rho_m).T

    # Hessian w.r.t x, vectorized over inequalities
    def aug_lag_hessian(self, x, l = None, m = None, rho_l = None, rho_m = None):
        x = np.asarray(x, dtype=float).reshape((self.x_dim, 1))
        rho_l = np.ones((self.eq_dim, 1)) if rho_l is None else np.asarray(rho_l, dtype = float).reshape((self.eq_dim, 1))
        rho_m = np.ones((self.ineq_dim, 1)) if rho_m is None else np.asarray(rho_m, dtype = float).reshape((self.ineq_dim, 1))
        H = np.copy(self.ddf_ddx(x).astype(float))
        if self.eq_exist:
            dh_dx = self.dh_dx(x)
            ddh_dx_term = general_utils.finite_diff_der(lambda x: self.dh_dx(x).T @ l, x)
            H = np.copy(H + ddh_dx_term + np.einsum('k,ki,kj->ij', rho_l[:,0], dh_dx, dh_dx))
        if self.ineq_exist:
            dg_dx = self.dg_dx(x)
            g_val = self.g(x)
            g_mask = (g_val > 0).astype(float).flatten()
            ddg_dx_term = general_utils.finite_diff_der(lambda x: self.dg_dx(x).T @ m, x)
            # Vectorized outer product sum over active inequalities
            H = np.copy(H + ddg_dx_term + np.einsum('k,k,ki,kj->ij', rho_m.flatten(), g_mask, dg_dx, dg_dx))
        # if self.eq_exist:
        #     dh_dx = self.dh_dx(x)
        #     H = np.copy(H + dh_dx.T @ (rho_l * dh_dx))
        # if self.ineq_exist:
        #     dg_dx = self.dg_dx(x)
        #     g_val = self.g(x).flatten()
        #     active = (g_val > 0.0)
        #     if np.any(active):
        #         dg_active = dg_dx[active, :]
        #         rho_active = rho_m.flatten()[active]
        #         H = np.copy(H + dg_active.T @ (rho_active[:, None] * dg_active))
        return H

    # # the Augmented Lagrangian using softplus
    # def aug_lag(self, x, l = None, m = None, rho_l = None, rho_m = None):
    #     x = np.asarray(x, dtype = float).reshape((self.x_dim, 1))
    #     l = np.zeros((self.eq_dim, 1)) if l is None else np.asarray(l, dtype = float).reshape((self.eq_dim, 1))
    #     m = np.zeros((self.ineq_dim, 1)) if m is None else np.asarray(m, dtype = float).reshape((self.ineq_dim, 1))
    #     rho_l = np.ones((self.eq_dim, 1)) if rho_l is None else np.asarray(rho_l, dtype = float).reshape((self.eq_dim, 1))
    #     rho_m = np.ones((self.ineq_dim, 1)) if rho_m is None else np.asarray(rho_m, dtype = float).reshape((self.ineq_dim, 1))
    #     L = np.copy(self.f(x))
    #     if self.eq_exist:
    #         h_val = self.h(x)
    #         L = np.copy(L + l.T @ h_val + 0.5 * np.sum(rho_l * h_val**2))
    #     if self.ineq_exist:
    #         g_val = self.g(x)
    #         sp = general_utils.softplus(g_val, general_utils.sp_param)
    #         L = np.copy(L + m.T @ g_val + 0.5 * np.sum(rho_m * sp**2))
    #     return L

    # # the derivative of the augmented Lagrangian with respect to the decision variables x
    # def aug_lag_der(self, x, l = None, m = None, rho_l = None, rho_m = None):
    #     return self.aug_lag_grad2(x, l, m, rho_l, rho_m).T

    # # the gradient of the augmented Lagrangian with respect to the decision variables x (the transpose of its derivative)
    # def aug_lag_grad(self, x, l = None, m = None, rho_l = None, rho_m = None):
    #     x = np.asarray(x, dtype = float).reshape((self.x_dim, 1))
    #     l = np.zeros((self.eq_dim, 1)) if l is None else np.asarray(l, dtype = float).reshape((self.eq_dim, 1))
    #     m = np.zeros((self.ineq_dim, 1)) if m is None else np.asarray(m, dtype = float).reshape((self.ineq_dim, 1))
    #     rho_l = np.ones((self.eq_dim, 1)) if rho_l is None else np.asarray(rho_l, dtype = float).reshape((self.eq_dim, 1))
    #     rho_m = np.ones((self.ineq_dim, 1)) if rho_m is None else np.asarray(rho_m, dtype = float).reshape((self.ineq_dim, 1))
    #     grad = np.copy(self.df_dx(x).T)
    #     if self.eq_exist:
    #         grad = np.copy(grad + self.dh_dx(x).T @ (l + rho_l * self.h(x)))
    #     if self.ineq_exist:
    #         g_val = self.g(x)
    #         sp = general_utils.softplus(g_val, general_utils.sp_param)
    #         sp_der = general_utils.softplus_der(g_val, general_utils.sp_param)
    #         grad = np.copy(grad + self.dg_dx(x).T @ (m + rho_m * sp * sp_der))
    #     return grad

    # # the Hessian of the augmented Lagrangian with respect to the decision variables x,
    # # ignoring the second derivatives of the constraints in the penalty terms
    # def aug_lag_hessian(self, x, l = None, m = None, rho_l = None, rho_m = None):
    #     x = np.asarray(x, dtype = float).reshape((self.x_dim, 1))
    #     l = np.zeros((self.eq_dim, 1)) if l is None else np.asarray(l, dtype = float).reshape((self.eq_dim, 1))
    #     m = np.zeros((self.ineq_dim, 1)) if m is None else np.asarray(m, dtype = float).reshape((self.ineq_dim, 1))
    #     rho_l = np.ones((self.eq_dim, 1)) if rho_l is None else np.asarray(rho_l, dtype = float).reshape((self.eq_dim, 1))
    #     rho_m = np.ones((self.ineq_dim, 1)) if rho_m is None else np.asarray(rho_m, dtype = float).reshape((self.ineq_dim, 1))
    #     H = np.copy(self.ddf_ddx(x).astype(float))
    #     if self.eq_exist:
    #         dh_dx = self.dh_dx(x)
    #         ddh_dx_term = general_utils.finite_diff_der(lambda z: self.dh_dx(z).T @ l, x)
    #         H = np.copy(H + ddh_dx_term + np.einsum('k,ki,kj->ij', rho_l[:, 0], dh_dx, dh_dx))
    #     if self.ineq_exist:
    #         dg_dx = self.dg_dx(x)
    #         g_val = self.g(x)
    #         sp = general_utils.softplus(g_val, general_utils.sp_param)
    #         sp_der = general_utils.softplus_der(g_val, general_utils.sp_param)
    #         sp_dder = general_utils.softplus_dder(g_val, general_utils.sp_param)
    #         ddg_dx_term = general_utils.finite_diff_der(lambda z: self.dg_dx(z).T @ m, x)
    #         coeff = rho_m[:, 0] * (sp[:, 0] * sp_dder[:, 0] + sp_der[:, 0]**2)
    #         H = np.copy(H + ddg_dx_term + np.einsum('k,ki,kj->ij', coeff, dg_dx, dg_dx))
    #     return H

    def newton_with_tricks(self, x, l, m, rho_l, rho_m):
        x = np.asarray(x, dtype = float).reshape((self.x_dim, 1))
        l = np.asarray(l, dtype = float).reshape((self.eq_dim, 1))
        m = np.asarray(m, dtype = float).reshape((self.ineq_dim, 1))
        rho_l = np.asarray(rho_l, dtype = float).reshape((self.eq_dim, 1))
        rho_m = np.asarray(rho_m, dtype = float).reshape((self.ineq_dim, 1))

        delta_x = np.inf * np.ones_like(x)
        reg_par = self.newton_reg_par
        iter_counter = 0

        while np.linalg.norm(delta_x) >= self.newton_tol and iter_counter <= self.newton_iter_max:
            grad = self.aug_lag_grad(x, l, m, rho_l, rho_m)
            H = self.aug_lag_hessian(x, l, m, rho_l, rho_m)

            # regularize Hessian until it is SPD
            H_reg = np.copy(H)
            while True:
                try:
                    L = np.linalg.cholesky(H_reg)
                    break
                except np.linalg.LinAlgError:
                    H_reg = H + reg_par * np.eye(H.shape[0])
                    reg_par *= self.newton_reg_factor

            # solve using Cholesky factors
            # y = np.linalg.solve(L, -grad)
            # delta_x = np.linalg.solve(L.T, y)
            y = solve_triangular(L, -grad, lower = True, check_finite = False)
            delta_x = solve_triangular(L.T, y, lower = False, check_finite = False)

            # Armijo backtracking line search
            s = 1.0
            f_x0 = self.aug_lag(x, l, m, rho_l, rho_m)
            while True:
                f_x1 = self.aug_lag(x + s * delta_x, l, m, rho_l, rho_m)
                if f_x1 <= f_x0 + 0.1 * s * (grad.T @ delta_x).item():
                    break
                s *= 0.5

            x = x + s * delta_x
            iter_counter += 1

        return x

    def QP_custom_solver(self, x0 = None, l0 = None, m0 = None, rho_l0 = None, rho_m0 = None):
        count_iter = 0

        x = np.zeros((self.x_dim, 1)) if x0 is None else np.asarray(x0, dtype = float).reshape((self.x_dim, 1))
        l = np.zeros((self.eq_dim, 1)) if l0 is None else np.asarray(l0, dtype = float).reshape((self.eq_dim, 1))
        m = np.zeros((self.ineq_dim, 1)) if m0 is None else np.asarray(m0, dtype = float).reshape((self.ineq_dim, 1))
        rho_l = np.ones((self.eq_dim, 1)) if rho_l0 is None else np.asarray(rho_l0, dtype = float).reshape((self.eq_dim, 1))
        rho_m = np.ones((self.ineq_dim, 1)) if rho_m0 is None else np.asarray(rho_m0, dtype = float).reshape((self.ineq_dim, 1))

        for _ in range(self.qp_iter_max):
            x_new = self.newton_with_tricks(x, l, m, rho_l, rho_m)
            l = l + rho_l * self.h(x_new)
            m = np.maximum(0, m + rho_m * self.g(x_new))
            rho_l = self.qp_penalty_factor * rho_l
            rho_m = self.qp_penalty_factor * rho_m

            if (
                np.linalg.norm(x - x_new) <= self.qp_tol
                and np.all(np.abs(self.h(x_new)) <= self.qp_tol)
                and np.all(self.g(x_new) <= self.qp_tol)
            ):
                x = x_new
                break

            x = x_new
            count_iter += 1

        return np.asarray(x, dtype = float).reshape(-1), count_iter

    def _qp_data(self, sparse = False):
        P = self.Q.reshape(self.x_dim, self.x_dim)
        q = self.q.reshape(self.x_dim,)
        G = None if self.explicit_ineq_dim == 0 else self.C.reshape(self.explicit_ineq_dim, self.x_dim)
        h = None if self.explicit_ineq_dim == 0 else self.d.reshape(self.explicit_ineq_dim,)
        A = None if self.eq_dim == 0 else self.A.reshape(self.eq_dim, self.x_dim)
        b = None if self.eq_dim == 0 else self.b.reshape(self.eq_dim,)
        lb = None if self.lb is None else self.lb.reshape(self.x_dim,)
        ub = None if self.ub is None else self.ub.reshape(self.x_dim,)
        if sparse:
            P = csc_matrix(P)
            if G is not None:
                G = csc_matrix(G)
            if A is not None:
                A = csc_matrix(A)
        return P, q, G, h, A, b, lb, ub

    def QP_lib_solver(self, solver_name):
        solver_name = solver_name.lower()
        if solver_name == "proxsuite":
            solver_name = "proxqp"
        if solver_name not in ["daqp", "cvxopt", "ecos", "scs", "osqp", "proxqp"]:
            return None
        sparse_backend = solver_name in ["osqp", "scs", "proxqp"]
        P, q, G, h, A, b, lb, ub = self._qp_data(sparse = sparse_backend)
        kwargs = {}
        if solver_name == "proxqp":
            kwargs["backend"] = "sparse" if sparse_backend else "dense"
        x = solve_qp(
            P = P,
            q = q,
            G = G,
            h = h,
            A = A,
            b = b,
            lb = lb,
            ub = ub,
            solver = solver_name,
            **kwargs,
        )
        if x is None:
            return None
        return np.asarray(x, dtype = float).reshape(-1)


if __name__ == "__main__":
    # QP problem definition
    rng = np.random.default_rng(seed = None)
    x_dim = 7
    eq_dim = 3
    ineq_dim = 7
    M = rng.standard_normal((x_dim, x_dim))
    Q = (1/2) * (M.T @ M)
    Q = Q + (-np.min(np.linalg.eigvals(Q)) + 1.0) * np.eye(x_dim)
    q = rng.standard_normal((x_dim, 1))
    A = rng.standard_normal((eq_dim, x_dim))
    b = rng.standard_normal((eq_dim, 1))
    C = rng.standard_normal((ineq_dim, x_dim))
    d = rng.standard_normal((ineq_dim, 1))
    lb = None
    ub = None

    qp = QP(Q, q, A, b, C, d, lb, ub)

    # solve the QP problem using our custom solver
    time_start_our_solver = time.perf_counter()
    x_our, count_iter = qp.QP_custom_solver()
    time_end_our_solver = time.perf_counter()
    time_total_our_solver = time_end_our_solver - time_start_our_solver
    print("\nOur results:")
    if np.any(np.abs(qp.h(x_our)) >= qp.qp_tol):
        print("Equality constraints Ax-b are not satisfied. No solution found.")
        print("Ax-b: ", qp.h(x_our).reshape(-1))
    if np.any(qp.g(x_our) > qp.qp_tol):
        print("Inequality constraints Cx-d are not satisfied. No solution found.")
        print("Cx-d: ", qp.g(x_our).reshape(-1))
    if np.all(np.abs(qp.h(x_our)) < qp.qp_tol) and np.all(qp.g(x_our) <= qp.qp_tol):
        print(f"All constraints are satisfied and a solution has been found! Number of steps needed (max = {qp.qp_iter_max}): {count_iter}")
        print("Equality constraints Ax-b: ", qp.h(x_our).reshape(-1))
        print("Inequality constraints Cx-d: ", qp.g(x_our).reshape(-1))
        print("Solution: ", x_our.reshape(-1))
        print("Objective: ", qp.f(x_our).reshape(-1))

    # solve the QP problem using library solver
    if len(sys.argv) > 1:
        lib_solver_name = sys.argv[1]
    else:
        lib_solver_name = "proxqp"  # using one of the solvers ["daqp", "cvxopt", "ecos", "scs", "osqp", "proxqp"]
    time_start_lib_solver = time.perf_counter()
    x_lib = qp.QP_lib_solver(lib_solver_name)
    time_end_lib_solver = time.perf_counter()
    time_total_lib_solver = time_end_lib_solver - time_start_lib_solver
    print(f"\nResults of library solver {lib_solver_name}:")
    if x_lib is not None:
        print("Equality constraints Ax-b: ", qp.h(x_lib).reshape(-1))
        print("Inequality constraints Cx-d: ", qp.g(x_lib).reshape(-1))
        print("Solution: ", x_lib.reshape(-1))
        print("Objective: ", qp.f(x_lib).reshape(-1))
    else:
        print("No solution found.")

    # Comparing the results
    print(f"\nTime needed for our solver: {(1e3 * time_total_our_solver):.2f} msec")
    print(f"Time needed for library solver {lib_solver_name}: {(1e3 * time_total_lib_solver):.2f} msec")
    print(f"Library solver {lib_solver_name} is {(time_total_our_solver / time_total_lib_solver):.2f} times faster than our solver !")
