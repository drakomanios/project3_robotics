import numpy as np
from scipy.stats import qmc


# softplus approximates relu = max(0, x) with softplus function
sp_param = 1e10
def softplus(x, a):  # approximation of relu = max(0, x)
    # stable version of (1/a) * log(1 + exp(a*x))
    z = a * x
    return (1.0 / a) * (np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z))))
    # return (1/a) * np.log(1.0 + np.exp(z))

# derivative of softplus function
def softplus_der(x, a):  # sigmoid
    # stable sigmoid
    z = a * x
    sig = np.empty_like(z)
    pos_mask, neg_mask = z>=0, z<0
    sig[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))  # for the positive elements of z
    sig[neg_mask] = np.exp(z[neg_mask]) / (1.0 + np.exp(z[neg_mask]))  # for the negative elemenets of z
    return sig
    # return 1.0 / (1.0 + np.exp(-z))

# second derviative of softplus function
def softplus_dder(x, a):
    # stable_version of softplus second derivative
    z = a * x
    sig = np.empty_like(z)
    pos_mask, neg_mask = z>=0, z<0
    sig[pos_mask] = a * np.exp(-z[pos_mask]) / (1.0 + np.exp(-z[pos_mask]))**2  # for the positive elements of z
    sig[neg_mask] = a * np.exp(z[neg_mask]) / (1.0 + np.exp(z[neg_mask]))**2  # for the negative elemetns of z
    return sig
    # return a * np.exp(z) / (1.0 + np.exp(z))**2

# transpose of matrix, shapes converted like this: (n, m) -> (m, n) and (p, n, m) -> (p, m, n)
def tr(m):
    return np.swapaxes(m, -1, -2)

# implement finite differences
def finite_diff_der(f, x, eps = 1e-5):
    # f and x are both column vectors
    der = np.zeros((f(x).shape[0], x.shape[0]))
    for i in range((x.shape[0])):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += eps
        x2[i] -= eps
        der[:, i] = (f(x1) - f(x2)).flatten() / (2.*eps)
    return der

# project onto bounded rectangular area
def project_to_bounds(g, lower_bounds, upper_bounds):
    return np.maximum(lower_bounds, np.minimum(g, upper_bounds))  # project the solution onto the feasible region specified by the bounds

# find the maximum eigenvalue and the corresponding eigenvector of a matrix A, using power iteration and Rayleigh quotient
def find_max_eig(A, tol = 1e-8, maxiter = 100):
    bk = np.random.rand(A.shape[1])
    for _ in range(maxiter):
        A_bk = A @ bk
        bk_new = A_bk / np.linalg.norm(A_bk)
        if np.linalg.norm(bk_new - bk) < tol:
            break
        bk = np.copy(bk_new)
    mk = bk.T @ A @ bk
    return mk, bk

# find the minimum eigenvalue and the corresponding eigenvector of a symmetric matrix A, using inverse iteration
def find_min_eig_sym(A, tol = 1e-8, maxiter = 100):
    x0 = np.random.rand(A.shape[1])
    x = x0 / np.linalg.norm(x0)
    n = A.shape[0]
    for _ in range(maxiter):
        y = np.linalg.solve(A, x)
        x = y / np.linalg.norm(y)
        mu = x.T @ A @ x
        if np.linalg.norm(A @ x - mu * x) < tol:
            break
    return mu, x

# # only for testing of max_eig and min_eig
# for k in range(iters):
#     auglag_H = auglag.aug_lag_hessian(x_s[k], l_s[k], m_s[k], rho_l_s[k], rho_m_s[k])
#     max_eig, _ = utils.find_max_eig(auglag_H)
#     min_eig, _ = utils.find_min_eig_sym(auglag_H)
#     eigenvalues, _ = np.linalg.eig(auglag_H)
#     max_eig_numpy = eigenvalues[np.argmax(eigenvalues)]
#     min_eig_numpy = eigenvalues[np.argmin(eigenvalues)]
#     ratio = max_eig/min_eig
#     ratio_numpy = max_eig_numpy/min_eig_numpy
#     print(ratio, ratio_numpy)
