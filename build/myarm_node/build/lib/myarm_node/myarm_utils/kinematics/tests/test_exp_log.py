import numpy as np
from scipy.linalg import expm, logm
import pinocchio as pin
from ..SE3_functions import Exp, exp, Log, Log2, hat


def test_exp_log():
    np.random.seed(0)
    print_align_gap = 35
    
    Exp_success_counter = 0
    Log_success_counter = 0
    error_tol = 1e-10
    tries = 1000

    for k in range(tries):
        tau = np.random.randn(6,)
        exp1 = Exp(tau)
        exp2 = exp(hat(tau))
        exp_scipy = expm(hat(tau))
        if np.allclose(exp1, exp2, atol = error_tol) and \
            np.allclose(exp1, exp_scipy, atol = error_tol):
            Exp_success_counter += 1
        M = Exp(tau)
        M_pin = pin.SE3(M[:3, :3], M[:3, 3])
        log1 = Log(M)
        log2 = Log2(M)
        log_scipy = (lambda l: np.block([l[2, 1], l[0, 2], l[1, 0], l[:3, 3]]))(logm(M))  # logm breaks in edge cases
        log_pin = (lambda l: np.block([l[3:], l[:3]]))(np.array(pin.log(M_pin)))
        if np.allclose(log1, log_pin, atol = error_tol) and \
            np.allclose(log2, log_pin, atol = error_tol) and \
            np.allclose(log_scipy, log_pin, atol = error_tol):
            Log_success_counter += 1

    print(f"{str('Exp success: '):<{print_align_gap}} {Exp_success_counter:>{print_align_gap}}/{tries}")
    print(f"{str('Log success: '):<{print_align_gap}} {Log_success_counter:>{print_align_gap}}/{tries}")
    return (Exp_success_counter, Log_success_counter), tries


if __name__ == "__main__":
    test_exp_log()