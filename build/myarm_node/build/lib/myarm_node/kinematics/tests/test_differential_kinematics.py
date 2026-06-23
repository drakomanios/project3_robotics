import numpy as np
from ..poe_fkine import poe_fk
from ..poe_diffkine import poe_dk, adjoint
from ..parameters import N, q_lb, q_ub


def test_differential_kinematics():
    np.random.seed(0)
    print_align_gap = 35
    
    success_counter = 0
    error_tol = 1e-10
    tries = 1000

    for k in range(tries):
        q = np.random.randn(N,)
        q = np.minimum(np.maximum(q, q_lb), q_ub)
        q_dot = np.random.randn(N,)
        Twb = poe_fk(q, "space", "world")
        Tsb = poe_fk(q, "space", "space")
        vwb = poe_dk(q, q_dot, "space", "world")
        vsb = poe_dk(q, q_dot, "space", "space")
        vbb = poe_dk(q, q_dot, "space", "body")
        vsb_2 = adjoint(Tsb @ np.linalg.inv(Twb)) @ vwb
        vbb_2 = adjoint(np.linalg.inv(Twb)) @ vwb
        if np.linalg.norm(vsb - vsb_2) < error_tol and \
            np.linalg.norm(vbb - vbb_2) < error_tol:
            success_counter += 1

    print(f"{str('Differential kinematics success: '):<{print_align_gap}} {success_counter:>{print_align_gap}}/{tries}")
    return success_counter, tries


if __name__ == "__main__":
    test_differential_kinematics()
