import numpy as np
from ..poe_fkine import poe_fk
from ..poe_diffkine import jacobian
from ..parameters import N, q_lb, q_ub, Tws


def test_jacobians():
    np.random.seed(0)
    print_align_gap = 35
    
    Jw_success_counter = 0
    Js_success_counter = 0
    Jb_success_counter = 0
    tries = 1000
    error_tol = 1e-10

    for k in range(tries):
        q = np.random.randn(N,)
        q = np.minimum(np.maximum(q, q_lb), q_ub)
        Tsb = np.linalg.inv(Tws) @ poe_fk(q, "space", "world")
        J_world_from_space = jacobian(q, "space", "world")
        J_world_from_body = jacobian(q, "body", "world")
        J_space = jacobian(q, "space", "space")
        J_space_from_body = jacobian(q, "body", "space")
        J_body  = jacobian(q, "body", "body")
        J_body_from_space = jacobian(q, "space", "body")
        if np.allclose(J_world_from_space, J_world_from_body, atol = error_tol):
            Jw_success_counter += 1
        if np.allclose(J_space, J_space_from_body, atol = error_tol):
            Js_success_counter += 1
        if np.allclose(J_body, J_body_from_space, atol = error_tol):
            Jb_success_counter += 1

    print(f"{str('World Jacobian success: '):<{print_align_gap}} {Jw_success_counter:>{print_align_gap}}/{tries}")
    print(f"{str('Space Jacobian success: '):<{print_align_gap}} {Js_success_counter:>{print_align_gap}}/{tries}")
    print(f"{str('Body Jacobian success: '):<{print_align_gap}} {Jb_success_counter:>{print_align_gap}}/{tries}")
    return (Jw_success_counter, Js_success_counter, Jb_success_counter), tries


if __name__ == "__main__":
    test_jacobians()