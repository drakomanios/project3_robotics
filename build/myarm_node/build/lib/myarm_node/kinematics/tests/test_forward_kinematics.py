from pathlib import Path
import numpy as np
from ..urdf_fkine import urdf_extract, urdf_fk_frame
from ..poe_fkine import poe_fk
from ..dh_fkine import compute_forward_kinematics_until_frame
from ..SE3_functions import Log
from ..parameters import N, screws, Tws, q_lb, q_ub


def test_forward_kinematics():
    np.random.seed(0)
    urdf_path = (
        Path(__file__).resolve().parents[1]
        / "urdf"
        / "myarm_300_pi.urdf"
    )

    print_align_gap = 35
    
    success_counter = 0
    error_tol = 1e-3
    tries = 1000

    model, data = urdf_extract(str(urdf_path))
    for _ in range(tries):
        q = np.random.randn(N,)
        q = np.minimum(np.maximum(q, q_lb), q_ub)
        Twb_poe_space = poe_fk(q, "space", "world")
        Twb_poe_body = poe_fk(q, "body", "world")
        Twb_urdf = urdf_fk_frame(model, data, q, "endeffector", Tws)
        Twb_dh = compute_forward_kinematics_until_frame(q)
        if np.linalg.norm(Log(Twb_poe_space @ np.linalg.inv(Twb_poe_body))) < error_tol and \
            np.linalg.norm(Log(Twb_poe_space @ np.linalg.inv(Twb_urdf))) < error_tol and \
            np.linalg.norm(Log(Twb_poe_space @ np.linalg.inv(Twb_dh))) < error_tol:
            success_counter += 1

    print(f"{str('Forward kinematics success: '):<{print_align_gap}} {success_counter:>{print_align_gap}}/{tries}")
    return success_counter, tries


if __name__ == "__main__":
    test_forward_kinematics()
