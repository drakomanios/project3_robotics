import camera.camera_utils as camera_utils
import numpy as np
import json
import os

cwd = os.getcwd()
parameters = json.loads(open(os.path.join(cwd, "assets/camera/camera_params.json")).read())
camera_matrix , dist_coeffs = np.array(parameters["camera_parameters"]["4"])
