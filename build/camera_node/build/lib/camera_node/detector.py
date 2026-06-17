import assets.camera_utils as camera_utils
import cv2
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import PoseStamped
import json
import os

cwd = os.path.getcwd()
parameters = json.loads(open(os.path.join(cwd, "assets/camera_params.json")).read())
print(parameters["4"])
