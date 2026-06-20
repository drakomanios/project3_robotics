import cv2
from cv2 import aruco
import numpy as np
import json
import os

MARKER_SIZE = 0.031  # meters

MARKER_WORLD_POSITIONS = {
    0: np.array([0.00, 0.00, 0.00], dtype=np.float32)
}

CAMERA_RESOLUTION = 480
ASPECT_RATIO = 1.3334


cwd = os.getcwd()

with open(
    os.path.join(cwd, "assets/camera/camera_params.json"),
    "r"
) as f:
    params = json.load(f)

params = params["camera_parameters"]["4"]

intrinsic_matrix = np.array(
    params["camera_matrix"],
    dtype=np.float64
)

dist_coeffs = np.array(
    params["dist_coeffs"],
    dtype=np.float64
)

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

detector_params = aruco.DetectorParameters()

detector = aruco.ArucoDetector(
    dictionary,
    detector_params
)

cam = cv2.VideoCapture(1)

cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION)
cam.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    int(CAMERA_RESOLUTION * ASPECT_RATIO)
)

if not cam.isOpened():
    raise RuntimeError("Could not open camera.")

half = MARKER_SIZE / 2

def marker_world_corners(center):
    """
    Returns the four marker corners in world coordinates.

    Corner order matches OpenCV ArUco:
    top-left
    top-right
    bottom-right
    bottom-left
    """

    x, y, z = center

    return np.array([
        [x - half, y + half, z],
        [x + half, y + half, z],
        [x + half, y - half, z],
        [x - half, y - half, z]
    ], dtype=np.float32)

saved = False

camera_rot = []
camera_tran = []

for i in range(3):

    ret, frame = cam.read()

    if not ret:
        break

    display = frame.copy()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:

        aruco.drawDetectedMarkers(
            display,
            corners,
            ids
        )

        world_points = []
        image_points = []

        for marker_id, marker_corners in zip(
            ids.flatten(),
            corners
        ):

            if marker_id not in MARKER_WORLD_POSITIONS:
                continue

            world_corners = marker_world_corners(
                MARKER_WORLD_POSITIONS[marker_id]
            )

            world_points.extend(world_corners)
            image_points.extend(marker_corners[0])

            center_px = marker_corners[0].mean(axis=0)

            cv2.putText(
                display,
                f"ID {marker_id}",
                (int(center_px[0]), int(center_px[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            world_points = np.array(
                world_points,
                dtype=np.float32
            )

            image_points = np.array(
                image_points,
                dtype=np.float32
            )

            success, rvec, tvec = cv2.solvePnP(
                world_points,
                image_points,
                intrinsic_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:

                R_wc, _ = cv2.Rodrigues(rvec)

                # camera pose in world frame
                R_cw = R_wc.T

                camera_position_world = (
                    -R_cw @ tvec
                ).flatten()

                camera_rot.append(R_cw)
                camera_tran.append(camera_position_world)

                print("\n========================")
                print("Camera Position (World)")
                print(camera_position_world)

                print("\nRotation World->Camera")
                print(R_wc)

camera_tran = np.array(camera_tran)
camera_tran = camera_tran.mean(axis=0)

camera_rot = np.array(camera_rot)
camera_rot = camera_rot.mean(axis=2)

extrinsic_data = {
                    "rotation_camera_to_world":
                        R_cw.tolist(),

                    "camera_position_world":
                        camera_tran.tolist()
                }

with open(
    "extrinsic_parameters.json",
    "w"
) as f:
    json.dump(
        extrinsic_data,
        f,
        indent=4
    )

print(
    "\nSaved extrinsic_parameters.json"
)

cam.release()
cv2.destroyAllWindows()