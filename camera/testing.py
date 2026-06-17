import cv2
import numpy as np
import json
import os


CUBE_SIZE = 0.024 #in meters
CAMERA_RESOLUTION = 480
ASPECT_RATIO = 1.3334

cwd = os.getcwd()
parameters = json.loads(open(os.path.join(cwd, "assets/camera/camera_params.json")).read())
parameters = parameters["camera_parameters"]["4"]
camera_matrix = np.array(parameters['camera_matrix'])
print(camera_matrix)
dist_coeffs = np.array(parameters['dist_coeffs'])

# Open the default camera
cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, int(ASPECT_RATIO * CAMERA_RESOLUTION))

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

lower_yellow = np.array([40*0.5, 30*2.55, 50*2.55])
upper_yellow = np.array([70*0.5, 100*2.55, 100*2.55])

camera_pos = np.array([0.23,0.02,0.27]) # CONFIRM EVERY TIME

Rcw = np.eye(3) # ROTATION MIGHT CHANGE , eye temporarily

def screen_space_to_world_space(x:float,y:float) -> tuple[float,float]:
    
    cube_pos_screenspace = np.array([x,y,0])
    cube_pos_cameraspace = camera_matrix @ cube_pos_screenspace
    cube_pos_worldspace = Rcw @ cube_pos_cameraspace - camera_pos
    x_world , y_world = cube_pos_worldspace[0:2]
    
    return x_world , y_world

139160.56 
104418.04

i = 0
while True:
    i += 1
    ret, frame = cam.read()

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Write the frame to the output file
    # out.write(frame)

    mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
    
    # Filter the yellow region
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)

        # Ignore tiny detections/noise
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            x_cube_center = (2*x+w)/2
            y_cube_center = (2*y+w)/2

            # Draw bounding box
            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Optional: show area
            cv2.putText(
                result,
                f"{int(area)}, {x_cube_center, y_cube_center}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            if i%20:
                a,b = screen_space_to_world_space(x_cube_center,y_cube_center)
                print(np.round(a,2)*1e-3,np.round(b,2)*1e-3)

    # Show frames
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Yellow Mask', mask)
    cv2.imshow('Yellow Filtered Result', result)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()