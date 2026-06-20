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
intrinsic_matrix = np.array(parameters['camera_matrix'])
print(intrinsic_matrix)
dist_coeffs = np.array(parameters['dist_coeffs'])

# Open the default camera
cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, int(ASPECT_RATIO * CAMERA_RESOLUTION))

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

lower_yellow = np.array([40*0.5, 20*2.55, 40*2.55])
upper_yellow = np.array([70*0.5, 100*2.55, 100*2.55])

with open("extrinsic_parameters.json", "r") as f:
    ext_params = json.load(f)
R_cw = np.array(ext_params["rotation_camera_to_world"])
camera_pos = np.array(ext_params["camera_position_world"])

def screen_space_to_world_space(u:float,v:float,size:float,width:float) -> tuple[float,float]:
    '''
    Estimates the real world xy position for an object of known size

    **Parameters**:
    - u : float , x coordinate in screenspace
    - v : float , y coordinate in screenspace
    - size : float, length of object in meters
    - width : float, width of the object in pixels

    **Returns**: 
    - x coordinate in worldspace
    - y coordinate in worldspace
    - z coordinate in worldspace
    '''
    pts = np.array([[[u, v]]])

    normalized = cv2.undistortPoints(
        pts,
        intrinsic_matrix,
        dist_coeffs
    )

    ray = np.array([normalized[0,0,0], normalized[0,0,1], 1.0])
    print(ray)

    #depth estimation DOES NOT WORK
    fx = intrinsic_matrix[0,0]
    z = fx * size / width
    print(z)

    # z = camera_pos[-1] - CUBE_SIZE

    cube_pos_cameraspace = ray * z
    cube_pos_worldspace = R_cw @ cube_pos_cameraspace + camera_pos
    x_world , y_world , z_world = cube_pos_worldspace
    
    return x_world , y_world, z_world

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
    result = cv2.GaussianBlur(result,(3,3),0,0)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray,80,200)

    lines = cv2.HoughLinesP(edges,1,np.pi/180,20,minLineLength=10, maxLineGap=100)
    edge_widths = []
    if lines == None:
        continue
    for line in lines:
        x1,y1,x2,y2 = line[0]
        edge_widths.append(np.sqrt((x1-x2)**2 + (y1-y2)**2))
        cv2.line(result,(x1,y1),(x2,y2),color=(0,0,255),thickness=3)
    width_estimate = np.mean(edge_widths)

    # Sobel gradients
    # sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    # sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # # Gradient magnitude
    # magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    # magnitude = cv2.normalize(
    #     magnitude,
    #     None,
    #     0,
    #     255,
    #     cv2.NORM_MINMAX
    # ).astype(np.uint8)

    # # Threshold edges
    # _, edges = cv2.threshold(
    #     magnitude,
    #     100,
    #     255,
    #     cv2.THRESH_BINARY
    # )

    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)

        # Ignore tiny detections/noise
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            x_cube_center = (2*x+w)/2
            y_cube_center = (2*y+h)/2

            # Draw bounding box
            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # show area
            cv2.putText(
                result,
                f"{int(area)}, {x_cube_center, y_cube_center}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            if i%20 == 0:
                x,y,z = screen_space_to_world_space(x_cube_center,y_cube_center,size=CUBE_SIZE,width=width_estimate)
                print(f"object xyz:",np.round(x*100,4),np.round(y*100,4),np.round(z*100,4))

    # Show frames
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Yellow Mask', mask)
    cv2.imshow('Yellow Filtered Result', result)
    # cv2.imshow("Sobel Magnitude", magnitude)
    cv2.imshow("Canny Edges", edges)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
# out.release()
cv2.destroyAllWindows()