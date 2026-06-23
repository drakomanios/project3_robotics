from pymycobot.myarm import MyArm
import numpy as np
import time


# establish connection with myarm
def connect(port = "/dev/ttyAMA0", baudrate = 115200, timeout = 0.1):
    myarm = MyArm(port = port, baudrate = baudrate, timeout = timeout)
    time.sleep(0.1)
    # set the arm to zero configuration
    myarm.set_encoders([2048, 2048, 2048, 2048, 2048, 2048, 2048], 50)
    return myarm

# execute joint sequence
def exec_joint_seq_openloop(myarm, joint_seq, dt, command_mode = 0):
    # set command mode: 1 always execute latest command first, 0 to execute in a queue
    myarm.set_fresh_mode(mode = command_mode)
    time.sleep(0.1)
    for k in range(len(joint_seq)):
        joints = np.rad2deg(joint_seq[k])
        myarm.send_angles(angles = joints.tolist(), speed = 50)
        time.sleep(dt)

# joint_angles_list = myarm.get_angles()
