import rclpy
import tf2_ros
import time
import numpy as np
import math
from rclpy.node import Node
from std_msgs.msg import String
# from myarm_node.myarm_utils.myarm_connect import connect
# from myarm_node.myarm_utils.kinematics.SE3_utils import quat_to_rot
from myarm_node.myarm_utils.fullpose_IK_solve_methods import poe_ik_dls_qp, poe_ik_dls_lm
from myarm_node.myarm_utils.kinematics.parameters import q_lb, q_ub

from myarm_node.myarm_utils.kinematics.poe_fkine import poe_fk
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

import socket
import json


class MyArmNode(Node):
    def __init__(self):
        super().__init__("myarm_node")

        # TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Declare parameter for the connection port
        # self.declare_parameter("port", "/dev/ttyAMA0")
        # port = self.get_parameter("port").value

        # # Declare parameter for the connection baud rate
        # self.declare_parameter("baudrate", 115200)
        # baudrate = self.get_parameter("baudrate").value

        # # Declare parameter for the speed of myarm joints 
        # self.declare_parameter("speed", 80)
        self.speed = 80

        HOST = "192.168.0.134"      # Pi IP
        PORT = 5000

        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((HOST, PORT))

        # Print the connection settings
        # self.get_logger().info(f"MyArm Connect! -> port: {port}, baudrate: {baudrate}!")

        # Create myarm object
        # self.myarm = connect(port, baudrate, 1.0)

        self.get_logger().info("Sampling workspace... Please wait...")
        
        num_samples = 2000
        robot_joints = len(q_lb)
        
        # 1. Παράγουμε τυχαίες γωνίες εντός ορίων για όλα τα δείγματα
        # Σχήμα: (num_samples, robot_joints) -> π.χ. (2000, 7)
        self.sampled_qs = q_lb + (q_ub - q_lb) * np.random.rand(num_samples, robot_joints)
        
        # 2. Υπολογίζουμε τα Forward Kinematics για κάθε δείγμα για να βρούμε το (x, y, z)
        self.sampled_xyz = np.zeros((num_samples, 3))
        
        for i in range(num_samples):
            # Υπολογισμός Forward Kinematics για το δείγμα i
            T = poe_fk(self.sampled_qs[i], frame_in="space", frame_ref="space")
            self.sampled_xyz[i] = T[:3, 3] # Κρατάμε μόνο τη θέση X, Y, Z
            
        self.get_logger().info(f"Successfully sampled {num_samples} workspace positions!")

        self.q0_list = np.zeros(7, dtype=np.float64)
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_z = 0.0

        # self.publisher_ = self.create_publisher(JointState, '/joint_states', 10)
        # timer_period = 0.1  # seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.subscriber = self.create_subscription(PoseStamped, '/detected_obstacle', self.obstacle_callback, 10)

        self.relay_publisher = self.create_publisher(String , '/relay_command', 10)

        self.has_caught = False
        self.can_catch = True
        self.returning = False



    # def timer_callback(self):
    #     msg = JointState()
    #     msg.header.stamp = self.get_clock().now().to_msg()
    #     msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    #     joint_angles = self.myarm.get_angles()
    #     # print(f"Publishing joint angles: {joint_angles}")
    #     positions = []
    #     if type(joint_angles) is int:
    #         self.get_logger().warn(f"Failed to get joint angles from myarm. Error code: {joint_angles}")
    #         return 

    #     for angle in joint_angles:
    #         positions.append(round(math.radians(angle),3))
    #     msg.position = positions
    #     self.publisher_.publish(msg)
    #     # self.get_logger().info('Publishing: "%s"' % msg.position)


    def obstacle_callback(self, msg):

        if self.has_caught or self.returning:
            return
        
        # Extract the position of the detected obstacle from the message
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        last_pos = np.array([self.last_x,self.last_y])
        curr_pos = np.array([x,y])

        if np.allclose(last_pos,curr_pos,atol=2e-2):
            return

        # Print the position of the detected obstacle
        self.get_logger().info(f"Detected obstacle at: x={x}, y={y}, z={z}")

        # Call the goto function to move the robot arm to the detected obstacle position
        # self.speed = 80
        self.goto(-x, -y, 0.05)
        # self.speed = 30
        self.catch(-x,-y,0.05,0.025)

        self.last_x = x
        self.last_y = y
        self.last_z = z


    
    def goto(self, x, y, z, reset_q0 = True):
        self.can_catch = False

        #LAST MINUTE ADJUSTMENTS
        # x -= 0.04 # more is right
        # y += 0.04 # more is down

        target_xyz = np.array([x,y,z])
        distances = np.linalg.norm(self.sampled_xyz - target_xyz, axis=1)
        
        closest_index = np.argmin(distances)
        # self.q0_list = self.sampled_qs[closest_index]

        self.get_logger().info(f"sampled xyz : {self.sampled_xyz[closest_index]}")

        # self.q0_list = np.deg2rad([80.25, -64.22, 91.66, -69.57, 65.16, -97.32, 99.44], dtype = np.float64)
        if reset_q0:    
            self.q0_list = np.zeros(7, dtype=np.float64)

        Twbd = np.diag([1., -1., -1. ,1.])
        Twbd[:, 3] = np.array([x, y, z, 1.])
        print(Twbd)

        # Solve inverse kinematics
        self.get_logger().warn("Starting inverse kinematics ...")
        q_rad, ik_success = poe_ik_dls_qp(self.q0_list, Twbd, "space", "world", 1.0, 1_000, 5e-4, 1e-4, "custom")
        # q_rad, ik_success = poe_ik_dls_lm(self.q0_list, Twbd, "space", "world", 100, 1e-3, 1e-2)

        if ik_success:
            # Convert radians to degrees and send command to myarm
            q_deg = [round(math.degrees(q),3) for q in q_rad]
            self.get_logger().info(f"SUCCESSFUL inverse kinematics! Found q = {np.round(q_deg, 2)} (degrees) \n sending command to myarm ...")
            packet = {
                "type": "angles",
                "angles": q_deg,
                "speed": self.speed
            }

            self.client.sendall(json.dumps(packet).encode())
            self.q0_list = q_rad
            time.sleep(3)
            if not self.returning:
                self.get_logger().info(f"attempting to catch")
                self.can_catch = True
        else:
            self.get_logger().error("inverse kinematics reached maximum iteration!")
            q_deg = [round(math.degrees(q),4) for q in q_rad]

    def catch(self,x,y,z,max_descent):
        self.returning = False
        if self.can_catch:
            msg = String()
            msg.data = "catch"
            self.relay_publisher.publish(msg)
            self.speed = 20 # be more gentle when catching , high speed wiggles a lot
            self.goto(x,y,z-max_descent, reset_q0 = False)
            self.can_catch = False
            self.goto(x,y,z, reset_q0 = False)
            time.sleep(2)
            self.speed = 80
            self.returning = True
            self.goto(0,0.1,0.1,reset_q0 = True)
            time.sleep(2)
            msg.data = "release"
            self.relay_publisher.publish(msg)
            time.sleep(1)
            msg.data = "off"
            self.relay_publisher.publish(msg)
            self.returning = False

def main(args = None):
    node = None
    try:
        rclpy.init(args = args)
        node = MyArmNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
