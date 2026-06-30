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
        self.speed = 80

        HOST = "192.168.0.134"      # Pi IP
        PORT = 5000

        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((HOST, PORT))

        self.get_logger().info("Sampling workspace... Please wait...")
        
        num_samples = 2000
        robot_joints = len(q_lb)
        
        self.sampled_qs = q_lb + (q_ub - q_lb) * np.random.rand(num_samples, robot_joints)
        self.sampled_xyz = np.zeros((num_samples, 3))
        
        for i in range(num_samples):
            T = poe_fk(self.sampled_qs[i], frame_in="space", frame_ref="space")
            self.sampled_xyz[i] = T[:3, 3]
            
        self.get_logger().info(f"Successfully sampled {num_samples} workspace positions!")

        self.q0_list = np.zeros(7, dtype=np.float64)
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_z = 0.0

        self.completed_targets = []
        self.target_tolerance = 0.02    # 3 cm
        
        self.subscriber = self.create_subscription(PoseStamped, '/detected_obstacle', self.obstacle_callback, 10)

        self.relay_publisher = self.create_publisher(String , '/relay_command', 10)

        self.has_caught = False
        self.can_catch = True
        self.returning = False

    def is_completed(self, pos):
        for p in self.completed_targets:
            if np.linalg.norm(pos - p) < self.target_tolerance:
                return True
        return False

    def obstacle_callback(self, msg):

        if self.has_caught or self.returning:
            return
        
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        curr_pos = np.array([x,y])

        if self.is_completed(curr_pos):
            return

        self.get_logger().info(f"Detected obstacle at: x={x}, y={y}, z={z}")

        ik_success = self.goto(-x, -y, 0.05)
        if ik_success: self.catch(-x,-y,0.05,0.025)

        self.completed_targets.append(curr_pos)

    
    def goto(self, x, y, z, reset_q0 = True):
        self.can_catch = False

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
        else:
            self.get_logger().error("inverse kinematics reached maximum iteration!")
            q_deg = [round(math.degrees(q),4) for q in q_rad]

        return ik_success

    def catch(self,x,y,z,max_descent):
        self.get_logger().info(f"attempting to catch")
        self.returning = False
        msg = String()
        msg.data = "catch"
        self.relay_publisher.publish(msg)
        self.speed = 20 # be more gentle when catching , high speed wiggles a lot
        ik_success = self.goto(x,y,z-max_descent, reset_q0 = False)
        self.speed = 80
        if ik_success:
            time.sleep(2)
            self.can_catch = False
            self.has_caught = True
            self.goto(x,y,z, reset_q0 = False)
            time.sleep(2)
        else:
            msg.data = "off"
            self.relay_publisher.publish(msg)
            self.get_logger().warn("could not get to catching position")
        self.returning = True
        self.goto(0,0.1,0.1,reset_q0 = False)
        time.sleep(2)
        if self.has_caught:
            self.completed_targets.append(np.array([x,y]))
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
