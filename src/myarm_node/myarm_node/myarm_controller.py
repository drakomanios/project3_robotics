import rclpy
import tf2_ros
import time
import numpy as np
import math
from rclpy.node import Node
from std_msgs.msg import String
from myarm_node.myarm_utils.myarm_connect import connect
# from myarm_node.myarm_utils.kinematics.SE3_utils import quat_to_rot
from myarm_node.myarm_utils.fullpose_IK_solve_methods import poe_ik_dls_qp, poe_ik_dls_lm
# from myarm_node.myarm_utils.poe_fkine import poe_fk
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped


class MyArmNode(Node):
    def __init__(self):
        super().__init__("myarm_node")

        # TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Declare parameter for the connection port
        self.declare_parameter("port", "/dev/ttyAMA0")
        port = self.get_parameter("port").value

        # Declare parameter for the connection baud rate
        self.declare_parameter("baudrate", 115200)
        baudrate = self.get_parameter("baudrate").value

        # Declare parameter for the speed of myarm joints 
        self.declare_parameter("speed", 80)
        self.speed = self.get_parameter("speed").value

        # Print the connection settings
        self.get_logger().info(f"MyArm Connect! -> port: {port}, baudrate: {baudrate}!")

        # Create myarm object
        self.myarm = connect(port, baudrate, 1.0)

        self.q0_list = np.zeros(7, dtype=np.float64)
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_z = 0.0

        self.publisher_ = self.create_publisher(JointState, '/joint_states', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.subscriber = self.create_subscription(PoseStamped, '/detected_obstacle', self.obstacle_callback, 10)

        self.relay_publisher = self.create_publisher(String , '/relay_command', 10)

        self.has_caught = False
        self.can_catch = True


    def timer_callback(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        joint_angles = self.myarm.get_angles()
        # print(f"Publishing joint angles: {joint_angles}")
        positions = []
        if type(joint_angles) is int:
            self.get_logger().warn(f"Failed to get joint angles from myarm. Error code: {joint_angles}")
            return 

        for angle in joint_angles:
            positions.append(round(math.radians(angle),3))
        msg.position = positions
        self.publisher_.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg.position)


    def obstacle_callback(self, msg):
        # Extract the position of the detected obstacle from the message
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        # Print the position of the detected obstacle
        self.get_logger().info(f"Detected obstacle at: x={x}, y={y}, z={z}")

        # Call the goto function to move the robot arm to the detected obstacle position
        self.goto(-x, -y, 0.05)
        # self.catch(-x,-y,0.07,0.04)

    
    def goto(self, x, y, z):
        self.q0_list = np.zeros(7, dtype=np.float64)

        # Build desired end-effector pose in myarm base (world) frame
        Twbd = np.diag([1., -1., -1. ,1.])
        Twbd[:, 3] = np.array([x, y, z, 1.])
        print(Twbd)

        # Solve inverse kinematics
        self.get_logger().warn("Starting inverse kinematics ...")
        q_rad, ik_success = poe_ik_dls_qp(self.q0_list, Twbd, "space", "world", 1.3, 200, 1e-4, 1e-3, "custom")
        # q_rad, ik_success = poe_ik_dls_lm(self.q0_list, Twbd, "space", "world", 50, 1e-2, 1e-2)

        if ik_success:
            # Convert radians to degrees and send command to myarm
            q_deg = [round(math.degrees(q),3) for q in q_rad]
            self.get_logger().info(f"SUCCESSFUL inverse kinematics! Found q = {np.round(q_deg, 2)} (degrees) \n sending command to myarm ...")
            self.myarm.send_angles(q_deg, self.speed)
            time.sleep(1)
            self.can_catch = True
        else:
            self.get_logger().info("inverse kinematics reached maximum iteration!")
            q_deg = [round(math.degrees(q),4) for q in q_rad]
            # self.get_logger().info("retrying from a closer initial position")
            # self.myarm.send_angles(q_deg, self.speed)
            # self.q0_list = np.copy(q_rad)

    def catch(self,x,y,z,max_descent):
        if self.can_catch:
            msg = String()
            msg.data = "catch"
            self.relay_publisher.publish(msg)
            self.goto(x,y,z-max_descent)
            self.can_catch = False
            self.goto(0,0,0.8)
            msg.data = "release"
            self.relay_publisher.publish(msg)
            time.sleep(1)

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
