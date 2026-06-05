import rclpy
import tf2_ros
import numpy as np
import math
from rclpy.node import Node
from std_msgs.msg import String
from myarm_node.myarm_utils.myarm_connect import connect
from myarm_node.myarm_utils.SE3_utils import quat_to_rot
from myarm_node.myarm_utils.solve_inv_kine import solve_ik
from sensor_msgs.msg import JointState
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

        # Create publisher and timer for the joint states
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

        # Create subscription for the detected obstacle

        self.q0_list = None
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_z = 0.0
    
    def publish_myarm_state(self) -> None:
        """
        This function is periodically called by the timer. It publishes the current state (joint angles)
        of the arm's joints to the /joint_states topic.
        """

        # Create message with time stamp
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        # Set joint names
        msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

        # Read the current positions of the joints (in degrees)
        ############# ENTER CODE HERE ###################
        
        
        #################################################
       
        # Convert the joint angles to radians
        positions = []
        for angle_deg in joint_angles_list:
            angle_rad = round(math.radians(angle_deg), 3)
            positions.append(angle_rad)

        # Set joint positions
        msg.position = positions

        # Publish the message
        ############# ENTER CODE HERE ###################
        
        
        #################################################

    def follow_obstacle(self, msg: PoseStamped):
        """
        Control myarm to follow the detected obstacle.
        The obstacle pose is transformed from lidar frame to myarm base frame.
        """

        self.speed = self.get_parameter("speed").value

        # Transform from lidar frame to myarm base (world) frame
        try:
            transform = self.tf_buffer.lookup_transform(
                "myarm_base_frame",
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout = rclpy.duration.Duration(seconds = 0.1)
            )
        except tf2_ros.TransformException as e:
            self.get_logger().error(f"TF lookup failed: {e}!")
            return

        # Translation of lidar frame origin expressed in myarm base (world) frame
        t = transform.transform.translation
        t_bl = np.array([t.x, t.y, t.z])

        # Rotation of lidar frame expressed in myarm base (world) frame
        rq = transform.transform.rotation
        R_bl = quat_to_rot(np.array([rq.x, rq.y, rq.z, rq.w]))

        # Obstacle's point in lidar frame
        p_lidar = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        # Transform obstacle's point from lidar frame into myarm base (world) frame
        ############# ENTER CODE HERE ###################
        
        
        #################################################
        posx_w, posy_w, posz_w = p_base

        # Build desired end-effector pose in myarm base (world) frame
        ############# ENTER CODE HERE ###################
        
        
        #################################################
 
        # Solve inverse kinematics
        if abs(posx_w - self.last_x) > 0.015 or abs(posy_w - self.last_y) > 0.015 or abs(posz_w - self.last_z) > 0.015:
            self.get_logger().info(f"{Twbd}")
            self.get_logger().warn("Starting inverse kinematics ...")
            q_rad, ik_success = solve_ik(Twbd, "space", self.q0_list, 1e-5)

            if ik_success:
                # Convert radians → degrees and send command to myarm
                q_deg = [math.degrees(q) for q in q_rad]
                self.myarm.send_angles(q_deg, self.speed)
                self.q0_list = [np.copy(q_rad), np.array([-1.557, 1.177, -0.532, -1.209, 0.612, -0.940, -2.180])]
                self.get_logger().info(f"SUCCESSFUL inverse kinematics! Found q = {np.round(q_deg, 2)} (degrees)")
            else:
                self.get_logger().info("FAILED inverse kinematics!")
            
            # Keep the last obstacle's position
            self.last_x = posx_w
            self.last_y = posy_w
            self.last_z = posz_w


def main(args = None):
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
