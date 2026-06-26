import rclpy
from rclpy.node import Node

from std_msgs.msg import String
import RPi.GPIO as GPIO


class RelayController(Node):

    RELAY1_PIN = 20   # Pump
    RELAY2_PIN = 26   # Switch

    def __init__(self):
        super().__init__("relay_controller")

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        GPIO.setup(self.RELAY1_PIN, GPIO.OUT)
        GPIO.setup(self.RELAY2_PIN, GPIO.OUT)

        self.set_relays(False, False)

        self.create_subscription(
            String,
            "/relay_command",
            self.command_callback,
            10
        )

        self.get_logger().info("Relay controller ready.")

    def relay_output(self, pin, state):
        """
        state=True  -> Relay ON
        state=False -> Relay OFF
        Active LOW relay board.
        """
        GPIO.output(pin, GPIO.LOW if state else GPIO.HIGH)

    def set_relays(self, relay1, relay2):
        self.relay_output(self.RELAY1_PIN, relay1)
        self.relay_output(self.RELAY2_PIN, relay2)

    def command_callback(self, msg):
        cmd = msg.data.lower().strip()

        if cmd == "off":
            # Pump OFF, Switch OFF
            self.set_relays(False, False)

        elif cmd == "catch":
            # Pump ON, Switch OFF
            self.set_relays(False, True)

        elif cmd == "release":
            # Pump ON, Switch ON
            self.set_relays(True, True)

        else:
            self.get_logger().warn(f"Unknown command: '{cmd}'")
            return

        self.get_logger().info(f"Command executed: {cmd}")

    def destroy_node(self):
        self.set_relays(False, False)
        GPIO.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = RelayController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()