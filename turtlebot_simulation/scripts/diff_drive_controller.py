#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

class DiffDriveController(Node):
    def __init__(self):
        # Initialize the node with the name 'diff_drive_controller'
        super().__init__('diff_drive_controller')

        # Robot physical constants
        self.wheel_radius = 0.035
        self.wheel_base_distance = 0.230

        # Subscriber for velocity commands (cmd_vel)
        self.sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Publisher for the resulting wheel velocities
        self.pub = self.create_publisher(
            Float64MultiArray, 
            'cmd_wheel_vel', 
            10
        )

        self.get_logger().info("Differential Drive Controller (Twist to Wheels) has started.")

    def cmd_vel_callback(self, msg):
        # Extract linear velocity (v) and angular velocity (w)
        v = msg.linear.x
        w = msg.angular.z

        # Inverse Kinematics calculation
        # v_left  = (v - w * L/2) / R
        # v_right = (v + w * L/2) / R
        half_base = (w * self.wheel_base_distance) / 2.0
        
        left_wheel_velocity = (v - half_base) / self.wheel_radius
        right_wheel_velocity = (v + half_base) / self.wheel_radius

        # Prepare and publish the message
        out_msg = Float64MultiArray()
        out_msg.data = [left_wheel_velocity, right_wheel_velocity]
        self.pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DiffDriveController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()