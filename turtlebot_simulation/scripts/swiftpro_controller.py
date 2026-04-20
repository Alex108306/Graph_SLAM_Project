#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

class SwiftproController(Node):
    def __init__(self):
        # Initialize the node with its name
        super().__init__('swiftpro_controller')

        self.declare_parameter('joint2_name', 'swiftpro/joint2')
        self.declare_parameter('joint3_name', 'swiftpro/joint3')

        self.joint2_name = self.get_parameter('joint2_name').value
        self.joint3_name = self.get_parameter('joint3_name').value

        # Joint state subscriber
        # In ROS 2, we provide (msg_type, topic, callback, qos_profile)
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Passive joint position publisher
        self.passive_joint_pub = self.create_publisher(
            Float64MultiArray,
            'command',
            10
        )
        
        self.get_logger().info("Swiftpro Passive Joint Controller has been started.")

    def joint_state_callback(self, msg):
        # Ensure we have the expected number of active joints
        if msg.name.index(self.joint2_name) is None or msg.name.index(self.joint3_name) is None:
            self.get_logger().warn("Expected joint names not found in JointState message.")
            return
        
        # Convert angles from Radians to Degrees
        # Assuming active joints are at indices 1 and 2 based on your original logic
        joint2_angle = msg.position[msg.name.index(self.joint2_name)] / math.pi * 180.0
        joint3_angle = msg.position[msg.name.index(self.joint3_name)] / math.pi * 180.0
        
        # Kinematic calculation for passive joints
        alpha2 = 90.0 - joint2_angle
        alpha3 = joint3_angle - 3.8
        
        pj1 = (alpha2 + alpha3) - 176.11 + 90.0
        pj2 = -90.0 + alpha2
        pj3 = joint2_angle
        pj5 = 90.0 - (alpha2 + alpha3 + 3.8)
        pj7 = 176.11 - 180.0 - alpha3
        pj8 = 48.39 + alpha3 - 44.55

        # Prepare output message
        msg_out = Float64MultiArray()
        # Convert back to Radians for the controller
        msg_out.data = [
            pj1 / 180.0 * math.pi,
            pj2 / 180.0 * math.pi, 
            pj3 / 180.0 * math.pi,
            pj5 / 180.0 * math.pi,
            pj7 / 180.0 * math.pi,
            pj8 / 180.0 * math.pi
        ]
        
        self.passive_joint_pub.publish(msg_out)

def main(args=None):
    rclpy.init(args=args)
    node = SwiftproController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()