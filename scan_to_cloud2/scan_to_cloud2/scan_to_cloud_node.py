#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from laser_geometry import LaserProjection
import tf2_ros
import tf2_geometry_msgs

class ScanToCloudNode(Node):
    def __init__(self):
        super().__init__('scan_to_cloud_node')
        self.laser_projector = LaserProjection()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/turtlebot/scan',
            self.scan_callback,
            10
        )

        self.cloud_pub = self.create_publisher(
            PointCloud2,
            '/turtlebot/cloud',
            10
        )

    def scan_callback(self, scan_msg: LaserScan):
        try:
            cloud_msg = self.laser_projector.projectLaser(scan_msg)
            self.cloud_pub.publish(cloud_msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to project laser scan: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ScanToCloudNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
