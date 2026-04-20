#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

class DifferentialDrive(Node):
    def __init__(self):
        super().__init__('differential_drive')

        # 1. Declare and Get Parameters
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('wheel_left_joint_name', 'turtlebot/wheel_left_joint')
        self.declare_parameter('wheel_right_joint_name', 'turtlebot/wheel_right_joint')

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.left_wheel_name = self.get_parameter('wheel_left_joint_name').value
        self.right_wheel_name = self.get_parameter('wheel_right_joint_name').value

        # Robot constants
        self.wheel_radius = 0.035
        self.wheel_base_distance = 0.230
        self.wheel_vel_noise = np.array([0.01, 0.01])

        # State Variables
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.P = np.diag([0.01, 0.01, 0.1])
        
        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0
        self.left_wheel_received = False
        self.first_received = False
        self.last_time = self.get_clock().now()

        # 2. Publishers, Subscribers, and TF Broadcaster
        self.js_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 20)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 20)
        self.tf_br = TransformBroadcaster(self)

        self.get_logger().info(f"Odom Node Started. Base: {self.base_frame}, Odom: {self.odom_frame}")

    def quaternion_from_euler(self, roll, pitch, yaw):
        """Helper to replace tf.transformations.quaternion_from_euler"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        return [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy
        ]

    def f(self, dt, wheel_velocities):
        lin_vel = wheel_velocities * self.wheel_radius
        v = (lin_vel[0] + lin_vel[1]) / 2.0
        w = (lin_vel[1] - lin_vel[0]) / self.wheel_base_distance
        
        x_k = self.x + np.cos(self.th) * v * dt
        y_k = self.y + np.sin(self.th) * v * dt
        th_k = self.th + w * dt
        return np.array([x_k, y_k, th_k])
    
    def Jfx(self, dt, wheel_velocities):
        lin_vel = wheel_velocities * self.wheel_radius
        v = (lin_vel[0] + lin_vel[1]) / 2.0
        return np.array([[1, 0, -v * np.sin(self.th) * dt],
                         [0, 1, v * np.cos(self.th) * dt],
                         [0, 0, 1]])
    
    def Jfw(self, dt):
        return np.array([[np.cos(self.th) * dt / 2.0, np.cos(self.th) * dt / 2.0],
                         [np.sin(self.th) * dt / 2.0, np.sin(self.th) * dt / 2.0],
                         [-dt / self.wheel_base_distance, dt / self.wheel_base_distance]])

    def joint_state_callback(self, msg):
        self.left_wheel_velocity = msg.velocity[msg.name.index(self.left_wheel_name)]
        self.right_wheel_velocity = msg.velocity[msg.name.index(self.right_wheel_name)]
        
        current_time = self.get_clock().now()
        
        if not self.first_received:
            self.first_received = True
            self.last_time = current_time
            return

        # Calculate dt in seconds
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        if dt <= 0: return

        # Integrate Position
        wheel_velocities = np.array([self.left_wheel_velocity, self.right_wheel_velocity])
        self.x, self.y, self.th = self.f(dt, wheel_velocities)

        # Propagate Covariance
        F = self.Jfx(dt, wheel_velocities)
        W = self.Jfw(dt)
        Q = np.diag(self.wheel_vel_noise)
        self.P = F @ self.P @ F.T + W @ Q @ W.T

        # Calculate velocities for odom message
        v = (wheel_velocities.sum() * self.wheel_radius) / 2.0
        w = (wheel_velocities[1] - wheel_velocities[0]) * self.wheel_radius / self.wheel_base_distance

        # Publish Odom
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        
        q = self.quaternion_from_euler(0, 0, self.th)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        # Fill 6x6 covariance matrix (flattened)
        cov = np.zeros((6, 6))
        cov[0:2, 0:2] = self.P[0:2, 0:2]
        cov[0:2, 5] = self.P[0:2, 2]
        cov[5, 0:2] = self.P[2, 0:2]
        cov[5, 5] = self.P[2, 2]
        odom.pose.covariance = cov.flatten().tolist()

        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w
        self.odom_pub.publish(odom)

        # Broadcast TF
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_br.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = DifferentialDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

    





