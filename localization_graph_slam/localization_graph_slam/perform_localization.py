import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import MarkerArray

import gtsam
from gtsam.symbol_shorthand import L, X


import numpy as np
from math import atan2, sin, cos

from .utils import quaternion_from_euler, euler_from_quaternion, quaternion_ned_to_enu
from .line_extraction import SplitAndMerge

class GraphSlam(Node):

    def __init__(self):
        super().__init__('graph_slam')

        # Initialize frame
        self.world_frame = "world_enu"
        self.base_footprint_frame = "turtlebot/base_footprint"

        # Initialize parameters of the robot
        self.base_length = 0.23
        self.radius_wheel = 0.035
        # self.base_length *= 1.1 # Adding noise to the base length parameter
        # self.radius_wheel *= 1.1
        self.covariance_wheel_encoder = np.diag(np.array([0.01**2, 0.01**2]))
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        # self.Pk = np.zeros((3, 3))
        self.Pk = np.diag([0.1, 0.1, 0.1])
        self.intialize_theta = False

        # Defining node publisher and subcriber
        self.joint_states_sub = self.create_subscription(JointState, "/turtlebot/joint_states", self.joint_state_callback, 20)
        self.imu_sub = self.create_subscription(Imu, "/turtlebot/sensors/imu_data", self.recieve_imu, 20)
        self.lidar_sub = self.create_subscription(LaserScan, "/turtlebot/scan", self.receive_lidar, 20)
        self.odom_ground_truth = self.create_subscription(Odometry, "/turtlebot/odom_ground_truth", self.recieve_odom_ground_truth, 20)
        self.odom_pub = self.create_publisher(Odometry, "/turtlebot/odom", 20)
        self.odom_ground_truth_enu_pub = self.create_publisher(Odometry, "/turtlebot/odom_ground_truth_enu", 20)
        self.tf_br = TransformBroadcaster(self)

        # Initialize the clock and the last time variable
        self.first_time = True
        self.last_time = self.get_clock().now()

        # Initialize value and flag for IMU update
        self.imu_update_flag = False
        self.imu_orientation = 0.0
        self.imu_covariance = np.zeros((1, 1))

        # Initialize value and flag for LiDAR update
        self.lidar_update_flag = False
        self.lidar_covariance = np.array([[0.01**2, 0.0], [0.0, 0.01**2]])

        # Initialize relative pose and covariance accumulated
        self.rel_disp = np.zeros((3, 1))
        self.rel_cov = np.zeros((3, 3))

        # Initialize graph slam
        self.isam2 = gtsam.ISAM2()
        self.graph = gtsam.NonlinearFactorGraph()
        self.i = 0
        self.initialize = True
        self.key_frame_update = 20
        self.num_min_key = 10
        self.k = 0

        # Initialize line extraction
        self.line_extractor = SplitAndMerge(0.01)
        self.polar_coordinates = []
        self.polar_covariances = []

        # Position of line feature in the map frame
        self.line_feature_map = []
        self.line_feature_cov_map = []


    # Function receive imu infomation data
    def recieve_imu(self, msg):
        # Extract orientation and covariance from IMU message
        orientation_ned = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        orientation_enu = quaternion_ned_to_enu(orientation_ned)
        cov_orientation = msg.orientation_covariance

        # Convert quaternion to Euler angles and extract yaw (theta)
        self.imu_orientation = euler_from_quaternion(orientation_enu[0], orientation_enu[1], orientation_enu[2], orientation_enu[3])[2]  # Adjust for initial orientation
        self.imu_covariance = np.array([[cov_orientation[8]]])  # Assuming covariance for yaw is at index 8
        self.imu_update_flag = True

        if not self.intialize_theta:
            self.theta = self.imu_orientation
            self.intialize_theta = True 

    # Function receive lidar information data
    def receive_lidar(self, msg):
        # Transform lidar range and angle data to Cartesian coordinates and extract line segments
        lidar_points = self.line_extractor.transform_lidar_to_cartesian(msg)
        line_segments = []
        line_segments = self.line_extractor.split(lidar_points, line_segments)
        line_segments = self.line_extractor.merge(line_segments)

        # Calculate polar coordinates and covariance of line segments
        self.polar_coordinates = self.line_extractor.calculate_polar_coordinates(line_segments)
        for line in line_segments:
            cov_line = self.line_extractor.calculate_line_covariance(line)
            self.polar_covariances.append(cov_line)
        self.lidar_update_flag = True
    
    # Define motion model
    def f(self, wheel_velocity, dt):

        # Transfer from velocity wheel to robot velocity
        linear_velocity = 1/2 * self.radius_wheel * (wheel_velocity[0, 0] + wheel_velocity[1, 0])
        angular_velocity = (self.radius_wheel/self.base_length) * (-wheel_velocity[0, 0] + wheel_velocity[1, 0])
        
        # Predict the robot pose within timestep dt
        x_k = self.x + cos(self.theta) * linear_velocity * dt
        y_k = self.y + sin(self.theta) * linear_velocity * dt
        theta_k = self.theta + angular_velocity * dt
        if theta_k > np.pi:
            theta_k -= 2 * np.pi
        elif theta_k < -np.pi:
            theta_k += 2 * np.pi
        
        return x_k, y_k, theta_k
    
    # Jacobian of motion model with respect to state
    def Jfx(self, wheel_velocity, dt):
        # Transfer from velocity wheel to robot velocity
        linear_velocity = 1/2 * self.radius_wheel * (wheel_velocity[0, 0] + wheel_velocity[1, 0])

        return np.array([[1, 0, -sin(self.theta) * linear_velocity * dt],
                         [0, 1, cos(self.theta) * linear_velocity * dt],
                         [0, 0, 1]])
    
    # Jacobian of motion model with respect to noise
    def Jfw(self, wheel_velocity, dt):
        return np.array([[1/2 * cos(self.theta) * dt, 1/2 * cos(self.theta) * dt],
                         [1/2 * sin(self.theta) * dt, 1/2 * sin(self.theta) * dt],
                         [- self.radius_wheel/self.base_length * dt, self.radius_wheel/self.base_length * dt]])
    
    # Jacobian of motion model with respect to relative pose
    def Jfx_rel(self, wheel_velocity, dt):
        linear_velocity = 1/2 * self.radius_wheel * (wheel_velocity[0, 0] + wheel_velocity[1, 0])

        return np.array([
            [1, 0, -sin(self.rel_disp[2, 0]) * linear_velocity * dt],
            [0, 1, cos(self.rel_disp[2, 0]) * linear_velocity * dt],
            [0, 0, 1]
        ])
    
    # Jacobianof motion model with respect to odometry noise
    def Jfw_rel(self, dt):
        return np.array([[1/2 * cos(self.rel_disp[2, 0]) * dt, 1/2 * cos(self.rel_disp[2, 0]) * dt],
                         [1/2 * sin(self.rel_disp[2, 0]) * dt, 1/2 * sin(self.rel_disp[2, 0]) * dt],
                         [- self.radius_wheel/self.base_length * dt, self.radius_wheel/self.base_length * dt]])
    
    # Observation model
    def h(self, xk_bar):
        return np.array([xk_bar[2]])
    
    # Jacobian of observation model with respect to state
    def Hk(self):
        return np.array([[0, 0, 1]])
    
    # Jacobian of observation model with respect to noise
    def Vk(self):
        return np.identity(1)
    
    # Prediction function
    def Prediction(self, wheel_velocity, dt):
        # Predict the position of the robot and the covariance
        x_bar, y_bar, theta_bar = self.f(wheel_velocity, dt)

        linear_velocity = 1/2 * self.radius_wheel * (wheel_velocity[0, 0] + wheel_velocity[1, 0])
        angular_velocity = (self.radius_wheel / self.base_length) * (-wheel_velocity[0, 0] + wheel_velocity[1, 0])

        dx_rel = cos(self.rel_disp[2, 0]) * linear_velocity * dt
        dy_rel = sin(self.rel_disp[2, 0]) * linear_velocity * dt
        dtheta_rel = angular_velocity * dt

        self.rel_disp += np.array([[dx_rel], [dy_rel], [dtheta_rel]])

        Jfx = self.Jfx(wheel_velocity, dt)
        Jfw = self.Jfw(wheel_velocity, dt)
        Pk_bar = Jfx @ self.Pk @ Jfx.T + Jfw @ self.covariance_wheel_encoder @ Jfw.T
        xk_bar = np.array([x_bar, y_bar, theta_bar]).reshape(3,1)

        Jfx_rel = self.Jfx_rel(wheel_velocity, dt)
        Jfw_rel = self.Jfw_rel(dt)

        self.rel_cov = Jfx_rel @ self.rel_cov @ Jfx_rel.T + Jfw_rel @ self.covariance_wheel_encoder @ Jfw_rel.T

        return xk_bar, Pk_bar

    # Update function
    def Update(self, xk_bar, Pk_bar):

        # Get matrix and value for updating process
        Hk = self.Hk()
        Vk = self.Vk()
        zk = np.array([[self.imu_orientation]])
        Rk = self.imu_covariance

        # Update process
        self.i += 1
        rel_pos = gtsam.Pose2(self.rel_disp[0,0], self.rel_disp[1,0], self.rel_disp[2,0])
        OdometryNoise = gtsam.noiseModel.Gaussian.Covariance(self.rel_cov + 1e-7 * np.eye(3))  # Adding a small value to prevent zero covariance
        sigma_heading = float(np.sqrt(Rk[0,0] + 1e-7))  # Adding a small value to prevent zero covariance
        pose_key_prev = gtsam.symbol('X', self.i-1)
        pose_key_curr = gtsam.symbol('X', self.i)
        self.graph.add(gtsam.BetweenFactorPose2(pose_key_prev, pose_key_curr, rel_pos, OdometryNoise))
        self.graph.add(gtsam.PoseRotationPrior2D(pose_key_curr, gtsam.Rot2(zk[0,0]), gtsam.noiseModel.Isotropic.Sigma(1, sigma_heading)))
        self.initial.insert(pose_key_curr, gtsam.Pose2(xk_bar[0,0], xk_bar[1,0], xk_bar[2,0]))
        self.k += 1
        if self.i > self.num_min_key and self.k >= self.key_frame_update:

            self.k = 0
            # Check update in the first time
            if self.initialize == True:
                optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial)
                self.initial = optimizer.optimize()
                self.initialize = False

            self.isam2.update(self.graph, self.initial)

            self.graph = gtsam.NonlinearFactorGraph()
            self.initial = gtsam.Values()

            full_graph = self.isam2.getFactorsUnsafe()

            results = self.isam2.calculateEstimate()
            pose = results.atPose2(pose_key_curr)
            xk = np.array([[pose.x()], [pose.y()], [pose.theta()]])
            marginals = gtsam.Marginals(full_graph, results)

            all_keys = gtsam.KeyVector()
            all_keys.append(pose_key_curr)

            Pk = marginals.jointMarginalCovariance(all_keys).fullMatrix()
        else:
            xk = xk_bar
            Pk = Pk_bar
        # self.get_logger().info(f"Updated Pose: x={xk[0,0]:.4f}, y={xk[1,0]:.4f}, theta={xk[2,0]:.4f}")
        # self.get_logger().info(f"Current Pose: x={self.x:.4f}, y={self.y:.4f}, theta={self.theta:.4f}")
        self.rel_disp = np.zeros((3, 1))
        self.rel_cov = np.zeros((3, 3))

        return xk, Pk

    
    def joint_state_callback(self, msg):
        if not self.intialize_theta:
            return
        current_time = self.get_clock().now()

        if self.first_time:
            self.first_time = False
            self.last_time = current_time
            PriorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.zeros(3) + 1e-7)
            pose_key = gtsam.symbol('X', self.i)
            self.graph.add(gtsam.PriorFactorPose2(pose_key, gtsam.Pose2(0, 0, self.theta), PriorNoise))
            self.initial = gtsam.Values()
            self.initial.insert(pose_key, gtsam.Pose2(0, 0, self.theta))
            return

        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        if dt <=0:
            return
        
        left_wheel_velocity = msg.velocity[0]
        right_wheel_velocity = msg.velocity[1]

        # Adding noise to the wheel encoder sensor
        wheel_velocity = np.array([[left_wheel_velocity], [right_wheel_velocity]]) + np.random.normal(np.zeros((2,1)), np.array([np.sqrt(self.covariance_wheel_encoder.diagonal())]).T)

        # Predict pose of robot with covariance
        xk_bar, Pk_bar = self.Prediction(wheel_velocity, dt)

        # Update function so use predict
        if self.imu_update_flag == True:
            xk, Pk = self.Update(xk_bar, Pk_bar)
            self.x = xk[0, 0]
            self.y = xk[1, 0]
            self.theta = xk[2, 0]
            self.Pk = Pk
            self.imu_update_flag = False
        else:
            self.x = xk_bar[0, 0]
            self.y = xk_bar[1, 0]
            self.theta = xk_bar[2, 0]
            self.Pk = Pk_bar
    
        # Transfer from velocity wheel to robot velocity
        linear_velocity = 1/2 * self.radius_wheel * (wheel_velocity[0, 0] + wheel_velocity[1, 0])
        angular_velocity = (self.radius_wheel/self.base_length) * (-wheel_velocity[0, 0] + wheel_velocity[1, 0])

        # Publish odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = self.world_frame
        odom_msg.child_frame_id = self.base_footprint_frame

        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0

        q = quaternion_from_euler(0, 0, self.theta)
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]

        odom_msg.twist.twist.linear.x = linear_velocity
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = angular_velocity

        # Fill 6x6 covariance matrix (flattened)
        cov = np.zeros((6, 6))
        cov[0:2, 0:2] = self.Pk[0:2, 0:2]
        cov[0:2, 5] = self.Pk[0:2, 2]
        cov[5, 0:2] = self.Pk[2, 0:2]
        cov[5, 5] = self.Pk[2, 2]
        odom_msg.pose.covariance = cov.flatten().tolist()

        self.odom_pub.publish(odom_msg)

        # Broadcast TF
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = self.world_frame
        t.child_frame_id = self.base_footprint_frame
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_br.sendTransform(t)

    def recieve_odom_ground_truth(self, msg):
        odom_ground_truth = msg
        x = odom_ground_truth.pose.pose.position.x
        y = odom_ground_truth.pose.pose.position.y
        linear_velocity = odom_ground_truth.twist.twist.linear.x
        angular_velocity = odom_ground_truth.twist.twist.angular.z

        q_x = odom_ground_truth.pose.pose.orientation.x
        q_y = odom_ground_truth.pose.pose.orientation.y
        q_z = odom_ground_truth.pose.pose.orientation.z
        q_w = odom_ground_truth.pose.pose.orientation.w
        _, _, theta = euler_from_quaternion(q_x, q_y, q_z, q_w)
        theta = np.pi - theta
        q = quaternion_from_euler(0 , 0, theta)

        current_time = self.get_clock().now()

        odom_ground_truth_enu_msg = Odometry()
        odom_ground_truth_enu_msg.header.stamp = current_time.to_msg()
        odom_ground_truth_enu_msg.header.frame_id = self.world_frame
        odom_ground_truth_enu_msg.child_frame_id = self.base_footprint_frame

        odom_ground_truth_enu_msg.pose.pose.position.x = -x
        odom_ground_truth_enu_msg.pose.pose.position.y = y
        odom_ground_truth_enu_msg.pose.pose.position.z = 0.0

        odom_ground_truth_enu_msg.pose.pose.orientation.x = q[0]
        odom_ground_truth_enu_msg.pose.pose.orientation.y = q[1]
        odom_ground_truth_enu_msg.pose.pose.orientation.z = q[2]
        odom_ground_truth_enu_msg.pose.pose.orientation.w = q[3]

        odom_ground_truth_enu_msg.twist.twist.linear.x = linear_velocity
        odom_ground_truth_enu_msg.twist.twist.linear.y = 0.0
        odom_ground_truth_enu_msg.twist.twist.linear.z = 0.0
        odom_ground_truth_enu_msg.twist.twist.angular.x = 0.0
        odom_ground_truth_enu_msg.twist.twist.angular.y = 0.0
        odom_ground_truth_enu_msg.twist.twist.angular.z = angular_velocity

        # Fill 6x6 covariance matrix (flattened)
        cov = np.zeros((6, 6))
        odom_ground_truth_enu_msg.pose.covariance = cov.flatten().tolist()

        self.odom_ground_truth_enu_pub.publish(odom_ground_truth_enu_msg)




def main(args=None):
    rclpy.init(args=args)

    graph_slam = GraphSlam()

    rclpy.spin(graph_slam)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    graph_slam.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()