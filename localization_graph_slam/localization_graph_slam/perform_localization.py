import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped, Point
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import MarkerArray, Marker

import gtsam
from gtsam.symbol_shorthand import L, X


import numpy as np
from math import atan2, sin, cos

from .utils import quaternion_from_euler, euler_from_quaternion, quaternion_ned_to_enu, warp_angle
from .line_extraction import SplitAndMerge
from .data_association import DataAssociation

def CartesianToPolar(cartesian_coordinate):
    x_f = cartesian_coordinate[0]
    y_f = cartesian_coordinate[1]
    range_f = np.sqrt(x_f**2 + y_f**2)
    theta_f = warp_angle(atan2(y_f, x_f))
    return [range_f, theta_f]

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
        self.Pk = np.diag([0.1, 0.1, 0.1])
        self.intialize_theta = False

        # Defining node publisher and subcriber
        self.joint_states_sub = self.create_subscription(JointState, "/turtlebot/joint_states", self.joint_state_callback, 20)
        self.imu_sub = self.create_subscription(Imu, "/turtlebot/sensors/imu_data", self.recieve_imu, 20)
        self.lidar_sub = self.create_subscription(LaserScan, "/turtlebot/scan", self.receive_lidar, 20)
        self.odom_ground_truth = self.create_subscription(Odometry, "/turtlebot/odom_ground_truth", self.recieve_odom_ground_truth, 20)
        self.odom_pub = self.create_publisher(Odometry, "/turtlebot/odom", 20)
        self.odom_ground_truth_enu_pub = self.create_publisher(Odometry, "/turtlebot/odom_ground_truth_enu", 20)
        self.visualize_map_pub = self.create_publisher(MarkerArray, "/turtlebot/line_features", 20)
        self.arm_controller_pub = self.create_publisher(
            Float64MultiArray,
            '/turtlebot/swiftpro/joint_velocity_controller/command',
            10,
        )
        self.tf_br = TransformBroadcaster(self)

        # Initialize the clock and the last time variable
        self.first_time = True
        self.last_time = self.get_clock().now()

        # Initialize joint state buffer
        self.joint_state_buffer = []
        self.joint_state_time_buffer = []

        # Initialize value and flag for IMU update
        self.imu_update_flag = False
        self.imu_orientation = 0.0
        self.imu_covariance = np.array([[0.01**2]])
        self.imu_buffer = []
        self.imu_time_buffer = []

        # Initialize value and flag for LiDAR update
        self.lidar_update_flag = False
        self.lidar_covariance = np.array([[0.01**2, 0.0], [0.0, 0.01**2]])
        self.line_buffer = []
        self.line_time_buffer = []

        # Initialize relative pose and covariance accumulated
        self.rel_disp = np.zeros((3, 1))
        self.rel_cov = np.zeros((3, 3))

        # Initialize graph slam
        self.isam2 = gtsam.ISAM2()
        self.graph = gtsam.NonlinearFactorGraph()
        self.i = 0
        self.initialize = True
        self.key_frame_update = 5
        self.num_min_key = 3
        self.k = 0

        # Initialize line extraction
        self.line_extractor = SplitAndMerge(0.01)
        self.polar_coordinates = []
        self.polar_covariances = []

        # Position of line feature in the map frame
        self.line_feature_map = []
        self.line_feature_cov_map = []
        self.new_feature = 0

        # Data association initialize
        self.H = []
        self.confidence_level = 0.99
        self.data_association = None

    # Function receive imu infomation data
    def recieve_imu(self, msg):
        # Extract orientation and covariance from IMU message
        orientation_ned = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        orientation_enu = quaternion_ned_to_enu(orientation_ned)
        cov_orientation = msg.orientation_covariance
        # self.get_logger().info(f"Received IMU orientation: {orientation_enu}, covariance: {cov_orientation[8]}")  # Log the yaw covariance

        # Convert quaternion to Euler angles and extract yaw (theta)
        self.imu_orientation = euler_from_quaternion(orientation_enu[0], orientation_enu[1], orientation_enu[2], orientation_enu[3])[2]  # Adjust for initial orientation
        self.imu_covariance = np.array([[cov_orientation[8]]])  # Assuming covariance for yaw is at index 8
        self.imu_update_flag = False

        # Store IMU data in buffer
        self.imu_buffer.append([self.imu_orientation, self.imu_covariance])
        self.imu_time_buffer.append(self.get_clock().now().nanoseconds / 1e9)
        if len(self.imu_buffer) > 5:  # Limit buffer size
            self.imu_buffer.pop(0)
            self.imu_time_buffer.pop(0)

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
        polar_coordinates = self.line_extractor.calculate_polar_coordinates(line_segments)
        # Align to robot local frame
        polar_coordinates = [[polar[0], warp_angle(polar[1] - np.pi)] for polar in polar_coordinates]
        polar_covariances = []
        for line in line_segments:
            cov_line = self.line_extractor.calculate_covariance_matrix(line)
            polar_covariances.append(cov_line)
        self.lidar_update_flag = True

        self.line_buffer.append([polar_coordinates, polar_covariances])
        self.line_time_buffer.append(self.get_clock().now().nanoseconds / 1e9)
        if len(self.line_buffer) > 5:  # Limit buffer size
            self.line_buffer.pop(0)
            self.line_time_buffer.pop(0)

        # Debug log
        # for i, polar_coordinate in enumerate(polar_coordinates):
        #     self.get_logger().info(f"Received line feature {i}: range {polar_coordinate[0]}, angle {polar_coordinate[1]}")
    
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
    
    def PolarToCartesian(self, polar_coordinate):
        range_f = polar_coordinate[0]
        theta_f = polar_coordinate[1]
        x_f = range_f * cos(theta_f)
        y_f = range_f * sin(theta_f)
        return np.array([x_f, y_f])
    
    def LineFeatureError(self, measurement, this, values, jacobians):
        pose = values.atPose2(this.keys()[0])
        landmark = values.atPoint2(this.keys()[1])

        x, y, theta = pose.x(), pose.y(), pose.theta()
        rho_f, alpha_f = landmark[0], landmark[1]

        rho_pred  = rho_f - cos(alpha_f)*x - sin(alpha_f)*y
        alpha_pred = warp_angle(alpha_f - theta)

        error = np.array([rho_pred - measurement[0],
                          warp_angle(alpha_pred - measurement[1])])

        if jacobians is not None:
            # d(error)/d(pose): shape (2, 3)
            jacobians[0] = np.array([
                [ -cos(alpha_f),  -sin(alpha_f), 0.0],
                [ 0.0,           0.0,         -1.0]
            ])
            # d(error)/d(landmark [rho_f, alpha_f]): shape (2, 2)
            jacobians[1] = np.array([
                [ 1.0,  sin(alpha_f)*x - cos(alpha_f)*y],
                [ 0.0, 1.0]
            ])
        return error
    
    def AddNewFeature(self):
        num_feature = len(self.line_feature_map)
        num_old_feature = num_feature - self.new_feature
        for i in range(num_old_feature, num_feature):
            polar_feature = self.line_feature_map[i]
            polar_cov_feature = self.line_feature_cov_map[i]
            noise_model = gtsam.noiseModel.Gaussian.Covariance(polar_cov_feature)
            feature_key = gtsam.symbol('L', i)
            self.graph.add(gtsam.PriorFactorPoint2(feature_key, gtsam.Point2(polar_feature[0], polar_feature[1]), noise_model))
            self.initial.insert(feature_key, gtsam.Point2(polar_feature[0], polar_feature[1]))
    
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
        # self.get_logger().info(f"Updating graph with IMU update: {self.imu_update_flag}, LiDAR update: {self.lidar_update_flag}")  # Log the update flags

        # Get matrix and value for updating process
        Hk = self.Hk()
        Vk = self.Vk()
        zk = np.array([[self.imu_orientation]])
        Rk = self.imu_covariance

        # Update process
        # Add pose factor to the graph
        self.i += 1
        rel_pos = gtsam.Pose2(self.rel_disp[0,0], self.rel_disp[1,0], self.rel_disp[2,0])
        OdometryNoise = gtsam.noiseModel.Gaussian.Covariance(self.rel_cov + 1e-7 * np.identity(3))  # Adding a small value to prevent zero covariance
        sigma_heading = float(np.sqrt(Rk[0,0] + 1e-7))  # Adding a small value to prevent zero covariance
        pose_key_prev = gtsam.symbol('X', self.i-1)
        pose_key_curr = gtsam.symbol('X', self.i)
        self.graph.add(gtsam.BetweenFactorPose2(pose_key_prev, pose_key_curr, rel_pos, OdometryNoise))


        if self.imu_update_flag:
            # self.get_logger().info(f"Adding IMU factor with orientation: {zk[0,0]}, covariance: {Rk[0,0]}")  # Log the IMU measurement and covariance
            self.graph.add(gtsam.PoseRotationPrior2D(pose_key_curr, gtsam.Rot2(zk[0,0]), gtsam.noiseModel.Isotropic.Sigma(1, sigma_heading)))
            self.k += 1
        
        self.initial.insert(pose_key_curr, gtsam.Pose2(xk_bar[0,0], xk_bar[1,0], xk_bar[2,0]))

        if self.lidar_update_flag:
            # self.get_logger().info(f"Number of line features observed: {len(self.polar_coordinates)}")
            for j in range(len(self.H)):
                if self.H[j] != None:
                    measurement = np.array([self.polar_coordinates[j][0], self.polar_coordinates[j][1]])
                    noise_model = gtsam.noiseModel.Gaussian.Covariance(self.polar_covariances[j])
                    feature_key = gtsam.symbol('L', self.H[j])
                    keys = gtsam.KeyVector()
                    keys.append(pose_key_curr)
                    keys.append(feature_key)
                    m = measurement.copy()
                    self.graph.add(gtsam.CustomFactor(noise_model, keys, lambda this, values, jacobians, meas=m: self.LineFeatureError(meas, this, values, jacobians)))
            if len(self.H) > 0 and self.imu_update_flag == False:
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
            for j in range(len(self.line_feature_map)):
                feature_key = gtsam.symbol('L', j)
                all_keys.append(feature_key)

            Pk_full = marginals.jointMarginalCovariance(all_keys).fullMatrix()
            Pk = Pk_full[0:3, 0:3]
            for j in range(len(self.line_feature_map)):
                feature_key = gtsam.symbol('L', j)
                line_feature = results.atPoint2(feature_key)
                self.line_feature_map[j] = [line_feature[0], line_feature[1]]
                self.line_feature_cov_map[j] = Pk_full[3+2*j:3+2*j+2, 3+2*j:3+2*j+2]
        else:
            xk = xk_bar
            Pk = Pk_bar

        self.rel_disp = np.zeros((3, 1))
        self.rel_cov = np.zeros((3, 3))

        return xk, Pk

    
    def joint_state_callback(self, msg):
        arm_control = Float64MultiArray()
        arm_control.data = [0.0, 0.0, -1.0, 0.0]
        self.arm_controller_pub.publish(arm_control)
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
        # self.get_logger().info(f"Time difference between joint state messages: {dt} seconds")  # Log the time difference between messages
        self.last_time = current_time
        if dt <=0:
            return
        
        left_wheel_velocity = msg.velocity[0]
        right_wheel_velocity = msg.velocity[1]

        # Adding noise to the wheel encoder sensor
        wheel_velocity = np.array([[left_wheel_velocity], [right_wheel_velocity]]) + np.random.normal(np.zeros((2,1)), np.array([np.sqrt(self.covariance_wheel_encoder.diagonal())]).T)

        current_time_in_sec = current_time.nanoseconds / 1e9
        # Try to synchronize the latest IMU and LiDAR data with the current joint state data
        if self.imu_update_flag and len(self.imu_time_buffer) > 0:
            while len(self.imu_time_buffer) > 0 and abs(self.imu_time_buffer[0] - current_time_in_sec) > 0.001:
                self.imu_time_buffer.pop(0)
                self.imu_buffer.pop(0)
            if len(self.imu_buffer) > 0:
                # self.get_logger().info(f"IMU time difference: {abs(self.imu_time_buffer[0] - current_time_in_sec)}")
                self.imu_orientation = self.imu_buffer[0][0]
                self.imu_covariance = self.imu_buffer[0][1]
            else:
                self.imu_update_flag = False
        
        if self.lidar_update_flag and len(self.line_time_buffer) > 0:
            while len(self.line_time_buffer) > 0 and abs(self.line_time_buffer[0] - current_time_in_sec) > 0.001:
                self.line_time_buffer.pop(0)
                self.line_buffer.pop(0)
            if len(self.line_buffer) > 0:
                # self.get_logger().info(f"LiDAR time difference: {abs(self.line_time_buffer[0] - current_time_in_sec)}")
                self.polar_coordinates = self.line_buffer[0][0]
                self.polar_covariances = self.line_buffer[0][1]
            else:
                self.lidar_update_flag = False

        # Predict pose of robot with covariance
        xk_bar, Pk_bar = self.Prediction(wheel_velocity, dt)

        # Data association
        if self.lidar_update_flag:
            self.data_association = DataAssociation(self.confidence_level, self.line_feature_map, self.line_feature_cov_map)
            self.H = self.data_association.DataAssociation(xk_bar, Pk_bar, self.polar_coordinates, self.polar_covariances)
        
        self.get_logger().info(f"Number of associated features: {len(self.H)}")  # Log the number of associated features
        
        # Check if robot moving or is translating larger than some threshold to update the graph
        trans_delta = np.linalg.norm(self.rel_disp[0:2, 0])
        rot_delta = abs(self.rel_disp[2, 0])
        should_add_key_frame = (trans_delta > 0.02) or (rot_delta > 0.02)

        # robot_moving = (abs(wheel_velocity[0, 0]) > 0.01) or (abs(wheel_velocity[1, 0]) > 0.01)
        # should_add_key_frame = should_add_key_frame and (not robot_moving)
        # Update function 
        if should_add_key_frame and (self.imu_update_flag == True or self.lidar_update_flag == True):
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
        
        # Add new feature to the map
        if self.lidar_update_flag:
            unsociated_features, unsociated_features_cov = self.data_association.GetUnassociatedFeatures(self.polar_coordinates, self.polar_covariances, self.H)
            self.new_feature = len(unsociated_features)
            if len(unsociated_features) > 0:
                xk = np.array([[self.x], [self.y], [self.theta]])
                self.data_association.AddmultipleNewFeatures(xk, self.Pk, unsociated_features, unsociated_features_cov)
                self.line_feature_map = self.data_association.map_feature
                self.line_feature_cov_map = self.data_association.map_feature_cov
                self.AddNewFeature()
            self.lidar_update_flag = False
        
        # for i, line in enumerate(self.line_feature_map):
        #     self.get_logger().info(f"Line feature {i} in the map: rho {line[0]}, alpha {line[1]}")

        self.get_logger().info(f"Number of features in the map: {len(self.line_feature_map)}")
        self.visualize_map()
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
        x = odom_ground_truth.pose.pose.position.x - 2.150009315590629
        y = odom_ground_truth.pose.pose.position.y - 1.3995580194955188
        # self.get_logger().info(f"Received ground truth odometry: x {x}, y {y}")  # Log the received ground truth position
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
    
    def visualize_map(self):
        marker_array = MarkerArray()
        for i, feature in enumerate(self.line_feature_map):
            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "line_features"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            rho_f, alpha_f = feature
            x_f = rho_f * cos(alpha_f)
            y_f = rho_f * sin(alpha_f)
            p1 = Point()
            p1.x = x_f + 0.5 * cos(alpha_f + np.pi/2)
            p1.y = y_f + 0.5 * sin(alpha_f + np.pi/2)
            p1.z = 0.0
            p2 = Point()
            p2.x = x_f - 0.5 * cos(alpha_f + np.pi/2)
            p2.y = y_f - 0.5 * sin(alpha_f + np.pi/2)
            p2.z = 0.0
            marker.points.append(p1)
            marker.points.append(p2)
            marker.scale.x = 0.05
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
        self.visualize_map_pub.publish(marker_array)




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