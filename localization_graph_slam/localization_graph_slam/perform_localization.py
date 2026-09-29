import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped, Point, Pose, PoseArray
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from visualization_msgs.msg import MarkerArray, Marker
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor, FloatingPointRange, IntegerRange

import gtsam
from gtsam.symbol_shorthand import L, X


import numpy as np
from math import atan2, sin, cos
import math

from .utils import quaternion_from_euler, euler_from_quaternion, quaternion_ned_to_enu, warp_angle
from .line_extraction import SplitAndMerge
from .data_association import DataAssociation
from .feature_candidate_tracker import FeatureCandidateTracker

def CartesianToPolar(cartesian_coordinate):
    x_f = cartesian_coordinate[0]
    y_f = cartesian_coordinate[1]
    range_f = np.sqrt(x_f**2 + y_f**2)
    theta_f = warp_angle(atan2(y_f, x_f))
    return [range_f, theta_f]

class GraphSlam(Node):

    def __init__(self):
        super().__init__('graph_slam')

        self.declare_parameter('mode', 'sim')
        self.mode = self.get_parameter('mode').get_parameter_value().string_value

        # Initialize frame
        if self.mode == 'sim':
            self.world_frame = "world_enu"
            self.base_footprint_frame = "turtlebot/base_footprint"
            # Sim has no separate wheel-odom TF, so we publish world->base directly.
            self.odom_frame = None
        else:
            # REP-105: localization owns map->odom, wheel odometry owns odom->base.
            # If your stack already has a frame called "map", change this to e.g. "slam_map".
            self.world_frame = "map"
            self.odom_frame = "odom"
            self.base_footprint_frame = "base_footprint"

        # ---- Declare tunable parameters (rqt_reconfigure compatible) ----
        self._declare_tunable_parameters()
        # ---- Load tunable parameters into instance state ----
        self._load_tunable_parameters()

        # Robot pose state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.Pk = np.diag([0.01, 0.01, 0.1])
        self.intialize_theta = False

        # Defining node publisher and subcriber
        if self.mode == 'sim':
            self.joint_states_sub = self.create_subscription(JointState, "/turtlebot/joint_states", self.joint_state_callback, 20)
            self.imu_sub = self.create_subscription(Imu, "/turtlebot/sensors/imu_data", self.recieve_imu, 20)
            self.lidar_sub = self.create_subscription(LaserScan, "/turtlebot/scan", self.receive_lidar, 20)
            self.odom_ground_truth = self.create_subscription(Odometry, "/turtlebot/odom_ground_truth", self.recieve_odom_ground_truth, 20)
            self.odom_pub = self.create_publisher(Odometry, "/turtlebot/odom", 20)
            self.odom_ground_truth_enu_pub = self.create_publisher(Odometry, "/turtlebot/odom_ground_truth_enu", 20)
            self.visualize_map_pub = self.create_publisher(MarkerArray, "/turtlebot/line_features", 20)
            self.line_error_pub_ = self.create_publisher(MarkerArray, "/turtlebot/line_feature_errors", 20)
            self.visualize_association_pub = self.create_publisher(MarkerArray, "/turtlebot/line_associations", 20)
            self.arm_controller_pub = self.create_publisher(
                Float64MultiArray,
                '/turtlebot/swiftpro/joint_velocity_controller/command',
                10,
            )
        else:
            self.joint_states_sub = self.create_subscription(JointState, "/turtlebot/joint_states", self.joint_state_callback, 20)
            self.imu_sub = self.create_subscription(Imu, "/turtlebot/imu", self.recieve_imu, 20)
            self.lidar_sub = self.create_subscription(LaserScan, "/turtlebot/scan", self.receive_lidar, 20)
            self.odom_pub = self.create_publisher(Odometry, "/turtlebot/odom_localization", 20)
            self.visualize_map_pub = self.create_publisher(MarkerArray, "/turtlebot/line_features", 20)
            self.line_error_pub_ = self.create_publisher(MarkerArray, "/turtlebot/line_feature_errors", 20)
            self.visualize_association_pub = self.create_publisher(MarkerArray, "/turtlebot/line_associations", 20)

        self.tf_br = TransformBroadcaster(self)
        # TF listener used in real mode to look up odom->base for the map->odom correction.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialize the clock and the last time variable
        self.first_time = True
        self.last_time = None

        # Initialize joint state buffer
        self.joint_state_buffer = []
        self.joint_state_time_buffer = []

        # Initialize value and flag for IMU update
        self.imu_update_flag = False
        self.imu_orientation = 0.0
        self.imu_buffer = []
        self.imu_time_buffer = []

        # Initialize value and flag for LiDAR update
        self.lidar_update_flag = False
        self.line_buffer = []
        self.line_time_buffer = []
        self.line_segments = None

        # Initialize relative pose and covariance accumulated
        self.rel_disp = np.zeros((3, 1))
        self.rel_cov = np.zeros((3, 3))

        # Previous robot pose
        self.prev_pose = None
        self.prev_cov = None

        # Initialize graph slam.
        self.isam2 = gtsam.ISAM2()
        self.graph = gtsam.NonlinearFactorGraph()
        self.i = 0
        self.initialize = True
        self.k = 0

        # Initialize line extraction (parameters loaded above)
        self.line_extractor = SplitAndMerge(
            distance_threshold=self.get_parameter('line_split_distance_threshold').value,
            mode=self.mode,
            min_points=self.get_parameter('line_min_points').value,
            merge_angle_threshold=math.radians(self.get_parameter('line_merge_angle_deg').value),
            merge_distance_threshold=self.get_parameter('line_merge_distance_threshold').value,
            measurement_sigma_r=self.get_parameter('line_measurement_sigma_r').value,
        )
        self.polar_coordinates = []
        self.polar_covariances = []

        # Position of line feature in the map frame
        self.line_feature_map = []
        self.line_feature_cov_map = []
        self.line_segments_map = []
        self.new_feature = 0

        # Data association initialize
        self.H = []
        self.data_association = None

        # Persistency: only promote a candidate line to the map after it has been
        # re-observed enough times. Filters out spurious / dirty extractions.
        self.feature_candidates = FeatureCandidateTracker(
            min_observations=self.feature_min_observations,
            match_rho_threshold=self.feature_candidate_match_rho,
            match_alpha_threshold=self.feature_candidate_match_alpha,
            max_frames_unseen=self.feature_candidate_max_unseen,
        )

        # Initialize time accumulate
        self.time_accumulate = 0.0

        # Latest robot velocity (updated every joint_state_callback; used by receive_lidar)
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        # Register dynamic parameter callback so rqt_reconfigure can retune at runtime
        self.add_on_set_parameters_callback(self._on_parameters_set)

    # ------------------------------------------------------------------
    # Tunable-parameter plumbing (rqt_reconfigure)
    # ------------------------------------------------------------------
    def _declare_tunable_parameters(self):
        """Declare every parameter that can be changed live from rqt_reconfigure."""
        def f_desc(description, lo=None, hi=None, step=0.0):
            d = ParameterDescriptor(description=description)
            if lo is not None and hi is not None:
                d.floating_point_range = [FloatingPointRange(from_value=float(lo), to_value=float(hi), step=float(step))]
            return d

        def i_desc(description, lo=None, hi=None, step=1):
            d = ParameterDescriptor(description=description)
            if lo is not None and hi is not None:
                d.integer_range = [IntegerRange(from_value=int(lo), to_value=int(hi), step=int(step))]
            return d

        # Robot kinematics
        self.declare_parameter('base_length',  0.23, f_desc('Wheelbase (m) between left/right wheel contact points', 0.05, 1.0))
        self.declare_parameter('radius_wheel', 0.035, f_desc('Wheel radius (m)', 0.005, 0.5))
        self.declare_parameter('wheel_encoder_sigma', 2.0, f_desc('Std-dev of wheel encoder velocity noise (rad/s)', 0.0, 10.0))

        # IMU
        self.declare_parameter('imu_sigma', 0.05, f_desc('Std-dev of IMU yaw measurement (rad)', 0.0, 1.0))

        # Sensor sync thresholds (s)
        default_sync = 0.05 if self.mode == 'sim' else 0.2
        self.declare_parameter('sync_time_thrsh_imu',  default_sync, f_desc('Max age (s) to accept buffered IMU measurement', 0.0, 2.0))
        self.declare_parameter('sync_time_thrsh_line', default_sync, f_desc('Max age (s) to accept buffered LiDAR measurement', 0.0, 2.0))

        # Graph SLAM key-frame strategy
        self.declare_parameter('key_frame_update', 2, i_desc('Number of accumulated updates before iSAM2 step', 1, 50))
        self.declare_parameter('num_min_key',      2, i_desc('Minimum pose keys before first iSAM2 step', 1, 50))
        self.declare_parameter('keyframe_trans_threshold', 0.30, f_desc('Translation (m) that triggers a new key frame', 0.0, 2.0))
        self.declare_parameter('keyframe_rot_threshold_deg', 5.0, f_desc('Rotation (deg) that triggers a new key frame', 0.0, 90.0))
        self.declare_parameter('keyframe_time_threshold', 2.0, f_desc('Idle time (s) that triggers a new key frame', 0.0, 30.0))

        # Line extraction (Split-and-Merge)
        self.declare_parameter('line_split_distance_threshold', 0.03, f_desc('Max point-to-line distance (m) before splitting', 0.001, 1.0))
        self.declare_parameter('line_min_points', 25, i_desc('Minimum number of points to keep a line segment', 2, 200))
        self.declare_parameter('line_min_length', 0.25, f_desc('Minimum line length (m) kept after extraction', 0.0, 5.0))
        self.declare_parameter('line_merge_angle_deg', 10.0, f_desc('Max heading delta (deg) for merging two lines', 0.0, 90.0))
        self.declare_parameter('line_merge_distance_threshold', 0.10, f_desc('Max rho delta (m) for merging two lines', 0.0, 1.0))
        self.declare_parameter('line_measurement_sigma_r', 0.05, f_desc('Endpoint xy std-dev (m) used for line covariance', 0.0, 1.0))

        # Data association
        self.declare_parameter('da_confidence_level', 0.95, f_desc('Chi^2 confidence level for ICNN gating', 0.5, 0.999))
        self.declare_parameter('da_duplicate_rho_threshold', 0.02, f_desc('Duplicate-feature gate on rho (m)', 0.0, 2.0))
        self.declare_parameter('da_duplicate_alpha_threshold', 0.01, f_desc('Duplicate-feature gate on alpha (rad)', 0.0, 3.1416))
        self.declare_parameter('da_measurement_sigma_rho_floor',   0.04, f_desc('Lower floor on rho std-dev   (m) before DA', 0.0, 1.0))
        self.declare_parameter('da_measurement_sigma_alpha_floor', 0.06, f_desc('Lower floor on alpha std-dev (rad) before DA', 0.0, 1.0))

        # Debug / diagnostics
        self.declare_parameter('debug', False, ParameterDescriptor(description='Enable verbose per-frame logs and publish /debug/* diagnostic topics'))

        # Feature persistency (candidate tracker)
        self.declare_parameter('feature_min_observations', 5, i_desc('Times a line must be re-observed before it enters the map', 1, 20))
        self.declare_parameter('feature_candidate_match_rho',   0.1, f_desc('Candidate match gate on rho (m) — looser than the duplicate gate', 0.0, 2.0))
        self.declare_parameter('feature_candidate_match_alpha', 0.05, f_desc('Candidate match gate on alpha (rad)', 0.0, 1.5708))
        self.declare_parameter('feature_candidate_max_unseen_frames', 8, i_desc('Drop a candidate after this many keyframes without a re-observation', 1, 100))

    def _load_tunable_parameters(self):
        """Copy parameter values into instance attributes used by the algorithm."""
        gp = lambda n: self.get_parameter(n).value

        self.base_length  = float(gp('base_length'))
        self.radius_wheel = float(gp('radius_wheel'))
        sigma_enc = float(gp('wheel_encoder_sigma'))
        self.covariance_wheel_encoder = np.diag([sigma_enc**2, sigma_enc**2])

        sigma_imu = float(gp('imu_sigma'))
        self.imu_covariance = np.array([[sigma_imu**2]])

        self.sync_time_thrsh_imu  = float(gp('sync_time_thrsh_imu'))
        self.sync_time_thrsh_line = float(gp('sync_time_thrsh_line'))

        self.key_frame_update = int(gp('key_frame_update'))
        self.num_min_key      = int(gp('num_min_key'))
        self.keyframe_trans_threshold = float(gp('keyframe_trans_threshold'))
        self.keyframe_rot_threshold   = math.radians(float(gp('keyframe_rot_threshold_deg')))
        self.keyframe_time_threshold  = float(gp('keyframe_time_threshold'))

        self.line_min_length = float(gp('line_min_length'))

        self.confidence_level             = float(gp('da_confidence_level'))
        self.duplicate_rho_threshold      = float(gp('da_duplicate_rho_threshold'))
        self.duplicate_alpha_threshold    = float(gp('da_duplicate_alpha_threshold'))
        self.measurement_sigma_rho_floor   = float(gp('da_measurement_sigma_rho_floor'))
        self.measurement_sigma_alpha_floor = float(gp('da_measurement_sigma_alpha_floor'))

        self.debug = bool(gp('debug'))

        self.feature_min_observations        = int(gp('feature_min_observations'))
        self.feature_candidate_match_rho     = float(gp('feature_candidate_match_rho'))
        self.feature_candidate_match_alpha   = float(gp('feature_candidate_match_alpha'))
        self.feature_candidate_max_unseen    = int(gp('feature_candidate_max_unseen_frames'))

    def _on_parameters_set(self, params):
        """rqt_reconfigure / `ros2 param set` callback: apply changes live."""
        for param in params:
            name, value = param.name, param.value
            if name == 'base_length':
                self.base_length = float(value)
            elif name == 'radius_wheel':
                self.radius_wheel = float(value)
            elif name == 'wheel_encoder_sigma':
                sigma = float(value)
                self.covariance_wheel_encoder = np.diag([sigma**2, sigma**2])
            elif name == 'imu_sigma':
                sigma = float(value)
                self.imu_covariance = np.array([[sigma**2]])
            elif name == 'sync_time_thrsh_imu':
                self.sync_time_thrsh_imu = float(value)
            elif name == 'sync_time_thrsh_line':
                self.sync_time_thrsh_line = float(value)
            elif name == 'key_frame_update':
                self.key_frame_update = int(value)
            elif name == 'num_min_key':
                self.num_min_key = int(value)
            elif name == 'keyframe_trans_threshold':
                self.keyframe_trans_threshold = float(value)
            elif name == 'keyframe_rot_threshold_deg':
                self.keyframe_rot_threshold = math.radians(float(value))
            elif name == 'keyframe_time_threshold':
                self.keyframe_time_threshold = float(value)
            elif name == 'line_split_distance_threshold':
                self.line_extractor.distance_threshold = float(value)
            elif name == 'line_min_points':
                self.line_extractor.min_points = int(value)
            elif name == 'line_min_length':
                self.line_min_length = float(value)
            elif name == 'line_merge_angle_deg':
                self.line_extractor.merge_angle_threshold = math.radians(float(value))
            elif name == 'line_merge_distance_threshold':
                self.line_extractor.merge_distance_threshold = float(value)
            elif name == 'line_measurement_sigma_r':
                self.line_extractor.measurement_sigma_r = float(value)
            elif name == 'da_confidence_level':
                self.confidence_level = float(value)
            elif name == 'da_duplicate_rho_threshold':
                self.duplicate_rho_threshold = float(value)
            elif name == 'da_duplicate_alpha_threshold':
                self.duplicate_alpha_threshold = float(value)
            elif name == 'da_measurement_sigma_rho_floor':
                self.measurement_sigma_rho_floor = float(value)
            elif name == 'da_measurement_sigma_alpha_floor':
                self.measurement_sigma_alpha_floor = float(value)
            elif name == 'feature_min_observations':
                self.feature_min_observations = int(value)
                if self.feature_candidates is not None:
                    self.feature_candidates.min_observations = self.feature_min_observations
            elif name == 'feature_candidate_match_rho':
                self.feature_candidate_match_rho = float(value)
                if self.feature_candidates is not None:
                    self.feature_candidates.match_rho_threshold = self.feature_candidate_match_rho
            elif name == 'feature_candidate_match_alpha':
                self.feature_candidate_match_alpha = float(value)
                if self.feature_candidates is not None:
                    self.feature_candidates.match_alpha_threshold = self.feature_candidate_match_alpha
            elif name == 'feature_candidate_max_unseen_frames':
                self.feature_candidate_max_unseen = int(value)
                if self.feature_candidates is not None:
                    self.feature_candidates.max_frames_unseen = self.feature_candidate_max_unseen
            elif name == 'debug':
                self.debug = bool(value)
        return SetParametersResult(successful=True)

    # Function receive imu infomation data
    def recieve_imu(self, msg):
        # Extract orientation and covariance from IMU message
        if self.mode == 'sim':
            orientation_ned = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
            orientation_enu = quaternion_ned_to_enu(orientation_ned)
        else:
            orientation_enu = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

        # Convert quaternion to Euler angles and extract yaw (theta)
        self.imu_orientation = euler_from_quaternion(orientation_enu[0], orientation_enu[1], orientation_enu[2], orientation_enu[3])[2]  # Adjust for initial orientation
        self.imu_update_flag = True
        ang_vel = msg.angular_velocity.z

        # Store IMU data in buffer
        self.imu_buffer.append([self.imu_orientation, self.imu_covariance, ang_vel])
        imu_time = rclpy.time.Time.from_msg(msg.header.stamp)
        self.imu_time_buffer.append(imu_time.nanoseconds / 1e9)
        if len(self.imu_buffer) > 5:  # Limit buffer size
            self.imu_buffer.pop(0)
            self.imu_time_buffer.pop(0)

        if not self.intialize_theta:
            self.theta = self.imu_orientation
            self.intialize_theta = True 

    def _broadcast_map_to_odom_correction(self, current_time):
        # Compute T_map_odom = T_map_base * inv(T_odom_base) and broadcast it.
        # On the very first ticks the odom->base TF may not be available yet;
        # in that case we skip rather than publish a stale/zero correction.
        try:
            tf_odom_base = self.tf_buffer.lookup_transform(
                self.odom_frame, self.base_footprint_frame,
                rclpy.time.Time())  # latest available
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f"map->odom correction skipped: cannot look up {self.odom_frame}->{self.base_footprint_frame}: {e}",
                throttle_duration_sec=2.0,
            )
            return

        ox = tf_odom_base.transform.translation.x
        oy = tf_odom_base.transform.translation.y
        oq = tf_odom_base.transform.rotation
        _, _, o_theta = euler_from_quaternion(oq.x, oq.y, oq.z, oq.w)

        # inv(T_odom_base) in 2D: rotate -theta and negate the rotated translation.
        c, s = cos(o_theta), sin(o_theta)
        ibx = -(c * ox + s * oy)
        iby = -(-s * ox + c * oy)
        ib_theta = -o_theta

        # T_map_base composed with inv(T_odom_base) -> T_map_odom
        cm, sm = cos(self.theta), sin(self.theta)
        mo_x = cm * ibx - sm * iby + self.x
        mo_y = sm * ibx + cm * iby + self.y
        mo_theta = warp_angle(self.theta + ib_theta)
        mo_q = quaternion_from_euler(0.0, 0.0, mo_theta)

        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = self.world_frame   # "map"
        t.child_frame_id = self.odom_frame     # "odom"
        t.transform.translation.x = mo_x
        t.transform.translation.y = mo_y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = mo_q[0]
        t.transform.rotation.y = mo_q[1]
        t.transform.rotation.z = mo_q[2]
        t.transform.rotation.w = mo_q[3]
        self.tf_br.sendTransform(t)

    # Function receive lidar information data
    def receive_lidar(self, msg):
        # Transform lidar range and angle data to Cartesian coordinates and extract line segments
        lidar_points = self.line_extractor.transform_lidar_to_cartesian(msg)
        line_segments = []
        line_segments = self.line_extractor.split(lidar_points, line_segments)
        line_segments = self.line_extractor.merge(line_segments)
        line_segments = [
            line for line in line_segments
            if np.linalg.norm(np.array(line[1]) - np.array(line[0])) >= self.line_min_length
        ]

        # Calculate polar coordinates and covariance of line segments
        polar_coordinates = self.line_extractor.calculate_polar_coordinates(line_segments)
        # Align to robot local frame
        polar_coordinates = [[polar[0], polar[1]] for polar in polar_coordinates]
        polar_covariances = []
        for line in line_segments:
            cov_line = self.line_extractor.calculate_covariance_matrix(line)
            cov_line = 0.5 * (cov_line + cov_line.T)
            cov_line = cov_line + 1e-9 * np.eye(2)
            polar_covariances.append(cov_line)
        self.lidar_update_flag = True

        self.line_buffer.append([polar_coordinates, polar_covariances, line_segments])
        line_time = rclpy.time.Time.from_msg(msg.header.stamp)
        self.line_time_buffer.append(line_time.nanoseconds / 1e9)
        if len(self.line_buffer) > 5:  # Limit buffer size
            self.line_buffer.pop(0)
            self.line_time_buffer.pop(0)
    
    # Define motion model
    def f(self, wheel_velocity, dt):

        # Transfer from velocity wheel to robot velocity
        linear_velocity = 1/2 * self.radius_wheel * (wheel_velocity[0, 0] + wheel_velocity[1, 0])
        angular_velocity = (self.radius_wheel/self.base_length) * (-wheel_velocity[0, 0] + wheel_velocity[1, 0])
        
        # Predict the robot pose within timestep dt (exact ICC integration)
        theta_k = self.theta + angular_velocity * dt
        x_k = self.x + cos(self.theta) * linear_velocity * dt
        y_k = self.y + sin(self.theta) * linear_velocity * dt
        if theta_k > np.pi:
            theta_k -= 2 * np.pi
        elif theta_k < -np.pi:
            theta_k += 2 * np.pi
        
        return x_k, y_k, theta_k
    
    # Jacobian of motion model with respect to state
    def Jfx(self, wheel_velocity, dt):
        linear_velocity = 1/2 * self.radius_wheel * (wheel_velocity[0, 0] + wheel_velocity[1, 0])
        dxk_dtheta = -sin(self.theta) * linear_velocity * dt
        dyk_dtheta =  cos(self.theta) * linear_velocity * dt
        return np.array([[1, 0, dxk_dtheta],
                         [0, 1, dyk_dtheta],
                         [0, 0, 1]])
    
    # Jacobian of motion model with respect to noise
    def Jfw(self, wheel_velocity, dt):
        return np.array([[1/2 * self.radius_wheel * cos(self.theta) * dt, 1/2 * self.radius_wheel * cos(self.theta) * dt],
                         [1/2 * self.radius_wheel * sin(self.theta) * dt, 1/2 * self.radius_wheel * sin(self.theta) * dt],
                         [- self.radius_wheel/self.base_length * dt, self.radius_wheel/self.base_length * dt]])
    
    # Jacobian of motion model with respect to relative pose
    def Jfx_rel(self, wheel_velocity, dt):
        linear_velocity = 1/2 * self.radius_wheel * (wheel_velocity[0, 0] + wheel_velocity[1, 0])
        dxk_dtheta = -sin(self.rel_disp[2, 0]) * linear_velocity * dt
        dyk_dtheta =  cos(self.rel_disp[2, 0]) * linear_velocity * dt
        return np.array([
            [1, 0, dxk_dtheta],
            [0, 1, dyk_dtheta],
            [0, 0, 1]
        ])
    
    # Jacobianof motion model with respect to odometry noise
    def Jfw_rel(self, wheel_velocity, dt):
        return np.array([[1/2 * self.radius_wheel * cos(self.rel_disp[2, 0]) * dt, 1/2 * self.radius_wheel * cos(self.rel_disp[2, 0]) * dt],
                         [1/2 * self.radius_wheel * sin(self.rel_disp[2, 0]) * dt, 1/2 * self.radius_wheel * sin(self.rel_disp[2, 0]) * dt],
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
                [ -cos(alpha_f-theta),  -sin(alpha_f-theta), 0.0],
                [ 0.0,           0.0,         -1.0]
            ])
            # d(error)/d(landmark [rho_f, alpha_f]): shape (2, 2)
            jacobians[1] = np.array([
                [ 1.0,  sin(alpha_f)*x - cos(alpha_f)*y],
                [ 0.0, 1.0]
            ])
        return error
    
    def AddNewFeature(self, pose_key_curr, measurements, covariances):
        num_feature = len(self.line_feature_map)
        num_old_feature = num_feature - self.new_feature
        for local_idx, feature_idx in enumerate(range(num_old_feature, num_feature)):
            polar_feature = self.line_feature_map[feature_idx]
            weak_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5, math.radians(20)]))
            feature_key = gtsam.symbol('L', feature_idx)
            self.graph.add(gtsam.PriorFactorPoint2(feature_key, gtsam.Point2(polar_feature[0], polar_feature[1]), weak_noise))

            measurement = np.array(measurements[local_idx])
            noise_model = gtsam.noiseModel.Gaussian.Covariance(covariances[local_idx])

            keys = gtsam.KeyVector()
            keys.append(pose_key_curr)
            keys.append(feature_key)

            m = measurement.copy()
            self.initial.insert(feature_key, gtsam.Point2(polar_feature[0], polar_feature[1]))
            self.graph.add(
                gtsam.CustomFactor(
                    noise_model,
                    keys,
                    lambda this, values, jacobians, meas=m: self.LineFeatureError(meas, this, values, jacobians),
                )
            )
    
    # Prediction function
    def Prediction(self, wheel_velocity, dt):
        # Predict the position of the robot and the covariance
        x_bar, y_bar, theta_bar = self.f(wheel_velocity, dt)

        linear_velocity = 1/2 * self.radius_wheel * (wheel_velocity[0, 0] + wheel_velocity[1, 0])
        angular_velocity = (self.radius_wheel / self.base_length) * (-wheel_velocity[0, 0] + wheel_velocity[1, 0])

        Jfx_rel = self.Jfx_rel(wheel_velocity, dt)
        Jfw_rel = self.Jfw_rel(wheel_velocity, dt)

        dx_rel = cos(self.rel_disp[2, 0]) * linear_velocity * dt
        dy_rel = sin(self.rel_disp[2, 0]) * linear_velocity * dt
        dtheta_rel = angular_velocity * dt

        self.rel_disp += np.array([[dx_rel], [dy_rel], [dtheta_rel]])

        Jfx = self.Jfx(wheel_velocity, dt)
        Jfw = self.Jfw(wheel_velocity, dt)

        if self.mode == "sim":
            Q_enc = self.covariance_wheel_encoder
        else:
            Q_enc = self.covariance_wheel_encoder

        Pk_bar = Jfx @ self.Pk @ Jfx.T + Jfw @ Q_enc @ Jfw.T
        xk_bar = np.array([x_bar, y_bar, theta_bar]).reshape(3,1)

        self.rel_cov = Jfx_rel @ self.rel_cov @ Jfx_rel.T + Jfw_rel @ Q_enc @ Jfw_rel.T

        return xk_bar, Pk_bar  

    # Update function
    def Update(self, xk_bar, Pk_bar):

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
            self.graph.add(gtsam.PoseRotationPrior2D(pose_key_curr, gtsam.Rot2(zk[0,0]), gtsam.noiseModel.Isotropic.Sigma(1, sigma_heading)))
            self.k += 1
        
        self.initial.insert(pose_key_curr, gtsam.Pose2(xk_bar[0,0], xk_bar[1,0], xk_bar[2,0]))

        if self.lidar_update_flag:
            for j in range(len(self.H)):
                if self.H[j] is not None and self.H[j] >= 0:
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
            Pk_body = Pk_full[0:3, 0:3]
            c, s = cos(pose.theta()), sin(pose.theta())
            J = np.array([[c, -s, 0],
                          [s,  c, 0],
                          [0,  0, 1]])
            Pk = J @ Pk_body @ J.T

            
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

        # Anchor the next BetweenFactor on the keyframe we just produced,
        # otherwise every odometry edge is computed relative to X0 and the
        # graph diverges.
        self.prev_pose = [float(xk[0, 0]), float(xk[1, 0]), float(xk[2, 0])]
        self.prev_cov = Pk

        return xk, Pk

    
    def joint_state_callback(self, msg):
        
        # Only get msg of two wheel encoders not from arm joints
        if self.mode == 'real':
            position = msg.position
            if len(position) != 2:
                return

        # In simulation have to fall the arm back to avoid collision
        if self.mode == 'sim':
            arm_control = Float64MultiArray()
            arm_control.data = [0.0, 0.0, -1.0, 0.0]
            self.arm_controller_pub.publish(arm_control)
        if not self.intialize_theta:
            return
        
        current_time = rclpy.time.Time.from_msg(msg.header.stamp)

        if self.first_time:
            self.first_time = False
            self.last_time = current_time
            self.prev_pose = [0, 0, self.theta]
            self.prev_cov = self.Pk
            PriorNoise = gtsam.noiseModel.Diagonal.Sigmas(1e-7*np.ones(3))
            pose_key = gtsam.symbol('X', self.i)
            self.graph.add(gtsam.PriorFactorPose2(pose_key, gtsam.Pose2(0, 0, self.theta), PriorNoise))
            self.initial = gtsam.Values()
            self.initial.insert(pose_key, gtsam.Pose2(0, 0, self.theta))
            return

        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        if dt <=0:
            return
        
        self.time_accumulate += dt
        
        if self.mode == 'sim':
            left_wheel_velocity = msg.velocity[0]
            right_wheel_velocity = msg.velocity[1]
        else:
            velocity = msg.velocity
            if len(velocity) == 2:
                left_wheel_velocity = msg.velocity[0]
                right_wheel_velocity = msg.velocity[1]


        self.linear_velocity = 0.5 * self.radius_wheel * (left_wheel_velocity + right_wheel_velocity)
        self.angular_velocity = (self.radius_wheel / self.base_length) * (-left_wheel_velocity + right_wheel_velocity)

        current_time_in_sec = current_time.nanoseconds / 1e9

        # Adding noise to the wheel encoder sensor
        wheel_velocity = np.array([[left_wheel_velocity], [right_wheel_velocity]])

        # Try to synchronize the latest IMU and LiDAR data with the current joint state data
        if self.imu_update_flag and len(self.imu_time_buffer) > 0:
            while len(self.imu_time_buffer) > 0 and abs(self.imu_time_buffer[0] - current_time_in_sec) > self.sync_time_thrsh_imu:
                self.imu_time_buffer.pop(0)
                self.imu_buffer.pop(0)
            if len(self.imu_buffer) > 0:
                # Use newest (closest to current time) surviving measurement
                self.imu_orientation = self.imu_buffer[-1][0]
                self.imu_covariance = self.imu_buffer[-1][1]
            else:
                self.imu_update_flag = False

        lidar_timestamp = current_time_in_sec
        if self.lidar_update_flag and len(self.line_time_buffer) > 0:
            while len(self.line_time_buffer) > 0 and abs(self.line_time_buffer[0] - current_time_in_sec) > self.sync_time_thrsh_line:
                # self.get_logger().debug(f"Time_delay lidar: {abs(self.line_time_buffer[0] - current_time_in_sec)}")
                self.line_time_buffer.pop(0)
                self.line_buffer.pop(0)
            if len(self.line_buffer) > 0:
                # Use newest (closest to current time) surviving measurement
                self.polar_coordinates = self.line_buffer[-1][0]
                self.polar_covariances = self.line_buffer[-1][1]
                self.line_segments = self.line_buffer[-1][2]
                lidar_timestamp = self.line_time_buffer[-1]
            else:
                self.lidar_update_flag = False

        # Predict pose of robot with covariance
        xk_bar, Pk_bar = self.Prediction(wheel_velocity, dt)

        # Data association
        if self.lidar_update_flag:
            self.data_association = DataAssociation(
                self.confidence_level,
                self.line_feature_map,
                self.line_feature_cov_map,
                self.line_segments_map,
                duplicate_rho_threshold=self.duplicate_rho_threshold,
                duplicate_alpha_threshold=self.duplicate_alpha_threshold,
                measurement_sigma_rho_floor=self.measurement_sigma_rho_floor,
                measurement_sigma_alpha_floor=self.measurement_sigma_alpha_floor,
            )
            self.H = self.data_association.DataAssociation(xk_bar, Pk_bar, self.polar_coordinates, self.polar_covariances)

        # Check if robot moving or is translating larger than some threshold to update the graph
        trans_delta = np.linalg.norm(self.rel_disp[0:2, 0])
        rot_delta = abs(self.rel_disp[2, 0])
        should_add_key_frame = (
            trans_delta > self.keyframe_trans_threshold
            or rot_delta > self.keyframe_rot_threshold
            or self.time_accumulate > self.keyframe_time_threshold
        )

        # Update function 
        if should_add_key_frame and (self.imu_update_flag == True or self.lidar_update_flag == True):
            xk, Pk = self.Update(xk_bar, Pk_bar)
            self.x = xk[0, 0]
            self.y = xk[1, 0]
            self.theta = xk[2, 0]
            self.Pk = Pk
            self.imu_update_flag = False
            self.time_accumulate = 0.0
        else:
            self.x = xk_bar[0, 0]
            self.y = xk_bar[1, 0]
            self.theta = xk_bar[2, 0]
            self.Pk = Pk_bar
        
        # Add new feature to the map (via the persistency tracker)
        if self.lidar_update_flag and should_add_key_frame:
            # Grow any already-mapped lines that this scan extends past their
            # current visualized endpoints.
            self._extend_associated_line_segments()

            unsociated_features, unsociated_features_cov, unsociated_line_segments = self.data_association.GetUnassociatedFeatures(self.line_segments, self.polar_coordinates, self.polar_covariances, self.H)
            xk = np.array([[self.x], [self.y], [self.theta]])

            # Funnel unassociated observations into the candidate pool. Only
            # candidates that have been seen >= feature_min_observations times
            # get promoted to actual map landmarks this frame.
            (promoted_world_polar,
             promoted_world_cov,
             promoted_line_world,
             promoted_last_obs,
             promoted_last_obs_cov) = self.feature_candidates.update(
                xk, self.Pk,
                unsociated_features, unsociated_features_cov, unsociated_line_segments,
            )
            self.new_feature = len(promoted_world_polar)
            if self.new_feature > 0:
                # Append directly using the tracker's fused world-frame estimates
                # rather than the single-shot projection in AddmultipleNewFeatures.
                self.line_feature_map.extend(promoted_world_polar)
                self.line_feature_cov_map.extend(promoted_world_cov)
                self.line_segments_map.extend(promoted_line_world)
                # Keep the DA instance's lists in sync for the next frame.
                self.data_association.map_feature = self.line_feature_map
                self.data_association.map_feature_cov = self.line_feature_cov_map
                self.data_association.map_line_segments = self.line_segments_map

                pose_key_curr = gtsam.symbol('X', self.i)
                self.AddNewFeature(pose_key_curr, promoted_last_obs, promoted_last_obs_cov)
            self.lidar_update_flag = False

        self.visualize_map()
        self.visualize_boundary_error_lines_map()
        self.visualize_associations()
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

        # Broadcast TF.

        if self.odom_frame is None:
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
        else:
            self._broadcast_map_to_odom_correction(current_time)

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

        for i, line in enumerate(self.line_segments_map):
            start_point, end_point = line[0], line[1]

            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "line_features"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0

            marker.points.append(
                Point(x=float(start_point[0]), y=float(start_point[1]), z=0.0)
            )
            marker.points.append(
                Point(x=float(end_point[0]), y=float(end_point[1]), z=0.0)
            )

            marker_array.markers.append(marker)

        self.visualize_map_pub.publish(marker_array)

    def visualize_associations(self):
        # For each observation associated to a map line: draw the observed
        # line transformed into the world frame, a connector to the associated
        # map line, and a text label with the line's world-frame polar
        # coordinates (rho, alpha).
        if self.line_segments is None or not self.H:
            return

        marker_array = MarkerArray()

        clear = Marker()
        clear.header.frame_id = self.world_frame
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.ns = "association"
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        c, s = cos(self.theta), sin(self.theta)
        marker_id = 1
        stamp = self.get_clock().now().to_msg()

        def make_line(points, width):
            nonlocal marker_id
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = stamp
            m.ns = "association"
            m.id = marker_id; marker_id += 1
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = width
            m.color.a = 1.0
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 1.0
            for p in points:
                m.points.append(Point(x=float(p[0]), y=float(p[1]), z=0.05))
            return m

        def make_text(text, position):
            nonlocal marker_id
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = stamp
            m.ns = "association"
            m.id = marker_id; marker_id += 1
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.scale.z = 0.12
            m.color.a = 1.0
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 1.0
            m.pose.position.x = float(position[0])
            m.pose.position.y = float(position[1])
            m.pose.position.z = 0.25
            m.pose.orientation.w = 1.0
            m.text = text
            return m

        for j, line in enumerate(self.line_segments):
            if j >= len(self.H):
                break
            map_idx = self.H[j]
            if map_idx is None or map_idx < 0 or map_idx >= len(self.line_segments_map):
                continue

            start_local, end_local = line[0], line[1]
            sx_w = c * start_local[0] - s * start_local[1] + self.x
            sy_w = s * start_local[0] + c * start_local[1] + self.y
            ex_w = c * end_local[0] - s * end_local[1] + self.x
            ey_w = s * end_local[0] + c * end_local[1] + self.y

            map_start, map_end = self.line_segments_map[map_idx]

            # Observed line (transformed into world frame).
            marker_array.markers.append(make_line([(sx_w, sy_w), (ex_w, ey_w)], 0.05))

            # Connector from observed-line midpoint to map-line midpoint.
            obs_mid = (0.5 * (sx_w + ex_w), 0.5 * (sy_w + ey_w))
            map_mid = (0.5 * (map_start[0] + map_end[0]), 0.5 * (map_start[1] + map_end[1]))
            marker_array.markers.append(make_line([obs_mid, map_mid], 0.02))

            # Polar position of the observed line in the world frame.
            rho_obs, alpha_obs = self.polar_coordinates[j]
            alpha_w = warp_angle(alpha_obs + self.theta)
            rho_w = rho_obs + cos(alpha_w) * self.x + sin(alpha_w) * self.y

            # obs_text = (
            #     f"obs[{j}] -> L{map_idx}\n"
            #     f"base:  rho={rho_obs:+.2f} alpha={alpha_obs:+.1f}rad\n"
            # )
            # marker_array.markers.append(make_text(obs_text, obs_mid))

            rho_map, alpha_map = self.line_feature_map[map_idx]
            map_text = f"L{map_idx}\nrho={rho_map:+.2f} alpha={math.degrees(alpha_map):+.1f}deg"
            marker_array.markers.append(make_text(map_text, map_mid))

        self.visualize_association_pub.publish(marker_array)

    def _extend_associated_line_segments(self):
        # For each observation associated to an existing map line, grow that
        # line's stored endpoints if the observation reaches farther along the
        # line direction. The (rho, alpha) parameters are untouched — only the
        # visualized [start, end] segment is widened, never shrunk.
        if self.line_segments is None or not self.H:
            return

        c, s = cos(self.theta), sin(self.theta)
        for j, line in enumerate(self.line_segments):
            if j >= len(self.H):
                break
            map_idx = self.H[j]
            if map_idx is None or map_idx < 0 or map_idx >= len(self.line_segments_map):
                continue

            sx_l, sy_l = line[0]
            ex_l, ey_l = line[1]
            obs_start_w = np.array([c * sx_l - s * sy_l + self.x,
                                    s * sx_l + c * sy_l + self.y])
            obs_end_w   = np.array([c * ex_l - s * ey_l + self.x,
                                    s * ex_l + c * ey_l + self.y])

            rho, alpha = self.line_feature_map[map_idx]
            tangent = np.array([-sin(alpha), cos(alpha)])
            anchor  = rho * np.array([cos(alpha), sin(alpha)])

            map_start = np.array(self.line_segments_map[map_idx][0], dtype=float)
            map_end   = np.array(self.line_segments_map[map_idx][1], dtype=float)
            projections = [
                float(np.dot(p - anchor, tangent))
                for p in (map_start, map_end, obs_start_w, obs_end_w)
            ]
            s_min, s_max = min(projections), max(projections)

            new_start = anchor + s_min * tangent
            new_end   = anchor + s_max * tangent
            self.line_segments_map[map_idx] = [new_start.tolist(), new_end.tolist()]

        if self.data_association is not None:
            self.data_association.map_line_segments = self.line_segments_map

    def _regularize_covariance(self, cov: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        cov_sym = 0.5 * (cov + cov.T)
        return cov_sym + np.eye(2) * eps
    
    def _curve_boundary_points(self, line: list, cov_line: np.ndarray, num_samples: int = 100) -> tuple:
        start_point = np.array(line[0], dtype=float)
        end_point = np.array(line[1], dtype=float)
        line_vec = end_point - start_point
        line_len = np.linalg.norm(line_vec)
        if line_len <= 1e-9:
            return [], []
        samples_per_meter = 80.0
        min_samples = 40
        max_samples = 400
        num_samples = int(np.clip(math.ceil(samples_per_meter * line_len), min_samples, max_samples))

        tangent = line_vec / line_len
        # Match the line normal convention used by fit_line / atan2(b, a).
        normal = np.array([tangent[1], -tangent[0]])
        theta = math.atan2(normal[1], normal[0])

        cov_reg = self._regularize_covariance(cov_line)
        samples_s = np.linspace(0.0, line_len, num_samples)
        boundary_plus = []
        boundary_minus = []
        for s in samples_s:
            point_nominal = start_point + s * tangent
            jacobian = np.array([1.0, point_nominal[0] * math.sin(theta) - point_nominal[1] * math.cos(theta)])
            variance_dist = float(jacobian @ cov_reg @ jacobian.T)
            variance_dist = max(variance_dist, 0.0)
            confidence_dist = math.sqrt(5.991464547107979 * variance_dist)
            boundary_plus.append(point_nominal + confidence_dist * normal)
            boundary_minus.append(point_nominal - confidence_dist * normal)

        return boundary_plus, boundary_minus
    
    def visualize_boundary_error_lines_map(self):
        """
        Visualize line segments in RVIZ2 using MarkerArray
        """

        lines_error_visualization = MarkerArray()
        for i, line in enumerate(self.line_segments_map):
            start_point, end_point = line[0], line[1]
            cov_line = self.line_feature_cov_map[i]
            boundary_plus, boundary_minus = self._curve_boundary_points([start_point, end_point], cov_line)
            if len(boundary_plus) > 1 and len(boundary_minus) > 1:
                marker = Marker()
                marker.header.frame_id = self.world_frame
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "line_uncertainty_curve"
                marker.id = 3000 * i
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.scale.x = 0.02
                marker.color.a = 0.85
                marker.color.r = 1.0
                marker.color.g = 0.85
                marker.color.b = 0.0
                marker.points = [Point(x=float(p[0]), y=float(p[1]), z=0.0) for p in boundary_plus]
                lines_error_visualization.markers.append(marker)
                marker = Marker()
                marker.header.frame_id = self.world_frame
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "line_uncertainty_curve"
                marker.id = 3000 * i + 1
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.scale.x = 0.02
                marker.color.a = 0.85
                marker.color.r = 1.0
                marker.color.g = 0.85
                marker.color.b = 0.0
                marker.points = [Point(x=float(p[0]), y=float(p[1]), z=0.0) for p in boundary_minus]
                lines_error_visualization.markers.append(marker)
        
        self.line_error_pub_.publish(lines_error_visualization)


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