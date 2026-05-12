import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

import numpy as np
import math

from scipy.linalg import block_diag
from .utils import warp_angle

class SplitAndMerge:
    def __init__(self, distance_threshold: float, lidar_flip_pi: bool = True, decrement_angle: bool = True):
        self.distance_threshold = distance_threshold
        self.lidar_flip_pi = lidar_flip_pi
        self.decrement_angle = decrement_angle
    
    def transform_lidar_to_cartesian(self, lidar_msg):
        """
        Transform lidar from Polar cooridnates to Catersian coordinates

        Args: 
            lidar_msg: lidar message received from /turtlebot/scan topic
        
        Return:
            List of array points in 2D Catersian coordinate (x,y)
        """
        lidar_points = []
        lidar_range = lidar_msg.ranges
        angle_increment = lidar_msg.angle_increment
        if self.lidar_flip_pi:
            angle_min = lidar_msg.angle_min + math.pi  # Rotating 180 degree to align with robot local frame.
        else:
            angle_min = lidar_msg.angle_min
        current_angle = angle_min
        for i in range(len(lidar_range)):
            if math.isinf(lidar_range[i]):
                if self.decrement_angle:
                    current_angle -= angle_increment
                else:
                    current_angle += angle_increment
                continue
            x_point = np.cos(current_angle) * lidar_range[i]
            y_point = np.sin(current_angle) * lidar_range[i]
            if self.decrement_angle:
                current_angle -= angle_increment
            else:
                current_angle += angle_increment
            lidar_points.append(np.array([x_point, y_point]))
        
        return lidar_points
    
    def fit_line(self, point1: np.array, point2: np.array) -> tuple:
        """
        Fit a line between two points

        Args:
            point1, point2: Two points to fit a line between in 2D Catersian coordinate (x,y)
        
        Return:
            Line parameters (a, b, c) for the line equation ax + by + c = 0
        """
            
        a = (point2[1] - point1[1])
        b = (point1[0] - point2[0])
        c = (point1[1] * point2[0] - point1[0] * point2[1])
        
        return a, b, c
    
    def distance_from_line(self, point: np.array, line_params: tuple) -> float:
        """
        Calculate the distance from a point to a line
        
        Args:
            point: A point in 2D Catersian coordinate (x, y)
            line_params: Line parameters (a, b, c) for the line equation ax + by + c = 0
        
        Return:
            Distance from the point to the line
        """

        a, b, c = line_params
        distance = abs(a * point[0] + b * point[1] + c) / math.sqrt(a**2 + b**2)
        return distance

    def split(self, list_of_points: list, line_segments: list) -> list:
        """
        Split the lines from the list of points using Split and Merge algorithm

        Args:
            list_of_points: List of 2D points Lidar in Catersian coordinate with respect to the robot local frame
        
        Return:
            List of line segments, where each line segment is represented by a tuple of two points (start_point, end_point)
        """

        # If the number of points is less than or equal to 15, we consider it is not a line
        if len(list_of_points) <= 15:
            return line_segments

        start_point = list_of_points[0]
        end_point = list_of_points[-1]

        line_params = self.fit_line(start_point, end_point)
        max_distance = 0.0
        key_furtherest = 0

        # Iterate split the line segment until the distance from the point to the line is less than the threshold
        for i, point in enumerate(list_of_points):
            distance = self.distance_from_line(point, line_params)
            if distance > max_distance:
                max_distance = distance
                key_furtherest = i
        
        if max_distance >= self.distance_threshold:
            left_list_points = list_of_points[0:key_furtherest+1]
            right_list_points = list_of_points[key_furtherest:]
            line_segments = self.split(left_list_points, line_segments)
            line_segments = self.split(right_list_points, line_segments)
        else:
            line_segment = [start_point, end_point]
            line_segments.append(line_segment)
        
        return line_segments

    def merge(self, line_segments: list) -> list:
        """
        Merge the line segments that are approximately collinear

        Args:
            line_segments: List of line segments, where each line segment is
            represented by [start_point, end_point]

        Return:
            List of merged line segments
        """

        angle_approve_merge = 0.087  # 5 degrees in radians
        distance_threshold_between_lines = 0.05  # 5 cm

        def can_merge(line_1, line_2):
            line_1_params = self.fit_line(line_1[0], line_1[1])
            line_2_params = self.fit_line(line_2[0], line_2[1])

            distance_line_1 = self.distance_from_line([0.0, 0.0], line_1_params)
            distance_line_2 = self.distance_from_line([0.0, 0.0], line_2_params)
            diff_distance = abs(distance_line_1 - distance_line_2)
            if diff_distance >= distance_threshold_between_lines:
                return False

            angle_1 = math.atan2(line_1_params[1], line_1_params[0])
            angle_2 = math.atan2(line_2_params[1], line_2_params[0])
            angle_diff = abs(math.atan2(math.sin(angle_1 - angle_2), math.cos(angle_1 - angle_2)))

            return angle_diff < angle_approve_merge

        if len(line_segments) <= 1:
            return line_segments

        current_segments = list(line_segments)

        while True:
            changed = False
            merged_segments = []
            i = 0

            while i < len(current_segments):
                if i < len(current_segments) - 1 and can_merge(current_segments[i], current_segments[i + 1]):
                    merged_segments.append([current_segments[i][0], current_segments[i + 1][1]])
                    changed = True
                    i += 2
                else:
                    merged_segments.append(current_segments[i])
                    i += 1

            # Optional closed-loop merge: merge last and first if they are compatible
            if len(merged_segments) > 1 and can_merge(merged_segments[-1], merged_segments[0]):
                merged_segments[0] = [merged_segments[-1][0], merged_segments[0][1]]
                merged_segments.pop()
                changed = True

            current_segments = merged_segments

            if not changed:
                break

        return current_segments
        
    def calculate_polar_coordinates(self, line_segments: list) -> list:
        """
        Calculate the polar coordinates (distance and angle) of the line segments with respect to the robot local frame
        Args:
            line_segments: List of line segments, where each line segment is represented by a tuple of two points (start_point, end_point)
        Return:
            List of polar coordinates (distance, angle) of the line segments with respect to the robot local frame
        """
        
        polar_coordinates = []

        for i, line in enumerate(line_segments):
            line_params = self.fit_line(line[0], line[1])
            distance = self.distance_from_line([0.0, 0.0], line_params)
            angle = warp_angle(math.atan2(line_params[1], line_params[0]) - np.pi)
            polar_coordinates.append([distance, angle])

        return polar_coordinates

    def calculate_covariance_matrix(self, line: list) -> np.ndarray:
        """
        Calculate the covariance matrix of the line segments with respect to the robot local frame
        Args:
            line: A line segment represented by a tuple of two points (start_point, end_point)
        Return:
            Covariance matrix of the line segment with respect to the robot local frame
        """
        sigma_r = 0.07 # 30 cm
        cov_max_2D_point = np.array([[sigma_r**2, 0.0], [0.0, sigma_r**2]])

        line_params = self.fit_line(line[0], line[1])
        a, b, c = line_params
        dr_da = -abs(c) * a / math.sqrt((a**2 + b**2)**3)
        dr_db = -abs(c) * b / math.sqrt((a**2 + b**2)**3)
        dr_dc = math.copysign(1.0, c) / math.sqrt(a**2 + b**2)
        dtheta_da = -b / (a**2 + b**2)
        dtheta_db = a / (a**2 + b**2)
        da_dy1 = -1
        da_dy2 = 1
        db_dx1 = 1
        db_dx2 = -1
        dc_dx1 = -line[1][1]
        dc_dy1 = line[1][0]
        dc_dx2 = line[0][1]
        dc_dy2 = -line[0][0]
        J = np.array([
            [dr_dc * dc_dx1 + dr_db * db_dx1, dr_dc * dc_dy1 + dr_da * da_dy1, dr_dc * dc_dx2 + dr_db * db_dx2, dr_dc * dc_dy2 + dr_da * da_dy2],
            [dtheta_db * db_dx1, dtheta_da * da_dy1, dtheta_db * db_dx2, dtheta_da * da_dy2]
        ])
        cov_line = J @ block_diag(cov_max_2D_point, cov_max_2D_point) @ J.T

        return cov_line


class LineExtractionNode(Node):
    def __init__(self):
        super().__init__('extract_line')

        self.declare_parameter('mode', 'sim')
        self.mode = self.get_parameter('mode').get_parameter_value().string_value.lower()
        if self.mode not in ['sim', 'real']:
            self.get_logger().warn(f"Unknown mode '{self.mode}', fallback to 'sim'.")
            self.mode = 'sim'

        if self.mode == 'sim':
            self.lidar_coordinate = "turtlebot/base_footprint"
            self.lidar_flip_pi = True
            self.lidar_decrement_angle = True
        else:
            self.lidar_coordinate = "base_footprint"
            self.lidar_flip_pi = False
            self.lidar_decrement_angle = False

        self.laser_sub_ = self.create_subscription(LaserScan, '/turtlebot/scan', self.receive_lidar_scan, 20)
        self.marker_array_pub_ = self.create_publisher(MarkerArray, '/turtlebot/scan_points', 20)
        self.line_array_pub_ = self.create_publisher(MarkerArray, '/turtlebot/line_segments', 20)
        self.create_timer(0.5, self.visualize_lidar_points)
        self.create_timer(0.5, self.visualize_line_segments)
        self.lidar_points = None
    
    def transform_lidar_to_cartesian(self, lidar_msg):
        """
        Transform lidar from Polar cooridnates to Catersian coordinates

        Args: 
            lidar_msg: lidar message received from /turtlebot/scan topic
        
        Return:
            List of array points in 2D Catersian coordinate (x,y)
        """
        lidar_points = []
        lidar_range = lidar_msg.ranges
        angle_increment = lidar_msg.angle_increment
        if self.lidar_flip_pi:
            angle_min = lidar_msg.angle_min + math.pi  # Rotating 180 degree to align with robot local frame.
        else:
            angle_min = lidar_msg.angle_min
        current_angle = angle_min
        for i in range(len(lidar_range)):
            if math.isinf(lidar_range[i]):
                if self.lidar_decrement_angle:
                    current_angle -= angle_increment
                else:
                    current_angle += angle_increment
                continue
            x_point = np.cos(current_angle) * lidar_range[i]
            y_point = np.sin(current_angle) * lidar_range[i]
            if self.lidar_decrement_angle:
                current_angle -= angle_increment
            else:
                current_angle += angle_increment
            lidar_points.append(np.array([x_point, y_point]))
        
        return lidar_points
    
    def receive_lidar_scan(self, msg):
        """
        Recieve lidar message from /turtlebot/scan topic

        Args:
            lidar_msg: lidar message received from /turtlebot/scan topic
        Return:
            List of array points in 2D Catersian coordinate (x,y)
        """
        lidar_msg = msg
        self.lidar_points = self.transform_lidar_to_cartesian(lidar_msg)

    def visualize_lidar_points(self):
        """
        Visualize lidar points in RVIZ2 using MarkerArray
        """

        if self.lidar_points == None:
            return
        
        marker_array = MarkerArray()
        for i, point in enumerate(self.lidar_points):
            marker = Marker()
            marker.header.frame_id = self.lidar_coordinate
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "lidar_points"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
        
        self.marker_array_pub_.publish(marker_array)
    
    def visualize_line_segments(self):
        """
        Visualize line segments in RVIZ2 using MarkerArray
        """

        if self.lidar_points == None:
            return

        # Extract line segments from lidar points using Split and Merge algorithm
        split_and_merge = SplitAndMerge(0.01, self.lidar_flip_pi, self.lidar_decrement_angle)
        line_segments = []
        line_segments = split_and_merge.split(self.lidar_points, line_segments)
        line_segments = split_and_merge.merge(line_segments)

        polar_coordinates = split_and_merge.calculate_polar_coordinates(line_segments)
        angle_list = [polar[1] for polar in polar_coordinates]
        self.get_logger().info(f"{angle_list}")
        
        lines_visualization = MarkerArray()
        for i, line in enumerate(line_segments):
            marker = Marker()
            marker.header.frame_id = self.lidar_coordinate
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "line_segments"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.points = []
            start_point = line[0]
            end_point = line[1]
            marker.points.append(Point(x=start_point[0], y=start_point[1], z=0.0))
            marker.points.append(Point(x=end_point[0], y=end_point[1], z=0.0))
            marker.scale.x = 0.05
            marker.color.a = 1.0
            marker.color.r = i / len(line_segments)
            marker.color.g = 1.0 - i / len(line_segments)
            marker.color.b = i / len(line_segments)
            lines_visualization.markers.append(marker)
        
            cov_line = split_and_merge.calculate_covariance_matrix([start_point, end_point])
            sigma_r = math.sqrt(cov_line[0, 0])
            # self.get_logger().info(f"Line {i}: sigma_r = {sigma_r}, sigma_theta = {math.sqrt(cov_line[1, 1])}")
            sigma_theta = math.sqrt(cov_line[1, 1])

            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            L = math.hypot(dx, dy)
            if L > 0:
                direction = np.array([dx, dy]) / L
                normal = np.array([-direction[1], direction[0]])

                for sign in [+1.0, -1.0]:
                    marker = Marker()
                    marker.header.frame_id = self.lidar_coordinate
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = "line_uncertainty_shift"
                    marker.id = 1000 * i + (0 if sign > 0 else 1)
                    marker.type = Marker.LINE_STRIP
                    marker.action = Marker.ADD
                    marker.scale.x = 0.02
                    marker.color.a = 0.5
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    offset = normal * sigma_r * sign
                    marker.points.append(Point(x=start_point[0] + offset[0], y=start_point[1] + offset[1], z=0.0))
                    marker.points.append(Point(x=end_point[0] + offset[0], y=end_point[1] + offset[1], z=0.0))
                    lines_visualization.markers.append(marker)

                mid = (start_point + end_point) / 2.0
                theta = math.atan2(dy, dx)
                for sign in [+1.0, -1.0]:
                    theta_offset = theta + sign * sigma_theta
                    dir_offset = np.array([math.cos(theta_offset), math.sin(theta_offset)])
                    half = 0.5 * L * dir_offset
                    q1 = mid - half
                    q2 = mid + half

                    marker = Marker()
                    marker.header.frame_id = self.lidar_coordinate
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = "line_uncertainty_angle"
                    marker.id = 2000 * i + (0 if sign > 0 else 1)
                    marker.type = Marker.LINE_STRIP
                    marker.action = Marker.ADD
                    marker.scale.x = 0.02
                    marker.color.a = 0.5
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0
                    marker.points.append(Point(x=q1[0], y=q1[1], z=0.0))
                    marker.points.append(Point(x=q2[0], y=q2[1], z=0.0))
                    lines_visualization.markers.append(marker)
        
        self.line_array_pub_.publish(lines_visualization)

def main(args=None):
    
    # Initialize ROS 2
    rclpy.init(args=args)
    line_extraction_node = LineExtractionNode()

    # Spin the node to keep it active and responsive to callbacks
    rclpy.spin(line_extraction_node)

    # Clean up and shutdown ROS 2
    line_extraction_node.destroy_node()
    rclpy.shutdown()
