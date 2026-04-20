from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='turtlebot',
        description='Name of the robot'
    )

    base_name_arg = DeclareLaunchArgument(
        'base_name',
        default_value='kobuki',
        description='Name of the base'
    )

    arm_name_arg = DeclareLaunchArgument(
        'arm_name',
        default_value='swiftpro',
        description='Name of the arm'
    )

    return LaunchDescription([
        robot_name_arg,
        base_name_arg,
        arm_name_arg
    ])
