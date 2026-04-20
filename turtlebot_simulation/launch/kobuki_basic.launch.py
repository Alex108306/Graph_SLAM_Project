import math
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 1. Locate packages
    pkg_turtlebot_sim = FindPackageShare('turtlebot_simulation')
    pkg_kobuki_desc = FindPackageShare('kobuki_description')

    # 2. Declare Arguments
    declare_robot_name = DeclareLaunchArgument('robot_name', default_value='turtlebot')
    declare_sim_data = DeclareLaunchArgument('simulation_data', 
        default_value=PathJoinSubstitution([pkg_turtlebot_sim, 'resources']))
    declare_scenario = DeclareLaunchArgument('scenario_description', 
        default_value=PathJoinSubstitution([pkg_turtlebot_sim, 'scenarios', 'kobuki_basic.scn']))
    declare_sim_rate = DeclareLaunchArgument('simulation_rate', default_value='1000.0')
    declare_window_res_x = DeclareLaunchArgument('window_resolution_x', default_value='1200')
    declare_window_res_y = DeclareLaunchArgument('window_resolution_y', default_value='800')
    declare_quality = DeclareLaunchArgument('rendering_quality', default_value='high')

    # 3. Process Xacro
    robot_description_content = Command([
        'xacro ', 
        PathJoinSubstitution([pkg_kobuki_desc, 'urdf', 'kobuki_standalone.urdf.xacro'])
    ])
    robot_description = {'robot_description': robot_description_content}

    # 4. Group actions for Namespacing
    robot_namespace_group = GroupAction([
        PushRosNamespace(LaunchConfiguration('robot_name')),

        # Stonefish Simulator Node
        Node(
            package='stonefish_ros2',
            executable='stonefish_simulator',
            name='stonefish_simulator',
            output='screen',
            arguments=[
                LaunchConfiguration('simulation_data'),
                LaunchConfiguration('scenario_description'),
                LaunchConfiguration('simulation_rate'),
                LaunchConfiguration('window_resolution_x'),
                LaunchConfiguration('window_resolution_y'),
                LaunchConfiguration('rendering_quality')
            ],
            parameters=[{'robot_name': LaunchConfiguration('robot_name')}],
            remappings=[
                ('stonefish_simulator/joint_states', 'joint_states')
            ]
        ),

        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[robot_description]
        ),

        # Navigation 
        Node(
            package='turtlebot_simulation',
            executable='diff_drive_odometry.py',
            name='diff_drive_odometry',
            output='screen',
            parameters=[{
                'odom_frame': 'world_enu',
                'base_frame': PathJoinSubstitution([LaunchConfiguration('robot_name'), 'base_footprint']),
                'wheel_left_joint_name': PathJoinSubstitution([LaunchConfiguration('robot_name'), 'wheel_left_joint']),
                'wheel_right_joint_name': PathJoinSubstitution([LaunchConfiguration('robot_name'), 'wheel_right_joint'])
            }]
        ),

        # Control
        Node(
            package='turtlebot_simulation',
            executable='diff_drive_controller.py',
            name='diff_drive_controller',
            output='screen'
        ),

        # RVIZ
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='enu_to_ned_broadcaster',
            arguments=['0.0', '0.0', '0.0', str(math.pi/2), '0.0', str(math.pi), 'world_enu', 'world_ned']
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', PathJoinSubstitution([pkg_kobuki_desc, 'rviz', 'kobuki.rviz'])]
        )
    ])

    return LaunchDescription([
        declare_robot_name,
        declare_sim_data,
        declare_scenario,
        declare_sim_rate,
        declare_window_res_x,
        declare_window_res_y,
        declare_quality,
        robot_namespace_group
    ])