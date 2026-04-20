from launch_ros.substitutions import FindPackageShare
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('turtlebot_simulation'),
                    'launch',
                    'turtlebot_basic.launch.py'
                ])
            ]),
            launch_arguments = {
                'scenario_description' : PathJoinSubstitution([FindPackageShare('turtlebot_simulation'), 'scenarios', 'turtlebot_hoi_circuit2.scn'])
            }.items()
        )
    ])