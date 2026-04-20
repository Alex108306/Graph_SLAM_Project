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
                    'kobuki_basic.launch.py'
                ])
            ]),
            launch_arguments = {
                'scenario_description' : PathJoinSubstitution([FindPackageShare('turtlebot_simulation'), 'scenarios', 'pr_world.scn'])
            }.items()
        )
    ])

# <node pkg="kobuki_node" type="beacons2marker.py" name="beacons2_marker" output="screen"/>
# <node pkg="tf" type="static_transform_publisher" name="beacons2_marker_tf" args="0 0 0 0 0 0 world world_ned 100"/>