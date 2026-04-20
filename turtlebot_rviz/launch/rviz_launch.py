import launch
from launch import LaunchDescription
from launch.actions import OpaqueFunction
from launch_ros.actions import Node
import rclpy
from rclpy.node import Node as RclpyNode

def check_topics(context, *args, **kwargs):
    rclpy.init(args=None)
    node = RclpyNode("turtlebot_rviz_checker")
    existing_topics = [t[0] for t in node.get_topic_names_and_types()]
    node.destroy_node()
    rclpy.shutdown()

    remappings = []
    if '/turtlebot/tf' in existing_topics:
        remappings.append(('/tf', '/turtlebot/tf'))
    if '/turtlebot/tf_static' in existing_topics:
        remappings.append(('/tf_static', '/turtlebot/tf_static'))

    return [
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            remappings=remappings
        )
    ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=check_topics)
    ])
