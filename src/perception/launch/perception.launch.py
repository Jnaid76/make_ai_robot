from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='perception',
            executable='yolo_detector',
            name='yolo_detector',
            output='screen',
            parameters=[{
                'use_sim_time': False,
            }]
        )
    ])

