#!/usr/bin/env python3
"""
ROS2 launch file for Mission 1: Navigate to Toilet and Bark.

This launch file starts the navigate_to_toilet_and_bark node which:
1. Navigates to the toilet location (x=-7.22, y=-0.54, yaw=1.5708)
2. Monitors alignment with toilet
3. Publishes "bark" to /robot_dog/speech when conditions are met

Usage:
    ros2 launch language_command_handler navigate_to_toilet_and_bark.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    use_astar_arg = DeclareLaunchArgument(
        'use_astar',
        default_value='true',
        description='Use A* path planning (requires map)'
    )

    inflation_radius_arg = DeclareLaunchArgument(
        'inflation_radius',
        default_value='0.5',
        description='Robot radius for collision avoidance (meters)'
    )

    simplify_path_arg = DeclareLaunchArgument(
        'simplify_path',
        default_value='true',
        description='Simplify A* path to reduce waypoints'
    )

    # Create the mission node
    navigate_to_toilet_node = Node(
        package='language_command_handler',
        executable='navigate_to_toilet_and_bark.py',
        name='navigate_to_toilet_and_bark',
        output='screen',
        parameters=[{
            'use_astar': LaunchConfiguration('use_astar'),
            'inflation_radius': LaunchConfiguration('inflation_radius'),
            'simplify_path': LaunchConfiguration('simplify_path'),
        }]
    )

    ld = LaunchDescription()
    ld.add_action(use_astar_arg)
    ld.add_action(inflation_radius_arg)
    ld.add_action(simplify_path_arg)
    ld.add_action(navigate_to_toilet_node)

    return ld

