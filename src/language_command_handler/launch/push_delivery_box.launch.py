#!/usr/bin/env python3
"""
ROS2 launch file for Mission 4: Push Delivery Box to Goal Area.

This launch file starts the push_delivery_box node which:
1. Navigates to 3 detection positions sequentially
2. Waits 3s at each position for YOLO to detect delivery box
3. If box detected within 2m: pushes forward 3m
4. Mission success when box is pushed into goal area

Usage:
    ros2 launch language_command_handler push_delivery_box.launch.py
    ros2 launch language_command_handler push_delivery_box.launch.py inflation_radius:=0.4
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
        default_value='0.35',
        description='Robot radius for collision avoidance (meters)'
    )

    simplify_path_arg = DeclareLaunchArgument(
        'simplify_path',
        default_value='true',
        description='Simplify A* path to reduce waypoints'
    )

    # Create the mission node
    push_box_node = Node(
        package='language_command_handler',
        executable='push_delivery_box.py',
        name='push_delivery_box',
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
    ld.add_action(push_box_node)

    return ld
