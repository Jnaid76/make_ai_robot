#!/usr/bin/env python3
"""
ROS2 launch file for Mission 3: Navigate to Colored Cone.

This launch file starts the navigate_to_cone node which:
1. Navigates to viewing position (x=1.0198, y=12.9420, yaw=1.57)
2. Detects colored cones using CV
3. Approaches the specified cone
4. Publishes "bark" to /robot_dog/speech when within 3 meters

Usage:
    ros2 launch language_command_handler navigate_to_cone.launch.py target_color:=red
    ros2 launch language_command_handler navigate_to_cone.launch.py target_color:=blue
    ros2 launch language_command_handler navigate_to_cone.launch.py target_color:=green
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    target_color_arg = DeclareLaunchArgument(
        'target_color',
        default_value='red',
        description='Target cone color (red, blue, or green)'
    )

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
    navigate_to_cone_node = Node(
        package='language_command_handler',
        executable='navigate_to_cone.py',
        name='navigate_to_cone',
        output='screen',
        parameters=[{
            'target_color': LaunchConfiguration('target_color'),
            'use_astar': LaunchConfiguration('use_astar'),
            'inflation_radius': LaunchConfiguration('inflation_radius'),
            'simplify_path': LaunchConfiguration('simplify_path'),
        }]
    )

    ld = LaunchDescription()
    ld.add_action(target_color_arg)
    ld.add_action(use_astar_arg)
    ld.add_action(inflation_radius_arg)
    ld.add_action(simplify_path_arg)
    ld.add_action(navigate_to_cone_node)

    return ld
