#!/usr/bin/env python3
"""
ROS2 launch file for Mission 2: Find and Identify Edible Food.

This launch file starts the find_edible_food node which:
1. Searches through 20 hospital rooms sequentially
2. Performs 360° scans to detect food objects
3. Prioritizes edible food (apple, banana, pizza) over rotten
4. Approaches detected food within 1m
5. Publishes "bark" to /robot_dog/speech when edible food is found

Usage:
    ros2 launch language_command_handler find_edible_food.launch.py
    ros2 launch language_command_handler find_edible_food.launch.py inflation_radius:=0.4
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
    find_edible_food_node = Node(
        package='language_command_handler',
        executable='find_edible_food.py',
        name='find_edible_food',
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
    ld.add_action(find_edible_food_node)

    return ld
