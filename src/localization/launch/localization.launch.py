#!/usr/bin/env python3

"""
Unified launch file for complete localization system.
Launches both global and odometry localizers together.

Usage:
    ros2 launch localization localization.launch.py
    ros2 launch localization localization.launch.py x:=5.0 y:=3.0 yaw:=1.57
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments for initial pose
    declare_x_cmd = DeclareLaunchArgument(
        'x',
        default_value='0.0',
        description='Initial x position in map frame (meters)'
    )

    declare_y_cmd = DeclareLaunchArgument(
        'y',
        default_value='1.0',
        description='Initial y position in map frame (meters)'
    )

    declare_yaw_cmd = DeclareLaunchArgument(
        'yaw',
        default_value='0.0',
        description='Initial yaw angle in map frame (radians)'
    )

    # Get launch configuration
    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')
    yaw = LaunchConfiguration('yaw')

    # Odometry localizer node (ICP-based scan matching)
    odom_localizer_node = Node(
        package='localization',
        executable='odom_localizer_node.py',
        name='odom_localizer_node',
        output='screen',
        emulate_tty=True
    )

    # Global localizer node (particle filter)
    global_localizer_node = Node(
        package='localization',
        executable='global_localizer_node.py',
        name='global_localizer_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'x': x,
            'y': y,
            'yaw': yaw
        }]
    )

    # Create launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_x_cmd)
    ld.add_action(declare_y_cmd)
    ld.add_action(declare_yaw_cmd)

    # Add nodes
    ld.add_action(odom_localizer_node)
    ld.add_action(global_localizer_node)

    return ld
