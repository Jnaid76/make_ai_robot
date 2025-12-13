#!/usr/bin/env python3
"""
ROS2 launch file for navigating to a goal position.

This launch file accepts goal coordinates as parameters and starts the navigate_to_goal node.

Usage:
    ros2 launch language_command_handler navigate_to_goal.launch.py goal_x:=2.0 goal_y:=1.0 goal_yaw:=0.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    goal_x_arg = DeclareLaunchArgument(
        'goal_x',
        default_value='0.0',
        description='Goal x position in meters'
    )
    
    goal_y_arg = DeclareLaunchArgument(
        'goal_y',
        default_value='0.0',
        description='Goal y position in meters'
    )
    
    goal_yaw_arg = DeclareLaunchArgument(
        'goal_yaw',
        default_value='0.0',
        description='Goal yaw orientation in radians'
    )

    use_astar_arg = DeclareLaunchArgument(
        'use_astar',
        default_value='true',
        description='Use A* path planning (requires map)'
    )

    # Create node with parameters
    navigate_to_goal_node = Node(
        package='language_command_handler',
        executable='navigate_to_goal.py',
        name='navigate_to_goal',
        output='screen',
        parameters=[{
            'goal_x': LaunchConfiguration('goal_x'),
            'goal_y': LaunchConfiguration('goal_y'),
            'goal_yaw': LaunchConfiguration('goal_yaw'),
            'use_astar': LaunchConfiguration('use_astar'),
        }]
    )
    
    ld = LaunchDescription()
    ld.add_action(goal_x_arg)
    ld.add_action(goal_y_arg)
    ld.add_action(goal_yaw_arg)
    ld.add_action(use_astar_arg)
    ld.add_action(navigate_to_goal_node)
    
    return ld

