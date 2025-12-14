#!/usr/bin/env python3
"""
A* Path Planner ROS2 Node

This node subscribes to the map and robot pose, and plans collision-free paths
using the A* algorithm. It publishes planned paths to /local_path for the path tracker.

Usage:
    ros2 run path_tracker astar_path_planner_node.py --ros-args -p goal_x:=2.0 -p goal_y:=1.0
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path, OccupancyGrid
import numpy as np
from .astar_planner import AStarPlanner


class AStarPathPlannerNode(Node):
    """
    ROS2 node for A* path planning with collision avoidance.
    """
    
    def __init__(self):
        super().__init__('astar_path_planner')
        
        # Declare parameters
        self.declare_parameter('goal_x', 0.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_yaw', 0.0)
        self.declare_parameter('inflation_radius', 0.3)  # Robot radius in meters
        self.declare_parameter('simplify_path', True)
        
        # Get parameters
        self.goal_x = self.get_parameter('goal_x').get_parameter_value().double_value
        self.goal_y = self.get_parameter('goal_y').get_parameter_value().double_value
        self.goal_yaw = self.get_parameter('goal_yaw').get_parameter_value().double_value
        inflation_radius = self.get_parameter('inflation_radius').get_parameter_value().double_value
        self.simplify_path = self.get_parameter('simplify_path').get_parameter_value().bool_value
        
        # Initialize A* planner
        self.planner = AStarPlanner(inflation_radius=inflation_radius)
        
        # State
        self.map_received = False
        self.robot_pose = None
        self.path_planned = False
        
        # Subscribe to map
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        
        # Subscribe to robot pose
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/go1_pose',
            self.pose_callback,
            10
        )
        
        # Publish path
        self.path_pub = self.create_publisher(
            Path,
            '/local_path',
            10
        )
        
        self.get_logger().info('A* Path Planner Node initialized')
        self.get_logger().info(f'Goal: x={self.goal_x:.3f}, y={self.goal_y:.3f}, yaw={math.degrees(self.goal_yaw):.1f}°')
        self.get_logger().info('Waiting for map and robot pose...')
    
    def map_callback(self, msg: OccupancyGrid):
        """
        Callback for map updates.
        """
        if self.map_received:
            return  # Only process once
        
        self.get_logger().info('Map received')
        
        # Extract map metadata
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        width = msg.info.width
        height = msg.info.height
        
        # Convert map data to numpy array
        # OccupancyGrid data is stored row-major, starting from (0,0)
        map_data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        
        # Set map in planner
        map_metadata = {
            'resolution': resolution,
            'origin_x': origin_x,
            'origin_y': origin_y,
            'width': width,
            'height': height
        }
        
        self.planner.set_map(map_data, map_metadata)
        self.map_received = True
        
        self.get_logger().info(f'Map loaded: {width}x{height}, resolution={resolution:.3f}m, origin=({origin_x:.2f}, {origin_y:.2f})')
        
        # Try to plan path if we have robot pose
        if self.robot_pose is not None:
            self.plan_and_publish_path()
    
    def pose_callback(self, msg: PoseStamped):
        """
        Callback for robot pose updates.
        """
        if self.robot_pose is None:
            self.robot_pose = msg
            self.get_logger().info(
                f'Robot pose received: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}'
            )
            # Try to plan path if we have map
            if self.map_received and not self.path_planned:
                self.plan_and_publish_path()
        else:
            self.robot_pose = msg
    
    def quaternion_to_yaw(self, quaternion):
        """Convert quaternion to yaw angle."""
        siny_cosp = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion."""
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q
    
    def plan_and_publish_path(self):
        """
        Plan path using A* and publish it.
        """
        if not self.map_received or self.robot_pose is None:
            return
        
        if self.path_planned:
            return  # Only plan once
        
        start_x = self.robot_pose.pose.position.x
        start_y = self.robot_pose.pose.position.y
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('Planning path with A* algorithm...')
        self.get_logger().info(f'  From: ({start_x:.3f}, {start_y:.3f})')
        self.get_logger().info(f'  To:   ({self.goal_x:.3f}, {self.goal_y:.3f})')
        
        # Plan path
        path_waypoints = self.planner.plan_path(start_x, start_y, self.goal_x, self.goal_y)
        
        if path_waypoints is None or len(path_waypoints) == 0:
            self.get_logger().error('Failed to plan path!')
            return
        
        self.get_logger().info(f'A* found path with {len(path_waypoints)} waypoints')
        
        # Simplify path if requested
        if self.simplify_path and len(path_waypoints) > 2:
            original_count = len(path_waypoints)
            path_waypoints = self.planner.simplify_path(path_waypoints)
            self.get_logger().info(f'Simplified path: {original_count} -> {len(path_waypoints)} waypoints')
        
        # Convert to ROS Path message
        path_msg = self.waypoints_to_path(path_waypoints)
        
        # Publish path
        self.path_pub.publish(path_msg)
        self.path_planned = True
        
        self.get_logger().info(f'Path published with {len(path_msg.poses)} waypoints')
        self.get_logger().info('=' * 60)
    
    def waypoints_to_path(self, waypoints):
        """
        Convert list of waypoints to ROS Path message.
        
        Args:
            waypoints: List of (x, y) tuples
            
        Returns:
            Path message
        """
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()
        
        for i, (x, y) in enumerate(waypoints):
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "map"
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            
            pose_stamped.pose.position.x = float(x)
            pose_stamped.pose.position.y = float(y)
            pose_stamped.pose.position.z = 0.0
            
            # Calculate orientation towards next waypoint
            if i < len(waypoints) - 1:
                next_x, next_y = waypoints[i + 1]
                yaw = math.atan2(next_y - y, next_x - x)
            else:
                # Last waypoint uses goal yaw
                yaw = self.goal_yaw
            
            pose_stamped.pose.orientation = self.yaw_to_quaternion(yaw)
            path.poses.append(pose_stamped)
        
        return path


def main(args=None):
    rclpy.init(args=args)
    node = AStarPathPlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down A* path planner node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

