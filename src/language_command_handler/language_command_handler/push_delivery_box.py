#!/usr/bin/env python3
"""
ROS2 node for Mission 4: Push Delivery Box to Goal Area.

This node:
1. Navigates to 3 detection positions sequentially
2. At each position: waits 3s for YOLO to detect "delivery_box" within 2m
3. If detected: pushes forward exactly 3m → MISSION SUCCESS
4. Otherwise: continues to next position
5. Mission ends after successful push or checking all 3 positions

Mission Requirements:
- Navigate to detection waypoints using A* path planning
- Use YOLO detector outputs (/detections/labels, /detections/distance)
- Wait 3 seconds at each position for detection
- Validate box is within 2m range before pushing
- Push exactly 3m forward when conditions are met
"""

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav_msgs.msg import Path, OccupancyGrid
from std_msgs.msg import String, Float32
import numpy as np
import os
import sys

# Import A* planner
try:
    from ament_index_python.packages import get_package_share_directory
    path_tracker_share = get_package_share_directory('path_tracker')
    path_tracker_python_path = os.path.join(path_tracker_share, '..', '..', 'src', 'path_tracker', 'path_tracker')
    if os.path.exists(path_tracker_python_path):
        sys.path.insert(0, path_tracker_python_path)
    from astar_planner import AStarPlanner
except (ImportError, Exception) as e:
    try:
        from path_tracker.astar_planner import AStarPlanner
    except ImportError:
        print(f"Warning: Could not import AStarPlanner ({e}). Using straight-line path planning.")
        AStarPlanner = None


class PushDeliveryBox(Node):
    """
    Mission 4 node: Navigate to positions, detect delivery box, push to goal.
    """

    # Detection positions
    # Format: {'x': meters, 'y': meters, 'yaw': radians}
    DETECTION_POSITIONS = [
        {'x': 5.0, 'y': -3.0, 'yaw': math.pi},       # Position 1: South side, yaw 90°
        {'x': 2.0, 'y': 0.0, 'yaw': math.pi/2},      # Position 2: West of box, yaw 0°
        {'x': 5.0, 'y': 3.0, 'yaw': 0},              # Position 3: North side, yaw -90°
    ]

    # Mission parameters
    DETECTION_TIMEOUT = 3.0        # seconds to wait at each position
    DETECTION_RANGE = 2.0          # meters - max distance to box
    PUSH_DISTANCE = 3.0            # meters - how far to push
    TARGET_LABEL = 'delivery_box'  # YOLO label
    ARRIVAL_THRESHOLD = 0.5        # meters - navigation arrival distance
    PUSH_SPEED = 0.2               # m/s - pushing speed

    # Mission states
    STATE_NAVIGATING_TO_POSITION = 0  # Navigate to detection position
    STATE_WAITING_FOR_DETECTION = 1   # Wait 3s for YOLO detection
    STATE_PUSHING_BOX = 2              # Push forward 3m
    STATE_NEXT_POSITION = 3            # Move to next position
    STATE_MISSION_SUCCESS = 4          # Box pushed successfully
    STATE_MISSION_FAILED = 5           # All positions checked, no success

    def __init__(self):
        super().__init__('push_delivery_box')

        # Parameters
        self.declare_parameter('inflation_radius', 0.35)
        self.declare_parameter('use_astar', True)
        self.declare_parameter('simplify_path', True)

        inflation_radius = self.get_parameter('inflation_radius').get_parameter_value().double_value
        self.use_astar = self.get_parameter('use_astar').get_parameter_value().bool_value
        self.simplify_path = self.get_parameter('simplify_path').get_parameter_value().bool_value

        # State variables
        self.robot_pose = None
        self.map_received = False
        self.path_published = False
        self.state = self.STATE_NAVIGATING_TO_POSITION
        self.current_position_index = 0
        self.current_goal_x = None
        self.current_goal_y = None
        self.current_goal_yaw = None

        # Detection state
        self.latest_labels = []
        self.latest_distance = -1.0
        self.detection_start_time = None

        # Push state
        self.push_start_x = None
        self.push_start_y = None

        # Navigation state (for rotation)
        self.is_rotating = False
        self.rotation_complete = False

        # Initialize A* planner
        self.planner = None
        if self.use_astar and AStarPlanner is not None:
            self.planner = AStarPlanner(inflation_radius=inflation_radius)
            self.get_logger().info(f'A* planner initialized (inflation={inflation_radius}m, simplify={self.simplify_path})')
        else:
            self.get_logger().info('A* planner disabled, using straight-line paths')

        # QoS profile for map (TRANSIENT_LOCAL for latched topics)
        map_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=1
        )

        # Subscribe to map for A* planning
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )

        # Subscribe to robot pose
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/go1_pose',
            self.pose_callback,
            10
        )

        # Subscribe to YOLO detector outputs
        self.labels_sub = self.create_subscription(
            String,
            '/detections/labels',
            self.labels_callback,
            10
        )

        self.distance_sub = self.create_subscription(
            Float32,
            '/detections/distance',
            self.distance_callback,
            10
        )

        # Publish path for navigation
        self.path_pub = self.create_publisher(
            Path,
            '/local_path',
            10
        )

        # Publish velocity commands
        self.vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Timer for state machine updates
        self.timer = self.create_timer(0.5, self.state_machine_update)

        # Timer for rotation control
        self.rotation_timer = self.create_timer(0.1, self.rotation_callback)

        self.get_logger().info('Push delivery box mission initialized. Waiting for robot pose and map...')

    def map_callback(self, msg: OccupancyGrid):
        """Callback for map updates."""
        if self.map_received or self.planner is None:
            return

        self.get_logger().info('Map received! Initializing A* planner...')

        # Convert OccupancyGrid to numpy array
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        # Reshape to 2D grid
        map_data = np.array(msg.data, dtype=np.int8).reshape((height, width))

        # Create map metadata dictionary
        map_metadata = {
            'resolution': resolution,
            'origin_x': origin_x,
            'origin_y': origin_y,
            'width': width,
            'height': height
        }

        # Initialize planner with map
        self.planner.set_map(map_data, map_metadata)
        self.map_received = True
        self.get_logger().info(f'A* planner ready! Map size: {width}x{height}, resolution: {resolution}m')

    def pose_callback(self, msg: PoseStamped):
        """Callback for robot pose updates."""
        self.robot_pose = msg

    def labels_callback(self, msg: String):
        """Callback for YOLO detection labels."""
        if msg.data == 'None' or msg.data == '':
            self.latest_labels = []
        else:
            self.latest_labels = [label.strip() for label in msg.data.split(',')]

    def distance_callback(self, msg: Float32):
        """Callback for YOLO detection distance."""
        self.latest_distance = msg.data

    def state_machine_update(self):
        """Main state machine callback."""
        if self.robot_pose is None:
            return

        current_x = self.robot_pose.pose.position.x
        current_y = self.robot_pose.pose.position.y
        current_yaw = self.quaternion_to_yaw(self.robot_pose.pose.orientation)

        # STATE 0: Navigate to detection position
        if self.state == self.STATE_NAVIGATING_TO_POSITION:
            # Set current goal if not set
            if self.current_goal_x is None:
                pos = self.DETECTION_POSITIONS[self.current_position_index]
                self.current_goal_x = pos['x']
                self.current_goal_y = pos['y']
                self.current_goal_yaw = pos['yaw']
                self.path_published = False
                self.rotation_complete = False
                self.is_rotating = True
                self.get_logger().info(f'Navigating to position {self.current_position_index + 1}: ({self.current_goal_x:.2f}, {self.current_goal_y:.2f}), yaw={math.degrees(self.current_goal_yaw):.1f}°')

            # Check if arrived at position
            distance_to_goal = math.sqrt((current_x - self.current_goal_x)**2 + (current_y - self.current_goal_y)**2)
            if distance_to_goal < self.ARRIVAL_THRESHOLD:
                self.get_logger().info(f'Arrived at position {self.current_position_index + 1}. Starting detection wait...')
                self.state = self.STATE_WAITING_FOR_DETECTION
                self.detection_start_time = self.get_clock().now()
                # Stop robot
                twist = Twist()
                self.vel_pub.publish(twist)

        # STATE 1: Wait for YOLO detection
        elif self.state == self.STATE_WAITING_FOR_DETECTION:
            # Check detection
            box_detected = self.TARGET_LABEL in self.latest_labels
            distance_valid = 0 < self.latest_distance <= self.DETECTION_RANGE

            if box_detected and distance_valid:
                self.get_logger().info(f'Box detected at {self.latest_distance:.2f}m! Starting push...')
                self.state = self.STATE_PUSHING_BOX
                self.push_start_x = current_x
                self.push_start_y = current_y
            else:
                # Check timeout
                elapsed = (self.get_clock().now() - self.detection_start_time).nanoseconds / 1e9
                if elapsed >= self.DETECTION_TIMEOUT:
                    self.get_logger().info(f'Detection timeout at position {self.current_position_index + 1}. Moving to next position...')
                    self.state = self.STATE_NEXT_POSITION

        # STATE 2: Push box forward
        elif self.state == self.STATE_PUSHING_BOX:
            # Calculate distance traveled
            distance_traveled = math.sqrt((current_x - self.push_start_x)**2 + (current_y - self.push_start_y)**2)

            if distance_traveled >= self.PUSH_DISTANCE:
                self.get_logger().info(f'Push complete! Traveled {distance_traveled:.2f}m. MISSION SUCCESS!')
                self.state = self.STATE_MISSION_SUCCESS
                # Stop robot
                twist = Twist()
                self.vel_pub.publish(twist)
            else:
                # Continue pushing forward
                twist = Twist()
                twist.linear.x = self.PUSH_SPEED
                twist.angular.z = 0.0
                self.vel_pub.publish(twist)

        # STATE 3: Move to next position
        elif self.state == self.STATE_NEXT_POSITION:
            self.current_position_index += 1
            if self.current_position_index >= len(self.DETECTION_POSITIONS):
                self.get_logger().info('All positions checked. MISSION FAILED - box not found.')
                self.state = self.STATE_MISSION_FAILED
            else:
                # Reset for next position
                self.current_goal_x = None
                self.current_goal_y = None
                self.current_goal_yaw = None
                self.state = self.STATE_NAVIGATING_TO_POSITION

        # STATE 4 & 5: Mission complete (success or failed)
        elif self.state in [self.STATE_MISSION_SUCCESS, self.STATE_MISSION_FAILED]:
            # Stop robot
            twist = Twist()
            self.vel_pub.publish(twist)
            # Mission ended, keep node alive

    def rotation_callback(self):
        """Handle rotation before navigation."""
        if not self.is_rotating or self.rotation_complete:
            return

        if self.robot_pose is None or self.current_goal_yaw is None:
            return

        current_yaw = self.quaternion_to_yaw(self.robot_pose.pose.orientation)
        yaw_error = self.normalize_angle(self.current_goal_yaw - current_yaw)

        # Check if rotation is complete
        if abs(yaw_error) < 0.1:  # 0.1 radian threshold (~5.7 degrees)
            if not self.rotation_complete:
                self.rotation_complete = True
                self.is_rotating = False
                self.get_logger().info('Rotation complete. Generating path...')

                # Generate and publish path
                if not self.path_published:
                    self.generate_and_publish_path(
                        self.current_goal_x,
                        self.current_goal_y,
                        self.current_goal_yaw
                    )
                    self.path_published = True

            # Stop rotation
            twist = Twist()
            self.vel_pub.publish(twist)
        else:
            # Continue rotating
            twist = Twist()
            twist.angular.z = 0.3 if yaw_error > 0 else -0.3
            self.vel_pub.publish(twist)

    def generate_and_publish_path(self, goal_x, goal_y, goal_yaw):
        """Generate path using A* and publish to /local_path."""
        if self.robot_pose is None:
            return

        start_x = self.robot_pose.pose.position.x
        start_y = self.robot_pose.pose.position.y
        start_yaw = self.quaternion_to_yaw(self.robot_pose.pose.orientation)

        waypoints = []

        # Try A* planning if available and map is loaded
        if self.use_astar and self.planner is not None and self.map_received:
            try:
                waypoints = self.planner.plan_path(start_x, start_y, goal_x, goal_y, start_yaw)
                if waypoints:
                    self.get_logger().info(f'A* path generated with {len(waypoints)} waypoints')
                else:
                    self.get_logger().warn('A* path planning failed! Using straight-line.')
            except Exception as e:
                self.get_logger().error(f'A* planning error: {e}. Using straight-line.')
                waypoints = []

        # Fallback to straight-line path
        if not waypoints:
            waypoints = [(start_x, start_y, start_yaw), (goal_x, goal_y, goal_yaw)]
            self.get_logger().info('Using straight-line path')

        # Densify path for smooth tracking (3 points per meter)
        densified_waypoints = self.densify_path(waypoints, points_per_meter=3.0)

        # Convert to Path message
        path_msg = self.waypoints_to_path(densified_waypoints, goal_yaw)

        # Publish path
        self.path_pub.publish(path_msg)
        self.get_logger().info(f'Published path with {len(path_msg.poses)} poses')

    def densify_path(self, waypoints, points_per_meter=3.0):
        """Add intermediate points between waypoints for smoother tracking."""
        if len(waypoints) < 2:
            return waypoints

        densified = []
        for i in range(len(waypoints) - 1):
            x1, y1 = waypoints[i][0], waypoints[i][1]
            x2, y2 = waypoints[i + 1][0], waypoints[i + 1][1]

            # Add current waypoint
            densified.append(waypoints[i])

            # Calculate distance
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Skip if distance is too small (rotation waypoint)
            if dist < 0.01:
                continue

            # Calculate number of intermediate points
            num_points = max(1, int(dist * points_per_meter))

            # Add intermediate points
            for j in range(1, num_points):
                t = j / num_points
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                # Interpolate yaw if available
                if len(waypoints[i]) > 2 and len(waypoints[i + 1]) > 2:
                    yaw1 = waypoints[i][2]
                    yaw2 = waypoints[i + 1][2]
                    yaw = yaw1 + t * self.normalize_angle(yaw2 - yaw1)
                    densified.append((x, y, yaw))
                else:
                    densified.append((x, y))

        # Add final waypoint
        densified.append(waypoints[-1])
        return densified

    def waypoints_to_path(self, waypoints, goal_yaw):
        """Convert waypoints to Path message."""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for i, wp in enumerate(waypoints):
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = 'D'  # Forward motion

            pose_stamped.pose.position.x = wp[0]
            pose_stamped.pose.position.y = wp[1]
            pose_stamped.pose.position.z = 0.0

            # Use waypoint yaw if available, otherwise compute from direction
            if len(wp) > 2 and wp[2] is not None:
                yaw = wp[2]
            elif i < len(waypoints) - 1:
                dx = waypoints[i + 1][0] - wp[0]
                dy = waypoints[i + 1][1] - wp[1]
                yaw = math.atan2(dy, dx)
            elif goal_yaw is not None:
                yaw = goal_yaw
            else:
                # Fallback: use yaw from previous waypoint or 0
                yaw = 0.0

            pose_stamped.pose.orientation = self.yaw_to_quaternion(yaw)
            path_msg.poses.append(pose_stamped)

        return path_msg

    @staticmethod
    def quaternion_to_yaw(q: Quaternion) -> float:
        """Convert quaternion to yaw angle."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def yaw_to_quaternion(yaw: float) -> Quaternion:
        """Convert yaw angle to quaternion."""
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = PushDeliveryBox()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
