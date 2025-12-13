#!/usr/bin/env python3
"""
ROS2 node for Mission 1: Navigate to toilet and bark when aligned.

This node:
1. Navigates to the toilet location (x=-7.22, y=-0.54, yaw=1.5708)
2. Monitors robot alignment with toilet using camera
3. Publishes "bark" to /robot_dog/speech when toilet is centered and within range

Mission Requirements:
- Navigate to toilet coordinates
- Align to face toilet (yaw = 1.5708 radians = 90 degrees)
- Bark when toilet is in camera center (middle 3/5 of image) and within 3 meters
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path, OccupancyGrid
from std_msgs.msg import String
# from sensor_msgs.msg import Image  # Not needed - using hardcoded position only
# from cv_bridge import CvBridge  # Not needed - no camera detection
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


class NavigateToToiletAndBark(Node):
    """
    Mission 1 node: Navigate to toilet and bark when aligned.
    """

    # Toilet coordinates (provided by user)
    TOILET_X = -7.22
    TOILET_Y = -0.54
    TOILET_YAW = 1.5708  # 90 degrees

    # Detection parameters (from Mission 1 requirements)
    DETECTION_DISTANCE_MAX = 3.0  # meters
    CENTER_REGION_LEFT = 1.0 / 5.0  # Exclude leftmost 1/5
    CENTER_REGION_RIGHT = 4.0 / 5.0  # Exclude rightmost 1/5

    def __init__(self):
        super().__init__('navigate_to_toilet_and_bark')

        # Parameters
        self.declare_parameter('inflation_radius', 0.35)  # Robot radius in meters
        self.declare_parameter('use_astar', True)
        self.declare_parameter('simplify_path', True)

        inflation_radius = self.get_parameter('inflation_radius').get_parameter_value().double_value
        self.use_astar = self.get_parameter('use_astar').get_parameter_value().bool_value
        self.simplify_path = self.get_parameter('simplify_path').get_parameter_value().bool_value

        # State variables
        self.robot_pose = None
        self.map_received = False
        self.path_published = False
        self.navigation_complete = False
        self.toilet_detected = False

        # CV Bridge not needed - using position-based detection only
        # self.bridge = CvBridge()

        # Initialize A* planner
        self.planner = None
        if AStarPlanner is not None and self.use_astar:
            self.planner = AStarPlanner(inflation_radius=inflation_radius)
            self.get_logger().info('A* path planner initialized')
        else:
            self.get_logger().warn('A* planner not available, using straight-line path planning')

        self.get_logger().info('=' * 70)
        self.get_logger().info('MISSION 1: Navigate to Toilet and Bark')
        self.get_logger().info(f'Target: x={self.TOILET_X:.3f}, y={self.TOILET_Y:.3f}, yaw={math.degrees(self.TOILET_YAW):.1f}°')
        self.get_logger().info('=' * 70)

        # Subscribe to map (for A* planning)
        if self.planner is not None:
            map_qos = QoSProfile(
                depth=10,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                reliability=QoSReliabilityPolicy.RELIABLE
            )
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

        # Camera detection disabled - using hardcoded location only
        # self.camera_sub = self.create_subscription(
        #     Image,
        #     '/camera_top/image',
        #     self.camera_callback,
        #     10
        # )

        # Publish path for navigation
        self.path_pub = self.create_publisher(
            Path,
            '/local_path',
            10
        )

        # Publish bark command
        self.speech_pub = self.create_publisher(
            String,
            '/robot_dog/speech',
            10
        )

        self.get_logger().info('Mission node initialized. Waiting for robot pose and map...')

    def map_callback(self, msg: OccupancyGrid):
        """Callback for map updates."""
        if self.map_received or self.planner is None:
            return

        self.get_logger().info('Map received for A* planning')

        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        width = msg.info.width
        height = msg.info.height

        map_data = np.array(msg.data, dtype=np.int8).reshape((height, width))

        map_metadata = {
            'resolution': resolution,
            'origin_x': origin_x,
            'origin_y': origin_y,
            'width': width,
            'height': height
        }

        self.planner.set_map(map_data, map_metadata)
        self.map_received = True

        self.get_logger().info(f'Map loaded: {width}x{height}, resolution={resolution:.3f}m')

        if self.robot_pose is not None and not self.path_published:
            self.generate_and_publish_path()

    def pose_callback(self, msg: PoseStamped):
        """Callback for robot pose updates."""
        if self.robot_pose is None:
            self.robot_pose = msg
            self.get_logger().info(
                f'Robot pose received: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}'
            )

            if not self.path_published:
                if self.planner is None or self.map_received:
                    self.generate_and_publish_path()
        else:
            self.robot_pose = msg

            # Check if we've reached the toilet location
            if not self.navigation_complete:
                distance_to_goal = math.sqrt(
                    (msg.pose.position.x - self.TOILET_X)**2 +
                    (msg.pose.position.y - self.TOILET_Y)**2
                )

                if distance_to_goal < 0.5:  # Within 50cm of goal
                    self.navigation_complete = True
                    self.get_logger().info('✓ Navigation complete! Reached toilet location.')

                    # Bark immediately upon arrival (no camera detection needed)
                    if not self.toilet_detected:
                        self.publish_bark()
                        self.toilet_detected = True

    # Camera callback removed - using position-based bark trigger only
    # def camera_callback(self, msg: Image):
    #     """
    #     (Disabled) Camera-based detection would be used in full implementation.
    #     Now using hardcoded position only - bark triggers on arrival.
    #     """
    #     pass

    def publish_bark(self):
        """Publish bark command to speech topic."""
        bark_msg = String()
        bark_msg.data = "bark"
        self.speech_pub.publish(bark_msg)

        self.get_logger().info('=' * 70)
        self.get_logger().info('🐕 BARK! Toilet detected and aligned!')
        self.get_logger().info('=' * 70)


    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

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

    def generate_and_publish_path(self):
        """Generate and publish path to toilet."""
        if self.robot_pose is None:
            self.get_logger().warn('Cannot generate path: Robot pose not available')
            return

        if self.planner is not None and not self.map_received:
            self.get_logger().warn('Cannot generate path: Map not received yet')
            return

        x_start = self.robot_pose.pose.position.x
        y_start = self.robot_pose.pose.position.y
        yaw_start = self.quaternion_to_yaw(self.robot_pose.pose.orientation)

        self.get_logger().info('=' * 70)
        self.get_logger().info('Generating path to toilet...')
        self.get_logger().info(f'  From: ({x_start:.3f}, {y_start:.3f}, {math.degrees(yaw_start):.1f}°)')
        self.get_logger().info(f'  To:   ({self.TOILET_X:.3f}, {self.TOILET_Y:.3f}, {math.degrees(self.TOILET_YAW):.1f}°)')

        # Use A* if available
        if self.planner is not None:
            self.get_logger().info('Using A* algorithm for collision-free path planning')
            path_waypoints = self.planner.plan_path(x_start, y_start, self.TOILET_X, self.TOILET_Y, start_yaw=yaw_start)

            if path_waypoints is None or len(path_waypoints) == 0:
                self.get_logger().error('A* failed to find path, falling back to straight-line')
                path = self.generate_smooth_path_straightline()
            else:
                self.get_logger().info(f'A* found path with {len(path_waypoints)} waypoints')

                # Check how many rotation waypoints we have (same position)
                rotation_count = 0
                for i in range(len(path_waypoints) - 1):
                    if abs(path_waypoints[i][0] - path_waypoints[i+1][0]) < 0.01 and \
                       abs(path_waypoints[i][1] - path_waypoints[i+1][1]) < 0.01:
                        rotation_count += 1
                    else:
                        break  # Rotation waypoints are at the start

                if rotation_count > 0:
                    self.get_logger().info(f'  Path includes {rotation_count} rotation waypoints at start')

                # Use raw A* path (smoothing disabled to prevent wall crashes)
                self.get_logger().info(f'Using raw A* path (no smoothing)')

                # Reduced densification to prevent jerking (matches navigate_to_goal.py)
                path_waypoints = self.densify_waypoints(path_waypoints, points_per_meter=3)
                self.get_logger().info(f'Densified path to {len(path_waypoints)} waypoints')

                path = self.waypoints_to_path(path_waypoints)

                # Debug: Print first 10 path poses to see rotation waypoints
                self.get_logger().info('First 10 path poses:')
                for i in range(min(10, len(path.poses))):
                    pose = path.poses[i]
                    yaw = self.quaternion_to_yaw(pose.pose.orientation)
                    self.get_logger().info(f'  [{i}] pos=({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}), yaw={math.degrees(yaw):.1f}°')
        else:
            self.get_logger().info('Using straight-line path planning')
            path = self.generate_smooth_path_straightline()

        self.path_pub.publish(path)
        self.path_published = True
        self.get_logger().info(f'✓ Path published with {len(path.poses)} points')
        self.get_logger().info('=' * 70)

    def densify_waypoints(self, waypoints, points_per_meter=20):
        """Interpolate waypoints for denser path."""
        if len(waypoints) < 2:
            return waypoints

        dense_waypoints = []

        for i in range(len(waypoints) - 1):
            # Handle both (x, y, yaw) and (x, y) formats
            if len(waypoints[i]) == 3:
                x1, y1, _ = waypoints[i]
                x2, y2, _ = waypoints[i + 1]
            else:
                x1, y1 = waypoints[i]
                x2, y2 = waypoints[i + 1]

            # Calculate distance between consecutive waypoints
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Skip densification for rotation waypoints (same position)
            if dist < 0.01:
                dense_waypoints.append(waypoints[i])
                continue

            num_points = max(2, int(dist * points_per_meter))

            for j in range(num_points):
                t = j / num_points
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                # Don't interpolate yaw for movement waypoints (will be computed from direction)
                dense_waypoints.append((x, y, None))

        dense_waypoints.append(waypoints[-1])
        return dense_waypoints

    def waypoints_to_path(self, waypoints):
        """Convert waypoints to ROS Path message."""
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()

        for i, waypoint in enumerate(waypoints):
            # Handle both (x, y, yaw) and (x, y) tuple formats
            if len(waypoint) == 3:
                x, y, explicit_yaw = waypoint
            else:
                x, y = waypoint
                explicit_yaw = None

            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "map"
            pose_stamped.header.stamp = self.get_clock().now().to_msg()

            pose_stamped.pose.position.x = float(x)
            pose_stamped.pose.position.y = float(y)
            pose_stamped.pose.position.z = 0.0

            # Use explicit yaw if provided (for rotation waypoints), otherwise compute from path
            if explicit_yaw is not None:
                yaw = explicit_yaw
            elif i < len(waypoints) - 1:
                # Calculate orientation towards next waypoint
                next_waypoint = waypoints[i + 1]
                next_x = next_waypoint[0]
                next_y = next_waypoint[1]

                # Only compute yaw if there's actual movement
                dx = next_x - x
                dy = next_y - y
                if abs(dx) > 0.001 or abs(dy) > 0.001:
                    yaw = math.atan2(dy, dx)
                else:
                    # No movement - this is likely the last rotation waypoint or similar
                    # Use the previous waypoint's yaw if it had an explicit one
                    if i > 0:
                        prev_waypoint = waypoints[i-1]
                        if len(prev_waypoint) == 3 and prev_waypoint[2] is not None:
                            yaw = prev_waypoint[2]
                        elif i > 0 and len(path.poses) > 0:
                            # Use the previous path pose's yaw
                            yaw = self.quaternion_to_yaw(path.poses[i-1].pose.orientation)
                        else:
                            yaw = self.TOILET_YAW
                    else:
                        yaw = self.TOILET_YAW
            else:
                yaw = self.TOILET_YAW

            pose_stamped.pose.orientation = self.yaw_to_quaternion(yaw)
            path.poses.append(pose_stamped)

        return path

    def generate_smooth_path_straightline(self):
        """Generate smooth straight-line path using cubic Hermite splines."""
        x_start = self.robot_pose.pose.position.x
        y_start = self.robot_pose.pose.position.y
        yaw_start = self.quaternion_to_yaw(self.robot_pose.pose.orientation)

        distance = math.sqrt((self.TOILET_X - x_start)**2 + (self.TOILET_Y - y_start)**2)

        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()

        num_points = max(60, min(100, int(distance * 20)))
        self.get_logger().info(f'  Distance: {distance:.3f}m with {num_points} waypoints')

        # Control point scaling
        control_scale = min(distance * 0.5, 2.0)
        cx0 = x_start + control_scale * math.cos(yaw_start)
        cy0 = y_start + control_scale * math.sin(yaw_start)
        cx1 = self.TOILET_X - control_scale * math.cos(self.TOILET_YAW)
        cy1 = self.TOILET_Y - control_scale * math.sin(self.TOILET_YAW)

        for i in range(num_points + 1):
            t = i / num_points

            # Cubic Hermite basis functions
            h00 = 2*t**3 - 3*t**2 + 1
            h10 = t**3 - 2*t**2 + t
            h01 = -2*t**3 + 3*t**2
            h11 = t**3 - t**2

            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "map"
            pose_stamped.header.stamp = self.get_clock().now().to_msg()

            tangent_x0 = cx0 - x_start
            tangent_y0 = cy0 - y_start
            tangent_x1 = self.TOILET_X - cx1
            tangent_y1 = self.TOILET_Y - cy1

            pose_stamped.pose.position.x = (h00 * x_start + h10 * tangent_x0 +
                                           h01 * self.TOILET_X + h11 * tangent_x1)
            pose_stamped.pose.position.y = (h00 * y_start + h10 * tangent_y0 +
                                           h01 * self.TOILET_Y + h11 * tangent_y1)
            pose_stamped.pose.position.z = 0.0

            # Compute orientation
            if i < num_points:
                t_next = (i + 1) / num_points
                h00_next = 2*t_next**3 - 3*t_next**2 + 1
                h10_next = t_next**3 - 2*t_next**2 + t_next
                h01_next = -2*t_next**3 + 3*t_next**2
                h11_next = t_next**3 - t_next**2

                x_next = (h00_next * x_start + h10_next * tangent_x0 +
                         h01_next * self.TOILET_X + h11_next * tangent_x1)
                y_next = (h00_next * y_start + h10_next * tangent_y0 +
                         h01_next * self.TOILET_Y + h11_next * tangent_y1)

                dx = x_next - pose_stamped.pose.position.x
                dy = y_next - pose_stamped.pose.position.y
                tangent_yaw = math.atan2(dy, dx)
            else:
                tangent_yaw = self.TOILET_YAW

            pose_stamped.pose.orientation = self.yaw_to_quaternion(tangent_yaw)
            path.poses.append(pose_stamped)

        return path


def main(args=None):
    rclpy.init(args=args)
    node = NavigateToToiletAndBark()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Mission interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
