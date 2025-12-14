#!/usr/bin/env python3
"""
ROS2 node for Mission 3: Navigate to colored cone.

This node:
1. Navigates to viewing position (x=1.0198, y=12.9420, yaw=1.57)
2. Detects colored cones (red, blue, green) using basic CV
3. Approaches the specified cone
4. Publishes "bark" when within 3 meters

Mission Requirements:
- Navigate to viewing position to see all cones
- Use color-based detection to identify target cone
- Approach the target cone
- Bark when within 3 meters of the cone
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav_msgs.msg import Path, OccupancyGrid
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
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


class NavigateToCone(Node):
    """
    Mission 3 node: Navigate to colored cone and bark.
    """

    # Viewing position coordinates (where robot can see all 3 cones)
    VIEWING_X = 1.0198
    VIEWING_Y = 12.9420
    VIEWING_YAW = 1.57  # Face the cones

    # Detection parameters
    DETECTION_DISTANCE_MAX = 3.0  # meters
    APPROACH_DISTANCE = 0.5  # How close to get before stopping (meters)

    # Color detection ranges (HSV)
    COLOR_RANGES = {
        'red': [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),    # Lower red range
            (np.array([170, 100, 100]), np.array([180, 255, 255]))  # Upper red range
        ],
        'blue': [
            (np.array([100, 100, 100]), np.array([130, 255, 255]))
        ],
        'green': [
            (np.array([40, 100, 100]), np.array([80, 255, 255]))
        ]
    }

    # Mission states
    STATE_NAVIGATING_TO_VIEW = 0
    STATE_SEARCHING_CONE = 1
    STATE_APPROACHING_CONE = 2
    STATE_MISSION_COMPLETE = 3

    def __init__(self):
        super().__init__('navigate_to_cone')

        # Parameters
        self.declare_parameter('target_color', 'red')  # Default to red cone
        self.declare_parameter('inflation_radius', 0.35)
        self.declare_parameter('use_astar', True)
        self.declare_parameter('simplify_path', True)

        self.target_color = self.get_parameter('target_color').get_parameter_value().string_value.lower()
        inflation_radius = self.get_parameter('inflation_radius').get_parameter_value().double_value
        self.use_astar = self.get_parameter('use_astar').get_parameter_value().bool_value
        self.simplify_path = self.get_parameter('simplify_path').get_parameter_value().bool_value

        # State variables
        self.robot_pose = None
        self.map_received = False
        self.path_published = False
        self.current_state = self.STATE_NAVIGATING_TO_VIEW

        # Cone detection variables
        self.cone_detected = False
        self.cone_center_x = 0
        self.cone_center_y = 0
        self.cone_distance = float('inf')
        self.last_cone_detection_time = None

        # CV Bridge
        self.bridge = CvBridge()

        # Initialize A* planner
        self.planner = None
        if AStarPlanner is not None and self.use_astar:
            self.planner = AStarPlanner(inflation_radius=inflation_radius)
            self.get_logger().info('A* path planner initialized')
        else:
            self.get_logger().warn('A* planner not available, using straight-line path planning')

        self.get_logger().info('=' * 70)
        self.get_logger().info(f'MISSION 3: Navigate to {self.target_color.upper()} Cone')
        self.get_logger().info(f'Viewing position: x={self.VIEWING_X:.3f}, y={self.VIEWING_Y:.3f}, yaw={math.degrees(self.VIEWING_YAW):.1f}°')
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

        # Subscribe to camera
        self.camera_sub = self.create_subscription(
            Image,
            '/camera_top/image',
            self.camera_callback,
            10
        )

        # Subscribe to depth camera
        self.depth_sub = self.create_subscription(
            Image,
            '/camera_top/depth',
            self.depth_callback,
            10
        )

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

        # Publish velocity commands (for fine adjustment)
        self.vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Store depth image
        self.depth_image = None

        # Timer for state machine updates
        self.timer = self.create_timer(0.5, self.state_machine_update)

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
            self.generate_and_publish_path(self.VIEWING_X, self.VIEWING_Y, self.VIEWING_YAW)

    def pose_callback(self, msg: PoseStamped):
        """Callback for robot pose updates."""
        if self.robot_pose is None:
            self.robot_pose = msg
            self.get_logger().info(
                f'Robot pose received: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}'
            )

            if not self.path_published and self.current_state == self.STATE_NAVIGATING_TO_VIEW:
                if self.planner is None or self.map_received:
                    self.generate_and_publish_path(self.VIEWING_X, self.VIEWING_Y, self.VIEWING_YAW)
        else:
            self.robot_pose = msg

    def depth_callback(self, msg: Image):
        """Callback for depth image."""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')

    def camera_callback(self, msg: Image):
        """Callback for camera image - detect colored cones."""
        if self.current_state not in [self.STATE_SEARCHING_CONE, self.STATE_APPROACHING_CONE]:
            return

        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Convert to HSV for color detection
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Get color ranges for target color
            color_ranges = self.COLOR_RANGES.get(self.target_color, [])

            # Create mask for target color
            mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
            for lower, upper in color_ranges:
                color_mask = cv2.inRange(hsv_image, lower, upper)
                mask = cv2.bitwise_or(mask, color_mask)

            # Clean up mask with morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                # Find largest contour (likely the cone)
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)

                # Only process if area is significant
                if area > 500:  # Minimum area threshold
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(largest_contour)

                    # Calculate center
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Get distance from depth image
                    if self.depth_image is not None:
                        # Sample depth at center of detection
                        depth_height, depth_width = self.depth_image.shape
                        depth_x = int(center_x * depth_width / cv_image.shape[1])
                        depth_y = int(center_y * depth_height / cv_image.shape[0])

                        # Get depth value (in meters)
                        distance = self.depth_image[depth_y, depth_x]

                        # Validate distance
                        if not np.isnan(distance) and distance > 0 and distance < 10.0:
                            self.cone_detected = True
                            self.cone_center_x = center_x
                            self.cone_center_y = center_y
                            self.cone_distance = distance
                            self.last_cone_detection_time = self.get_clock().now()

                            self.get_logger().info(
                                f'{self.target_color.upper()} cone detected at ({center_x}, {center_y}), '
                                f'distance: {distance:.2f}m, area: {area:.0f}'
                            )
                        else:
                            self.cone_detected = False
                    else:
                        # No depth image, assume detected
                        self.cone_detected = True
                        self.cone_center_x = center_x
                        self.cone_center_y = center_y
                        self.last_cone_detection_time = self.get_clock().now()
                else:
                    self.cone_detected = False
            else:
                self.cone_detected = False

        except Exception as e:
            self.get_logger().error(f'Error processing camera image: {e}')

    def state_machine_update(self):
        """Update state machine."""
        if self.robot_pose is None:
            return

        if self.current_state == self.STATE_NAVIGATING_TO_VIEW:
            # Check if reached viewing position
            distance_to_view = math.sqrt(
                (self.robot_pose.pose.position.x - self.VIEWING_X)**2 +
                (self.robot_pose.pose.position.y - self.VIEWING_Y)**2
            )

            if distance_to_view < 0.5:  # Within 50cm of viewing position
                self.get_logger().info('Reached viewing position. Searching for cone...')
                self.current_state = self.STATE_SEARCHING_CONE
                self.path_published = False

        elif self.current_state == self.STATE_SEARCHING_CONE:
            if self.cone_detected:
                self.get_logger().info(f'Found {self.target_color} cone! Approaching...')
                self.current_state = self.STATE_APPROACHING_CONE

        elif self.current_state == self.STATE_APPROACHING_CONE:
            if self.cone_detected and self.cone_distance < self.DETECTION_DISTANCE_MAX:
                # Align with cone and approach
                self.approach_cone()

                # Check if close enough
                if self.cone_distance < self.APPROACH_DISTANCE:
                    self.publish_bark()
                    self.current_state = self.STATE_MISSION_COMPLETE
            else:
                # Lost sight of cone, go back to searching
                self.get_logger().warn('Lost sight of cone, searching again...')
                self.current_state = self.STATE_SEARCHING_CONE

    def approach_cone(self):
        """Approach the detected cone using visual servoing."""
        if not self.cone_detected:
            return

        twist = Twist()

        # Assume image width is 640 pixels (adjust if different)
        image_width = 640
        image_center = image_width / 2

        # Calculate error from image center
        error_x = self.cone_center_x - image_center

        # Proportional control for rotation
        kp_angular = 0.003  # Tune this value
        twist.angular.z = -kp_angular * error_x

        # Forward motion based on distance
        if self.cone_distance > self.APPROACH_DISTANCE:
            kp_linear = 0.3  # Tune this value
            twist.linear.x = kp_linear * (self.cone_distance - self.APPROACH_DISTANCE)
            # Limit forward speed
            twist.linear.x = min(twist.linear.x, 0.3)
        else:
            twist.linear.x = 0.0

        self.vel_pub.publish(twist)

    def publish_bark(self):
        """Publish bark command to speech topic."""
        bark_msg = String()
        bark_msg.data = "bark"
        self.speech_pub.publish(bark_msg)

        self.get_logger().info('=' * 70)
        self.get_logger().info(f'🐕 BARK! Reached {self.target_color.upper()} cone!')
        self.get_logger().info('=' * 70)

        # Stop the robot
        stop_msg = Twist()
        self.vel_pub.publish(stop_msg)

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

    def generate_and_publish_path(self, goal_x, goal_y, goal_yaw):
        """Generate and publish path to goal."""
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
        self.get_logger().info('Generating path...')
        self.get_logger().info(f'  From: ({x_start:.3f}, {y_start:.3f}, {math.degrees(yaw_start):.1f}°)')
        self.get_logger().info(f'  To:   ({goal_x:.3f}, {goal_y:.3f}, {math.degrees(goal_yaw):.1f}°)')

        # Use A* if available
        if self.planner is not None:
            self.get_logger().info('Using A* algorithm for collision-free path planning')
            path_waypoints = self.planner.plan_path(x_start, y_start, goal_x, goal_y, start_yaw=yaw_start)

            if path_waypoints is None or len(path_waypoints) == 0:
                self.get_logger().error('A* failed to find path, falling back to straight-line')
                path = self.generate_smooth_path_straightline(goal_x, goal_y, goal_yaw)
            else:
                self.get_logger().info(f'A* found path with {len(path_waypoints)} waypoints')
                path_waypoints = self.densify_waypoints(path_waypoints, points_per_meter=3)
                self.get_logger().info(f'Densified path to {len(path_waypoints)} waypoints')
                path = self.waypoints_to_path(path_waypoints, goal_yaw)
        else:
            self.get_logger().info('Using straight-line path planning')
            path = self.generate_smooth_path_straightline(goal_x, goal_y, goal_yaw)

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
            if len(waypoints[i]) == 3:
                x1, y1, _ = waypoints[i]
                x2, y2, _ = waypoints[i + 1]
            else:
                x1, y1 = waypoints[i]
                x2, y2 = waypoints[i + 1]

            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if dist < 0.01:
                dense_waypoints.append(waypoints[i])
                continue

            num_points = max(2, int(dist * points_per_meter))

            for j in range(num_points):
                t = j / num_points
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                dense_waypoints.append((x, y, None))

        dense_waypoints.append(waypoints[-1])
        return dense_waypoints

    def waypoints_to_path(self, waypoints, goal_yaw):
        """Convert waypoints to ROS Path message."""
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()

        for i, waypoint in enumerate(waypoints):
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

            if explicit_yaw is not None:
                yaw = explicit_yaw
            elif i < len(waypoints) - 1:
                next_waypoint = waypoints[i + 1]
                next_x = next_waypoint[0]
                next_y = next_waypoint[1]

                dx = next_x - x
                dy = next_y - y
                if abs(dx) > 0.001 or abs(dy) > 0.001:
                    yaw = math.atan2(dy, dx)
                else:
                    if i > 0:
                        prev_waypoint = waypoints[i-1]
                        if len(prev_waypoint) == 3 and prev_waypoint[2] is not None:
                            yaw = prev_waypoint[2]
                        elif i > 0 and len(path.poses) > 0:
                            yaw = self.quaternion_to_yaw(path.poses[i-1].pose.orientation)
                        else:
                            yaw = goal_yaw
                    else:
                        yaw = goal_yaw
            else:
                yaw = goal_yaw

            pose_stamped.pose.orientation = self.yaw_to_quaternion(yaw)
            path.poses.append(pose_stamped)

        return path

    def generate_smooth_path_straightline(self, goal_x, goal_y, goal_yaw):
        """Generate smooth straight-line path using cubic Hermite splines."""
        x_start = self.robot_pose.pose.position.x
        y_start = self.robot_pose.pose.position.y
        yaw_start = self.quaternion_to_yaw(self.robot_pose.pose.orientation)

        distance = math.sqrt((goal_x - x_start)**2 + (goal_y - y_start)**2)

        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()

        num_points = max(60, min(100, int(distance * 20)))
        self.get_logger().info(f'  Distance: {distance:.3f}m with {num_points} waypoints')

        control_scale = min(distance * 0.5, 2.0)
        cx0 = x_start + control_scale * math.cos(yaw_start)
        cy0 = y_start + control_scale * math.sin(yaw_start)
        cx1 = goal_x - control_scale * math.cos(goal_yaw)
        cy1 = goal_y - control_scale * math.sin(goal_yaw)

        for i in range(num_points + 1):
            t = i / num_points

            h00 = 2*t**3 - 3*t**2 + 1
            h10 = t**3 - 2*t**2 + t
            h01 = -2*t**3 + 3*t**2
            h11 = t**3 - t**2

            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "map"
            pose_stamped.header.stamp = self.get_clock().now().to_msg()

            tangent_x0 = cx0 - x_start
            tangent_y0 = cy0 - y_start
            tangent_x1 = goal_x - cx1
            tangent_y1 = goal_y - cy1

            pose_stamped.pose.position.x = (h00 * x_start + h10 * tangent_x0 +
                                           h01 * goal_x + h11 * tangent_x1)
            pose_stamped.pose.position.y = (h00 * y_start + h10 * tangent_y0 +
                                           h01 * goal_y + h11 * tangent_y1)
            pose_stamped.pose.position.z = 0.0

            if i < num_points:
                t_next = (i + 1) / num_points
                h00_next = 2*t_next**3 - 3*t_next**2 + 1
                h10_next = t_next**3 - 2*t_next**2 + t_next
                h01_next = -2*t_next**3 + 3*t_next**2
                h11_next = t_next**3 - t_next**2

                x_next = (h00_next * x_start + h10_next * tangent_x0 +
                         h01_next * goal_x + h11_next * tangent_x1)
                y_next = (h00_next * y_start + h10_next * tangent_y0 +
                         h01_next * goal_y + h11_next * tangent_y1)

                dx = x_next - pose_stamped.pose.position.x
                dy = y_next - pose_stamped.pose.position.y
                tangent_yaw = math.atan2(dy, dx)
            else:
                tangent_yaw = goal_yaw

            pose_stamped.pose.orientation = self.yaw_to_quaternion(tangent_yaw)
            path.poses.append(pose_stamped)

        return path


def main(args=None):
    rclpy.init(args=args)
    node = NavigateToCone()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Mission interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
