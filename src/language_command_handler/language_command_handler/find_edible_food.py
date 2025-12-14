#!/usr/bin/env python3
"""
ROS2 node for Mission 2: Find and Identify Edible Food.

This node:
1. Navigates through 9 hospital rooms sequentially
2. Performs 360° scan in each room to detect food objects
3. Prioritizes edible food (apple, banana, pizza) over rotten food
4. Approaches detected food within 1m using visual servoing
5. Publishes "bark" when edible food is found and stops mission
6. Continues to next room if only rotten/other objects found

Mission Requirements:
- Navigate to each room waypoint using A* path planning
- Use YOLO detector outputs (/detections/labels, /detections/distance)
- Perform 360° rotation scan to collect all detections
- Select best object (edible > rotten) and approach
- Bark when edible food is verified at close range
"""

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav_msgs.msg import Path, OccupancyGrid
from std_msgs.msg import String, Float32
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


class FindEdibleFood(Node):
    """
    Mission 2 node: Search 9 rooms to find edible food and bark.
    """

    # Room coordinates for food search mission
    # Format: {'x': meters, 'y': meters, 'yaw': radians}
    ROOM_COORDINATES = [
        {'x': -7.87, 'y': -5.55, 'yaw': -1.57},  # Room 1
        {'x': -7.99, 'y': -7.33, 'yaw': 1.57},   # Room 2
        {'x': -7.99, 'y': 21.14, 'yaw': 1.57},   # Room 3
        {'x': -7.99, 'y': 26.70, 'yaw': 3.14},   # Room 4
        {'x': 5.84, 'y': 28.55, 'yaw': -1.57},   # Room 5
        {'x': 8.34, 'y': 19.94, 'yaw': 1.57},    # Room 6
        {'x': 8.34, 'y': -5.87, 'yaw': 1.57},    # Room 7
        {'x': 10.88, 'y': 5.48, 'yaw': 0.0},     # Room 8
        {'x': 10.88, 'y': 10.56, 'yaw': 0.0},    # Room 9
    ]

    # Object categories (matching YOLO detector classes)
    EDIBLE_OBJECTS = ['banana', 'apple', 'pizza']
    ROTTEN_OBJECTS = ['rotten_banana', 'rotten_apple', 'rotten_pizza']
    OTHER_OBJECTS = ['stop_sign', 'nurse', 'cone_red', 'cone_green', 'cone_blue', 'delivery_box']

    # Detection parameters
    TARGET_APPROACH_DISTANCE = 1.0  # Stop at 1m from object
    ROOM_ARRIVAL_DISTANCE = 0.5     # Room reached when within 0.5m

    # Mission states
    STATE_NAVIGATING_TO_ROOM = 0    # Navigate to room waypoint using A*
    STATE_SEARCHING_FOR_FOOD = 1    # Perform 360° rotation scan
    STATE_SELECTING_TARGET = 2      # Select best object (edible > rotten)
    STATE_APPROACHING_OBJECT = 3    # Visual servoing to approach object
    STATE_VERIFYING_OBJECT = 4      # Check if edible and bark
    STATE_NEXT_ROOM = 5             # Move to next room
    STATE_MISSION_COMPLETE = 6      # Mission ended (success/failure)

    def __init__(self):
        super().__init__('find_edible_food')

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
        self.current_state = self.STATE_NAVIGATING_TO_ROOM
        self.current_room_index = 0
        self.mission_success = False
        self.rotation_complete = False

        # Detection variables
        self.latest_labels = []
        self.latest_distance = -1.0
        self.current_target_object = None
        self.last_detection_time = None

        # Scanning variables
        self.scan_detections = []
        self.search_start_yaw = None
        self.target_yaw = None  # For rotating to face selected object

        # Rotation parameters (from toilet navigation logic)
        self.rotation_threshold = 0.1  # radians (~5.7 degrees)
        self.rotation_speed = 0.3  # angular velocity for rotation
        self.current_goal_x = None
        self.current_goal_y = None
        self.current_goal_yaw = None

        # Initialize A* planner
        self.planner = None
        if AStarPlanner is not None and self.use_astar:
            self.planner = AStarPlanner(inflation_radius=inflation_radius)
            self.get_logger().info('A* path planner initialized')
        else:
            self.get_logger().warn('A* planner not available, using straight-line path planning')

        self.get_logger().info('=' * 70)
        self.get_logger().info('MISSION 2: Find and Identify Edible Food')
        self.get_logger().info(f'Total rooms to search: {len(self.ROOM_COORDINATES)}')
        self.get_logger().info(f'Edible objects: {self.EDIBLE_OBJECTS}')
        self.get_logger().info(f'Rotten objects: {self.ROTTEN_OBJECTS}')
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

        # Publish bark command
        self.speech_pub = self.create_publisher(
            String,
            '/robot_dog/speech',
            10
        )

        # Publish velocity commands (for scanning and approach)
        self.vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Timer for state machine updates
        self.timer = self.create_timer(0.5, self.state_machine_update)

        # Timer for rotation control (from toilet navigation logic)
        self.rotation_timer = self.create_timer(0.1, self.rotation_callback)

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

    def pose_callback(self, msg: PoseStamped):
        """Callback for robot pose updates."""
        self.robot_pose = msg

    def labels_callback(self, msg: String):
        """Callback for YOLO detection labels."""
        if msg.data == 'None':
            self.latest_labels = []
        else:
            self.latest_labels = [label.strip() for label in msg.data.split(',')]

        # Update last detection time if labels present
        if self.latest_labels:
            self.last_detection_time = self.get_clock().now()

    def distance_callback(self, msg: Float32):
        """Callback for distance to centered object."""
        self.latest_distance = msg.data
        if msg.data > 0:
            self.last_detection_time = self.get_clock().now()

    def quaternion_to_yaw(self, quaternion: Quaternion) -> float:
        """Convert quaternion to yaw angle in radians."""
        siny_cosp = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def yaw_to_quaternion(self, yaw: float) -> Quaternion:
        """Convert yaw angle to quaternion."""
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi] range."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def stop_robot(self):
        """Stop robot motion."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.vel_pub.publish(twist)

    def generate_and_publish_path(self, goal_x: float, goal_y: float, goal_yaw: float):
        """Generate path to goal using A* and publish to /local_path."""
        if self.robot_pose is None:
            self.get_logger().warn('No robot pose available')
            return

        start_x = self.robot_pose.pose.position.x
        start_y = self.robot_pose.pose.position.y
        start_yaw = self.quaternion_to_yaw(self.robot_pose.pose.orientation)

        self.get_logger().info(f'Planning path from ({start_x:.2f}, {start_y:.2f}) to ({goal_x:.2f}, {goal_y:.2f})')

        # Generate path using A* planner
        if self.planner is not None and self.map_received:
            waypoints = self.planner.plan_path(start_x, start_y, goal_x, goal_y, start_yaw=start_yaw)

            if not waypoints:
                self.get_logger().error('A* path planning failed! Skipping to next room.')
                return None

            # Add goal orientation to final waypoint
            waypoints[-1] = (goal_x, goal_y, goal_yaw)

        else:
            # Straight-line path (fallback)
            waypoints = [(start_x, start_y, start_yaw), (goal_x, goal_y, goal_yaw)]

        # Densify path for smooth tracking (3 points per meter)
        densified_waypoints = self.densify_path(waypoints, points_per_meter=3.0)

        # Create Path message using toilet navigation logic
        path_msg = self.waypoints_to_path(densified_waypoints, goal_yaw)

        self.path_pub.publish(path_msg)
        self.get_logger().info(f'Published path with {len(path_msg.poses)} waypoints')
        return path_msg

    def densify_path(self, waypoints: list, points_per_meter: float = 3.0) -> list:
        """Add intermediate points between waypoints for smooth tracking."""
        densified = []

        for i in range(len(waypoints) - 1):
            # Handle both (x, y, yaw) and (x, y) formats
            if len(waypoints[i]) == 3:
                x1, y1, yaw1 = waypoints[i]
            else:
                x1, y1 = waypoints[i]
                yaw1 = None

            if len(waypoints[i + 1]) == 3:
                x2, y2, yaw2 = waypoints[i + 1]
            else:
                x2, y2 = waypoints[i + 1]
                yaw2 = None

            densified.append((x1, y1, yaw1))

            # Calculate distance between waypoints
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Skip densification for rotation waypoints (same position)
            if dist < 0.01:
                continue

            num_points = max(1, int(dist * points_per_meter))

            # Add intermediate points
            for j in range(1, num_points):
                t = j / num_points
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                # Don't interpolate yaw for movement waypoints (will be computed from direction)
                densified.append((x, y, None))

        # Add final waypoint
        densified.append(waypoints[-1])

        return densified

    def waypoints_to_path(self, waypoints: list, goal_yaw: float) -> Path:
        """Convert waypoints to ROS Path message (from toilet navigation logic)."""
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
                    # No movement - use previous or goal yaw
                    if i > 0 and len(path.poses) > 0:
                        yaw = self.quaternion_to_yaw(path.poses[i-1].pose.orientation)
                    else:
                        yaw = goal_yaw
            else:
                yaw = goal_yaw

            pose_stamped.pose.orientation = self.yaw_to_quaternion(yaw)

            # Override frame_id to "D" (Drive/Forward) to prevent backward motion
            # The path tracker uses frame_id to determine gear:
            # "D" = forward, "R" = reverse, "N" = neutral
            # This ensures robot always rotates in place first, then moves forward
            pose_stamped.header.frame_id = "D"

            path.poses.append(pose_stamped)

        return path

    def rotation_callback(self):
        """Timer callback to handle initial rotation towards goal (from toilet navigation logic)."""
        if self.robot_pose is None:
            return

        # Only rotate when navigating to room and not yet rotated
        if self.current_state != self.STATE_NAVIGATING_TO_ROOM or self.rotation_complete:
            return

        # Set goal if not set yet (for current room)
        if self.current_goal_x is None and self.current_room_index < len(self.ROOM_COORDINATES):
            room = self.ROOM_COORDINATES[self.current_room_index]
            self.current_goal_x = room['x']
            self.current_goal_y = room['y']
            self.current_goal_yaw = room['yaw']

        if self.current_goal_x is None:
            return

        # Calculate angle to goal
        current_yaw = self.quaternion_to_yaw(self.robot_pose.pose.orientation)
        dx = self.current_goal_x - self.robot_pose.pose.position.x
        dy = self.current_goal_y - self.robot_pose.pose.position.y
        desired_yaw = math.atan2(dy, dx)

        # Calculate angle difference
        angle_diff = self.normalize_angle(desired_yaw - current_yaw)

        # Check if rotation is complete
        if abs(angle_diff) < self.rotation_threshold:
            if not self.rotation_complete:
                # Stop rotation
                stop_msg = Twist()
                self.vel_pub.publish(stop_msg)

                self.rotation_complete = True
                self.get_logger().info(f'✓ Rotation complete! Aligned with goal (error: {math.degrees(angle_diff):.1f}°)')

                # Now publish the navigation path
                if not self.path_published:
                    room = self.ROOM_COORDINATES[self.current_room_index]
                    self.generate_and_publish_path(room['x'], room['y'], room['yaw'])
        else:
            # Continue rotating (pure rotation, no forward motion)
            twist = Twist()
            twist.linear.x = 0.0  # No forward motion during rotation
            twist.angular.z = self.rotation_speed if angle_diff > 0 else -self.rotation_speed
            self.vel_pub.publish(twist)

            if not hasattr(self, '_last_rotation_log') or \
               (self.get_clock().now() - self._last_rotation_log).nanoseconds > 1e9:  # Log every 1 second
                self.get_logger().info(f'Rotating towards Room {self.current_room_index + 1}... (angle error: {math.degrees(angle_diff):.1f}°)')
                self._last_rotation_log = self.get_clock().now()

    def state_machine_update(self):
        """Main state machine update (called every 0.5s)."""
        if self.robot_pose is None:
            return

        # Execute state-specific logic
        if self.current_state == self.STATE_NAVIGATING_TO_ROOM:
            self.handle_navigating_to_room()
        elif self.current_state == self.STATE_SEARCHING_FOR_FOOD:
            self.handle_searching_for_food()
        elif self.current_state == self.STATE_SELECTING_TARGET:
            self.handle_selecting_target()
        elif self.current_state == self.STATE_APPROACHING_OBJECT:
            self.handle_approaching_object()
        elif self.current_state == self.STATE_VERIFYING_OBJECT:
            self.handle_verifying_object()
        elif self.current_state == self.STATE_NEXT_ROOM:
            self.handle_next_room()
        elif self.current_state == self.STATE_MISSION_COMPLETE:
            self.handle_mission_complete()

    def handle_navigating_to_room(self):
        """Navigate to current room waypoint."""
        # Wait for rotation to complete before checking arrival
        if not self.rotation_complete:
            return

        # Check if reached room
        room = self.ROOM_COORDINATES[self.current_room_index]
        current_x = self.robot_pose.pose.position.x
        current_y = self.robot_pose.pose.position.y
        distance_to_room = math.sqrt((current_x - room['x'])**2 + (current_y - room['y'])**2)

        if distance_to_room < self.ROOM_ARRIVAL_DISTANCE:
            self.get_logger().info(f'Arrived at Room {self.current_room_index + 1}. Starting search...')
            self.path_published = False
            self.current_state = self.STATE_SEARCHING_FOR_FOOD
            # Reset scan variables
            self.scan_detections = []
            self.search_start_yaw = None

    def handle_searching_for_food(self):
        """Perform 360° rotation scan to collect food detections."""
        current_yaw = self.quaternion_to_yaw(self.robot_pose.pose.orientation)

        # Initialize scan on first call
        if self.search_start_yaw is None:
            self.search_start_yaw = current_yaw
            self.scan_detections = []
            self.get_logger().info('Starting 360° food scan...')

        # Rotate robot
        twist = Twist()
        twist.angular.z = 0.3  # 0.3 rad/s rotation speed
        twist.linear.x = 0.0
        self.vel_pub.publish(twist)

        # Collect detected food objects
        if self.latest_labels:
            food_objects = [obj for obj in self.latest_labels
                            if obj in self.EDIBLE_OBJECTS + self.ROTTEN_OBJECTS]
            for obj in food_objects:
                # Avoid duplicates
                if obj not in [d['label'] for d in self.scan_detections]:
                    self.scan_detections.append({
                        'label': obj,
                        'yaw': current_yaw
                    })
                    self.get_logger().info(f'Detected: {obj} at yaw={current_yaw:.2f}')

        # Check if 360° rotation complete
        rotation_delta = abs(self.normalize_angle(current_yaw - self.search_start_yaw))
        if rotation_delta > 2 * math.pi - 0.2:  # 360° with tolerance
            self.stop_robot()
            self.get_logger().info(f'Scan complete. Found {len(self.scan_detections)} food object(s)')
            self.current_state = self.STATE_SELECTING_TARGET

    def handle_selecting_target(self):
        """Select best object from scan (prioritize edible over rotten)."""
        if not self.scan_detections:
            # No food found in this room
            self.get_logger().info('No food found in this room')
            self.current_state = self.STATE_NEXT_ROOM
            return

        # Prioritize edible food
        edible = [d for d in self.scan_detections if d['label'] in self.EDIBLE_OBJECTS]
        if edible:
            self.current_target_object = edible[0]['label']
            self.target_yaw = edible[0]['yaw']
            self.get_logger().info(f'Selected EDIBLE target: {self.current_target_object}')
        else:
            # Only rotten food found
            self.current_target_object = self.scan_detections[0]['label']
            self.target_yaw = self.scan_detections[0]['yaw']
            self.get_logger().info(f'Selected ROTTEN target: {self.current_target_object}')

        # Rotate back to face the selected object
        self.get_logger().info(f'Rotating to face object at yaw={self.target_yaw:.2f}...')
        self.current_state = self.STATE_APPROACHING_OBJECT
        self.last_detection_time = self.get_clock().now()

    def handle_approaching_object(self):
        """Approach detected object using visual servoing."""
        current_yaw = self.quaternion_to_yaw(self.robot_pose.pose.orientation)

        # First, rotate to face the object's remembered yaw
        if self.target_yaw is not None:
            yaw_error = self.normalize_angle(self.target_yaw - current_yaw)
            if abs(yaw_error) > 0.1:  # Not facing object yet
                twist = Twist()
                twist.angular.z = 0.3 * (1.0 if yaw_error > 0 else -1.0)
                twist.linear.x = 0.0
                self.vel_pub.publish(twist)
                return
            else:
                # Facing object, clear target yaw
                self.target_yaw = None
                self.stop_robot()
                time.sleep(0.3)  # Brief pause
                return

        # Now use distance feedback for approach
        if self.latest_distance < 0:
            # Object not centered yet - rotate slowly to center
            twist = Twist()
            twist.angular.z = 0.2  # Slow rotation to find object
            twist.linear.x = 0.0
            self.vel_pub.publish(twist)

            # Check if object lost for too long
            if self.last_detection_time is not None:
                time_since_detection = (self.get_clock().now() - self.last_detection_time).nanoseconds / 1e9
                if time_since_detection > 3.0:
                    self.get_logger().warn('Object lost for >3s. Returning to search.')
                    self.current_state = self.STATE_SEARCHING_FOR_FOOD
                    self.search_start_yaw = None
            return

        # Object is centered (distance > 0)
        if self.latest_distance > self.TARGET_APPROACH_DISTANCE + 0.1:
            # Approach object
            twist = Twist()
            twist.angular.z = 0.0  # Keep straight (object already centered)
            kp_linear = 0.3
            twist.linear.x = kp_linear * (self.latest_distance - self.TARGET_APPROACH_DISTANCE)
            twist.linear.x = min(twist.linear.x, 0.3)  # Max speed 0.3 m/s
            self.vel_pub.publish(twist)
        else:
            # Close enough (within 1m) - verify object
            self.stop_robot()
            self.get_logger().info(f'Reached object at {self.latest_distance:.2f}m. Verifying...')
            self.current_state = self.STATE_VERIFYING_OBJECT

    def handle_verifying_object(self):
        """Check if centered object is edible food and bark if yes."""
        # Wait briefly for stable detection
        time.sleep(0.5)

        # Check current target object type
        if self.current_target_object in self.EDIBLE_OBJECTS:
            # Success! Found edible food
            self.get_logger().info('=' * 70)
            self.get_logger().info(f'Mission SUCCESS: Found edible food ({self.current_target_object})')
            self.get_logger().info(f'Room: {self.current_room_index + 1}/{len(self.ROOM_COORDINATES)}')
            self.get_logger().info('=' * 70)

            # Publish bark
            speech_msg = String()
            speech_msg.data = 'bark'
            self.speech_pub.publish(speech_msg)

            self.mission_success = True
            self.current_state = self.STATE_MISSION_COMPLETE
        else:
            # Rotten or other object - continue search
            self.get_logger().info(f'Not edible: {self.current_target_object}. Moving to next room.')
            self.current_state = self.STATE_NEXT_ROOM

    def handle_next_room(self):
        """Move to next room or complete mission if all rooms searched."""
        self.current_room_index += 1

        if self.current_room_index >= len(self.ROOM_COORDINATES):
            # All rooms searched, no edible food found
            self.stop_robot()
            self.get_logger().error('=' * 70)
            self.get_logger().error(f'Mission FAILED: No edible food found in {len(self.ROOM_COORDINATES)} rooms')
            self.get_logger().error('=' * 70)
            self.mission_success = False
            self.current_state = self.STATE_MISSION_COMPLETE
            return

        # Navigate to next room - reset rotation variables
        self.get_logger().info(f'Moving to Room {self.current_room_index + 1}/{len(self.ROOM_COORDINATES)}')
        self.rotation_complete = False
        self.current_goal_x = None
        self.current_goal_y = None
        self.current_goal_yaw = None
        self.current_state = self.STATE_NAVIGATING_TO_ROOM

    def handle_mission_complete(self):
        """Mission ended - log final status."""
        self.stop_robot()

        if self.mission_success:
            self.get_logger().info('Mission node active - SUCCESS. Keep running or shutdown manually.')
        else:
            self.get_logger().info('Mission node active - FAILED. Keep running or shutdown manually.')

        # Keep node running (user can shutdown manually)
        # Could also call self.destroy_node() to auto-shutdown


def main(args=None):
    rclpy.init(args=args)
    node = FindEdibleFood()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down Mission 2 node.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
