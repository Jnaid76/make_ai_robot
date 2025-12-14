#!/usr/bin/env python3

"""
ROS2 global localizer node with Monte Carlo Localization (MCL)

This node implements a particle filter for global localization.
It publishes tf from map to odom frame and /go1_pose topic.

Transform chain:
- odom_localizer: tf from odom to base frame (ICP-based)
- global_localizer: tf from map to odom frame (particle filter)
- combined: tf from map to base frame

The particle filter uses:
- LiDAR scans for measurement updates
- Odometry for motion prediction
- Pre-computed likelihood field from occupancy grid map
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import TransformStamped, PoseStamped
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import OccupancyGrid
import tf_transformations
from tf2_ros import TransformBroadcaster, Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from rclpy.time import Time
from rosgraph_msgs.msg import Clock

from utils import (
    pose_to_matrix, transform_to_matrix, scan_to_pcd,
    initialize_particles, normalize_weights, resample_particles,
    estimate_pose_from_particles, predict_particles,
    transform_scan_to_map, compute_likelihood_field, compute_scan_likelihood
)

import numpy as np


class GlobalLocalizerNode(Node):
    def __init__(self):
        super().__init__('global_localizer',
                        allow_undeclared_parameters=True,
                        automatically_declare_parameters_from_overrides=True)
        self.get_logger().info('Global localizer node with MCL initialized')

        # ===== TF setup =====
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ===== QoS profile for map (transient local for late joiners) =====
        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # ===== Sensor subscriptions =====
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.imu_sub = self.create_subscription(Imu, '/imu_plugin/out', self.imu_callback, 10)

        # ===== Publishers =====
        self.pose_pub = self.create_publisher(PoseStamped, '/go1_pose', 10)

        # ===== Sensor data storage =====
        self.latest_scan = None
        self.latest_imu = None
        self.occupancy_grid = None
        self.likelihood_field = None

        # ===== Get initial pose from launch parameters =====
        # Parameters are already declared by automatically_declare_parameters_from_overrides
        x_init = self.get_parameter_or('x', rclpy.Parameter('x', rclpy.Parameter.Type.DOUBLE, 0.0)).value
        y_init = self.get_parameter_or('y', rclpy.Parameter('y', rclpy.Parameter.Type.DOUBLE, 1.0)).value
        yaw_init = self.get_parameter_or('yaw', rclpy.Parameter('yaw', rclpy.Parameter.Type.DOUBLE, 0.0)).value
        self.z = 0.33  # Height of robot base when standing

        self.get_logger().info(f'Initial pose: x={x_init}, y={y_init}, yaw={yaw_init}')

        # ===== Particle filter parameters =====
        self.num_particles = 500
        self.initial_variance = [0.5, 0.5, 0.2]  # [var_x, var_y, var_theta]

        # Motion model noise
        self.motion_noise = {
            'trans': 0.05,  # Proportional to translation
            'rot': 0.05     # Proportional to rotation
        }

        # Sensor model parameters
        self.sensor_params = {
            'z_hit': 0.95,
            'z_rand': 0.05,
            'sigma_hit': 0.2,
            'max_range': 30.0
        }

        # ===== Initialize particle filter =====
        initial_pose = [x_init, y_init, yaw_init]
        self.particles = initialize_particles(self.num_particles, initial_pose, self.initial_variance)
        self.get_logger().info(f'Initialized {self.num_particles} particles')

        # ===== State variables =====
        self.last_odom_base = None
        self.T_map_to_odom = None
        self.initialized = False
        self.update_count = 0

        # ===== Create timer for particle filter updates =====
        self.interval_tf_pub = 0.1  # 10 Hz
        self.tf_timer = self.create_timer(self.interval_tf_pub, self.tf_callback)

    def scan_callback(self, msg):
        """Store latest scan for measurement update"""
        self.latest_scan = msg

    def imu_callback(self, msg):
        """Store latest IMU data (currently not used, but available for future)"""
        self.latest_imu = msg

    def map_callback(self, msg):
        """Store map and pre-compute likelihood field"""
        self.occupancy_grid = msg
        self.get_logger().info('Map received, computing likelihood field...')
        self.likelihood_field = compute_likelihood_field(msg)
        self.get_logger().info('Likelihood field computed successfully')

    def get_odometry_delta(self):
        """
        Get change in odometry since last update.

        Returns:
            [dx, dy, dtheta]: Change in pose in odom frame, or None if not available
        """
        try:
            # Get current odom->base transform
            current_odom_base = self.tf_buffer.lookup_transform('odom', 'base', rclpy.time.Time())

            if self.last_odom_base is None:
                self.last_odom_base = current_odom_base
                return None

            # Compute relative motion
            T_old = transform_to_matrix(self.last_odom_base.transform)
            T_new = transform_to_matrix(current_odom_base.transform)
            T_delta = np.linalg.inv(T_old) @ T_new

            # Extract 2D motion
            dx = T_delta[0, 3]
            dy = T_delta[1, 3]
            dtheta = np.arctan2(T_delta[1, 0], T_delta[0, 0])

            self.last_odom_base = current_odom_base
            return [dx, dy, dtheta]

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().debug(f'Could not get odom delta: {e}')
            return None

    def update_map_to_odom(self, estimated_pose):
        """
        Compute and store map->odom transform.

        Args:
            estimated_pose: [x, y, theta] - robot pose in map frame
        """
        try:
            # Get current odom->base transform
            odom_to_base = self.tf_buffer.lookup_transform('odom', 'base', rclpy.time.Time())
            T_odom_to_base = transform_to_matrix(odom_to_base.transform)

            # Create T_map_to_base from estimated pose
            x, y, theta = estimated_pose
            q = tf_transformations.quaternion_from_euler(0, 0, theta)
            T_map_to_base = pose_to_matrix([x, y, self.z, q[0], q[1], q[2], q[3]])

            # Compute T_map_to_odom = T_map_to_base @ T_base_to_odom
            T_base_to_odom = np.linalg.inv(T_odom_to_base)
            self.T_map_to_odom = T_map_to_base @ T_base_to_odom

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'Could not update map->odom: {e}')

    def publish_transform(self):
        """Publish map->odom transform"""
        if self.T_map_to_odom is None:
            return

        translation = self.T_map_to_odom[:3, 3]
        quaternion = tf_transformations.quaternion_from_matrix(self.T_map_to_odom)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = 'map'
        tf_msg.child_frame_id = 'odom'
        tf_msg.transform.translation.x = translation[0]
        tf_msg.transform.translation.y = translation[1]
        tf_msg.transform.translation.z = translation[2]
        tf_msg.transform.rotation.x = quaternion[0]
        tf_msg.transform.rotation.y = quaternion[1]
        tf_msg.transform.rotation.z = quaternion[2]
        tf_msg.transform.rotation.w = quaternion[3]

        self.tf_broadcaster.sendTransform(tf_msg)

    def publish_pose(self, pose):
        """
        Publish estimated pose on /go1_pose.

        Args:
            pose: [x, y, theta]
        """
        x, y, theta = pose

        msg = PoseStamped()
        # Use timestamp from latest scan for proper simulation time
        msg.header.stamp = self.latest_scan.header.stamp
        msg.header.frame_id = 'map'
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = self.z

        q = tf_transformations.quaternion_from_euler(0, 0, theta)
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]

        self.pose_pub.publish(msg)

    def tf_callback(self):
        """
        Main particle filter update loop:
        1. Prediction: Move particles based on odometry
        2. Update: Weight particles based on scan matching
        3. Resample: If effective sample size is low
        4. Estimate: Compute pose from particles
        5. Publish: Update transforms and pose
        """
        # Wait for map
        if self.occupancy_grid is None or self.likelihood_field is None:
            self.get_logger().warn('Waiting for map...', throttle_duration_sec=2.0)
            return

        # Wait for scan
        if self.latest_scan is None:
            self.get_logger().warn('Waiting for scan data...', throttle_duration_sec=2.0)
            return

        # ===== STEP 1: Prediction (Motion Update) =====
        delta_pose = self.get_odometry_delta()
        if delta_pose is not None:
            # Only predict if there was motion
            motion_magnitude = np.sqrt(delta_pose[0]**2 + delta_pose[1]**2) + abs(delta_pose[2])
            if motion_magnitude > 1e-4:
                predict_particles(self.particles, delta_pose, self.motion_noise)

        # ===== STEP 2: Update (Measurement) =====
        # Convert scan to point cloud
        scan_pcd = scan_to_pcd(self.latest_scan)

        # Subsample scan for performance (use every 5th point)
        scan_pcd = scan_pcd[::5]

        if len(scan_pcd) < 10:
            self.get_logger().warn('Not enough scan points', throttle_duration_sec=2.0)
            return

        # Update particle weights based on scan matching
        for particle in self.particles:
            # Transform scan to map frame using particle pose
            scan_in_map = transform_scan_to_map(scan_pcd, particle)

            # Compute likelihood
            likelihood = compute_scan_likelihood(scan_in_map, self.likelihood_field, self.sensor_params)

            # Update weight
            particle['weight'] *= likelihood

        # Normalize weights
        normalize_weights(self.particles)

        # ===== STEP 3: Resampling =====
        # Compute effective sample size
        weights_squared = sum(p['weight']**2 for p in self.particles)
        n_eff = 1.0 / weights_squared if weights_squared > 0 else 0

        # Resample if effective sample size is too low
        if n_eff < len(self.particles) / 2:
            self.particles = resample_particles(self.particles)

        # ===== STEP 4: Pose Estimation =====
        estimated_pose = estimate_pose_from_particles(self.particles)

        # ===== STEP 5: Update and Publish =====
        self.update_map_to_odom(estimated_pose)
        self.publish_transform()
        self.publish_pose(estimated_pose)

        # Log progress
        self.update_count += 1
        if self.update_count % 50 == 0:
            self.get_logger().info(
                f'Update {self.update_count}: pose=({estimated_pose[0]:.2f}, {estimated_pose[1]:.2f}, '
                f'{estimated_pose[2]:.2f}), n_eff={n_eff:.0f}'
            )


if __name__ == '__main__':
    rclpy.init()
    node = GlobalLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
