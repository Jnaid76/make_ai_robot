#!/usr/bin/env python3

import os
from pathlib import Path
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from datetime import datetime


class DataCollector(Node):
    """
    Node for collecting training images from Gazebo simulation.
    Captures images at regular intervals and saves them to disk for labeling.
    """

    def __init__(self):
        super().__init__('data_collector')

        # Default save dir: prefer src/perception/perception/data/images/train in workspace
        # Fallbacks to installed package path or CWD/perception_data/images/train.
        default_save_dir = str(self._resolve_default_save_dir())

        # Declare parameters
        self.declare_parameter('save_dir', default_save_dir)
        self.declare_parameter('capture_interval', 0.5)  # seconds

        self.save_dir = self.get_parameter('save_dir').value
        self.capture_interval = self.get_parameter('capture_interval').value

        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.image_count = 0
        self.latest_image = None

        # Subscribe to camera
        self.rgb_sub = self.create_subscription(
            Image, '/camera_top/image', self.image_callback, 10)

        # Timer for periodic capture
        self.timer = self.create_timer(self.capture_interval, self.capture_image)

        self.get_logger().info('=' * 60)
        self.get_logger().info('Data Collector Initialized')
        self.get_logger().info(f'Saving images to: {self.save_dir}')
        self.get_logger().info(f'Capture interval: {self.capture_interval} seconds')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Place objects in Gazebo and move the robot around.')
        self.get_logger().info('Images will be captured automatically.')
        self.get_logger().info('Press Ctrl+C to stop.')
        self.get_logger().info('=' * 60)

    def _resolve_default_save_dir(self) -> Path:
        """Find a collaborator-friendly default save path."""
        pkg_file = Path(__file__).resolve()

        # 1) Try workspace src layout: <workspace>/src/perception/perception/data/images/train
        for parent in pkg_file.parents:
            if parent.name == 'install':
                ws_root = parent.parent
                candidate = ws_root / 'src' / 'perception' / 'perception' / 'data' / 'images' / 'train'
                return candidate

        # 2) Fallback to installed package location
        installed = pkg_file.parent / 'data' / 'images' / 'train'
        if installed.exists():
            return installed

        # 3) Last resort: relative to current working directory
        return Path.cwd() / 'perception_data' / 'images' / 'train'

    def image_callback(self, msg):
        """Store latest image"""
        self.latest_image = msg

    def capture_image(self):
        """Capture and save the latest image"""
        if self.latest_image is None:
            self.get_logger().warn('No image received yet...', throttle_duration_sec=5.0)
            return

        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, 'bgr8')

            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f'img_{timestamp}_{self.image_count:04d}.jpg'
            filepath = os.path.join(self.save_dir, filename)

            # Save image
            cv2.imwrite(filepath, cv_image)
            self.image_count += 1

            self.get_logger().info(
                f'✓ Saved image {self.image_count}: {filename} '
                f'({cv_image.shape[1]}x{cv_image.shape[0]})'
            )

        except Exception as e:
            self.get_logger().error(f'Error saving image: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('=' * 60)
        node.get_logger().info(f'Data collection complete!')
        node.get_logger().info(f'Total images collected: {node.image_count}')
        node.get_logger().info(f'Saved to: {node.save_dir}')
        node.get_logger().info('Next step: Label these images using LabelImg or Roboflow')
        node.get_logger().info('=' * 60)
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            # Context may already be shutdown by launch system; ignore.
            pass


if __name__ == '__main__':
    main()
