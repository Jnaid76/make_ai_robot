#!/usr/bin/env python3

"""
Trajectory recording script for localization evaluation.

This script subscribes to /go1_pose and records the trajectory to a text file
in the format required for submission: timestamp x y z qx qy qz qw

Usage:
    ros2 run localization record_trajectory.py <output_file>

Example:
    ros2 run localization record_trajectory.py trajectory_1.txt
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import sys


class TrajectoryRecorder(Node):
    def __init__(self, output_file):
        super().__init__('trajectory_recorder')
        self.output_file = output_file
        self.poses = []

        # Subscribe to /go1_pose
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/go1_pose',
            self.pose_callback,
            10
        )

        self.get_logger().info(f'Recording trajectory to {output_file}')
        self.get_logger().info('Press Ctrl+C to stop and save')

    def pose_callback(self, msg):
        """Record pose with timestamp"""
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w

        # Format: timestamp x y z qx qy qz qw
        line = f"{timestamp} {x} {y} {z} {qx} {qy} {qz} {qw}\n"
        self.poses.append(line)

        # Log progress every 100 poses
        if len(self.poses) % 100 == 0:
            self.get_logger().info(f'Recorded {len(self.poses)} poses')

    def save(self):
        """Save recorded trajectory to file"""
        if len(self.poses) == 0:
            self.get_logger().warn('No poses recorded!')
            return

        with open(self.output_file, 'w') as f:
            f.writelines(self.poses)

        self.get_logger().info(f'Saved {len(self.poses)} poses to {self.output_file}')


def main():
    if len(sys.argv) < 2:
        print("Usage: ros2 run localization record_trajectory.py <output_file>")
        print("Example: ros2 run localization record_trajectory.py trajectory_1.txt")
        sys.exit(1)

    output_file = sys.argv[1]

    rclpy.init()
    recorder = TrajectoryRecorder(output_file)

    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        print('\nStopping recording...')
    finally:
        recorder.save()
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
