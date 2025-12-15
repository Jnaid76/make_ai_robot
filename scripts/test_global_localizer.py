#!/usr/bin/env python3

"""
Test global_localizer_node.py with bag replay and compute ATE

This script:
1. Plays back a ROS2 bag file
2. Records estimated poses from /go1_pose
3. Compares with ground truth from /ground_truth/state
4. Computes ATE (Absolute Trajectory Error)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
import sys
from pathlib import Path


class TrajectoryRecorder(Node):
    def __init__(self):
        super().__init__('trajectory_recorder')
        
        # Subscribe to estimated pose and ground truth
        self.pose_sub = self.create_subscription(
            PoseStamped, '/go1_pose', self.pose_callback, 10
        )
        self.gt_sub = self.create_subscription(
            Odometry, '/ground_truth/state', self.gt_callback, 10
        )
        
        # Storage for trajectories
        self.estimated_poses = []
        self.ground_truth_poses = []
        
        self.get_logger().info('Trajectory recorder initialized')
        self.get_logger().info('Recording from /go1_pose and /ground_truth/state')
        
    def pose_callback(self, msg):
        """Record estimated pose"""
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        
        self.estimated_poses.append([timestamp, x, y, z])
        
        if len(self.estimated_poses) % 100 == 0:
            self.get_logger().info(f'Recorded {len(self.estimated_poses)} estimated poses')
    
    def gt_callback(self, msg):
        """Record ground truth pose"""
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        
        self.ground_truth_poses.append([timestamp, x, y, z])
    
    def save_trajectories(self, output_dir):
        """Save trajectories to TUM format files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save estimated trajectory
        est_file = output_dir / 'estimated_trajectory.txt'
        with open(est_file, 'w') as f:
            f.write('# timestamp x y z\n')
            for pose in self.estimated_poses:
                f.write(f'{pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f} {pose[3]:.6f}\n')
        
        # Save ground truth trajectory
        gt_file = output_dir / 'ground_truth_trajectory.txt'
        with open(gt_file, 'w') as f:
            f.write('# timestamp x y z\n')
            for pose in self.ground_truth_poses:
                f.write(f'{pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f} {pose[3]:.6f}\n')
        
        self.get_logger().info(f'Saved {len(self.estimated_poses)} estimated poses to {est_file}')
        self.get_logger().info(f'Saved {len(self.ground_truth_poses)} ground truth poses to {gt_file}')
        
        return str(est_file), str(gt_file)
    
    def compute_ate(self):
        """Compute ATE between estimated and ground truth trajectories"""
        if len(self.estimated_poses) == 0 or len(self.ground_truth_poses) == 0:
            self.get_logger().error('No poses recorded!')
            return None
        
        # Convert to numpy arrays
        est_array = np.array(self.estimated_poses)
        gt_array = np.array(self.ground_truth_poses)
        
        # Align trajectories by timestamp (simple nearest neighbor)
        aligned_errors = []
        
        for est_pose in est_array:
            est_time = est_pose[0]
            est_xyz = est_pose[1:4]
            
            # Find closest ground truth pose
            time_diffs = np.abs(gt_array[:, 0] - est_time)
            closest_idx = np.argmin(time_diffs)
            
            # Skip if time difference is too large (>0.1 sec)
            if time_diffs[closest_idx] > 0.1:
                continue
            
            gt_xyz = gt_array[closest_idx, 1:4]
            
            # Compute Euclidean error
            error = np.linalg.norm(est_xyz - gt_xyz)
            aligned_errors.append(error)
        
        if len(aligned_errors) == 0:
            self.get_logger().error('No aligned poses found!')
            return None
        
        # Compute statistics
        ate = np.mean(aligned_errors)
        std = np.std(aligned_errors)
        max_error = np.max(aligned_errors)
        min_error = np.min(aligned_errors)
        
        self.get_logger().info('='*60)
        self.get_logger().info('ATE Results:')
        self.get_logger().info(f'  Mean ATE: {ate:.4f} m')
        self.get_logger().info(f'  Std Dev:  {std:.4f} m')
        self.get_logger().info(f'  Max Error: {max_error:.4f} m')
        self.get_logger().info(f'  Min Error: {min_error:.4f} m')
        self.get_logger().info(f'  Aligned poses: {len(aligned_errors)}')
        self.get_logger().info('='*60)
        
        return {
            'ate': ate,
            'std': std,
            'max': max_error,
            'min': min_error,
            'n_aligned': len(aligned_errors)
        }


def main():
    rclpy.init()
    
    recorder = TrajectoryRecorder()
    
    print("\n" + "="*60)
    print("Testing global_localizer_node.py")
    print("="*60)
    print("\nInstructions:")
    print("1. Make sure map_server is running and activated")
    print("2. Make sure global_localizer_node is running")
    print("3. Play the bag file in another terminal:")
    print("   ros2 bag play data/trajectory1/ --clock")
    print("4. Wait for bag to finish playing")
    print("5. Press Ctrl+C to stop recording and compute ATE")
    print("="*60 + "\n")
    
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        print('\n\nStopping recorder and computing ATE...\n')
        
        # Save trajectories
        output_dir = 'results/global_localizer_test'
        est_file, gt_file = recorder.save_trajectories(output_dir)
        
        # Compute ATE
        results = recorder.compute_ate()
        
        if results:
            # Save results to file
            results_file = Path(output_dir) / 'ate_results.txt'
            with open(results_file, 'w') as f:
                f.write(f"ATE Results for global_localizer_node.py\n")
                f.write(f"{'='*60}\n")
                f.write(f"Mean ATE: {results['ate']:.4f} m\n")
                f.write(f"Std Dev:  {results['std']:.4f} m\n")
                f.write(f"Max Error: {results['max']:.4f} m\n")
                f.write(f"Min Error: {results['min']:.4f} m\n")
                f.write(f"Aligned poses: {results['n_aligned']}\n")
            
            print(f"\nResults saved to: {results_file}")
            print(f"Estimated trajectory: {est_file}")
            print(f"Ground truth trajectory: {gt_file}")
    
    finally:
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
