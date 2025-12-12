#!/usr/bin/env python3
"""
Offline Particle Filter (IMU-only) reading directly from a ros2 bag (.db3)

This script reads a rosbag2 SQLite3 DB (`.db3`) using `rosbag2_py.SequentialReader`,
extracts `/cmd_vel` and `/imu` messages, and runs the same simple IMU-heading-only
particle filter as the CSV-based skeleton.

Notes & requirements:
- Requires ROS2 Python packages available in the current Python environment:
  `rosbag2_py`, `rosidl_runtime_py`, `rclpy` (for serialization utilities).
- If `rosbag2_py` is not available, fall back to the CSV-based workflow.

Usage example:
  python3 scripts/offline_pf_from_bag.py --bag /path/to/trajectory1.db3 \
    --cmd_topic /cmd_vel --imu_topic /imu --out results/traj1_est_from_bag.txt

The script is commented to be easy to extend (e.g., add /scan handling later).
"""

import argparse
import math
import random
import sys
import time
import numpy as np

# try imports for bag reading and message deserialization
try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rosidl_runtime_py.utilities import get_message
    from rclpy.serialization import deserialize_message
    ROSBAG2_AVAILABLE = True
except Exception as e:
    ROSBAG2_AVAILABLE = False
    _rb_err = e

# ----------------- utility functions (same as CSV script) --------------------

def angle_normalize(z):
    return (z + math.pi) % (2 * math.pi) - math.pi


def quat_from_yaw(yaw):
    qx = 0.0
    qy = 0.0
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    return qx, qy, qz, qw


class ParticleFilter:
    def __init__(self, n_particles=500, motion_std=(0.02, 0.02, 0.01), imu_std=0.05):
        self.N = int(n_particles)
        self.particles = np.zeros((self.N, 3))
        self.weights = np.ones(self.N) / self.N
        self.motion_std = motion_std
        self.imu_std = imu_std

    def initialize(self, x0, y0, yaw0, spread=(0.1, 0.1, 0.1)):
        self.particles[:, 0] = np.random.normal(x0, spread[0], size=self.N)
        self.particles[:, 1] = np.random.normal(y0, spread[1], size=self.N)
        self.particles[:, 2] = np.random.normal(yaw0, spread[2], size=self.N)
        self.weights.fill(1.0 / self.N)

    def predict(self, v, omega, dt):
        dx = v * dt
        dtheta = omega * dt
        nx = np.random.normal(0.0, self.motion_std[0], size=self.N)
        ny = np.random.normal(0.0, self.motion_std[1], size=self.N)
        nth = np.random.normal(0.0, self.motion_std[2], size=self.N)
        self.particles[:, 0] += (dx * np.cos(self.particles[:, 2])) + nx
        self.particles[:, 1] += (dx * np.sin(self.particles[:, 2])) + ny
        self.particles[:, 2] += dtheta + nth
        self.particles[:, 2] = np.array([angle_normalize(a) for a in self.particles[:, 2]])

    def update_imu(self, meas_yaw):
        diffs = np.array([angle_normalize(meas_yaw - th) for th in self.particles[:, 2]])
        var = self.imu_std ** 2
        weights = (1.0 / math.sqrt(2 * math.pi * var)) * np.exp(-0.5 * (diffs ** 2) / var)
        self.weights *= weights
        s = np.sum(self.weights)
        if s <= 0:
            self.weights.fill(1.0 / self.N)
        else:
            self.weights /= s

    def update_scan(self, meas_yaw, scan_std=0.2):
        """Simple scan-based measurement update using scan-derived dominant heading.
        This is an approximate measurement model: we compute the principal direction
        of the scan point cloud and use the yaw difference to weight particles.
        """
        diffs = np.array([angle_normalize(meas_yaw - th) for th in self.particles[:, 2]])
        var = scan_std ** 2
        weights = (1.0 / math.sqrt(2 * math.pi * var)) * np.exp(-0.5 * (diffs ** 2) / var)
        self.weights *= weights
        s = np.sum(self.weights)
        if s <= 0:
            self.weights.fill(1.0 / self.N)
        else:
            self.weights /= s

    def neff(self):
        return 1.0 / np.sum(self.weights ** 2)

    def systematic_resample(self):
        positions = (np.arange(self.N) + random.random()) / self.N
        indexes = np.zeros(self.N, 'i')
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < self.N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles[:] = self.particles[indexes]
        self.weights.fill(1.0 / self.N)

    def estimate(self):
        x = np.average(self.particles[:, 0], weights=self.weights)
        y = np.average(self.particles[:, 1], weights=self.weights)
        sin_mean = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
        cos_mean = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
        yaw = math.atan2(sin_mean, cos_mean)
        return x, y, yaw


# ----------------- bag reading + main loop ----------------------------------

def bag_to_estimates(bag_path, cmd_topic, imu_topic, scan_topic, n_particles, motion_std, imu_std, scan_std, resample_thresh, out_path):
    if not ROSBAG2_AVAILABLE:
        print('rosbag2_py or required ROS Python utilities are not available in this environment.')
        print('Error:', _rb_err)
        print('Please run this script in a ROS2 environment with rosbag2_py installed, or use the CSV-based script.')
        sys.exit(1)

    # open bag
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    # build topic->type map
    topics_and_types = reader.get_all_topics_and_types()
    topic_type_map = {t.name: t.type for t in topics_and_types}

    # verify topics exist
    if cmd_topic not in topic_type_map:
        print(f'Warning: {cmd_topic} not in bag topics.')
    if imu_topic not in topic_type_map:
        print(f'Warning: {imu_topic} not in bag topics.')
    if scan_topic not in topic_type_map:
        print(f'Warning: {scan_topic} not in bag topics.')

    # prepare message classes for deserialization
    msg_class_cache = {}
    for top, typ in topic_type_map.items():
        try:
            msg_class_cache[top] = get_message(typ)
        except Exception:
            msg_class_cache[top] = None

    pf = ParticleFilter(n_particles=n_particles, motion_std=motion_std, imu_std=imu_std)

    # we'll initialize particles on first imu message using the bag's first timestamp and zero pose
    initialized = False
    last_time = None
    last_cmd = (0.0, 0.0, 0.0)  # (t, v, omega)
    estimates = []
    time_offset = None  # offset to convert Unix timestamp to relative time (like GT file)
    gt_time_offset = 56.0  # GT file starts at 56.0 seconds; align estimates to this reference

    # Read messages sequentially from bag
    while reader.has_next():
        (topic, serialized_msg, t_ns) = reader.read_next()
        t = t_ns * 1e-9
        # deserialize if we have a class
        msg_cls = msg_class_cache.get(topic)
        if msg_cls is None:
            continue
        try:
            msg = deserialize_message(serialized_msg, msg_cls)
        except Exception:
            # fallback: skip message if we cannot deserialize
            continue

        # handle cmd_vel
        if topic == cmd_topic:
            # geometry_msgs/msg/Twist expected
            try:
                v = float(msg.linear.x)
                omega = float(msg.angular.z)
            except Exception:
                v = 0.0
                omega = 0.0
            last_cmd = (t, v, omega)
            continue

        # handle imu
        if topic == imu_topic:
            # sensor_msgs/msg/Imu expected
            # extract orientation quaternion -> yaw
            try:
                qx = msg.orientation.x
                qy = msg.orientation.y
                qz = msg.orientation.z
                qw = msg.orientation.w
                yaw = math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
            except Exception:
                # if message does not have orientation, skip
                continue

            # On first IMU message, set time offset (convert absolute to relative time)
            if time_offset is None:
                time_offset = t  # first message time in absolute (Unix timestamp)

            # Compute relative time aligned with GT reference (GT starts at 56.0 seconds)
            t_rel = (t - time_offset) + gt_time_offset

            if not initialized:
                # initialize at origin or small perturbation — better to initialize at first GT if available
                pf.initialize(0.0, 1.0, yaw, spread=(0.05, 0.05, 0.2))
                initialized = True
                last_time = t
                # record first estimate with relative timestamp
                x_est, y_est, yaw_est = pf.estimate()
                qx_est, qy_est, qz_est, qw_est = quat_from_yaw(yaw_est)
                estimates.append((t_rel, x_est, y_est, 0.0, qx_est, qy_est, qz_est, qw_est))
                continue

            # compute dt
            dt = t - last_time
            if dt < 0:
                dt = 0.0
            last_time = t

            # use last_cmd (if available) to predict
            v = last_cmd[1] if last_cmd is not None else 0.0
            omega = last_cmd[2] if last_cmd is not None else 0.0
            pf.predict(v, omega, dt)
            pf.update_imu(yaw)

            # resample
            if pf.neff() < resample_thresh * pf.N:
                pf.systematic_resample()

            x_est, y_est, yaw_est = pf.estimate()
            qx_est, qy_est, qz_est, qw_est = quat_from_yaw(yaw_est)
            estimates.append((t_rel, x_est, y_est, 0.0, qx_est, qy_est, qz_est, qw_est))

        # handle scan (LaserScan)
        if topic == scan_topic:
            # sensor_msgs/msg/LaserScan expected
            try:
                ranges = np.array(msg.ranges)
                angle_min = float(msg.angle_min)
                angle_inc = float(msg.angle_increment)
            except Exception:
                continue

            # build xy points for valid ranges
            valid = np.isfinite(ranges) & (ranges > 0.01)
            if np.count_nonzero(valid) < 10:
                continue
            idxs = np.nonzero(valid)[0]
            angles = angle_min + idxs * angle_inc
            xs = ranges[valid] * np.cos(angles)
            ys = ranges[valid] * np.sin(angles)
            pts = np.vstack((xs, ys)).T

            # PCA to find dominant direction of scan points
            try:
                cov = np.cov(pts.T)
                eigvals, eigvecs = np.linalg.eig(cov)
                principal = eigvecs[:, np.argmax(eigvals)]
                scan_yaw = math.atan2(principal[1], principal[0])
            except Exception:
                continue

            # ensure time offset exists
            if time_offset is None:
                time_offset = t
            t_rel = (t - time_offset) + gt_time_offset

            # update PF with scan-derived yaw
            pf.update_scan(scan_yaw, scan_std=scan_std)

            # resample
            if pf.neff() < resample_thresh * pf.N:
                pf.systematic_resample()

            x_est, y_est, yaw_est = pf.estimate()
            qx_est, qy_est, qz_est, qw_est = quat_from_yaw(yaw_est)
            estimates.append((t_rel, x_est, y_est, 0.0, qx_est, qy_est, qz_est, qw_est))

    # write out estimates with relative timestamps
    with open(out_path, 'w') as f:
        for e in estimates:
            f.write('{:.9f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(*e))

    print('Wrote estimates to', out_path)


def main():
    parser = argparse.ArgumentParser(description='Offline PF reading rosbag2 .db3 directly (IMU-only)')
    parser.add_argument('--bag', required=True, help='Path to rosbag2 db3 file or directory')
    parser.add_argument('--cmd_topic', default='/cmd_vel')
    parser.add_argument('--imu_topic', default='/imu')
    parser.add_argument('--scan_topic', default='/scan')
    parser.add_argument('--out', default='traj_est_from_bag.txt')
    parser.add_argument('--n_particles', type=int, default=500)
    parser.add_argument('--motion_std', type=float, nargs=3, default=(0.02, 0.02, 0.01))
    parser.add_argument('--imu_std', type=float, default=0.05)
    parser.add_argument('--scan_std', type=float, default=0.25, help='std dev for scan-derived heading measurement')
    parser.add_argument('--resample_thresh', type=float, default=0.5)
    args = parser.parse_args()

    bag_to_estimates(args.bag, args.cmd_topic, args.imu_topic, args.scan_topic, args.n_particles, tuple(args.motion_std), args.imu_std, args.scan_std, args.resample_thresh, args.out)


if __name__ == '__main__':
    main()
