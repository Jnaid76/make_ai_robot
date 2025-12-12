#!/usr/bin/env python3
"""
Offline Particle Filter (IMU + LiDAR fusion)
- Uses /cmd_vel for velocity commands (optional)
- Uses /imu_plugin/out for IMU orientation (used to update yaw measurement and to compute dt via timestamps)
- Uses /scan for LiDAR measurement updates using MAP-BASED RAYCAST likelihood

Outputs estimates in same format as GT: time x y z qx qy qz qw
"""

import argparse
import math
import random
import sys
import numpy as np
import os

try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rosidl_runtime_py.utilities import get_message
    from rclpy.serialization import deserialize_message
    ROSBAG2_AVAILABLE = True
except Exception as e:
    ROSBAG2_AVAILABLE = False
    _rb_err = e

from scipy.spatial import cKDTree  # For legacy scan-to-scan fallback

# Import map-based raycast module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_raycast import create_map_raycast_model

# utilities

def angle_normalize(z):
    return (z + math.pi) % (2 * math.pi) - math.pi


def quat_from_yaw(yaw):
    qx = 0.0
    qy = 0.0
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    return qx, qy, qz, qw

class ParticleFilter:
    def __init__(self, n_particles=500, motion_std=(0.02,0.02,0.01), imu_std=0.05, 
                 occupancy_map=None, raycast_sensor=None, likelihood_model=None):
        self.N = int(n_particles)
        self.particles = np.zeros((self.N,3))
        self.weights = np.ones(self.N) / self.N
        self.motion_std = motion_std
        self.imu_std = imu_std
        
        # Map-based raycast components
        self.occupancy_map = occupancy_map
        self.raycast_sensor = raycast_sensor
        self.likelihood_model = likelihood_model

    def initialize(self, x0, y0, yaw0, spread=(0.1,0.1,0.1)):
        self.particles[:,0] = np.random.normal(x0, spread[0], size=self.N)
        self.particles[:,1] = np.random.normal(y0, spread[1], size=self.N)
        self.particles[:,2] = np.random.normal(yaw0, spread[2], size=self.N)
        self.weights.fill(1.0/self.N)

    def predict(self, v, omega, dt):
        dx = v * dt
        dtheta = omega * dt
        nx = np.random.normal(0.0, self.motion_std[0], size=self.N)
        ny = np.random.normal(0.0, self.motion_std[1], size=self.N)
        nth = np.random.normal(0.0, self.motion_std[2], size=self.N)
        self.particles[:,0] += (dx * np.cos(self.particles[:,2])) + nx
        self.particles[:,1] += (dx * np.sin(self.particles[:,2])) + ny
        self.particles[:,2] += dtheta + nth
        self.particles[:,2] = np.array([angle_normalize(a) for a in self.particles[:,2]])

    def update_imu(self, meas_yaw):
        diffs = np.array([angle_normalize(meas_yaw - th) for th in self.particles[:,2]])
        var = self.imu_std ** 2
        weights = (1.0 / math.sqrt(2 * math.pi * var)) * np.exp(-0.5 * (diffs ** 2) / var)
        self.weights *= weights
        s = np.sum(self.weights)
        if s <= 0 or not np.isfinite(s):
            self.weights.fill(1.0/self.N)
        else:
            self.weights /= s
    
    def weight_by_map_raycast(self, actual_ranges, scan_angles):
        """
        Update particle weights using map-based raycast measurement model.
        
        Args:
            actual_ranges: Actual LiDAR ranges (np.array)
            scan_angles: Scan angles relative to robot frame (np.array)
        """
        if self.raycast_sensor is None or self.likelihood_model is None:
            print("Warning: Raycast components not initialized, skipping map update")
            return
        
        # Compute expected ranges for each particle
        expected_ranges_list = []
        for i in range(self.N):
            particle_pose = self.particles[i]  # [x, y, yaw]
            expected_ranges = self.raycast_sensor.cast_rays(particle_pose, scan_angles)
            expected_ranges_list.append(expected_ranges)
        
        # Compute weights using likelihood model
        new_weights = self.likelihood_model.compute_weights(actual_ranges, expected_ranges_list)
        
        # Multiply with existing weights
        self.weights *= new_weights
        
        # Normalize
        s = np.sum(self.weights)
        if s <= 0 or not np.isfinite(s):
            self.weights.fill(1.0/self.N)
        else:
            self.weights /= s

    def weight_by_scan_distances(self, pts_local, pts_ref_kdtree, var):
        N = self.N
        dists = np.zeros(N)
        # transform pts_local by each particle and query kd-tree
        for i in range(N):
            x,y,yaw = self.particles[i]
            c = math.cos(yaw); s = math.sin(yaw)
            R = np.array([[c, -s],[s, c]])
            pts_world = (R @ pts_local.T).T + np.array([x,y])
            dd, _ = pts_ref_kdtree.query(pts_world, k=1)
            dists[i] = np.mean(dd)
        weights = np.exp(-0.5 * (dists**2) / var)
        self.weights *= weights
        s = np.sum(self.weights)
        if s <= 0 or not np.isfinite(s):
            self.weights.fill(1.0/self.N)
        else:
            self.weights /= s

    def neff(self):
        return 1.0 / np.sum(self.weights**2)

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
        self.weights.fill(1.0/self.N)

    def estimate(self):
        x = np.average(self.particles[:,0], weights=self.weights)
        y = np.average(self.particles[:,1], weights=self.weights)
        sin_mean = np.average(np.sin(self.particles[:,2]), weights=self.weights)
        cos_mean = np.average(np.cos(self.particles[:,2]), weights=self.weights)
        yaw = math.atan2(sin_mean, cos_mean)
        return x,y,yaw


def scan_to_points(ranges, angle_min, angle_inc, indices=None):
    r = np.array(ranges)
    if indices is None:
        idxs = np.where(np.isfinite(r) & (r>0.01))[0]
    else:
        idxs = np.array(indices)
        idxs = idxs[np.isfinite(r[idxs]) & (r[idxs]>0.01)]
    if idxs.size == 0:
        return np.zeros((0,2))
    angles = angle_min + idxs * angle_inc
    xs = r[idxs] * np.cos(angles)
    ys = r[idxs] * np.sin(angles)
    pts = np.vstack((xs, ys)).T
    return pts


def bag_to_estimates(bag_path, cmd_topic, imu_topic, scan_topic, n_particles, motion_std, imu_std, scan_std, n_beams, resample_thresh, out_path, map_yaml_path=None):
    if not ROSBAG2_AVAILABLE:
        print('rosbag2_py not available:', _rb_err)
        sys.exit(1)

    # Initialize map-based raycast if map provided
    occupancy_map, raycast_sensor, likelihood_model = None, None, None
    if map_yaml_path and os.path.exists(map_yaml_path):
        print(f"[INFO] Loading map from {map_yaml_path} for raycast-based localization")
        occupancy_map, raycast_sensor, likelihood_model = create_map_raycast_model(
            map_yaml_path,
            max_range=30.0,
            sigma_hit=scan_std,
            n_beams=n_beams
        )
        print("[INFO] Map-based raycast initialized successfully")
    else:
        print("[WARNING] No map provided, using legacy scan-to-scan method (not recommended)")

    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag_path, storage_id='sqlite3'), ConverterOptions('', ''))
    topics_and_types = reader.get_all_topics_and_types()
    topic_type_map = {t.name:t.type for t in topics_and_types}

    for top in (cmd_topic, imu_topic, scan_topic):
        if top not in topic_type_map:
            print(f'Warning: {top} not in bag topics.')

    msg_cls = {}
    for top,typ in topic_type_map.items():
        try:
            msg_cls[top] = get_message(typ)
        except Exception:
            msg_cls[top] = None

    pf = ParticleFilter(n_particles=n_particles, motion_std=motion_std, imu_std=imu_std,
                       occupancy_map=occupancy_map, raycast_sensor=raycast_sensor, 
                       likelihood_model=likelihood_model)

    initialized = False
    last_time = None
    last_cmd = (0.0,0.0,0.0)
    estimates = []
    time_offset = None
    gt_time_offset = 56.0

    full_count = None
    subsample_idxs = None

    prev_scan_world_pts = None
    prev_kd = None

    while reader.has_next():
        topic, serialized_msg, t_ns = reader.read_next()
        t = t_ns * 1e-9
        msg_c = msg_cls.get(topic)
        if msg_c is None:
            continue
        try:
            msg = deserialize_message(serialized_msg, msg_c)
        except Exception:
            continue

        if topic == cmd_topic:
            try:
                v = float(msg.linear.x)
                omega = float(msg.angular.z)
            except Exception:
                v = 0.0
                omega = 0.0
            last_cmd = (t, v, omega)
            continue

        if topic == imu_topic:
            # extract yaw from orientation
            try:
                qx = msg.orientation.x
                qy = msg.orientation.y
                qz = msg.orientation.z
                qw = msg.orientation.w
                yaw = math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
            except Exception:
                continue

            if time_offset is None:
                time_offset = t
            t_rel = (t - time_offset) + gt_time_offset

            if not initialized:
                pf.initialize(0.01, 0.98, yaw, spread=(0.05,0.05,0.2))
                initialized = True
                last_time = t
                x_est,y_est,yaw_est = pf.estimate()
                qx_est,qy_est,qz_est,qw_est = quat_from_yaw(yaw_est)
                estimates.append((t_rel, x_est, y_est, 0.0, qx_est,qy_est,qz_est,qw_est))
                continue

            dt = t - last_time
            if dt < 0:
                dt = 0.0
            last_time = t

            # predict using last cmd
            v = last_cmd[1] if last_cmd is not None else 0.0
            omega = last_cmd[2] if last_cmd is not None else 0.0
            pf.predict(v, omega, dt)

            # imu yaw update
            pf.update_imu(yaw)

            x_est,y_est,yaw_est = pf.estimate()
            qx_est,qy_est,qz_est,qw_est = quat_from_yaw(yaw_est)
            estimates.append((t_rel, x_est, y_est, 0.0, qx_est,qy_est,qz_est,qw_est))
            continue

        if topic == scan_topic:
            try:
                ranges = np.array(msg.ranges)
                angle_min = float(msg.angle_min)
                angle_inc = float(msg.angle_increment)
            except Exception:
                continue

            if full_count is None:
                full_count = len(ranges)
                if n_beams >= full_count:
                    subsample_idxs = np.arange(full_count)
                else:
                    subsample_idxs = np.linspace(0, full_count-1, n_beams, dtype=int)

            if time_offset is None:
                time_offset = t
            t_rel = (t - time_offset) + gt_time_offset

            # predict using last_cmd with dt since last_time
            if last_time is None:
                dt = 0.0
            else:
                dt = t - last_time
            last_time = t
            v = last_cmd[1] if last_cmd is not None else 0.0
            omega = last_cmd[2] if last_cmd is not None else 0.0
            if dt > 0:
                pf.predict(v, omega, dt)

            # if first scan and not initialized, init (should be initialized by IMU normally)
            if not initialized:
                pf.initialize(0.01, 0.98, 0.0, spread=(0.1,0.1,0.5))
                initialized = True

            # Use map-based raycast if available, otherwise fall back to scan-to-scan
            if raycast_sensor is not None:
                # Map-based measurement update
                # Compute scan angles for subsampled beams
                scan_angles = angle_min + subsample_idxs * angle_inc
                actual_ranges = ranges[subsample_idxs]
                
                # Filter valid ranges
                valid_mask = np.isfinite(actual_ranges) & (actual_ranges > 0.01) & (actual_ranges < 30.0)
                if np.sum(valid_mask) < 5:
                    # Not enough valid beams, skip this scan
                    x_est,y_est,yaw_est = pf.estimate()
                    qx,qy,qz,qw = quat_from_yaw(yaw_est)
                    estimates.append((t_rel, x_est, y_est, 0.0, qx,qy,qz,qw))
                    continue
                
                # Update weights using map raycast
                pf.weight_by_map_raycast(actual_ranges, scan_angles)
            else:
                # Legacy scan-to-scan method
                pts_local = scan_to_points(ranges, angle_min, angle_inc, indices=subsample_idxs)
                if pts_local.shape[0] < 5:
                    continue

                # if no prev_kd, build from current mean pose
                if prev_kd is None:
                    x_est,y_est,yaw_est = pf.estimate()
                    c = math.cos(yaw_est); s = math.sin(yaw_est)
                    R = np.array([[c, -s],[s, c]])
                    prev_scan_world_pts = (R @ pts_local.T).T + np.array([x_est,y_est])
                    prev_kd = cKDTree(prev_scan_world_pts)
                    # write current estimate
                    x_est,y_est,yaw_est = pf.estimate()
                    qx,qy,qz,qw = quat_from_yaw(yaw_est)
                    estimates.append((t_rel, x_est, y_est, 0.0, qx,qy,qz,qw))
                    continue

                # weight by scan distances
                var = scan_std ** 2
                pf.weight_by_scan_distances(pts_local, prev_kd, var)

                # update prev_kd using current mean
                x_est,y_est,yaw_est = pf.estimate()
                c = math.cos(yaw_est); s = math.sin(yaw_est)
                R = np.array([[c, -s],[s, c]])
                prev_scan_world_pts = (R @ pts_local.T).T + np.array([x_est,y_est])
                prev_kd = cKDTree(prev_scan_world_pts)

            # Resample if needed
            if pf.neff() < resample_thresh * pf.N:
                pf.systematic_resample()

            # Estimate and record
            x_est,y_est,yaw_est = pf.estimate()
            qx,qy,qz,qw = quat_from_yaw(yaw_est)
            estimates.append((t_rel, x_est, y_est, 0.0, qx,qy,qz,qw))

    # write out estimates
    with open(out_path, 'w') as f:
        for e in estimates:
            f.write('{:.9f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(*e))
    print('Wrote estimates to', out_path)


def main():
    parser = argparse.ArgumentParser(description='Offline PF IMU+LiDAR fusion with MAP-BASED RAYCAST (scan_std=0.15, n_particles=500, n_beams=144)')
    parser.add_argument('--bag', required=True)
    parser.add_argument('--map_yaml', default='src/go1_simulation/maps/hospital.yaml', 
                       help='Path to map .yaml file for raycast-based localization')
    parser.add_argument('--cmd_topic', default='/cmd_vel')
    parser.add_argument('--imu_topic', default='/imu_plugin/out')
    parser.add_argument('--scan_topic', default='/scan')
    parser.add_argument('--out', default='traj_est_imu_lidar.txt')
    parser.add_argument('--n_particles', type=int, default=500)
    parser.add_argument('--motion_std', type=float, nargs=3, default=(0.02,0.02,0.01))
    parser.add_argument('--imu_std', type=float, default=0.05)
    parser.add_argument('--scan_std', type=float, default=0.15)
    parser.add_argument('--n_beams', type=int, default=144)
    parser.add_argument('--resample_thresh', type=float, default=0.5)
    args = parser.parse_args()

    bag_to_estimates(args.bag, args.cmd_topic, args.imu_topic, args.scan_topic, args.n_particles, tuple(args.motion_std), args.imu_std, args.scan_std, args.n_beams, args.resample_thresh, args.out, args.map_yaml)

if __name__ == '__main__':
    main()
