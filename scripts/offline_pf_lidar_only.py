#!/usr/bin/env python3
"""
Offline Particle Filter (LiDAR-only, scan-to-scan likelihood)
Reads a rosbag2 .db3, uses /cmd_vel for motion prediction (optional) and /scan
for measurement updates. Measurement model: compare current scan (subset of beams)
transformed by each particle to the previous scan anchored at the previous filter mean
using KDTree nearest-neighbor distances.

Outputs estimates in same format as GT: time x y z qx qy qz qw
"""

import argparse
import math
import random
import sys
import numpy as np

try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rosidl_runtime_py.utilities import get_message
    from rclpy.serialization import deserialize_message
    ROSBAG2_AVAILABLE = True
except Exception as e:
    ROSBAG2_AVAILABLE = False
    _rb_err = e

from scipy.spatial import cKDTree

# minimal utilities

def angle_normalize(z):
    return (z + math.pi) % (2 * math.pi) - math.pi


def quat_from_yaw(yaw):
    qx = 0.0
    qy = 0.0
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    return qx, qy, qz, qw

class ParticleFilter:
    def __init__(self, n_particles=500, motion_std=(0.02,0.02,0.01)):
        self.N = int(n_particles)
        self.particles = np.zeros((self.N,3))  # x,y,yaw
        self.weights = np.ones(self.N) / self.N
        self.motion_std = motion_std

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

    def weight_by_scan_distances(self, pts_curr, pts_ref_kdtree, var):
        # pts_curr: (M,2) points in world frame for a particle; pts_ref_kdtree: KDTree built from reference
        # Return weights array shape (N,) computed via average NN distance per particle
        N = self.N
        M = pts_curr.shape[0]
        # compute for each particle: transform pts_curr by particle pose -> world
        # To vectorize, loop over particles (N * M might be moderate)
        dists = np.zeros(N)
        for i in range(N):
            x,y,yaw = self.particles[i]
            c = math.cos(yaw)
            s = math.sin(yaw)
            R = np.array([[c, -s],[s, c]])
            pts_world = (R @ pts_curr.T).T + np.array([x,y])
            # query kd-tree
            dd, _ = pts_ref_kdtree.query(pts_world, k=1)
            dists[i] = np.mean(dd)
        # convert mean distances to weights
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
    # ranges: array-like, indices: subset indices to sample
    r = np.array(ranges)
    if indices is None:
        idxs = np.where(np.isfinite(r) & (r>0.01))[0]
    else:
        idxs = np.array(indices)
        # filter invalid
        idxs = idxs[np.isfinite(r[idxs]) & (r[idxs]>0.01)]
    if idxs.size == 0:
        return np.zeros((0,2))
    angles = angle_min + idxs * angle_inc
    xs = r[idxs] * np.cos(angles)
    ys = r[idxs] * np.sin(angles)
    pts = np.vstack((xs, ys)).T
    return pts


def bag_to_estimates(bag_path, cmd_topic, scan_topic, n_particles, motion_std, scan_std, n_beams, resample_thresh, out_path):
    if not ROSBAG2_AVAILABLE:
        print('rosbag2_py not available:', _rb_err)
        sys.exit(1)

    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag_path, storage_id='sqlite3'), ConverterOptions('', ''))
    topics_and_types = reader.get_all_topics_and_types()
    topic_type_map = {t.name:t.type for t in topics_and_types}

    if cmd_topic not in topic_type_map:
        print(f'Warning: {cmd_topic} not in bag topics.')
    if scan_topic not in topic_type_map:
        print(f'Warning: {scan_topic} not in bag topics.')

    msg_cls = {}
    for top,typ in topic_type_map.items():
        try:
            msg_cls[top] = get_message(typ)
        except Exception:
            msg_cls[top] = None

    pf = ParticleFilter(n_particles=n_particles, motion_std=motion_std)

    initialized = False
    last_time = None
    last_cmd = (0.0,0.0,0.0)
    estimates = []
    time_offset = None
    gt_time_offset = 56.0

    # prepare beam subsample indices
    full_count = None
    subsample_idxs = None

    prev_scan_world_pts = None
    prev_mean_pose = None

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

        if topic == scan_topic:
            # process scan
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

            # compute relative time
            if time_offset is None:
                time_offset = t
            t_rel = (t - time_offset) + gt_time_offset

            # predict step using last_cmd
            if last_time is None:
                dt = 0.0
            else:
                dt = t - last_time
            last_time = t
            v = last_cmd[1] if last_cmd is not None else 0.0
            omega = last_cmd[2] if last_cmd is not None else 0.0
            if dt > 0:
                pf.predict(v, omega, dt)

            # convert scan to local points (subsample)
            pts_local = scan_to_points(ranges, angle_min, angle_inc, indices=subsample_idxs)
            if pts_local.shape[0] < 5:
                continue

            # if first scan: initialize particles and set prev_scan_world using mean pose
            if not initialized:
                # initialize at origin facing 0
                pf.initialize(0.0, 1.0, 0.0, spread=(0.1,0.1,0.5))
                initialized = True
                prev_mean_pose = pf.estimate()
                # transform prev scan points into world using prev_mean_pose
                x0,y0,yaw0 = prev_mean_pose
                c = math.cos(yaw0); s = math.sin(yaw0)
                R = np.array([[c, -s],[s, c]])
                prev_scan_world_pts = (R @ pts_local.T).T + np.array([x0,y0])
                prev_kd = cKDTree(prev_scan_world_pts)
                # record initial estimate
                x_est,y_est,yaw_est = pf.estimate()
                qx,qy,qz,qw = quat_from_yaw(yaw_est)
                estimates.append((t_rel, x_est, y_est, 0.0, qx,qy,qz,qw))
                continue

            # build KDTree from previous scan world points (already have prev_kd)
            # For each particle, transform local pts into world using particle pose and compute mean NN distance
            var = scan_std ** 2
            # weight particles
            pf.weight_by_scan_distances(pts_local, prev_kd, var)

            # resample
            if pf.neff() < resample_thresh * pf.N:
                pf.systematic_resample()

            # record estimate
            x_est,y_est,yaw_est = pf.estimate()
            qx,qy,qz,qw = quat_from_yaw(yaw_est)
            estimates.append((t_rel, x_est, y_est, 0.0, qx,qy,qz,qw))

            # update prev_scan_world using current mean pose
            prev_mean_pose = (x_est, y_est, yaw_est)
            c = math.cos(yaw_est); s = math.sin(yaw_est)
            R = np.array([[c, -s],[s, c]])
            prev_scan_world_pts = (R @ pts_local.T).T + np.array([x_est, y_est])
            prev_kd = cKDTree(prev_scan_world_pts)

    # write out estimates
    with open(out_path, 'w') as f:
        for e in estimates:
            f.write('{:.9f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(*e))
    print('Wrote estimates to', out_path)


def main():
    parser = argparse.ArgumentParser(description='Offline PF LiDAR-only (scan-to-scan)')
    parser.add_argument('--bag', required=True)
    parser.add_argument('--cmd_topic', default='/cmd_vel')
    parser.add_argument('--scan_topic', default='/scan')
    parser.add_argument('--out', default='traj_est_lidar_only.txt')
    parser.add_argument('--n_particles', type=int, default=500)
    parser.add_argument('--motion_std', type=float, nargs=3, default=(0.02,0.02,0.01))
    parser.add_argument('--scan_std', type=float, default=0.5)
    parser.add_argument('--n_beams', type=int, default=72, help='number of beams to subsample from full scan')
    parser.add_argument('--resample_thresh', type=float, default=0.5)
    args = parser.parse_args()

    bag_to_estimates(args.bag, args.cmd_topic, args.scan_topic, args.n_particles, tuple(args.motion_std), args.scan_std, args.n_beams, args.resample_thresh, args.out)

if __name__ == '__main__':
    main()
