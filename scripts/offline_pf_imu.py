#!/usr/bin/env python3
"""
Offline Particle Filter (IMU-heading only) - Skeleton implementation

Purpose:
- Offline processing script to perform localization using a particle filter
  that uses motion commands (`/cmd_vel`) as the motion model and IMU (yaw)
  as the measurement model.

Design choices (A - IMU-only):
- Motion model: integrate linear velocity (forward) and angular velocity
  (yaw rate) from `/cmd_vel` messages. Add zero-mean Gaussian noise when
  predicting particle states.
- Measurement model: use IMU-provided yaw (orientation) as a direct noisy
  observation of heading. Particle weights computed from angular error.
- Resampling: systematic resampling when effective sample size (ESS) drops
  below threshold (e.g., N/2).

Inputs (offline):
- Ground-truth poses file (for evaluation) in the same format as the dataset
  (time x y z qx qy qz qw). Example: `traj1_go1_pose_gt.txt`.
- Pre-extracted CSVs from the ros2 bag for `/cmd_vel` and `/imu` topics.
  (See README/usage below for extraction commands.)

Outputs:
- Estimated trajectory text file in the same format as GT: time x y z qx qy qz qw

Notes:
- This script is intentionally simple and well-commented to serve as a
  skeleton. It can and should be extended later to use LiDAR scans
  (`/scan`) for a more informative measurement model.
- Later extension to a ROS2 node is straightforward: wrap the predict/update
  loop inside a rclpy node that subscribes to topics in real time.

Usage example (after extracting CSVs from bag):

python3 scripts/offline_pf_imu.py \
  --cmd_vel data/cmd_vel.csv \
  --imu data/imu.csv \
  --gt data/traj1_go1_pose_gt.txt \
  --out results/traj1_est.txt \
  --n_particles 500

Helper: how to extract CSVs from a ros2 bag (offline):
# Terminal A: play bag (with clock)
ros2 bag play /path/to/trajectory1.db3 --clock
# Terminal B: while bag plays, dump topics to files
ros2 topic echo -p /cmd_vel > cmd_vel.csv
ros2 topic echo -p /imu > imu.csv
# stop bag play when done

The `-p` flag writes messages in a plain CSV-like representation; you may need
post-processing depending on exact output format. The script's CSV reader is
flexible (it will try to parse numeric values from each line).

"""

import argparse
import math
import random
import sys
import csv
from collections import deque
import numpy as np

# ----------------------------- Utility functions -----------------------------

def angle_normalize(z):
    """Normalize angle to [-pi, pi]."""
    return (z + math.pi) % (2 * math.pi) - math.pi


def quat_from_yaw(yaw):
    """Return quaternion (qx,qy,qz,qw) for a yaw-only rotation."""
    qx = 0.0
    qy = 0.0
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    return qx, qy, qz, qw


# ----------------------------- I/O helpers ----------------------------------

def read_gt_file(gt_path):
    """Read GT file and return list of (time, x, y, z, qx, qy, qz, qw).

    Expected whitespace-separated columns, first column is timestamp (seconds),
    then x y z qx qy qz qw.
    """
    data = []
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            if len(toks) < 8:
                continue
            t = float(toks[0])
            x = float(toks[1])
            y = float(toks[2])
            z = float(toks[3])
            qx = float(toks[4])
            qy = float(toks[5])
            qz = float(toks[6])
            qw = float(toks[7])
            data.append((t, x, y, z, qx, qy, qz, qw))
    return data


def read_cmd_vel_csv(path):
    """Read cmd_vel CSV produced by `ros2 topic echo -p /cmd_vel`.

    The file may contain entries where a line is python dict-like or CSV.
    We attempt to parse numeric tokens from each line and extract timestamp,
    linear.x and angular.z. If timestamp is not present in the dump, the
    script will simulate timestamps by assuming constant publish rate —
    however it's recommended to extract with timestamps.

    Returns list of (t, linear_x, linear_y, angular_z)
    """
    out = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        # Try to detect header-like lines; otherwise parse tokens
        for row in reader:
            if not row:
                continue
            # join row back in case ros2 printed YAML-like lines
            line = ','.join(row)
            # extract floats from line
            toks = [tok for tok in line.replace('\t', ' ').replace(':', ' ').replace(',', ' ').split()]
            nums = []
            for tok in toks:
                try:
                    nums.append(float(tok))
                except:
                    continue
            # heuristics: if we see at least 3 numbers, interpret as [t, lin_x, ang_z]
            if len(nums) >= 3:
                # Many ros2 -p dumps include header-like type info; we just map
                t = nums[0]
                linx = nums[1]
                angz = nums[2]
                # linear y maybe absent
                liny = 0.0
                if len(nums) >= 4:
                    liny = nums[2]
                    angz = nums[3] if len(nums) >= 4 else 0.0
                out.append((t, linx, liny, angz))
    # If no timestamps found, optionally create pseudo-times (not implemented)
    return out


def read_imu_csv(path):
    """Read imu CSV and return list of (t, yaw_rad).

    The CSV may contain quaternion fields or yaw directly. We try to detect
    qx,qy,qz,qw or angular_z and produce a yaw in radians.
    """
    out = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.replace('\t', ' ').replace(':', ' ').replace(',', ' ').split()
            nums = []
            for tok in toks:
                try:
                    nums.append(float(tok))
                except:
                    continue
            if not nums:
                continue
            # heuristics:
            # if 5+ numbers: assume t qx qy qz qw
            if len(nums) >= 5:
                t = nums[0]
                # find last 4 numbers as quaternion
                qx, qy, qz, qw = nums[-4], nums[-3], nums[-2], nums[-1]
                # convert quaternion to yaw
                # yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))  <-- for yaw around z
                ys = math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
                out.append((t, ys))
            elif len(nums) >= 3:
                # if provided t, ang_vel_z
                t = nums[0]
                yaw = nums[1]
                out.append((t, yaw))
    return out


# ----------------------------- Particle Filter ------------------------------

class ParticleFilter:
    def __init__(self, n_particles=500, motion_std=(0.02, 0.02, 0.01), imu_std=0.05):
        """
        n_particles: number of particles
        motion_std: tuple (std_x, std_y, std_theta) additive noise during predict
        imu_std: standard deviation (radians) for imu yaw observation model
        """
        self.N = int(n_particles)
        self.particles = np.zeros((self.N, 3))  # columns: x, y, theta
        self.weights = np.ones(self.N) / self.N
        self.motion_std = motion_std
        self.imu_std = imu_std

    def initialize(self, x0, y0, yaw0, spread=(0.1, 0.1, 0.1)):
        """Initialize particles around a given pose with gaussian spread."""
        self.particles[:, 0] = np.random.normal(x0, spread[0], size=self.N)
        self.particles[:, 1] = np.random.normal(y0, spread[1], size=self.N)
        self.particles[:, 2] = np.random.normal(yaw0, spread[2], size=self.N)
        self.weights.fill(1.0 / self.N)

    def predict(self, v, omega, dt):
        """Motion model: differential drive style integration for heading.

        x' = x + v * dt * cos(theta) + noise_x
        y' = y + v * dt * sin(theta) + noise_y
        theta' = theta + omega * dt + noise_theta
        """
        dx = v * dt
        dtheta = omega * dt
        # Add noise per particle
        nx = np.random.normal(0.0, self.motion_std[0], size=self.N)
        ny = np.random.normal(0.0, self.motion_std[1], size=self.N)
        nth = np.random.normal(0.0, self.motion_std[2], size=self.N)

        self.particles[:, 0] += (dx * np.cos(self.particles[:, 2])) + nx
        self.particles[:, 1] += (dx * np.sin(self.particles[:, 2])) + ny
        self.particles[:, 2] += dtheta + nth
        # normalize angles
        self.particles[:, 2] = np.array([angle_normalize(a) for a in self.particles[:, 2]])

    def update_imu(self, meas_yaw):
        """Weight particles by IMU yaw measurement (Gaussian in angle difference)."""
        diffs = np.array([angle_normalize(meas_yaw - th) for th in self.particles[:, 2]])
        # compute likelihoods
        var = self.imu_std ** 2
        weights = (1.0 / np.sqrt(2 * math.pi * var)) * np.exp(-0.5 * (diffs ** 2) / var)
        # multiply (Bayes) and normalize
        self.weights *= weights
        sumw = np.sum(self.weights)
        if sumw <= 0:
            # reinitialize weights uniformly to avoid NaNs
            self.weights.fill(1.0 / self.N)
        else:
            self.weights /= sumw

    def neff(self):
        return 1.0 / np.sum(self.weights ** 2)

    def systematic_resample(self):
        """Systematic resampling implementation."""
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
        # resample
        self.particles[:] = self.particles[indexes]
        self.weights.fill(1.0 / self.N)

    def estimate(self):
        """Return (x, y, yaw) estimate: weighted average for x,y; circular mean for yaw."""
        x = np.average(self.particles[:, 0], weights=self.weights)
        y = np.average(self.particles[:, 1], weights=self.weights)
        # circular mean for yaw
        sin_mean = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
        cos_mean = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
        yaw = math.atan2(sin_mean, cos_mean)
        return x, y, yaw


# ----------------------------- Main processing ------------------------------

def main():
    parser = argparse.ArgumentParser(description='Offline PF (IMU-only)')
    parser.add_argument('--cmd_vel', required=True, help='CSV file of cmd_vel (time, linear.x, linear.y, angular.z)')
    parser.add_argument('--imu', required=True, help='CSV file of IMU (time, quaternion or yaw)')
    parser.add_argument('--gt', required=True, help='GT pose txt file')
    parser.add_argument('--out', default='traj_est.txt', help='Output estimated trajectory (GT format)')
    parser.add_argument('--n_particles', type=int, default=500)
    parser.add_argument('--motion_std', type=float, nargs=3, default=(0.02, 0.02, 0.01), help='Motion noise std (x,y,theta)')
    parser.add_argument('--imu_std', type=float, default=0.05, help='IMU yaw noise std (radians)')
    parser.add_argument('--resample_threshold', type=float, default=0.5, help='ESS threshold fraction of N to trigger resampling')

    args = parser.parse_args()

    gt = read_gt_file(args.gt)
    if not gt:
        print('GT file empty or not readable:', args.gt)
        sys.exit(1)

    cmd_vel = read_cmd_vel_csv(args.cmd_vel)
    imu = read_imu_csv(args.imu)

    if not imu:
        print('IMU file empty or not readable:', args.imu)
        sys.exit(1)

    # Build indexable structures (deques) for cmd_vel
    cmd_q = deque(cmd_vel)

    pf = ParticleFilter(n_particles=args.n_particles, motion_std=args.motion_std, imu_std=args.imu_std)

    # Initialize particles at first GT pose (conservative spread)
    t0, x0, y0, z0, qx, qy, qz, qw = gt[0]
    # compute yaw from quaternion
    yaw0 = math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
    pf.initialize(x0, y0, yaw0, spread=(0.05, 0.05, 0.1))

    estimates = []

    # Process imu measurements sequentially: at each imu time, predict using latest cmd_vel and then update with imu yaw
    last_time = imu[0][0]
    cmd_last = (0.0, 0.0, 0.0, 0.0)
    cmd_idx = 0

    # Convert cmd_vel list to array for easier lookup
    cmd_arr = list(cmd_vel)

    for (t_imu, yaw_meas) in imu:
        dt = t_imu - last_time
        if dt < 0:
            dt = 0.0
        last_time = t_imu

        # find latest cmd_vel whose timestamp <= t_imu
        v = 0.0
        omega = 0.0
        # Simple scan from end for efficiency if timestamps are increasing
        while cmd_idx < len(cmd_arr) and cmd_arr[cmd_idx][0] <= t_imu:
            cmd_last = cmd_arr[cmd_idx]
            cmd_idx += 1
        if cmd_last:
            # cmd_last is (t, linx, liny, angz)
            v = cmd_last[1]
            omega = cmd_last[3] if len(cmd_last) > 3 else 0.0

        # Predict step
        pf.predict(v, omega, dt)

        # Update with IMU yaw
        pf.update_imu(yaw_meas)

        # Resample if necessary
        ess = pf.neff()
        if ess < args.resample_threshold * pf.N:
            pf.systematic_resample()

        # Record estimate
        x_est, y_est, yaw_est = pf.estimate()
        qx, qy, qz, qw = quat_from_yaw(yaw_est)
        # Save in GT-like format: time x y z qx qy qz qw
        estimates.append((t_imu, x_est, y_est, 0.0, qx, qy, qz, qw))

    # Write estimates to output file
    with open(args.out, 'w') as f:
        for e in estimates:
            f.write('{:.9f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(*e))

    print('Done. Wrote estimates to', args.out)


if __name__ == '__main__':
    main()
