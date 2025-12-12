#!/usr/bin/env python3
"""Simple trajectory comparison script (without evo dependency)"""

import sys
import math
import numpy as np

def read_trajectory(filepath):
    """Read trajectory file (time x y z qx qy qz qw)"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            if len(toks) < 8:
                continue
            try:
                t = float(toks[0])
                x = float(toks[1])
                y = float(toks[2])
                z = float(toks[3])
                data.append((t, x, y, z))
            except:
                continue
    return np.array(data)

def synchronize_trajectories(gt_data, est_data):
    """Synchronize by matching timestamps (nearest neighbor)
    
    If GT times are small (relative) and EST times are large (Unix timestamp),
    we compute an offset to align them.
    """
    gt_times = gt_data[:, 0]
    est_times = est_data[:, 0]
    
    # Check if we need to apply time offset
    # If GT times are small (< 1000) and EST times are large (> 1000000), assume relative vs absolute
    if np.max(gt_times) < 1000 and np.min(est_times) > 1000000:
        # Compute offset: offset = est_t0 - gt_t0
        offset = est_times[0] - gt_times[0]
        gt_times_adjusted = gt_times + offset
    else:
        gt_times_adjusted = gt_times
    
    # For each GT timestamp, find closest estimated timestamp
    synced_gt = []
    synced_est = []
    
    for i, gt_t in enumerate(gt_times_adjusted):
        # Find closest est_t
        idx = np.argmin(np.abs(est_times - gt_t))
        if abs(est_times[idx] - gt_t) < 1.0:  # within 1 second
            synced_gt.append(gt_data[i])
            synced_est.append(est_data[idx])
    
    return np.array(synced_gt), np.array(synced_est)

def compute_metrics(gt_data, est_data):
    """Compute simple metrics: ATE (Absolute Trajectory Error)"""
    # Compute position differences
    diffs = gt_data[:, 1:4] - est_data[:, 1:4]
    
    # ATE = RMSE of position differences
    ate = np.sqrt(np.mean(np.sum(diffs**2, axis=1)))
    
    # Per-axis errors
    ate_x = np.sqrt(np.mean(diffs[:, 0]**2))
    ate_y = np.sqrt(np.mean(diffs[:, 1]**2))
    ate_z = np.sqrt(np.mean(diffs[:, 2]**2))
    
    # Max errors
    max_err = np.max(np.linalg.norm(diffs, axis=1))
    min_err = np.min(np.linalg.norm(diffs, axis=1))
    mean_err = np.mean(np.linalg.norm(diffs, axis=1))
    
    return {
        'ATE': ate,
        'ATE_x': ate_x,
        'ATE_y': ate_y,
        'ATE_z': ate_z,
        'max_error': max_err,
        'min_error': min_err,
        'mean_error': mean_err,
        'n_synced': len(gt_data)
    }

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 script.py <gt_file> <est_file>')
        sys.exit(1)
    
    gt_file = sys.argv[1]
    est_file = sys.argv[2]
    
    print(f"Reading GT: {gt_file}")
    gt = read_trajectory(gt_file)
    print(f"  -> {len(gt)} poses")
    
    print(f"Reading EST: {est_file}")
    est = read_trajectory(est_file)
    print(f"  -> {len(est)} poses")
    
    print("\nSynchronizing trajectories...")
    gt_sync, est_sync = synchronize_trajectories(gt, est)
    print(f"  -> {len(gt_sync)} synchronized poses")
    
    if len(gt_sync) == 0:
        print("No synchronized poses found!")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("TRAJECTORY COMPARISON METRICS")
    print("="*60)
    
    metrics = compute_metrics(gt_sync, est_sync)
    print(f"ATE (Absolute Trajectory Error): {metrics['ATE']:.4f} m")
    print(f"  ATE_x: {metrics['ATE_x']:.4f} m")
    print(f"  ATE_y: {metrics['ATE_y']:.4f} m")
    print(f"  ATE_z: {metrics['ATE_z']:.4f} m")
    print(f"Mean position error: {metrics['mean_error']:.4f} m")
    print(f"Min position error:  {metrics['min_error']:.4f} m")
    print(f"Max position error:  {metrics['max_error']:.4f} m")
    print(f"Number of synchronized poses: {metrics['n_synced']}")
    print("="*60)
