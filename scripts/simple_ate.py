#!/usr/bin/env python3
"""Simple ATE calculator"""

import numpy as np
import sys

def load_trajectory(filename):
    """Load trajectory from file (timestamp x y z format)"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                timestamp = float(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                data.append([timestamp, x, y, z])
    return np.array(data)

def compute_ate(gt_file, est_file):
    """Compute ATE between ground truth and estimated trajectories"""
    
    # Load trajectories
    print(f"Loading GT: {gt_file}")
    gt = load_trajectory(gt_file)
    print(f"  -> {len(gt)} poses")
    
    print(f"Loading EST: {est_file}")
    est = load_trajectory(est_file)
    print(f"  -> {len(est)} poses")
    
    if len(gt) == 0 or len(est) == 0:
        print("Error: Empty trajectory!")
        return
    
    # Align by timestamp
    errors = []
    for est_pose in est:
        est_time = est_pose[0]
        est_xyz = est_pose[1:4]
        
        # Find closest GT pose
        time_diffs = np.abs(gt[:, 0] - est_time)
        closest_idx = np.argmin(time_diffs)
        
        # Skip if time diff > 0.1 sec
        if time_diffs[closest_idx] > 0.1:
            continue
        
        gt_xyz = gt[closest_idx, 1:4]
        error = np.linalg.norm(est_xyz - gt_xyz)
        errors.append(error)
    
    if len(errors) == 0:
        print("Error: No aligned poses!")
        return
    
    # Compute statistics
    errors = np.array(errors)
    ate = np.mean(errors)
    std = np.std(errors)
    max_err = np.max(errors)
    min_err = np.min(errors)
    
    print("\n" + "="*60)
    print("ATE Results:")
    print("="*60)
    print(f"Mean ATE:     {ate:.4f} m")
    print(f"Std Dev:      {std:.4f} m")
    print(f"Max Error:    {max_err:.4f} m")
    print(f"Min Error:    {min_err:.4f} m")
    print(f"Aligned:      {len(errors)} poses")
    print("="*60)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 simple_ate.py <gt_file> <est_file>")
        sys.exit(1)
    
    compute_ate(sys.argv[1], sys.argv[2])
