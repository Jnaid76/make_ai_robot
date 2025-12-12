#!/usr/bin/env python3
"""
Hyperparameter grid search for LiDAR-only PF
Runs offline_pf_lidar_only.py with various combinations of:
  - scan_std: [0.1, 0.25, 0.5, 1.0]
  - n_beams: [36, 72, 144]
  - n_particles: [500, 1000, 2000]

Collects ATE results and produces a summary table.
"""

import subprocess
import sys
import os
import re

# Grid search parameters
scan_stds = [0.1, 0.25, 0.5, 1.0]
n_beams_list = [36, 72, 144]
n_particles_list = [500, 1000, 2000]

bag_path = "data/trajectory1/trajectory1/trajectory1.db3"
gt_path = "data/trajectory1/trajectory1/traj1_go1_pose_gt.txt"
results_dir = "results"
script_path = "scripts/offline_pf_lidar_only.py"
compare_script = "scripts/compare_trajectories.py"

# ensure results dir exists
os.makedirs(results_dir, exist_ok=True)

results = []

total = len(scan_stds) * len(n_beams_list) * len(n_particles_list)
counter = 0

print(f"Running {total} combinations...")
print("=" * 100)

for scan_std in scan_stds:
    for n_beams in n_beams_list:
        for n_particles in n_particles_list:
            counter += 1
            out_file = f"{results_dir}/tuning_scan_std{scan_std}_beams{n_beams}_particles{n_particles}.txt"
            
            # run PF
            cmd = [
                "python3", script_path,
                "--bag", bag_path,
                "--cmd_topic", "/cmd_vel",
                "--scan_topic", "/scan",
                "--out", out_file,
                "--n_particles", str(n_particles),
                "--n_beams", str(n_beams),
                "--scan_std", str(scan_std)
            ]
            
            print(f"[{counter}/{total}] scan_std={scan_std}, n_beams={n_beams}, n_particles={n_particles}...", end=" ")
            sys.stdout.flush()
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode != 0:
                    print(f"FAILED (PF returned {result.returncode})")
                    continue
            except subprocess.TimeoutExpired:
                print("TIMEOUT")
                continue
            except Exception as e:
                print(f"ERROR: {e}")
                continue
            
            # evaluate
            cmd_compare = ["python3", compare_script, gt_path, out_file]
            try:
                result = subprocess.run(cmd_compare, capture_output=True, text=True, timeout=300)
                output = result.stdout
                
                # parse ATE from output
                ate_match = re.search(r'ATE \(Absolute Trajectory Error\): ([\d.]+)', output)
                ate_x_match = re.search(r'ATE_x: ([\d.]+)', output)
                ate_y_match = re.search(r'ATE_y: ([\d.]+)', output)
                mean_err_match = re.search(r'Mean position error: ([\d.]+)', output)
                
                if ate_match:
                    ate = float(ate_match.group(1))
                    ate_x = float(ate_x_match.group(1)) if ate_x_match else None
                    ate_y = float(ate_y_match.group(1)) if ate_y_match else None
                    mean_err = float(mean_err_match.group(1)) if mean_err_match else None
                    
                    results.append({
                        'scan_std': scan_std,
                        'n_beams': n_beams,
                        'n_particles': n_particles,
                        'ate': ate,
                        'ate_x': ate_x,
                        'ate_y': ate_y,
                        'mean_err': mean_err
                    })
                    print(f"ATE={ate:.4f}")
                else:
                    print("PARSE ERROR")
            except Exception as e:
                print(f"COMPARE ERROR: {e}")

print("=" * 100)
print("\nRESULTS SUMMARY")
print("=" * 100)

if not results:
    print("No successful runs.")
    sys.exit(1)

# sort by ATE (ascending)
results.sort(key=lambda x: x['ate'])

# print table header
print(f"{'Rank':<6} {'scan_std':<12} {'n_beams':<10} {'n_particles':<12} {'ATE':<10} {'ATE_x':<10} {'ATE_y':<10} {'Mean Err':<10}")
print("-" * 100)

for rank, r in enumerate(results, 1):
    print(f"{rank:<6} {r['scan_std']:<12.2f} {r['n_beams']:<10} {r['n_particles']:<12} {r['ate']:<10.4f} {r['ate_x']:<10.4f} {r['ate_y']:<10.4f} {r['mean_err']:<10.4f}")

print("-" * 100)
print(f"\nBest result: scan_std={results[0]['scan_std']}, n_beams={results[0]['n_beams']}, n_particles={results[0]['n_particles']}")
print(f"  ATE={results[0]['ate']:.4f} m")
print(f"  File: {results_dir}/tuning_scan_std{results[0]['scan_std']}_beams{results[0]['n_beams']}_particles{results[0]['n_particles']}.txt")
