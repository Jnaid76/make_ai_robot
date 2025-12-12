#!/usr/bin/env python3
"""
Grid search for IMU+LiDAR fusion script over n_particles and n_beams.
Fixed scan_std=0.1 (best from earlier search). Produces a small summary table.
"""
import subprocess
import sys
import os
import re

n_beams_list = [36, 72, 144]
n_particles_list = [500, 1000, 2000]
scan_std = 0.1

bag_path = "data/trajectory1/trajectory1/trajectory1.db3"
gt_path = "data/trajectory1/trajectory1/traj1_go1_pose_gt.txt"
results_dir = "results"
script_path = "scripts/offline_pf_fusion.py"
compare_script = "scripts/compare_trajectories.py"

os.makedirs(results_dir, exist_ok=True)

results = []

total = len(n_beams_list) * len(n_particles_list)
counter = 0

for n_beams in n_beams_list:
    for n_particles in n_particles_list:
        counter += 1
        out_file = f"{results_dir}/fusion_scanstd{scan_std}_beams{n_beams}_particles{n_particles}.txt"
        cmd = [
            "python3", script_path,
            "--bag", bag_path,
            "--cmd_topic", "/cmd_vel",
            "--imu_topic", "/imu_plugin/out",
            "--scan_topic", "/scan",
            "--out", out_file,
            "--n_particles", str(n_particles),
            "--n_beams", str(n_beams),
            "--scan_std", str(scan_std)
        ]
        print(f"[{counter}/{total}] beams={n_beams}, particles={n_particles} ... ", end='')
        sys.stdout.flush()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if r.returncode != 0:
                print(f"PF failed (code {r.returncode})")
                print(r.stderr)
                continue
        except Exception as e:
            print(f"PF run error: {e}")
            continue

        # evaluate
        cmd_compare = ["python3", compare_script, gt_path, out_file]
        try:
            rv = subprocess.run(cmd_compare, capture_output=True, text=True, timeout=300)
            out = rv.stdout
            ate_match = re.search(r'ATE \(Absolute Trajectory Error\): ([\d.]+)', out)
            ate_x_match = re.search(r'ATE_x: ([\d.]+)', out)
            ate_y_match = re.search(r'ATE_y: ([\d.]+)', out)
            mean_err_match = re.search(r'Mean position error: ([\d.]+)', out)
            if ate_match:
                ate = float(ate_match.group(1))
                ate_x = float(ate_x_match.group(1)) if ate_x_match else None
                ate_y = float(ate_y_match.group(1)) if ate_y_match else None
                mean_err = float(mean_err_match.group(1)) if mean_err_match else None
                results.append({'n_beams': n_beams, 'n_particles': n_particles, 'ate': ate, 'ate_x': ate_x, 'ate_y': ate_y, 'mean_err': mean_err, 'file': out_file})
                print(f"ATE={ate:.4f}")
            else:
                print("COMPARE PARSE ERROR")
        except Exception as e:
            print(f"Compare error: {e}")

# sort and print
if not results:
    print("No successful runs")
    sys.exit(1)

results.sort(key=lambda x: x['ate'])
print("\nSummary:\n")
print(f"{'Rank':<6} {'n_beams':<8} {'n_particles':<12} {'ATE':<8} {'ATE_x':<8} {'ATE_y':<8} {'MeanErr':<8}")
print('-'*70)
for i,r in enumerate(results,1):
    print(f"{i:<6} {r['n_beams']:<8} {r['n_particles']:<12} {r['ate']:<8.4f} {r['ate_x']:<8.4f} {r['ate_y']:<8.4f} {r['mean_err']:<8.4f}")

print('\nBest:', results[0])
