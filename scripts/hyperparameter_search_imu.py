#!/usr/bin/env python3
"""
Grid search for IMU-only PF over n_particles.
Uses `offline_pf_from_bag.py` with scan_topic disabled.
"""
import subprocess, sys, os, re

n_particles_list = [500, 1000, 2000]
bag_path = "data/trajectory1/trajectory1/trajectory1.db3"
gt_path = "data/trajectory1/trajectory1/traj1_go1_pose_gt.txt"
results_dir = "results"
script_path = "scripts/offline_pf_from_bag.py"
compare_script = "scripts/compare_trajectories.py"

os.makedirs(results_dir, exist_ok=True)
results = []

for n in n_particles_list:
    out_file = f"{results_dir}/imu_only_particles{n}.txt"
    cmd = [
        "python3", script_path,
        "--bag", bag_path,
        "--cmd_topic", "/cmd_vel",
        "--imu_topic", "/imu_plugin/out",
        "--scan_topic", "/no_scan_topic",  # disable scan processing
        "--out", out_file,
        "--n_particles", str(n)
    ]
    print(f"Running IMU-only n_particles={n} ...", end=' ')
    sys.stdout.flush()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            print(f"PF failed: {r.returncode}")
            print(r.stderr)
            continue
    except Exception as e:
        print("PF run error:", e)
        continue
    # compare
    try:
        rc = subprocess.run(["python3", compare_script, gt_path, out_file], capture_output=True, text=True, timeout=300)
        out = rc.stdout
        m = re.search(r'ATE \(Absolute Trajectory Error\): ([0-9.]+)', out)
        mx = re.search(r'ATE_x: ([0-9.]+)', out)
        my = re.search(r'ATE_y: ([0-9.]+)', out)
        mean = re.search(r'Mean position error: ([0-9.]+)', out)
        if m:
            results.append({'n_particles': n, 'ate': float(m.group(1)), 'ate_x': float(mx.group(1)), 'ate_y': float(my.group(1)), 'mean_err': float(mean.group(1)), 'file': out_file})
            print(f"ATE={m.group(1)}")
        else:
            print("Compare parse error")
    except Exception as e:
        print("Compare error", e)

# print summary
print('\nIMU-only results:')
print(f"{'n_particles':<12} {'ATE':<8} {'ATE_x':<8} {'ATE_y':<8} {'MeanErr':<8}")
print('-'*60)
for r in sorted(results, key=lambda x: x['ate']):
    print(f"{r['n_particles']:<12} {r['ate']:<8.4f} {r['ate_x']:<8.4f} {r['ate_y']:<8.4f} {r['mean_err']:<8.4f}")
