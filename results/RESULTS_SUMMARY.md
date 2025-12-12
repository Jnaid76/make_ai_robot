# Trajectory Estimation Results Summary

## Completed Baseline Experiments

### Experiment A: IMU-only Particle Filter
- **Method**: Motion model from cmd_vel + IMU heading measurement update
- **Configuration**: n_particles=500, motion_std=(0.02, 0.02, 0.01), imu_std=0.05
- **Output file**: `results/traj1_est_from_bag.txt`
- **Results**:
  - ATE: 1.5605 m
  - ATE_x: 1.4728 m
  - ATE_y: 0.5157 m
  - ATE_z: 0.0000 m
  - Mean position error: 1.3881 m
  - Min position error: 0.0052 m
  - Max position error: 3.5854 m
- **Analysis**: Pure IMU integration accumulates drift quickly without positional constraints.

### Experiment B1: Fusion (IMU predict + simple scan heading PCA update)
- **Method**: IMU heading from orientation + scan-derived heading via PCA
- **Configuration**: n_particles=500, motion_std=(0.02, 0.02, 0.01), imu_std=0.05, scan_std=0.25
- **Output file**: `results/traj1_est_from_bag_lidar_fused.txt`
- **Results**:
  - ATE: 2.3477 m
  - ATE_x: 0.9174 m
  - ATE_y: 2.1610 m
  - ATE_z: 0.0000 m
  - Mean position error: 1.9791 m
  - Min position error: 0.0064 m
  - Max position error: 4.8317 m
- **Analysis**: Simple PCA-based heading extraction did not improve results; likely due to noisy/ambiguous principal direction in sparse scans.

### Experiment B2: LiDAR-only (scan-to-scan KDTree NN)
- **Method**: Weak motion model from cmd_vel + scan-to-scan nearest-neighbor distance likelihood
- **Configuration**: n_particles=500, motion_std=(0.02, 0.02, 0.01), scan_std=0.5, n_beams=72
- **Output file**: `results/traj1_est_from_bag_lidar_only.txt`
- **Results**:
  - ATE: **0.8945 m** ← Best so far
  - ATE_x: 0.6037 m
  - ATE_y: 0.6601 m
  - ATE_z: 0.0000 m
  - Mean position error: 0.6793 m
  - Min position error: 0.0101 m
  - Max position error: 1.6972 m
- **Analysis**: Directly constraining particle positions via scan geometry (KDTree NN distances) significantly outperforms heading-only updates. This approach does not require a pre-built map.

## Quick Comparison Table

| Method | Configuration | ATE (m) | ATE_x (m) | ATE_y (m) | Mean Err (m) | Notes |
|--------|---------------|---------|-----------|-----------|--------------|-------|
| IMU-only | n_p=500, imu_std=0.05 | 1.5605 | 1.4728 | 0.5157 | 1.3881 | Baseline; pure integration drift |
| Fusion (PCA) | n_p=500, scan_std=0.25 | 2.3477 | 0.9174 | 2.1610 | 1.9791 | Poor; PCA heading unreliable |
| LiDAR-only (KDTree) | n_p=500, scan_std=0.5, beams=72 | **0.8945** | 0.6037 | 0.6601 | 0.6793 | **Best so far**; scan-to-scan NN |

## Ongoing: Hyperparameter Grid Search for LiDAR-only

Running exhaustive search over:
- `scan_std`: [0.1, 0.25, 0.5, 1.0]
- `n_beams`: [36, 72, 144]
- `n_particles`: [500, 1000, 2000]
- Total combinations: 36

Expected to find optimal settings within scan_std and particle count.

**Status**: Running... check `results/hyperparameter_search.log` for progress.

---

Generated: 2025-12-11
