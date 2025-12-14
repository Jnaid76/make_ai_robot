#!/usr/bin/env python3

"""
Utils like transforming pose and transform to 4x4 transformation matrix, converting laser scan to point cloud, ICP registration, etc.
"""

import tf_transformations
from sensor_msgs.msg import LaserScan
import numpy as np

def pose_to_matrix(pose):
    """Convert pose [x, y, z, qx, qy, qz, qw] to 4x4 transformation matrix."""
    T = np.eye(4)
    T[:3, 3] = pose[:3]  # translation
    quat = pose[3:]  # quaternion [qx, qy, qz, qw]
    T[:3, :3] = tf_transformations.quaternion_matrix(quat)[:3, :3]
    return T

def transform_to_matrix(transform):
    """Convert geometry_msgs Transform to 4x4 transformation matrix."""
    T = np.eye(4)
    T[0, 3] = transform.translation.x
    T[1, 3] = transform.translation.y
    T[2, 3] = transform.translation.z
    quat = [transform.rotation.x, transform.rotation.y, 
            transform.rotation.z, transform.rotation.w]
    T[:3, :3] = tf_transformations.quaternion_matrix(quat)[:3, :3]
    return T

def scan_to_pcd(msg: LaserScan):
    """
    Process ROS2 scan message to range and angle data.
    """
    # Get ranges and angles
    ranges = np.array(msg.ranges)
    angles = np.arange(
        msg.angle_min, 
        msg.angle_max + msg.angle_increment/2, 
        msg.angle_increment
        )

    # Ensure angles and ranges have the same number of elements
    if len(angles) > len(ranges):
        angles = angles[:len(ranges)]
    elif len(ranges) > len(angles):
        ranges = ranges[:len(angles)]

    # Filter out invalid range values (inf, nan, and out-of-bounds)
    valid_indices = np.where(
        (ranges > msg.range_min) & 
        (ranges < msg.range_max) & 
        np.isfinite(ranges)
    )[0]

    ranges = ranges[valid_indices]
    angles = angles[valid_indices]
    
    # Convert ranges and angles to 2d point cloud
    pcd = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
    return pcd

def icp_2d(previous_pcd, current_pcd, max_iterations, tolerance, distance_threshold):
    """
    Iterative Closest Point (ICP) algorithm for aligning 2D point clouds.

    Args:
        current_pcd: (N, 2) numpy array of 2D points representing the previous point cloud
        previous_pcd: (M, 2) numpy array of 2D points representing the current point cloud
        max_iterations: Maximum number of iterations to run ICP
        tolerance: tolerance to check for convergence
        distance_threshold: If provided, only consider point pairs within this distance
    """
    # Step 1: Initialize
    R, t = _initialize_alignment(current_pcd, previous_pcd)
    E_prev = np.inf
    errors = []
    
    for iteration in range(max_iterations):
        # Step 2: Apply current transformation
        aligned_current_pcd = _apply_alignment(current_pcd, R, t)
        
        # Step 3: Find correspondences
        pairs, distances = _return_closest_pairs(aligned_current_pcd, previous_pcd)
        
        # Step 3.5: Reject correspondences if threshold provided (Part c)
        if distance_threshold is not None:
            valid_mask = distances <= distance_threshold
            if np.sum(valid_mask) < 3:  # Need at least 3 points for alignment
                print(f"Warning: Only {np.sum(valid_mask)} valid correspondences. Stopping.")
                break
            pairs = pairs[valid_mask]
            distances = distances[valid_mask]
        
        # Step 4: Update alignment (get incremental transformation)
        R_new, t_new = _update_alignment(aligned_current_pcd[pairs[:, 0]], previous_pcd[pairs[:, 1]])
        
        # Step 5: Compose transformations
        # Total transformation: first apply (R, t), then apply (R_new, t_new)
        # So: R_total = R_new @ R, t_total = R_new @ t + t_new
        R = R_new @ R
        t = R_new @ t + t_new
        
        # Step 6: Compute error (MSE)
        E = np.mean(distances ** 2)
        errors.append(E)
        
        # Step 7: Check convergence
        if abs(E_prev - E) < tolerance:
            break
        
        E_prev = E
    
    # Convert R (2x2) and t (2,) to SE(2) transformation matrix (3x3)
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = t
    
    return T
        

def _initialize_alignment(current_pcd, previous_pcd):
    """Return the initial rotation and translation."""
    # Initialize with identity rotation
    R = np.eye(2)
    
    # Initialize translation by aligning centroids
    # This provides a much better starting point than t=0
    current_pcd_centroid = np.mean(current_pcd, axis=0)
    previous_pcd_centroid = np.mean(previous_pcd, axis=0)
    t = previous_pcd_centroid - current_pcd_centroid
    
    return R, t        

def _apply_alignment(current_pcd, R, t):
    """Apply the current alignment and return the aligned current_pcd points."""
    # Apply transformation: aligned = R @ current_pcd^T + t
    # current_pcd is (N, 2), R is (2, 2), t is (2,)
    # Result: (R @ current_pcd.T).T + t = current_pcd @ R.T + t
    return (R @ current_pcd.T).T + t


def _return_closest_pairs(current_pcd, previous_pcd):
    """Return the closest pairs of points between current_pcd and previous_pcd.
    
    For each point in current_pcd, find the closest point in previous_pcd.
    
    Args:
        current_pcd: (N, 2) numpy array of 2D points
        previous_pcd: (M, 2) numpy array of 2D points
    
    Returns:
        pairs: (N, 2) numpy array where pairs[i] = [current_pcd_idx, previous_pcd_idx]
        distances: (N,) numpy array of distances to closest points
    """
    # Compute pairwise distances using broadcasting
    # current_pcd[:, None, :] has shape (N, 1, 2)
    # previous_pcd[None, :, :] has shape (1, M, 2)
    # diff has shape (N, M, 2)
    diff = current_pcd[:, None, :] - previous_pcd[None, :, :]
    distances_matrix = np.linalg.norm(diff, axis=2)  # Shape: (N, M)
    
    # Find closest previous_pcd point for each current_pcd point
    closest_current_indices = np.argmin(distances_matrix, axis=1)  # Shape: (N,)
    distances = distances_matrix[np.arange(len(current_pcd)), closest_current_indices]  # Shape: (N,)
    
    # Create pairs: (current_pcd_idx, previous_pcd_idx)
    pairs = np.column_stack((np.arange(len(current_pcd)), closest_current_indices))
    
    return pairs, distances


def _update_alignment(current_pcd_paired, previous_pcd_paired):
    """Update the alignment and return the new rotation and translation.
    
    Args:
        current_pcd_paired: (K, 2) numpy array of paired points from previous point cloud
        previous_pcd_paired: (K, 2) numpy array of paired points from current point cloud
    """
    # Implement the SVD-based least squares solution
    # current_pcd_paired and previous_pcd_paired are already paired (same length)
    
    # Step 1: Compute centroids
    previous_centroid = np.mean(current_pcd_paired, axis=0)
    current_centroid = np.mean(previous_pcd_paired, axis=0)
    
    # Step 2: Center the points
    previous_centered = current_pcd_paired - previous_centroid
    current_centered = previous_pcd_paired - current_centroid
    
    # Step 3: Compute cross-covariance matrix H
    H = current_centered.T @ previous_centered
    
    # Step 4: Compute SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Step 5: Compute rotation    
    # Handle reflection case: ensure det(R) = +1 for 2D (2x2 matrix)
    R = U @ np.diag([1, np.linalg.det(U @ Vt)]) @ Vt
    
    # Step 6: Compute translation
    t = current_centroid - R @ previous_centroid
    return R, t


# ============================================================================
# Particle Filter Utilities
# ============================================================================

def initialize_particles(num_particles, initial_pose, variance):
    """
    Initialize particles around initial pose with Gaussian noise.

    Args:
        num_particles: Number of particles to create
        initial_pose: [x, y, theta] - initial pose estimate
        variance: [var_x, var_y, var_theta] - variance for each dimension

    Returns:
        particles: List of dicts with keys 'x', 'y', 'theta', 'weight'
    """
    particles = []
    for _ in range(num_particles):
        particle = {
            'x': initial_pose[0] + np.random.normal(0, np.sqrt(variance[0])),
            'y': initial_pose[1] + np.random.normal(0, np.sqrt(variance[1])),
            'theta': initial_pose[2] + np.random.normal(0, np.sqrt(variance[2])),
            'weight': 1.0 / num_particles
        }
        particles.append(particle)
    return particles


def normalize_weights(particles):
    """
    Normalize particle weights to sum to 1.

    Args:
        particles: List of particle dicts
    """
    total_weight = sum(p['weight'] for p in particles)
    if total_weight > 0:
        for p in particles:
            p['weight'] /= total_weight
    else:
        # If all weights are zero, reset to uniform
        uniform_weight = 1.0 / len(particles)
        for p in particles:
            p['weight'] = uniform_weight


def resample_particles(particles):
    """
    Low-variance systematic resampling to prevent particle depletion.

    Args:
        particles: List of particle dicts with weights

    Returns:
        new_particles: Resampled list of particles
    """
    N = len(particles)
    new_particles = []

    # Low-variance systematic resampling
    r = np.random.uniform(0, 1.0 / N)
    c = particles[0]['weight']
    i = 0

    for m in range(N):
        U = r + m / N
        while U > c and i < N - 1:
            i += 1
            c += particles[i]['weight']

        # Deep copy the selected particle
        new_particle = {
            'x': particles[i]['x'],
            'y': particles[i]['y'],
            'theta': particles[i]['theta'],
            'weight': 1.0 / N
        }
        new_particles.append(new_particle)

    return new_particles


def estimate_pose_from_particles(particles):
    """
    Estimate robot pose from particle cloud using weighted mean.
    Handles angle wrapping for theta using circular statistics.

    Args:
        particles: List of particle dicts with weights

    Returns:
        [x, y, theta]: Estimated pose
    """
    # Weighted mean for x, y
    x = sum(p['x'] * p['weight'] for p in particles)
    y = sum(p['y'] * p['weight'] for p in particles)

    # Circular mean for theta (handle angle wrapping)
    sin_sum = sum(np.sin(p['theta']) * p['weight'] for p in particles)
    cos_sum = sum(np.cos(p['theta']) * p['weight'] for p in particles)
    theta = np.arctan2(sin_sum, cos_sum)

    return [x, y, theta]


def predict_particles(particles, delta_pose, motion_noise):
    """
    Apply odometry-based motion model with noise to all particles.
    Uses decomposition into rotation-translation-rotation for better accuracy.

    Args:
        particles: List of particle dicts
        delta_pose: [dx, dy, dtheta] - change in pose from odometry
        motion_noise: Dict with 'trans' and 'rot' noise parameters
    """
    dx, dy, dtheta = delta_pose

    # Decompose motion into rotation-translation-rotation
    delta_trans = np.sqrt(dx**2 + dy**2)

    # Handle case when robot didn't move
    if delta_trans < 1e-6:
        delta_rot1 = 0.0
    else:
        delta_rot1 = np.arctan2(dy, dx)

    delta_rot2 = dtheta - delta_rot1

    # Normalize angles to [-pi, pi]
    delta_rot1 = np.arctan2(np.sin(delta_rot1), np.cos(delta_rot1))
    delta_rot2 = np.arctan2(np.sin(delta_rot2), np.cos(delta_rot2))

    # Apply motion model to each particle
    for p in particles:
        # Add noise to motion parameters
        noisy_rot1 = delta_rot1 + np.random.normal(0, motion_noise['rot'] * (abs(delta_rot1) + 0.1))
        noisy_trans = delta_trans + np.random.normal(0, motion_noise['trans'] * (delta_trans + 0.1))
        noisy_rot2 = delta_rot2 + np.random.normal(0, motion_noise['rot'] * (abs(delta_rot2) + 0.1))

        # Update particle pose
        p['theta'] += noisy_rot1
        p['x'] += noisy_trans * np.cos(p['theta'])
        p['y'] += noisy_trans * np.sin(p['theta'])
        p['theta'] += noisy_rot2

        # Normalize theta to [-pi, pi]
        p['theta'] = np.arctan2(np.sin(p['theta']), np.cos(p['theta']))


def transform_scan_to_map(scan_pcd, particle):
    """
    Transform scan points to map frame using particle pose.

    Args:
        scan_pcd: (N, 2) numpy array of scan points in base frame
        particle: Dict with 'x', 'y', 'theta'

    Returns:
        scan_in_map: (N, 2) numpy array of scan points in map frame
    """
    # Create 2D rotation matrix
    cos_theta = np.cos(particle['theta'])
    sin_theta = np.sin(particle['theta'])
    R = np.array([[cos_theta, -sin_theta],
                  [sin_theta, cos_theta]])

    # Transform: R @ scan_pcd.T + translation
    translation = np.array([particle['x'], particle['y']])
    scan_in_map = (R @ scan_pcd.T).T + translation

    return scan_in_map


def occupancy_grid_to_array(occupancy_grid):
    """
    Convert OccupancyGrid message to numpy array.

    Args:
        occupancy_grid: nav_msgs/OccupancyGrid message

    Returns:
        grid_array: 2D numpy array (height, width) with values 0-100
        metadata: Dict with 'resolution', 'origin_x', 'origin_y'
    """
    width = occupancy_grid.info.width
    height = occupancy_grid.info.height
    resolution = occupancy_grid.info.resolution
    origin_x = occupancy_grid.info.origin.position.x
    origin_y = occupancy_grid.info.origin.position.y

    # Convert data to numpy array and reshape
    grid_array = np.array(occupancy_grid.data, dtype=np.int8).reshape((height, width))

    metadata = {
        'resolution': resolution,
        'origin_x': origin_x,
        'origin_y': origin_y,
        'width': width,
        'height': height
    }

    return grid_array, metadata


def point_to_grid_index(x, y, metadata):
    """
    Convert world coordinates to grid indices.

    Args:
        x, y: World coordinates
        metadata: Dict with 'resolution', 'origin_x', 'origin_y'

    Returns:
        (grid_x, grid_y): Grid indices (can be out of bounds)
    """
    grid_x = int((x - metadata['origin_x']) / metadata['resolution'])
    grid_y = int((y - metadata['origin_y']) / metadata['resolution'])
    return grid_x, grid_y


def is_occupied(x, y, grid_array, metadata, threshold=65):
    """
    Check if a point in world coordinates is occupied in the map.

    Args:
        x, y: World coordinates
        grid_array: 2D numpy array from occupancy_grid_to_array
        metadata: Dict with map metadata
        threshold: Occupancy threshold (0-100)

    Returns:
        True if occupied, False otherwise
    """
    grid_x, grid_y = point_to_grid_index(x, y, metadata)

    # Check bounds
    if grid_x < 0 or grid_x >= metadata['width'] or grid_y < 0 or grid_y >= metadata['height']:
        return False

    # Check occupancy (map uses row-major order: [y, x])
    return grid_array[grid_y, grid_x] >= threshold


def compute_likelihood_field(occupancy_grid):
    """
    Pre-compute distance transform for efficient likelihood computation.
    Uses scipy's distance_transform_edt for fast computation.

    Args:
        occupancy_grid: nav_msgs/OccupancyGrid message

    Returns:
        likelihood_field: Dict with 'distances' array and metadata
    """
    from scipy import ndimage

    grid_array, metadata = occupancy_grid_to_array(occupancy_grid)

    # Create binary occupancy map (True = free, False = occupied)
    # Handle unknown (-1) as free space
    free_space = grid_array < 50  # Values < 50 are considered free

    # Compute distance transform (distance to nearest obstacle)
    # Note: distance_transform_edt computes distance to nearest False value
    distances = ndimage.distance_transform_edt(free_space) * metadata['resolution']

    likelihood_field = {
        'distances': distances,
        'metadata': metadata
    }

    return likelihood_field


def compute_scan_likelihood(scan_points, likelihood_field, sensor_params):
    """
    Compute probability of observing scan given map using likelihood field.

    Args:
        scan_points: (N, 2) numpy array of scan points in map frame
        likelihood_field: Dict from compute_likelihood_field
        sensor_params: Dict with 'z_hit', 'z_rand', 'sigma_hit', 'max_range'

    Returns:
        likelihood: Probability of observing this scan (0-1)
    """
    distances = likelihood_field['distances']
    metadata = likelihood_field['metadata']

    z_hit = sensor_params['z_hit']
    z_rand = sensor_params['z_rand']
    sigma_hit = sensor_params['sigma_hit']
    max_range = sensor_params['max_range']

    log_likelihood = 0.0
    valid_points = 0

    for point in scan_points:
        x, y = point
        grid_x, grid_y = point_to_grid_index(x, y, metadata)

        # Check bounds
        if grid_x < 0 or grid_x >= metadata['width'] or grid_y < 0 or grid_y >= metadata['height']:
            continue

        # Get distance to nearest obstacle from pre-computed field
        dist = distances[grid_y, grid_x]

        # Gaussian probability centered at obstacles (dist=0)
        gaussian_prob = np.exp(-0.5 * (dist / sigma_hit) ** 2) / (sigma_hit * np.sqrt(2 * np.pi))

        # Mixture model: hit + random
        prob = z_hit * gaussian_prob + z_rand / max_range

        # Avoid log(0)
        if prob > 1e-10:
            log_likelihood += np.log(prob)
            valid_points += 1

    # Return normalized likelihood
    if valid_points > 0:
        # Average log likelihood, then exponentiate
        avg_log_likelihood = log_likelihood / valid_points
        # Clip to prevent numerical issues
        avg_log_likelihood = np.clip(avg_log_likelihood, -100, 0)
        likelihood = np.exp(avg_log_likelihood)
    else:
        # No valid points, return small likelihood
        likelihood = 1e-10

    return likelihood