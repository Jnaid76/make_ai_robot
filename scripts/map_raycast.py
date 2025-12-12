#!/usr/bin/env python3
"""
Map-based Raycast Module for Particle Filter Localization

This module provides:
1. Occupancy grid map loading from .pgm + .yaml files
2. Bresenham raycast algorithm for computing expected LiDAR ranges
3. Likelihood field measurement model for particle weighting

The raycast approach simulates what the LiDAR *should* see from each particle's
hypothetical position, then compares with actual scan to compute likelihood.
"""

import numpy as np
import yaml
import math
from PIL import Image
from typing import Tuple, Optional


class OccupancyGridMap:
    """
    Occupancy grid map representation for raycast operations.
    
    Coordinate systems:
    - World coordinates: meters (e.g., x, y in [-50, 50])
    - Grid coordinates: pixels (e.g., row, col in [0, 1000])
    
    Map conventions (ROS2 standard):
    - 0: Free space
    - 100: Occupied
    - -1: Unknown
    """
    
    def __init__(self, map_yaml_path: str):
        """
        Load occupancy grid from .yaml metadata and .pgm image.
        
        Args:
            map_yaml_path: Path to .yaml file (e.g., "hospital.yaml")
        """
        # Load metadata
        with open(map_yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        self.resolution = float(metadata['resolution'])  # meters/pixel
        self.origin = np.array(metadata['origin'][:2])  # [x, y] in meters
        self.occupied_thresh = float(metadata.get('occupied_thresh', 0.65))
        self.free_thresh = float(metadata.get('free_thresh', 0.196))
        self.negate = int(metadata.get('negate', 0))
        
        # Load image
        import os
        map_dir = os.path.dirname(map_yaml_path)
        image_path = os.path.join(map_dir, metadata['image'])
        
        img = Image.open(image_path)
        img_array = np.array(img, dtype=np.float32)
        
        # Convert to occupancy probabilities [0, 1]
        # PGM: 0=black (occupied), 255=white (free)
        if self.negate == 0:
            # Standard: black=occupied, white=free
            occupancy_prob = 1.0 - (img_array / 255.0)
        else:
            # Inverted: white=occupied, black=free
            occupancy_prob = img_array / 255.0
        
        # Create occupancy grid: 0=free, 100=occupied, -1=unknown
        self.grid = np.full_like(occupancy_prob, -1, dtype=np.int8)
        self.grid[occupancy_prob <= self.free_thresh] = 0  # Free
        self.grid[occupancy_prob >= self.occupied_thresh] = 100  # Occupied
        
        self.height, self.width = self.grid.shape
        
        print(f"[OccupancyGridMap] Loaded map: {self.width}x{self.height} pixels")
        print(f"  Resolution: {self.resolution} m/pixel")
        print(f"  Origin: {self.origin}")
        print(f"  World bounds: x=[{self.origin[0]}, {self.origin[0] + self.width*self.resolution}], "
              f"y=[{self.origin[1]}, {self.origin[1] + self.height*self.resolution}]")
        print(f"  Occupied cells: {np.sum(self.grid == 100)} / {self.grid.size}")
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates (meters) to grid coordinates (pixels).
        
        Args:
            x, y: Position in world frame (meters)
            
        Returns:
            (row, col): Grid indices. May be out of bounds.
        """
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        # Note: row = grid_y, col = grid_x (image convention)
        return grid_y, grid_x
    
    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """
        Convert grid coordinates (pixels) to world coordinates (meters).
        
        Args:
            row, col: Grid indices
            
        Returns:
            (x, y): Position in world frame (meters)
        """
        x = self.origin[0] + col * self.resolution
        y = self.origin[1] + row * self.resolution
        return x, y
    
    def is_occupied(self, x: float, y: float) -> bool:
        """
        Check if world position is occupied.
        
        Args:
            x, y: World coordinates (meters)
            
        Returns:
            True if occupied or out of bounds, False if free
        """
        row, col = self.world_to_grid(x, y)
        if not self.is_valid_grid(row, col):
            return True  # Out of bounds treated as occupied
        return self.grid[row, col] == 100
    
    def is_valid_grid(self, row: int, col: int) -> bool:
        """Check if grid coordinates are within map bounds."""
        return 0 <= row < self.height and 0 <= col < self.width
    
    def get_occupancy(self, row: int, col: int) -> int:
        """Get occupancy value at grid position. Returns 100 if out of bounds."""
        if not self.is_valid_grid(row, col):
            return 100
        return self.grid[row, col]


class RaycastSensor:
    """
    Bresenham raycast-based LiDAR sensor model for particle filter.
    
    This simulates what a LiDAR sensor *would* see from a given pose,
    enabling comparison with actual sensor measurements for particle weighting.
    """
    
    def __init__(self, occupancy_map: OccupancyGridMap, 
                 max_range: float = 30.0,
                 occupied_threshold: int = 50):
        """
        Initialize raycast sensor.
        
        Args:
            occupancy_map: OccupancyGridMap instance
            max_range: Maximum LiDAR range (meters)
            occupied_threshold: Grid values >= this are considered obstacles
        """
        self.map = occupancy_map
        self.max_range = max_range
        self.occupied_threshold = occupied_threshold
    
    def bresenham_raycast(self, x0: float, y0: float, angle: float) -> float:
        """
        Cast a single ray from (x0, y0) at given angle until hitting obstacle or max_range.
        
        Uses Bresenham line algorithm for efficient grid traversal.
        
        Args:
            x0, y0: Starting position in world coordinates (meters)
            angle: Ray direction in world frame (radians)
            
        Returns:
            Range to first obstacle (meters), or max_range if no hit
        """
        # Compute endpoint at max_range
        x1 = x0 + self.max_range * math.cos(angle)
        y1 = y0 + self.max_range * math.sin(angle)
        
        # Convert to grid coordinates
        row0, col0 = self.map.world_to_grid(x0, y0)
        row1, col1 = self.map.world_to_grid(x1, y1)
        
        # Bresenham's line algorithm
        points = self._bresenham_line(col0, row0, col1, row1)
        
        # Traverse ray until obstacle
        first_point = True
        for col, row in points:
            # Skip starting cell (robot position)
            if first_point:
                first_point = False
                continue
            
            if not self.map.is_valid_grid(row, col):
                # Hit map boundary
                wx, wy = self.map.grid_to_world(row, col)
                return math.hypot(wx - x0, wy - y0)
            
            occupancy = self.map.grid[row, col]
            if occupancy >= self.occupied_threshold:
                # Hit obstacle
                wx, wy = self.map.grid_to_world(row, col)
                return math.hypot(wx - x0, wy - y0)
        
        # No obstacle found within max_range
        return self.max_range
    
    @staticmethod
    def _bresenham_line(x0: int, y0: int, x1: int, y1: int):
        """
        Generate grid cells along line from (x0,y0) to (x1,y1) using Bresenham algorithm.
        
        Args:
            x0, y0: Start grid coordinates
            x1, y1: End grid coordinates
            
        Yields:
            (x, y): Grid coordinates along the line
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            yield x, y
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    def cast_rays(self, pose: np.ndarray, scan_angles: np.ndarray) -> np.ndarray:
        """
        Cast multiple rays from given pose at specified angles.
        
        Args:
            pose: [x, y, yaw] in world frame
            scan_angles: Array of scan angles relative to robot frame (radians)
                        e.g., np.linspace(-π, π, 360) for 360-degree scan
        
        Returns:
            expected_ranges: Array of expected ranges (meters) for each angle
        """
        x, y, yaw = pose
        expected_ranges = np.zeros(len(scan_angles))
        
        for i, angle_rel in enumerate(scan_angles):
            # Convert to world frame
            angle_world = yaw + angle_rel
            expected_ranges[i] = self.bresenham_raycast(x, y, angle_world)
        
        return expected_ranges


class LikelihoodFieldModel:
    """
    Likelihood field measurement model for particle weighting.
    
    Compares actual LiDAR ranges with raycast-predicted ranges to compute
    particle likelihood using a Gaussian error model.
    """
    
    def __init__(self, sigma_hit: float = 0.2, 
                 z_hit: float = 0.95, 
                 z_rand: float = 0.05,
                 max_range: float = 30.0):
        """
        Initialize likelihood field model.
        
        Args:
            sigma_hit: Standard deviation for Gaussian error model (meters)
            z_hit: Weight for correct measurements
            z_rand: Weight for random/spurious measurements (for robustness)
            max_range: Maximum valid range (meters)
        """
        self.sigma_hit = sigma_hit
        self.z_hit = z_hit
        self.z_rand = z_rand
        self.max_range = max_range
        
        # Precompute normalization constant for Gaussian
        self.norm_factor = 1.0 / (math.sqrt(2 * math.pi) * sigma_hit)
    
    def compute_likelihood(self, 
                          actual_ranges: np.ndarray,
                          expected_ranges: np.ndarray) -> float:
        """
        Compute likelihood P(z | x, map) for a single particle.
        
        Args:
            actual_ranges: Actual LiDAR measurements (meters)
            expected_ranges: Raycast-predicted ranges from particle pose (meters)
            
        Returns:
            likelihood: Probability of observing actual_ranges given expected_ranges
        """
        # Filter valid measurements (not inf, not max_range)
        valid_mask = (actual_ranges < self.max_range - 0.01) & (actual_ranges > 0.01)
        
        if not np.any(valid_mask):
            # No valid measurements - return uniform likelihood
            return 1.0 / self.max_range
        
        actual = actual_ranges[valid_mask]
        expected = expected_ranges[valid_mask]
        
        # Compute range differences
        errors = actual - expected
        
        # Gaussian likelihood for each beam
        # P(z_i | x) = z_hit * N(0, sigma_hit) + z_rand * U(0, max_range)
        gaussian_prob = self.norm_factor * np.exp(-0.5 * (errors / self.sigma_hit) ** 2)
        uniform_prob = 1.0 / self.max_range
        
        beam_likelihoods = self.z_hit * gaussian_prob + self.z_rand * uniform_prob
        
        # Total likelihood: use MEAN of log-likelihoods to avoid underflow
        # This is equivalent to geometric mean of likelihoods
        log_likelihood = np.mean(np.log(beam_likelihoods + 1e-10))
        likelihood = np.exp(log_likelihood)
        
        return likelihood
    
    def compute_weights(self,
                       actual_ranges: np.ndarray,
                       expected_ranges_list: list) -> np.ndarray:
        """
        Compute weights for all particles given actual scan.
        
        Args:
            actual_ranges: Actual LiDAR scan (n_beams,)
            expected_ranges_list: List of expected ranges for each particle
                                 [(n_beams,), (n_beams,), ...]
            
        Returns:
            weights: Normalized weights for all particles (n_particles,)
        """
        n_particles = len(expected_ranges_list)
        raw_weights = np.zeros(n_particles)
        
        for i, expected_ranges in enumerate(expected_ranges_list):
            raw_weights[i] = self.compute_likelihood(actual_ranges, expected_ranges)
        
        # Normalize weights
        total = np.sum(raw_weights)
        if total < 1e-20:
            # All particles have near-zero likelihood - uniform weights
            return np.ones(n_particles) / n_particles
        
        return raw_weights / total


# Convenience function for integration
def create_map_raycast_model(map_yaml_path: str,
                             max_range: float = 30.0,
                             sigma_hit: float = 0.2,
                             n_beams: int = 144) -> Tuple[OccupancyGridMap, RaycastSensor, LikelihoodFieldModel]:
    """
    Factory function to create all components for map-based localization.
    
    Args:
        map_yaml_path: Path to map .yaml file
        max_range: Maximum LiDAR range (meters)
        sigma_hit: Measurement noise standard deviation (meters)
        n_beams: Number of beams to use from scan (for subsampling)
        
    Returns:
        (map, raycast_sensor, likelihood_model): Tuple of initialized components
    """
    occupancy_map = OccupancyGridMap(map_yaml_path)
    raycast_sensor = RaycastSensor(occupancy_map, max_range=max_range)
    likelihood_model = LikelihoodFieldModel(sigma_hit=sigma_hit, max_range=max_range)
    
    print(f"[MapRaycast] Initialized with max_range={max_range}m, sigma_hit={sigma_hit}m, n_beams={n_beams}")
    
    return occupancy_map, raycast_sensor, likelihood_model
