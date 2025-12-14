#!/usr/bin/env python3

import numpy as np


class DistanceEstimator:
    """Extract distance from depth image given bounding box"""

    @staticmethod
    def get_distance(depth_image: np.ndarray, bbox: tuple, method='median') -> float:
        """
        Extract distance to object from depth image.

        Args:
            depth_image: Depth image as numpy array (H, W) in meters
            bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates
            method: 'median', 'mean', or 'min' for distance calculation

        Returns:
            Distance in meters, or -1.0 if invalid
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Clamp to image boundaries
        h, w = depth_image.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        # Ensure valid bounding box
        if x2 <= x1 or y2 <= y1:
            return -1.0

        # Extract region of interest
        roi = depth_image[y1:y2, x1:x2]

        # Filter valid depth values (> 0, < inf, finite)
        valid_depths = roi[(roi > 0.0) & (roi < np.inf) & np.isfinite(roi)]

        if len(valid_depths) == 0:
            return -1.0  # No valid depth

        # Calculate distance based on method
        if method == 'median':
            return float(np.median(valid_depths))
        elif method == 'mean':
            return float(np.mean(valid_depths))
        elif method == 'min':
            return float(np.min(valid_depths))
        else:
            return float(np.median(valid_depths))
