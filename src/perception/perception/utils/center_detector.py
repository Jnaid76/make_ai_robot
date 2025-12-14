#!/usr/bin/env python3

class CenterDetector:
    """
    Determines if an object is in the center region of the image.
    Center region is defined as the middle 3/5 of the image width.
    Leftmost 1/5 and rightmost 1/5 are excluded.
    """

    def __init__(self, image_width: int):
        """
        Initialize center detector with image width.

        Args:
            image_width: Width of the image in pixels
        """
        self.image_width = image_width
        self.left_boundary = int(image_width * 0.2)
        self.right_boundary = int(image_width * 0.8)

    def is_centered(self, bbox_center_x: float) -> bool:
        """
        Check if bounding box center is in the center region.

        Args:
            bbox_center_x: X-coordinate of bounding box center

        Returns:
            True if in center region (between 0.2*width and 0.8*width)
        """
        return self.left_boundary <= bbox_center_x <= self.right_boundary

    def get_boundaries(self) -> tuple:
        """
        Get the left and right boundary x-coordinates.

        Returns:
            Tuple of (left_boundary, right_boundary)
        """
        return (self.left_boundary, self.right_boundary)
