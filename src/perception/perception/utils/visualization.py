#!/usr/bin/env python3

import cv2
import numpy as np


class DetectionVisualizer:
    """Draw bounding boxes and labels on images"""

    # Color map for classes (BGR format for OpenCV)
    COLORS = {
        'banana': (0, 255, 255),          # Yellow
        'apple': (0, 0, 255),             # Red
        'pizza': (0, 165, 255),           # Orange
        'rotten_banana': (0, 100, 100),   # Dark yellow
        'rotten_apple': (0, 0, 128),      # Dark red
        'rotten_pizza': (0, 82, 128),     # Dark orange
        'stop_sign': (0, 0, 255),         # Red
        'nurse': (255, 0, 255),           # Magenta
        'cone_red': (0, 0, 255),          # Red
        'cone_green': (0, 255, 0),        # Green
        'cone_blue': (255, 0, 0),         # Blue
        'delivery_box': (128, 128, 128),  # Gray
    }

    @staticmethod
    def draw_detections(image: np.ndarray, detections: list,
                        centered_idx: int = -1) -> np.ndarray:
        """
        Draw bounding boxes on image.

        Args:
            image: Input image (BGR)
            detections: List of dicts with keys: 'bbox', 'label', 'conf', 'distance'
            centered_idx: Index of centered object (highlighted differently)

        Returns:
            Annotated image
        """
        annotated = image.copy()

        for idx, det in enumerate(detections):
            bbox = det['bbox']  # (x1, y1, x2, y2)
            label = det['label']
            conf = det['conf']
            distance = det.get('distance', None)

            x1, y1, x2, y2 = map(int, bbox)

            # Choose color
            color = DetectionVisualizer.COLORS.get(label, (255, 255, 255))

            # Highlight centered object with thicker box
            thickness = 3 if idx == centered_idx else 2

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Create label text
            text = f"{label} {conf:.2f}"
            if distance is not None and distance > 0:
                text += f" {distance:.2f}m"

            # Draw label background
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - text_h - 4),
                          (x1 + text_w, y1), color, -1)

            # Draw label text
            cv2.putText(annotated, text, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Draw star marker for centered object
            if idx == centered_idx:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.drawMarker(annotated, (center_x, center_y), (0, 255, 0),
                               cv2.MARKER_STAR, 20, 2)

        return annotated
