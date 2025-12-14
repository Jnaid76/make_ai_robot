#!/usr/bin/env python3
"""
Visualize waypoints on hospital map
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Map parameters from hospital.yaml
RESOLUTION = 0.1  # meters per pixel
ORIGIN_X = -50.0  # meters
ORIGIN_Y = -50.0  # meters

# Waypoints (assuming start position around (0, 1))
waypoints = [
    (0.0, 1.0),      # Start position (approximate)
    (-2.0, 1.0),     # Move west, stay in corridor
    (-4.0, 1.0),     # Continue west
    (-5.0, 0.5),     # Turn to prepare going south
    (-5.0, -1.0),    # Go south
    (-5.0, -3.0),    # Continue south
    (-5.0, -5.0),    # Continue south
    (-5.0, -7.0),    # Continue south
    (-5.0, -8.5),    # Near corner
    (-5.5, -9.0),    # Turn corner
    (-7.3, -9.0),    # Go west along bottom corridor
    (-7.3, -7.0),    # Go north towards toilet
    (-7.3, -5.0),    # Continue north
    (-7.3, -3.0),    # Continue north
    (-7.3, -1.0),    # Continue north
    (-7.3, 0.0),     # Near toilet
    (-7.22, -0.54)   # Final toilet position
]

# Toilet location
toilet_pos = (-7.22, -0.54)

def world_to_pixel(x, y):
    """Convert world coordinates to pixel coordinates"""
    pixel_x = int((x - ORIGIN_X) / RESOLUTION)
    pixel_y = int((y - ORIGIN_Y) / RESOLUTION)
    return pixel_x, pixel_y

# Load map
map_path = '/home/jnaid/make_ai_robot/src/go1_simulation/maps/hospital.pgm'
map_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)

if map_img is None:
    print(f"Error: Could not load map from {map_path}")
    exit(1)

# Convert to BGR for colored visualization
map_colored = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)

# Flip vertically (pgm origin is top-left, but world origin is bottom-left)
map_colored = cv2.flip(map_colored, 0)

print(f"Map size: {map_colored.shape[1]}x{map_colored.shape[0]} pixels")
print(f"Map covers: {map_colored.shape[1] * RESOLUTION}m x {map_colored.shape[0] * RESOLUTION}m")
print(f"Origin: ({ORIGIN_X}, {ORIGIN_Y})")

# Draw waypoints
for i, (x, y) in enumerate(waypoints):
    px, py = world_to_pixel(x, y)
    
    # Check if pixel is within map bounds
    if 0 <= px < map_colored.shape[1] and 0 <= py < map_colored.shape[0]:
        # Draw circle for waypoint
        if i == 0:
            color = (0, 255, 0)  # Green for start
            label = "START"
        elif i == len(waypoints) - 1:
            color = (0, 0, 255)  # Red for toilet
            label = "TOILET"
        else:
            color = (255, 0, 0)  # Blue for waypoints
            label = f"WP{i}"
        
        cv2.circle(map_colored, (px, py), 5, color, -1)
        cv2.putText(map_colored, label, (px + 10, py + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        print(f"Waypoint {i}: ({x:.2f}, {y:.2f}) -> pixel ({px}, {py})")
        
        # Draw line to next waypoint
        if i < len(waypoints) - 1:
            next_x, next_y = waypoints[i + 1]
            next_px, next_py = world_to_pixel(next_x, next_y)
            cv2.line(map_colored, (px, py), (next_px, next_py), (255, 255, 0), 2)
    else:
        print(f"Warning: Waypoint {i} at ({x}, {y}) is outside map bounds")

# Draw coordinate axes at origin
origin_px, origin_py = world_to_pixel(0, 0)
if 0 <= origin_px < map_colored.shape[1] and 0 <= origin_py < map_colored.shape[0]:
    cv2.circle(map_colored, (origin_px, origin_py), 4, (255, 255, 255), -1)
    cv2.putText(map_colored, "ORIGIN (0,0)", (origin_px + 8, origin_py - 8), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# Flip back for correct display
map_colored = cv2.flip(map_colored, 0)

# Save and display
output_path = '/home/jnaid/make_ai_robot/waypoints_visualization.png'
cv2.imwrite(output_path, map_colored)
print(f"\nVisualization saved to: {output_path}")

# Display with matplotlib
plt.figure(figsize=(16, 16))
plt.imshow(cv2.cvtColor(map_colored, cv2.COLOR_BGR2RGB))
plt.title('Hospital Map with Waypoints', fontsize=16)
plt.xlabel('Pixel X', fontsize=12)
plt.ylabel('Pixel Y', fontsize=12)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', label='START (0, 1)'),
    Patch(facecolor='blue', label='Waypoints'),
    Patch(facecolor='red', label='TOILET (-7.22, -0.54)'),
    Patch(facecolor='cyan', label='Path')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig('/home/jnaid/make_ai_robot/waypoints_visualization_matplotlib.png', dpi=150)
print(f"Matplotlib visualization saved to: /home/jnaid/make_ai_robot/waypoints_visualization_matplotlib.png")

print("\nWaypoint Summary:")
print("=" * 60)
for i, (x, y) in enumerate(waypoints):
    if i == 0:
        label = "START"
    elif i == len(waypoints) - 1:
        label = "TOILET"
    else:
        label = f"WP{i}"
    print(f"{label:10s}: x={x:7.2f}m, y={y:7.2f}m")
print("=" * 60)
