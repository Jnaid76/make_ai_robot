#!/usr/bin/env python3
"""
A* Path Planning Visualization Tool

This script allows you to visualize A* path planning on a 2D map without running Gazebo.
You can interactively click start and goal positions to see the planned path.

Usage:
    python3 visualize_astar_path.py --map <map_yaml_file>

Example:
    python3 visualize_astar_path.py --map ../../../go1_simulation/maps/maze_room.yaml
"""

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
import cv2
import os
import sys

# Import A* planner
try:
    from astar_planner import AStarPlanner
except ImportError:
    print("Error: Could not import AStarPlanner. Make sure you're in the correct directory.")
    sys.exit(1)


class AStarVisualizer:
    """Interactive A* path planning visualizer."""

    def __init__(self, map_yaml_path, inflation_radius=0.35):
        """
        Initialize the visualizer.

        Args:
            map_yaml_path: Path to the map YAML file
            inflation_radius: Obstacle inflation radius in meters
        """
        self.inflation_radius = inflation_radius
        self.start_pos = None
        self.goal_pos = None
        self.start_yaw = 0.0

        # Load map
        self.map_data, self.map_metadata = self.load_map(map_yaml_path)

        # Initialize A* planner
        self.planner = AStarPlanner(inflation_radius=inflation_radius)
        self.planner.set_map(self.map_data, self.map_metadata)

        # Setup plot
        self.setup_plot()

    def load_map(self, yaml_path):
        """Load map from YAML file."""
        # Read YAML
        with open(yaml_path, 'r') as f:
            map_config = yaml.safe_load(f)

        # Get image path (relative to YAML file)
        yaml_dir = os.path.dirname(yaml_path)
        image_path = os.path.join(yaml_dir, map_config['image'])

        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Map image not found: {image_path}")

        map_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if map_img is None:
            raise ValueError(f"Failed to load map image: {image_path}")

        # Convert to occupancy grid (0-100)
        # In PGM: 255 = free, 0 = occupied, 205 = unknown
        # In occupancy grid: 0 = free, 100 = occupied
        map_data = np.zeros_like(map_img, dtype=np.int8)
        map_data[map_img == 0] = 100  # Occupied
        map_data[map_img == 254] = 0  # Free
        map_data[(map_img > 0) & (map_img < 254)] = -1  # Unknown

        # Create metadata
        height, width = map_data.shape
        metadata = {
            'resolution': map_config['resolution'],
            'origin_x': map_config['origin'][0],
            'origin_y': map_config['origin'][1],
            'width': width,
            'height': height
        }

        print(f"Loaded map: {width}x{height}, resolution={metadata['resolution']}m")
        print(f"Origin: ({metadata['origin_x']}, {metadata['origin_y']})")

        return map_data, metadata

    def setup_plot(self):
        """Setup matplotlib plot."""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))

        # Display map (flip y-axis to match ROS convention)
        extent = [
            self.map_metadata['origin_x'],
            self.map_metadata['origin_x'] + self.map_metadata['width'] * self.map_metadata['resolution'],
            self.map_metadata['origin_y'],
            self.map_metadata['origin_y'] + self.map_metadata['height'] * self.map_metadata['resolution']
        ]

        # Create colored map: white=free, black=occupied, gray=unknown
        display_map = np.zeros((*self.map_data.shape, 3))
        display_map[self.map_data == 0] = [1, 1, 1]  # Free = white
        display_map[self.map_data == 100] = [0, 0, 0]  # Occupied = black
        display_map[self.map_data == -1] = [0.5, 0.5, 0.5]  # Unknown = gray

        self.map_img = self.ax.imshow(display_map, origin='lower', extent=extent, zorder=0)

        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('A* Path Planning Visualization\n'
                         'Left Click: Set Start | Right Click: Set Goal | Middle Click: Clear')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

        # Initialize plot elements
        self.start_marker = None
        self.goal_marker = None
        self.path_line = None
        self.raw_path_line = None
        self.inflation_circles = []

        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Add instructions
        instructions = (
            "Controls:\n"
            "  Left Click: Set start position\n"
            "  Right Click: Set goal position\n"
            "  Middle Click: Clear path\n"
            "  'i': Toggle inflation visualization\n"
            "  'r': Toggle raw path\n"
            f"  Inflation radius: {self.inflation_radius}m"
        )
        self.ax.text(0.02, 0.98, instructions, transform=self.ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.8), fontsize=9)

        self.show_inflation = False
        self.show_raw_path = False

    def on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        if event.button == 1:  # Left click - set start
            self.start_pos = (x, y)
            print(f"Start: ({x:.2f}, {y:.2f})")
            self.update_markers()
            self.plan_path()

        elif event.button == 3:  # Right click - set goal
            self.goal_pos = (x, y)
            print(f"Goal: ({x:.2f}, {y:.2f})")
            self.update_markers()
            self.plan_path()

        elif event.button == 2:  # Middle click - clear
            self.clear_path()

    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'i':
            self.show_inflation = not self.show_inflation
            print(f"Inflation visualization: {self.show_inflation}")
            self.update_inflation_visualization()

        elif event.key == 'r':
            self.show_raw_path = not self.show_raw_path
            print(f"Raw path visualization: {self.show_raw_path}")
            self.plan_path()

    def update_markers(self):
        """Update start and goal markers."""
        # Remove old markers
        if self.start_marker:
            self.start_marker.remove()
        if self.goal_marker:
            self.goal_marker.remove()

        # Add new markers
        if self.start_pos:
            self.start_marker = Circle(self.start_pos, 0.15, color='green',
                                      label='Start', zorder=5)
            self.ax.add_patch(self.start_marker)

        if self.goal_pos:
            self.goal_marker = Circle(self.goal_pos, 0.15, color='red',
                                     label='Goal', zorder=5)
            self.ax.add_patch(self.goal_marker)

        self.fig.canvas.draw()

    def update_inflation_visualization(self):
        """Toggle obstacle inflation visualization."""
        # Remove old circles
        for circle in self.inflation_circles:
            circle.remove()
        self.inflation_circles = []

        if self.show_inflation:
            # Show inflated obstacles
            resolution = self.map_metadata['resolution']
            inflation_cells = int(self.inflation_radius / resolution)

            # Sample points to show (every 10th cell to avoid cluttering)
            for y in range(0, self.map_data.shape[0], 10):
                for x in range(0, self.map_data.shape[1], 10):
                    if self.map_data[y, x] > 50:  # Occupied cell
                        world_x = x * resolution + self.map_metadata['origin_x']
                        world_y = y * resolution + self.map_metadata['origin_y']

                        circle = Circle((world_x, world_y), self.inflation_radius,
                                      color='red', alpha=0.1, zorder=1)
                        self.ax.add_patch(circle)
                        self.inflation_circles.append(circle)

        self.fig.canvas.draw()

    def plan_path(self):
        """Plan and visualize path using A*."""
        if not self.start_pos or not self.goal_pos:
            return

        # Remove old path
        if self.path_line:
            self.path_line.remove()
            self.path_line = None
        if self.raw_path_line:
            self.raw_path_line.remove()
            self.raw_path_line = None

        # Plan path
        print("\n" + "="*60)
        print("Planning path...")
        path_waypoints = self.planner.plan_path(
            self.start_pos[0], self.start_pos[1],
            self.goal_pos[0], self.goal_pos[1],
            start_yaw=self.start_yaw,
            simplify=True,
            smooth_corners=True,
            add_rotation_waypoints=False
        )

        if path_waypoints is None:
            print("ERROR: No path found!")
            self.ax.set_title('A* Path Planning - NO PATH FOUND!', color='red')
            self.fig.canvas.draw()
            return

        # Also get raw path (before smoothing)
        raw_path = self.planner.plan_path(
            self.start_pos[0], self.start_pos[1],
            self.goal_pos[0], self.goal_pos[1],
            start_yaw=self.start_yaw,
            simplify=False,
            smooth_corners=False,
            add_rotation_waypoints=False
        )

        print(f"Path found with {len(path_waypoints)} waypoints")
        print(f"Raw path has {len(raw_path)} waypoints")

        # Calculate path length
        path_length = 0.0
        for i in range(len(path_waypoints) - 1):
            dx = path_waypoints[i+1][0] - path_waypoints[i][0]
            dy = path_waypoints[i+1][1] - path_waypoints[i][1]
            path_length += np.sqrt(dx**2 + dy**2)

        print(f"Path length: {path_length:.2f}m")

        # Analyze corners
        corners = []
        if len(path_waypoints) >= 3:
            for i in range(1, len(path_waypoints) - 1):
                v1_x = path_waypoints[i][0] - path_waypoints[i-1][0]
                v1_y = path_waypoints[i][1] - path_waypoints[i-1][1]
                v2_x = path_waypoints[i+1][0] - path_waypoints[i][0]
                v2_y = path_waypoints[i+1][1] - path_waypoints[i][1]

                len1 = np.sqrt(v1_x**2 + v1_y**2)
                len2 = np.sqrt(v2_x**2 + v2_y**2)

                if len1 > 1e-6 and len2 > 1e-6:
                    dot = (v1_x * v2_x + v1_y * v2_y) / (len1 * len2)
                    angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
                    if angle > 10:
                        corners.append((i, angle, path_waypoints[i]))

        if corners:
            print(f"Found {len(corners)} corners:")
            for idx, angle, pos in corners:
                print(f"  Corner at waypoint {idx}: {angle:.1f}° at ({pos[0]:.2f}, {pos[1]:.2f})")

        # Extract x, y coordinates
        path_x = [wp[0] for wp in path_waypoints]
        path_y = [wp[1] for wp in path_waypoints]

        # Plot smoothed path
        self.path_line, = self.ax.plot(path_x, path_y, 'b-', linewidth=2,
                                       label='Smoothed Path', zorder=3)

        # Plot waypoints
        self.ax.plot(path_x, path_y, 'bo', markersize=4, zorder=4)

        # Plot raw path if enabled
        if self.show_raw_path and raw_path:
            raw_x = [wp[0] for wp in raw_path]
            raw_y = [wp[1] for wp in raw_path]
            self.raw_path_line, = self.ax.plot(raw_x, raw_y, 'r--', linewidth=1,
                                              alpha=0.5, label='Raw Path', zorder=2)

        # Highlight corners
        if corners:
            corner_x = [c[2][0] for c in corners]
            corner_y = [c[2][1] for c in corners]
            self.ax.plot(corner_x, corner_y, 'ro', markersize=8,
                        label='Corners', zorder=6)

        # Check wall clearance at corners
        min_clearance = float('inf')
        for idx, angle, pos in corners:
            # Check clearance to nearest obstacle
            map_x, map_y = self.planner.world_to_map(pos[0], pos[1])

            # Search in a radius around corner
            search_radius = int(0.5 / self.map_metadata['resolution'])
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    check_x = map_x + dx
                    check_y = map_y + dy

                    if (0 <= check_x < self.map_metadata['width'] and
                        0 <= check_y < self.map_metadata['height']):
                        if self.map_data[check_y, check_x] > 50:
                            dist = np.sqrt(dx**2 + dy**2) * self.map_metadata['resolution']
                            min_clearance = min(min_clearance, dist)

        if min_clearance != float('inf'):
            print(f"Minimum wall clearance at corners: {min_clearance:.3f}m")
            if min_clearance < self.inflation_radius:
                print(f"WARNING: Clearance ({min_clearance:.3f}m) < inflation radius ({self.inflation_radius}m)!")
                print("         Robot may crash into walls at corners!")

        self.ax.legend()
        self.ax.set_title(f'A* Path Planning - Path Length: {path_length:.2f}m, '
                         f'{len(corners)} corners')
        self.fig.canvas.draw()
        print("="*60)

    def clear_path(self):
        """Clear path and markers."""
        self.start_pos = None
        self.goal_pos = None

        if self.start_marker:
            self.start_marker.remove()
            self.start_marker = None
        if self.goal_marker:
            self.goal_marker.remove()
            self.goal_marker = None
        if self.path_line:
            self.path_line.remove()
            self.path_line = None
        if self.raw_path_line:
            self.raw_path_line.remove()
            self.raw_path_line = None

        for circle in self.inflation_circles:
            circle.remove()
        self.inflation_circles = []

        self.ax.set_title('A* Path Planning Visualization\n'
                         'Left Click: Set Start | Right Click: Set Goal | Middle Click: Clear')
        self.fig.canvas.draw()
        print("Cleared path")

    def show(self):
        """Show the interactive plot."""
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize A* path planning on a 2D map')
    parser.add_argument('--map', type=str,
                       default='../../go1_simulation/maps/maze_room.yaml',
                       help='Path to map YAML file')
    parser.add_argument('--inflation', type=float, default=0.35,
                       help='Obstacle inflation radius in meters')

    args = parser.parse_args()

    # Check if map file exists
    if not os.path.exists(args.map):
        print(f"Error: Map file not found: {args.map}")
        print("Available maps:")
        maps_dir = os.path.dirname(args.map) or '../../go1_simulation/maps'
        if os.path.exists(maps_dir):
            for f in os.listdir(maps_dir):
                if f.endswith('.yaml'):
                    print(f"  {os.path.join(maps_dir, f)}")
        sys.exit(1)

    print("="*60)
    print("A* Path Planning Visualizer")
    print("="*60)

    visualizer = AStarVisualizer(args.map, inflation_radius=args.inflation)
    visualizer.show()


if __name__ == '__main__':
    main()
