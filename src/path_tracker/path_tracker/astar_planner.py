#!/usr/bin/env python3
"""
A* Path Planning Algorithm for Collision-Free Navigation

This module implements the A* algorithm for path planning on occupancy grid maps.
It finds the shortest collision-free path from start to goal position.

Author: CRAIP Team
"""

import math
import heapq
from typing import List, Tuple, Optional, Set
import numpy as np


class Node:
    """
    Represents a node in the A* search graph.
    """
    def __init__(self, x: int, y: int, g_cost: float = float('inf'), h_cost: float = 0.0, parent=None):
        self.x = x
        self.y = y
        self.g_cost = g_cost  # Cost from start to this node
        self.h_cost = h_cost  # Heuristic cost from this node to goal
        self.f_cost = g_cost + h_cost  # Total cost
        self.parent = parent  # Parent node for path reconstruction
    
    def __lt__(self, other):
        """For priority queue comparison"""
        if self.f_cost != other.f_cost:
            return self.f_cost < other.f_cost
        return self.h_cost < other.h_cost
    
    def __eq__(self, other):
        """For set operations"""
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        """For set operations"""
        return hash((self.x, self.y))
    
    def __repr__(self):
        return f"Node({self.x}, {self.y}, f={self.f_cost:.2f})"


class AStarPlanner:
    """
    A* path planner for occupancy grid maps.
    """
    
    def __init__(self, inflation_radius: float = 0.35):
        """
        Initialize A* planner.
        
        Args:
            inflation_radius: Robot radius in meters for obstacle inflation
        """
        self.inflation_radius = inflation_radius
        self.map_data = None
        self.map_metadata = None
        
    def set_map(self, occupancy_grid, map_metadata):
        """
        Set the occupancy grid map for planning.
        
        Args:
            occupancy_grid: 2D numpy array of occupancy values (0-100)
            map_metadata: Dictionary with 'resolution', 'origin_x', 'origin_y', 'width', 'height'
        """
        self.map_data = occupancy_grid
        self.map_metadata = map_metadata
        
    def world_to_map(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to map cell coordinates.
        
        Args:
            world_x: X coordinate in world frame (meters)
            world_y: Y coordinate in world frame (meters)
            
        Returns:
            Tuple of (map_x, map_y) cell indices
        """
        if self.map_metadata is None:
            raise ValueError("Map not set. Call set_map() first.")
        
        resolution = self.map_metadata['resolution']
        origin_x = self.map_metadata['origin_x']
        origin_y = self.map_metadata['origin_y']
        
        map_x = int((world_x - origin_x) / resolution)
        map_y = int((world_y - origin_y) / resolution)
        
        return map_x, map_y
    
    def map_to_world(self, map_x: int, map_y: int) -> Tuple[float, float]:
        """
        Convert map cell coordinates to world coordinates.
        
        Args:
            map_x: X cell index
            map_y: Y cell index
            
        Returns:
            Tuple of (world_x, world_y) in meters
        """
        if self.map_metadata is None:
            raise ValueError("Map not set. Call set_map() first.")
        
        resolution = self.map_metadata['resolution']
        origin_x = self.map_metadata['origin_x']
        origin_y = self.map_metadata['origin_y']
        
        world_x = map_x * resolution + origin_x
        world_y = map_y * resolution + origin_y
        
        return world_x, world_y
    
    def is_valid_cell(self, map_x: int, map_y: int) -> bool:
        """
        Check if a map cell is valid (within bounds and not occupied).
        
        Args:
            map_x: X cell index
            map_y: Y cell index
            
        Returns:
            True if cell is valid for path planning
        """
        if self.map_data is None:
            return False
        
        height, width = self.map_data.shape
        
        # Check bounds
        if map_x < 0 or map_x >= width or map_y < 0 or map_y >= height:
            return False
        
        # Check if cell is occupied (value > 50 means occupied)
        # Also check inflation radius around the cell
        cell_value = self.map_data[map_y, map_x]
        
        if cell_value > 50:  # Occupied
            return False
        
        # Inflate obstacles by checking nearby cells
        inflation_cells = int(self.inflation_radius / self.map_metadata['resolution'])
        for dy in range(-inflation_cells, inflation_cells + 1):
            for dx in range(-inflation_cells, inflation_cells + 1):
                check_x = map_x + dx
                check_y = map_y + dy
                
                if (check_x >= 0 and check_x < width and 
                    check_y >= 0 and check_y < height):
                    dist = math.sqrt(dx*dx + dy*dy) * self.map_metadata['resolution']
                    if dist <= self.inflation_radius:
                        if self.map_data[check_y, check_x] > 50:
                            return False
        
        return True
    
    def heuristic(self, node1: Node, node2: Node) -> float:
        """
        Calculate heuristic cost (Euclidean distance) between two nodes.
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            Heuristic cost in meters
        """
        dx = node1.x - node2.x
        dy = node1.y - node2.y
        distance = math.sqrt(dx*dx + dy*dy) * self.map_metadata['resolution']
        return distance
    
    def get_neighbors(self, node: Node) -> List[Node]:
        """
        Get valid neighboring nodes (8-connected).
        
        Args:
            node: Current node
            
        Returns:
            List of valid neighbor nodes
        """
        neighbors = []
        
        # 8-connected neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                new_x = node.x + dx
                new_y = node.y + dy
                
                if self.is_valid_cell(new_x, new_y):
                    neighbors.append(Node(new_x, new_y))
        
        return neighbors
    
    def get_move_cost(self, node1: Node, node2: Node) -> float:
        """
        Calculate movement cost between two adjacent nodes.
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            Movement cost in meters
        """
        dx = node2.x - node1.x
        dy = node2.y - node1.y
        
        # Diagonal moves cost more
        if abs(dx) == 1 and abs(dy) == 1:
            return math.sqrt(2) * self.map_metadata['resolution']
        else:
            return self.map_metadata['resolution']
    
    def plan_path(self, start_x: float, start_y: float,
                  goal_x: float, goal_y: float,
                  start_yaw: float = None,
                  simplify: bool = True,
                  smooth_corners: bool = True,
                  add_rotation_waypoints: bool = True) -> Optional[List[Tuple[float, float, float]]]:
        """
        Plan a path from start to goal using A* algorithm.

        This method automatically applies path simplification and corner smoothing
        to produce a navigation-ready path. If the robot is facing away from the goal,
        it adds rotation waypoints at the start to orient the robot towards the goal.

        Args:
            start_x: Start X coordinate in world frame (meters)
            start_y: Start Y coordinate in world frame (meters)
            goal_x: Goal X coordinate in world frame (meters)
            goal_y: Goal Y coordinate in world frame (meters)
            start_yaw: Starting yaw angle in radians (optional, for rotation waypoints)
            simplify: Whether to simplify the path (remove unnecessary waypoints)
            smooth_corners: Whether to smooth sharp corners
            add_rotation_waypoints: Whether to add rotation waypoints if facing away from goal

        Returns:
            List of (x, y, yaw) tuples in world coordinates, or None if no path found.
            For rotation waypoints, yaw specifies orientation. For movement waypoints, yaw is None.
        """
        if self.map_data is None or self.map_metadata is None:
            raise ValueError("Map not set. Call set_map() first.")

        # Check if robot needs initial rotation towards goal
        rotation_waypoints = []

        if start_yaw is not None and add_rotation_waypoints:
            direction_to_goal = math.atan2(goal_y - start_y, goal_x - start_x)
            # Use signed angle difference to determine rotation direction
            angle_diff_signed = self._normalize_angle(direction_to_goal - start_yaw)
            angle_diff = abs(angle_diff_signed)

            print(f"A*: Current yaw: {math.degrees(start_yaw):.1f}°, Direction to goal: {math.degrees(direction_to_goal):.1f}°")
            print(f"A*: Angle difference: {math.degrees(angle_diff):.1f}°")

            # If facing more than 45 degrees away from goal, add rotation waypoints
            # Lowered threshold from 90 to 45 degrees for smoother starts
            if angle_diff > math.pi / 4:  # 45 degrees
                print(f"A*: Robot not aligned with goal, adding rotation waypoints first")

                # Add rotation waypoints with gradually changing orientation
                num_turn_points = 8  # Increased from 5 for smoother rotation
                for i in range(num_turn_points):
                    # Interpolate from start_yaw to direction_to_goal
                    t = (i + 1) / num_turn_points
                    intermediate_yaw = start_yaw + t * angle_diff_signed
                    # Waypoint with explicit yaw: (x, y, yaw)
                    rotation_waypoints.append((start_x, start_y, intermediate_yaw))

                print(f"A*: Added {num_turn_points} rotation waypoints at ({start_x:.2f}, {start_y:.2f})")

        # Convert to map coordinates
        start_map_x, start_map_y = self.world_to_map(start_x, start_y)
        goal_map_x, goal_map_y = self.world_to_map(goal_x, goal_y)

        # Validate start and goal
        if not self.is_valid_cell(start_map_x, start_map_y):
            print(f"Warning: Start position ({start_x}, {start_y}) is in an occupied cell")
            return None

        if not self.is_valid_cell(goal_map_x, goal_map_y):
            print(f"Warning: Goal position ({goal_x}, {goal_y}) is in an occupied cell")
            return None

        # Initialize start and goal nodes
        start_node = Node(start_map_x, start_map_y, g_cost=0.0)
        goal_node = Node(goal_map_x, goal_map_y)

        start_node.h_cost = self.heuristic(start_node, goal_node)
        start_node.f_cost = start_node.g_cost + start_node.h_cost

        # Open set (priority queue) and closed set
        open_set = [start_node]
        heapq.heapify(open_set)
        closed_set: Set[Node] = set()

        # Dictionary to track best g_cost for each node
        g_costs = {(start_node.x, start_node.y): 0.0}

        # A* search
        while open_set:
            # Get node with lowest f_cost
            current = heapq.heappop(open_set)

            # Skip if already processed with better cost
            if (current.x, current.y) in closed_set:
                continue

            closed_set.add((current.x, current.y))

            # Check if goal reached
            if current.x == goal_node.x and current.y == goal_node.y:
                # Reconstruct raw path
                path = []
                node = current
                while node is not None:
                    world_x, world_y = self.map_to_world(node.x, node.y)
                    # Movement waypoints have None for yaw (will be computed from path direction)
                    path.append((world_x, world_y, None))
                    node = node.parent
                path.reverse()

                # Apply path post-processing
                # DISABLED: Smoothing was causing paths to cut through walls
                # if simplify and len(path) > 2:
                #     path = self.simplify_path(path, keep_corners=True)

                # # Apply Gaussian smoothing to reduce zigzag from grid movement
                # if len(path) > 2:
                #     path = self.gaussian_smooth_path(path, sigma=0.8, iterations=2)

                # if smooth_corners and len(path) > 2:
                #     path = self.smooth_corners(path, corner_radius=0.4)

                # Prepend rotation waypoints if needed
                if rotation_waypoints:
                    path = rotation_waypoints + path

                return path

            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                neighbor_key = (neighbor.x, neighbor.y)

                # Skip if already in closed set
                if neighbor_key in closed_set:
                    continue

                # Calculate tentative g_cost
                move_cost = self.get_move_cost(current, neighbor)

                # Add penalty for direction changes to encourage straighter paths
                direction_penalty = 0.0
                if current.parent is not None:
                    # Vector from parent to current
                    v1_x = current.x - current.parent.x
                    v1_y = current.y - current.parent.y
                    # Vector from current to neighbor
                    v2_x = neighbor.x - current.x
                    v2_y = neighbor.y - current.y

                    # If not a straight line, add penalty
                    if v1_x != 0 or v1_y != 0:
                        # Normalize and compute dot product
                        len1 = math.sqrt(v1_x**2 + v1_y**2)
                        len2 = math.sqrt(v2_x**2 + v2_y**2)
                        if len1 > 0 and len2 > 0:
                            dot = (v1_x * v2_x + v1_y * v2_y) / (len1 * len2)
                            # Penalty proportional to angle change (0 for straight, 1 for perpendicular)
                            direction_penalty = (1.0 - dot) * 0.05 * self.map_metadata['resolution']

                tentative_g = current.g_cost + move_cost + direction_penalty

                # Check if this is a better path
                if neighbor_key not in g_costs or tentative_g < g_costs[neighbor_key]:
                    neighbor.g_cost = tentative_g
                    neighbor.h_cost = self.heuristic(neighbor, goal_node)
                    # Slightly increase heuristic weight to prefer more direct paths
                    neighbor.f_cost = neighbor.g_cost + 1.05 * neighbor.h_cost
                    neighbor.parent = current

                    g_costs[neighbor_key] = tentative_g
                    heapq.heappush(open_set, neighbor)

        # No path found
        print("A*: No path found from start to goal")
        return None
    
    def gaussian_smooth_path(self, waypoints: List[Tuple[float, float, float]],
                            sigma: float = 0.8, iterations: int = 2) -> List[Tuple[float, float, float]]:
        """
        Smooth path using Gaussian filter to reduce zigzag from grid-based planning.

        Args:
            waypoints: List of (x, y, yaw) tuples (yaw can be None)
            sigma: Standard deviation for Gaussian kernel (higher = more smoothing)
            iterations: Number of smoothing passes

        Returns:
            Smoothed waypoints
        """
        if len(waypoints) < 3:
            return waypoints

        smoothed = list(waypoints)

        for _ in range(iterations):
            new_smoothed = [smoothed[0]]  # Keep first point fixed

            for i in range(1, len(smoothed) - 1):
                # Gaussian kernel weights for 5-point stencil
                weights = []
                points = []

                for offset in [-2, -1, 0, 1, 2]:
                    idx = max(0, min(len(smoothed) - 1, i + offset))
                    # Gaussian weight
                    w = math.exp(-(offset**2) / (2 * sigma**2))
                    weights.append(w)
                    points.append(smoothed[idx])

                # Normalize weights
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]

                # Weighted average of positions
                x = sum(w * p[0] for w, p in zip(weights, points))
                y = sum(w * p[1] for w, p in zip(weights, points))

                # Keep yaw as None (will be computed from path direction)
                new_smoothed.append((x, y, None))

            new_smoothed.append(smoothed[-1])  # Keep last point fixed
            smoothed = new_smoothed

        return smoothed

    def smooth_corners(self, waypoints: List[Tuple[float, float, float]],
                      corner_radius: float = 0.4) -> List[Tuple[float, float, float]]:
        """
        Smooth sharp corners in the path using arc interpolation.

        Args:
            waypoints: List of (x, y, yaw) tuples (yaw can be None)
            corner_radius: Radius for corner smoothing in meters

        Returns:
            Smoothed waypoints
        """
        if len(waypoints) < 3:
            return waypoints

        smoothed = [waypoints[0]]  # Keep first point

        for i in range(1, len(waypoints) - 1):
            prev = waypoints[i - 1]
            curr = waypoints[i]
            next_pt = waypoints[i + 1]

            # Vectors from current point (using only x, y)
            v1_x, v1_y = prev[0] - curr[0], prev[1] - curr[1]
            v2_x, v2_y = next_pt[0] - curr[0], next_pt[1] - curr[1]

            # Normalize vectors
            len1 = math.sqrt(v1_x**2 + v1_y**2)
            len2 = math.sqrt(v2_x**2 + v2_y**2)

            if len1 < 1e-6 or len2 < 1e-6:
                smoothed.append(curr)
                continue

            v1_x, v1_y = v1_x / len1, v1_y / len1
            v2_x, v2_y = v2_x / len2, v2_y / len2

            # Calculate angle between vectors
            dot = v1_x * v2_x + v1_y * v2_y
            angle = math.acos(max(-1.0, min(1.0, dot)))

            # If angle is sharp (> 10 degrees), add corner smoothing
            if angle > math.radians(10):
                # Calculate corner cut distance with larger cuts for safety
                cut_dist = min(corner_radius, len1 * 0.5, len2 * 0.5)

                # Points before and after corner
                before = (curr[0] + v1_x * cut_dist, curr[1] + v1_y * cut_dist)
                after = (curr[0] + v2_x * cut_dist, curr[1] + v2_y * cut_dist)

                # Add more arc waypoints for smoother curves
                num_arc_points = max(5, int(angle / math.radians(10)))
                for j in range(num_arc_points + 1):
                    t = j / num_arc_points
                    # Circular arc interpolation for smoother corners
                    # Use quadratic Bezier curve for better smoothing
                    t_inv = 1 - t
                    x = t_inv**2 * before[0] + 2 * t_inv * t * curr[0] + t**2 * after[0]
                    y = t_inv**2 * before[1] + 2 * t_inv * t * curr[1] + t**2 * after[1]
                    smoothed.append((x, y, None))
            else:
                smoothed.append(curr)

        smoothed.append(waypoints[-1])  # Keep last point
        return smoothed

    def simplify_path(self, path: List[Tuple[float, float, float]],
                     keep_corners: bool = True) -> List[Tuple[float, float, float]]:
        """
        Simplify path by removing unnecessary waypoints (line-of-sight check).

        Args:
            path: Original path as list of (x, y, yaw) tuples (yaw can be None)
            keep_corners: If True, preserve corner waypoints even if line of sight exists

        Returns:
            Simplified path
        """
        if len(path) <= 2:
            return path

        # First, identify corner points (significant direction changes)
        corners = {0, len(path) - 1}  # Always keep start and end

        if keep_corners and len(path) >= 3:
            for i in range(1, len(path) - 1):
                # Calculate vectors (using only x, y)
                v1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
                v2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])

                len1 = math.sqrt(v1[0]**2 + v1[1]**2)
                len2 = math.sqrt(v2[0]**2 + v2[1]**2)

                if len1 > 1e-6 and len2 > 1e-6:
                    # Normalize and compute dot product
                    v1_norm = (v1[0]/len1, v1[1]/len1)
                    v2_norm = (v2[0]/len2, v2[1]/len2)
                    dot = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]

                    # If direction change > 20 degrees, keep as corner
                    if dot < math.cos(math.radians(20)):
                        corners.add(i)

        simplified = [path[0]]
        i = 0

        while i < len(path) - 1:
            # Try to skip as many points as possible
            for j in range(len(path) - 1, i + 1, -1):
                # If there's a corner between i and j, don't skip past it
                has_corner = any(c in corners for c in range(i + 1, j))

                # has_line_of_sight expects (x, y) tuples
                if not has_corner and self.has_line_of_sight((path[i][0], path[i][1]), (path[j][0], path[j][1])):
                    simplified.append(path[j])
                    i = j
                    break
            else:
                # No line of sight, add next point
                i += 1
                if i < len(path):
                    simplified.append(path[i])

        return simplified
    
    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to [-pi, pi] range.

        Args:
            angle: Angle in radians

        Returns:
            Normalized angle in [-pi, pi]
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def has_line_of_sight(self, point1: Tuple[float, float],
                          point2: Tuple[float, float]) -> bool:
        """
        Check if there's a clear line of sight between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            True if line of sight is clear
        """
        x1, y1 = point1
        x2, y2 = point2
        
        # Convert to map coordinates
        map_x1, map_y1 = self.world_to_map(x1, y1)
        map_x2, map_y2 = self.world_to_map(x2, y2)
        
        # Bresenham's line algorithm to check all cells along the line
        dx = abs(map_x2 - map_x1)
        dy = abs(map_y2 - map_y1)
        sx = 1 if map_x1 < map_x2 else -1
        sy = 1 if map_y1 < map_y2 else -1
        err = dx - dy
        
        x, y = map_x1, map_y1
        
        while True:
            if not self.is_valid_cell(x, y):
                return False
            
            if x == map_x2 and y == map_y2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True

