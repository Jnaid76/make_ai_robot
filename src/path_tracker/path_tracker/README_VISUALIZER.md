# A* Path Planning Visualizer

This tool allows you to visualize and debug the A* path planning algorithm on a 2D map without running Gazebo.

## Features

- **Interactive path planning**: Click to set start/goal positions
- **Corner detection**: Automatically identifies and highlights sharp corners
- **Wall clearance analysis**: Checks if the robot will crash into walls at corners
- **Path smoothing visualization**: Compare raw vs. smoothed paths
- **Obstacle inflation visualization**: See the inflated obstacle boundaries
- **Real-time statistics**: Path length, number of corners, minimum clearance

## Usage

### Basic Usage

```bash
cd /home/tkweon426/craip_2025f_g4/src/path_tracker/path_tracker

# Visualize with maze_room map
python3 visualize_astar_path.py --map ../../go1_simulation/maps/maze_room.yaml

# Visualize with hospital map
python3 visualize_astar_path.py --map ../../go1_simulation/maps/hospital.yaml

# Custom inflation radius
python3 visualize_astar_path.py --map ../../go1_simulation/maps/maze_room.yaml --inflation 0.5
```

### Interactive Controls

- **Left Click**: Set start position (green marker)
- **Right Click**: Set goal position (red marker)
- **Middle Click**: Clear path
- **'i' key**: Toggle obstacle inflation visualization
- **'r' key**: Toggle raw (unsmoothed) path visualization

### Understanding the Output

#### Console Output
```
Planning path...
Path found with 25 waypoints
Raw path has 87 waypoints
Path length: 8.45m
Found 3 corners:
  Corner at waypoint 8: 78.3° at (2.34, 1.56)
  Corner at waypoint 15: 92.1° at (3.12, 2.88)
  Corner at waypoint 20: 45.7° at (4.56, 3.21)
Minimum wall clearance at corners: 0.28m
WARNING: Clearance (0.28m) < inflation radius (0.35m)!
         Robot may crash into walls at corners!
```

#### Visual Elements
- **Green circle**: Start position
- **Red circle**: Goal position
- **Blue line with dots**: Smoothed path (what the robot will follow)
- **Red dashed line**: Raw A* path (before smoothing) - toggle with 'r'
- **Red dots**: Detected corners (turns > 10°)
- **Red transparent circles**: Inflated obstacles - toggle with 'i'

#### Warning Signs
- **"NO PATH FOUND!"**: Start or goal is in an obstacle
- **"Clearance < inflation radius"**: Path is too close to walls - robot will likely crash!
- **Many sharp corners**: Path may be too zigzaggy - increase smoothing

## Debugging Wall Crashes

If your robot crashes into walls at corners:

1. **Check corner clearance**:
   - Look at the console output for "Minimum wall clearance"
   - If clearance < inflation_radius, the path is too tight!

2. **Visualize inflation**:
   - Press 'i' to see the inflated obstacle boundaries
   - Path should stay outside the red circles

3. **Common fixes**:
   - Increase `inflation_radius` (default: 0.35m)
   - Increase `corner_radius` in [astar_planner.py:343](../../../src/path_tracker/path_tracker/astar_planner.py#L343)
   - Lower corner smoothing threshold (currently 10°)
   - Increase Gaussian smoothing sigma

4. **Test different scenarios**:
   - Try narrow corridors
   - Try sharp 90° corners
   - Try diagonal approaches to corners

## Map Files

Available maps:
- `maze_room.yaml`: Room with maze-like corridors (good for testing tight spaces)
- `hospital.yaml`: Large hospital environment
- `empty.yaml`: Empty map for testing

## Tips

- **Test corner scenarios**: Right-click on the opposite side of a corner to see how the robot will navigate it
- **Check multiple paths**: Try different start/goal combinations to find problematic areas
- **Iterate quickly**: Use this tool to test parameter changes before running in Gazebo
- **Save screenshots**: Use matplotlib's save button to document problematic paths

## Requirements

- Python 3
- matplotlib
- numpy
- opencv-python (cv2)
- PyYAML

Install missing dependencies:
```bash
pip3 install matplotlib numpy opencv-python pyyaml
```
