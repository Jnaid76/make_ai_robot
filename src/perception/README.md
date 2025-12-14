# Perception Module - YOLOv8 Object Detection

ROS2 perception package for real-time object detection using YOLOv8 with RGB-D camera.

## Overview

This package provides:
- Object detection for competition missions (food, stop signs, cones, nurse, delivery box)
- Distance estimation using depth camera
- Center region detection (middle 3/5 of image)
- Mission-specific nodes (e.g., find_food_and_bark)

## Dependencies

### System Dependencies
```bash
sudo apt install python3-opencv
```

### Python Dependencies
```bash
cd /home/tkweon426/craip_2025f_g4/src/perception
pip install -r requirements.txt
```

This installs:
- ultralytics (YOLOv8)
- opencv-python
- torch, torchvision
- numpy

## Build

```bash
cd /home/tkweon426/craip_2025f_g4
colcon build --packages-select perception
source install/setup.bash
```

## Phase 2: Data Collection

### Object Classes (12 total)
- **Good food**: banana, apple, pizza
- **Bad food**: rotten_banana, rotten_apple, rotten_pizza
- **Mission objects**: stop_sign, nurse, cone_red, cone_green, cone_blue, delivery_box

### Step 1: Setup Gazebo Environment

You need to place the objects in your Gazebo world. Objects should be available in the `environment/models/` directory or you can spawn them programmatically.

**Terminal 1 - Launch Gazebo:**
```bash
ros2 launch go1_simulation go1.gazebo.launch.py use_gt_pose:=true
```

### Step 2: Collect Training Images

**Terminal 2 - Start Data Collection:**
```bash
source /home/tkweon426/craip_2025f_g4/install/setup.bash

# Collect training images (saves to data/images/train/)
ros2 launch perception data_collection.launch.py

# Or specify custom directory and interval:
ros2 launch perception data_collection.launch.py \
  save_dir:=/home/tkweon426/craip_2025f_g4/src/perception/data/images/train \
  capture_interval:=1.5
```

**Terminal 3 - Move Robot or Objects:**
```bash
# Option 1: Keyboard control
ros2 run go1_simulation move_go1_with_keyboard.py

# Option 2: Manual teleportation of objects in Gazebo GUI
# Option 3: Use service calls to spawn/move objects
```

### Data Collection Tips

**Diversity is key!** Collect images with:
- **Different distances**: 0.5m, 1m, 2m, 3m from camera
- **Different angles**: Front view, side view, 45° angles
- **Multiple objects**: Several objects in same frame
- **Partial occlusions**: Objects partially hidden
- **Various positions**: Floor, table, different heights

**Target dataset:**
- **Training**: 800-1200 images (80%)
- **Validation**: 200-300 images (20%)
- **Per class**: 100-150 images minimum
- **Total**: 1200-1800 images

### Step 3: Collect Validation Images

After collecting training images, collect validation set:

```bash
ros2 launch perception data_collection.launch.py \
  save_dir:=/home/tkweon426/craip_2025f_g4/src/perception/data/images/val \
  capture_interval:=2.0
```

Run for ~5-10 minutes in **different scenarios** than training.

### Step 4: Label Images with Roboflow

See `ROBOFLOW_GUIDE.md` for detailed instructions.

**Quick steps:**
1. Go to https://roboflow.com and create a project
2. Upload images from `data/images/train/` and `data/images/val/`
3. Define the 12 classes (see `data/classes.txt` for reference)
4. Label images using the web interface
5. Export dataset in **YOLOv8** format
6. Download and extract to replace `data/` directory
7. Verify with: `python3 perception/verify_dataset.py`

### Step 5: Verify Dataset Structure

After labeling, verify your dataset structure:

```bash
cd /home/tkweon426/craip_2025f_g4/src/perception/data

# Check image counts
ls images/train | wc -l  # Should be 800-1200
ls images/val | wc -l    # Should be 200-300

# Check label counts (should match image counts)
ls labels/train | wc -l
ls labels/val | wc -l
```

**Expected structure:**
```
data/
├── images/
│   ├── train/  (800-1200 .jpg files)
│   └── val/    (200-300 .jpg files)
└── labels/
    ├── train/  (800-1200 .txt files)
    └── val/    (200-300 .txt files)
```

Each `.txt` label file contains lines like:
```
0 0.5 0.5 0.2 0.3
```
Format: `<class_id> <x_center> <y_center> <width> <height>` (all normalized 0-1)

## Phase 3: Model Training

See `TRAINING_GUIDE.md` for detailed instructions.

**Quick start:**
```bash
cd /Users/zahra/craip_2025f_g4/src/perception

# Verify dataset is ready
python3 perception/verify_dataset.py

# Train model
python3 training/train_yolo.py
```

**Training works on macOS, Ubuntu, or cloud!** See `TRAINING_GUIDE.md` for details.

## Phase 4: Detection Implementation

After training, you'll implement:
- Full YOLO detector with RGB+Depth synchronization
- Bounding box visualization
- Distance estimation
- Center region detection

## Phase 5-7: Testing & Evaluation

- Interface viewer testing
- Mission node examples
- ROS bag evaluation
- Video recording for submission

## Usage (After Full Implementation)

### Run Perception System

```bash
# Terminal 1: Gazebo
ros2 launch go1_simulation go1.gazebo.launch.py use_gt_pose:=true

# Terminal 2: Perception
ros2 launch perception perception.launch.py

# Terminal 3: Interface Viewer
ros2 launch module_test interface_viewer.launch.py
```

### Run Mission Nodes

```bash
# Example: Find food and bark
ros2 launch perception find_food_and_bark.launch.py
```

## Topics

### Published
- `/camera/detections/image` - RGB with bounding boxes
- `/detections/labels` - Detected object labels (comma-separated)
- `/detections/distance` - Distance to centered object (meters)
- `/robot_dog/speech` - "bark" or "None" (from mission nodes)

### Subscribed
- `/camera_top/image` - RGB camera
- `/camera_top/depth` - Depth image

## Package Structure

```
perception/
├── perception/           # Python modules
│   ├── yolo_detector.py       # Main detection node
│   ├── data_collector.py      # Data collection
│   ├── distance_estimator.py  # Distance calculation
│   ├── find_food_and_bark.py # Mission node example
│   └── utils/
│       ├── center_detector.py
│       └── visualization.py
├── launch/              # Launch files
├── config/              # Configuration files
├── models/              # YOLO weights
├── data/                # Training data (gitignored)
└── training/            # Training scripts
```

## Current Status

✅ **Phase 1 Complete**: Package setup, skeleton detector
🔄 **Phase 2 In Progress**: Data collection ready, need to collect & label images
⏳ **Phase 3**: Model training (coming next)
⏳ **Phase 4**: Full detection implementation
⏳ **Phase 5-7**: Testing and evaluation

## Notes

- Data collection captures images every 2 seconds by default (adjustable)
- Aim for 100-150 images per class for good training results
- Use diverse scenarios (distances, angles, lighting)
- Label format must be YOLO (not PascalVOC)
- Recommended: Use Roboflow for faster labeling workflow

## Troubleshooting

**No images being saved?**
- Check that Gazebo is running
- Verify camera topic: `ros2 topic echo /camera_top/image --once`
- Check save directory permissions

**LabelImg not in YOLO format?**
- Click the "PascalVOC" button to toggle to "YOLO" format
- Verify .txt files contain normalized coordinates (0-1 range)

**Image counts don't match?**
- Every .jpg in images/ should have corresponding .txt in labels/
- Missing labels mean those images won't be used in training

## License

Apache 2.0
