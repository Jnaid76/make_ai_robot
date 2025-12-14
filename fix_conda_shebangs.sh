#!/bin/bash
# Fix shebangs to use conda Python after colcon build

CONDA_PYTHON="/home/tkweon426/anaconda3/envs/language_command_handler/bin/python3"

echo "Fixing shebangs to use conda Python..."

# Fix yolo_detector
if [ -f "install/perception/lib/perception/yolo_detector" ]; then
    sed -i "1s|.*|#!${CONDA_PYTHON}|" install/perception/lib/perception/yolo_detector
    echo "✓ Fixed yolo_detector"
fi

# Fix data_collector
if [ -f "install/perception/lib/perception/data_collector" ]; then
    sed -i "1s|.*|#!${CONDA_PYTHON}|" install/perception/lib/perception/data_collector
    echo "✓ Fixed data_collector"
fi

echo "Done! You can now run: ros2 launch perception perception.launch.py"
