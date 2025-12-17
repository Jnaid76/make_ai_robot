import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    # Get the path to the venv Python and the script
    workspace_dir = os.path.expanduser('~/make_ai_robot')
    venv_python = os.path.join(workspace_dir, 'perception_venv', 'bin', 'python3')
    script_path = os.path.join(workspace_dir, 'src', 'perception', 'perception', 'yolo_detector.py')
    
    # Build environment with ROS2 paths and venv
    env = os.environ.copy()
    ros_python_path = '/opt/ros/jazzy/lib/python3.12/site-packages'
    workspace_python_path = os.path.join(workspace_dir, 'install', 'perception', 'lib', 'python3.12', 'site-packages')
    
    # Prepend ROS2 paths to PYTHONPATH
    existing_path = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f"{ros_python_path}:{workspace_python_path}:{existing_path}"
    
    return LaunchDescription([
        ExecuteProcess(
            cmd=[venv_python, script_path],
            name='yolo_detector',
            output='screen',
            env=env,
        )
    ])

