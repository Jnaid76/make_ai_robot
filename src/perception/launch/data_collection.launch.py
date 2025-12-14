import importlib
from pathlib import Path
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Prefer workspace src path: <workspace>/src/perception/perception/data/images/train
    launch_file = Path(__file__).resolve()
    default_save_dir = None
    for parent in launch_file.parents:
        if parent.name == 'install':
            ws_root = parent.parent
            default_save_dir = ws_root / 'src' / 'perception' / 'perception' / 'data' / 'images' / 'train'
            break

    if default_save_dir is None:
        perception_pkg_path = Path(importlib.import_module('perception').__file__).resolve().parent
        default_save_dir = perception_pkg_path / 'data' / 'images' / 'train'

    default_save_dir = str(default_save_dir)

    return LaunchDescription([
        DeclareLaunchArgument(
            'save_dir',
            default_value=default_save_dir,
            description='Directory to save collected images'
        ),
        DeclareLaunchArgument(
            'capture_interval',
            default_value='0.5',
            description='Interval between image captures in seconds'
        ),

        Node(
            package='perception',
            executable='data_collector',
            name='data_collector',
            output='screen',
            parameters=[{
                'save_dir': LaunchConfiguration('save_dir'),
                'capture_interval': LaunchConfiguration('capture_interval'),
            }]
        )
    ])
