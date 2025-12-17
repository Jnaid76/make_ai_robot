from setuptools import setup
from glob import glob
import os

package_name = "perception"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name, package_name + ".utils"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=[
        "setuptools",
        "opencv-python",
        "ultralytics",
        "numpy",
        "torch",
        "torchvision",
    ],
    zip_safe=True,
    maintainer="tkweon426",
    maintainer_email="tkweon426@snu.ac.kr",
    description="YOLOv8-based perception module for object detection",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yolo_detector = perception.yolo_detector:main",
            "data_collector = perception.data_collector:main",
        ],
    },
)
