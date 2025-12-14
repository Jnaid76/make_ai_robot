#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber

from perception.utils.visualization import DetectionVisualizer
from perception.utils.center_detector import CenterDetector
from perception.distance_estimator import DistanceEstimator


class YOLODetector(Node):
    """
    YOLOv8 detector node for object detection with RGB-D camera.
    Uses trained model from training/best.pt for inference.
    """

    def __init__(self):
        super().__init__('yolo_detector')

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # Get package directory - works in both development and installed environments
        # Try multiple possible locations
        current_file = os.path.abspath(__file__)
        possible_roots = [
            os.path.dirname(os.path.dirname(current_file)),  # src/perception/
            os.path.join(os.path.expanduser('~'), 'craip_2025f_g4', 'src', 'perception'),  # Absolute path
            os.path.dirname(current_file),  # perception/perception/
        ]
        
        package_dir = None
        for root in possible_roots:
            test_model = os.path.join(root, 'training', 'best.pt')
            if os.path.exists(test_model):
                package_dir = root
                break
        
        if package_dir is None:
            # Use the first option as default
            package_dir = possible_roots[0]
            self.get_logger().warn(f'Could not find model, using default path: {package_dir}')
        
        model_path = os.path.join(package_dir, 'training', 'best.pt')
        classes_path = os.path.join(package_dir, 'perception', 'data', 'classes.txt')

        # Load YOLO model
        if not os.path.exists(model_path):
            self.get_logger().error(f'Model not found at {model_path}')
            raise FileNotFoundError(f'Model not found at {model_path}')
        
        self.get_logger().info(f'Loading YOLO model from {model_path}')
        self.model = YOLO(model_path)
        self.get_logger().info('YOLO model loaded successfully')

        # Load class names
        self.class_names = self._load_class_names(classes_path)
        self.get_logger().info(f'Loaded {len(self.class_names)} classes: {self.class_names}')

        # Publishers (required topics)
        self.image_pub = self.create_publisher(
            Image, '/camera/detections/image', 10)
        self.labels_pub = self.create_publisher(
            String, '/detections/labels', 10)
        self.distance_pub = self.create_publisher(
            Float32, '/detections/distance', 10)
        self.speech_pub = self.create_publisher(
            String, '/robot_dog/speech', 10)
        
        # Define edible objects (good food)
        self.edible_objects = ['banana', 'apple', 'pizza']

        # Subscribers for RGB and Depth (synchronized)
        rgb_sub = Subscriber(self, Image, '/camera_top/image')
        depth_sub = Subscriber(self, Image, '/camera_top/depth')

        # Synchronize RGB and Depth messages
        self.sync = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.image_callback)

        # Store latest depth image for fallback
        self.latest_depth = None
        self.latest_depth_header = None

        # Frame counter for logging
        self.frame_count = 0

        # Detection confidence threshold
        self.conf_threshold = 0.25

        self.get_logger().info('YOLO Detector initialized')
        self.get_logger().info('Subscribed to: /camera_top/image, /camera_top/depth')
        self.get_logger().info('Publishing to: /camera/detections/image, /detections/labels, /detections/distance, /robot_dog/speech')

    def _load_class_names(self, classes_path: str) -> list:
        """Load class names from classes.txt file"""
        if not os.path.exists(classes_path):
            self.get_logger().warn(f'Classes file not found at {classes_path}, using default names')
            return []
        
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        return classes

    def image_callback(self, rgb_msg: Image, depth_msg: Image):
        """Process synchronized RGB and Depth images"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')

            # Store latest depth for fallback
            self.latest_depth = depth_image
            self.latest_depth_header = depth_msg.header

            # Run YOLO inference
            results = self.model(cv_image, conf=self.conf_threshold, verbose=False)

            # Parse detections
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get class ID and confidence
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Get class name
                    if cls_id < len(self.class_names):
                        label = self.class_names[cls_id]
                    else:
                        label = f'class_{cls_id}'

                    # Calculate distance from depth image
                    bbox = (x1, y1, x2, y2)
                    distance = DistanceEstimator.get_distance(depth_image, bbox)

                    detections.append({
                        'bbox': bbox,
                        'label': label,
                        'conf': conf,
                        'distance': distance
                    })

            # Find centered object
            center_detector = CenterDetector(cv_image.shape[1])
            centered_idx = -1
            centered_distance = -1.0

            for idx, det in enumerate(detections):
                bbox = det['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                if center_detector.is_centered(center_x):
                    if centered_idx == -1 or det['conf'] > detections[centered_idx]['conf']:
                        centered_idx = idx
                        centered_distance = det['distance']

            # Visualize detections
            annotated_image = DetectionVisualizer.draw_detections(
                cv_image, detections, centered_idx)

            # Prepare labels string
            if detections:
                labels_list = [f"{det['label']}" for det in detections]
                labels_str = ','.join(labels_list)
            else:
                labels_str = 'None'

            # Publish results
            # 1. Annotated image
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, 'bgr8')
            annotated_msg.header = rgb_msg.header
            self.image_pub.publish(annotated_msg)

            # 2. Labels
            labels_msg = String()
            labels_msg.data = labels_str
            self.labels_pub.publish(labels_msg)

            # 3. Distance to centered object
            distance_msg = Float32()
            distance_msg.data = centered_distance if centered_idx >= 0 else -1.0
            self.distance_pub.publish(distance_msg)

            # 4. Speech (bark if edible object is centered)
            speech_msg = String()
            if centered_idx >= 0:
                centered_label = detections[centered_idx]['label']
                if centered_label in self.edible_objects:
                    speech_msg.data = 'bark'
                else:
                    speech_msg.data = 'None'
            else:
                speech_msg.data = 'None'
            self.speech_pub.publish(speech_msg)

            self.frame_count += 1

            # Log periodically (every 30 frames)
            if self.frame_count % 30 == 0:
                self.get_logger().info(
                    f'Frame {self.frame_count}: {len(detections)} detections, '
                    f'centered: {detections[centered_idx]["label"] if centered_idx >= 0 else "None"}'
                )

        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {e}')
        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {e}', exc_info=True)


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f'Shutting down. Processed {node.frame_count} frames.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
