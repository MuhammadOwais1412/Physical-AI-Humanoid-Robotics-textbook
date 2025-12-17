# Solutions for Module 3: The AI-Robot Brain (NVIDIA Isaac)

## Exercise 1: Isaac Sim Overview
**Problem**: Create a synthetic dataset using Isaac Sim.

**Solution**:

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.synthetic_utils import SyntheticDataHelper
import carb

class IsaacSyntheticData:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)

    def setup_scene(self):
        # Add a simple object to the scene
        add_reference_to_stage(
            usd_path="/Isaac/Props/Prims/prim.usda",
            prim_path="/World/Cube"
        )

    def generate_dataset(self):
        # Initialize synthetic data helper
        sd_helper = SyntheticDataHelper()

        # Configure the synthetic data pipeline
        sd_helper.initialize(
            camera_prim=omni.usd.get_context().get_stage().GetPrimAtPath("/World/Camera"),
            viewport_name="Viewport"
        )

        # Generate various synthetic data
        rgb_data = sd_helper.get_rgb_data()
        depth_data = sd_helper.get_depth_data()
        segmentation_data = sd_helper.get_segmentation_data()

        return {
            'rgb': rgb_data,
            'depth': depth_data,
            'segmentation': segmentation_data
        }

# Usage
dataset_generator = IsaacSyntheticData()
dataset_generator.setup_scene()
synthetic_data = dataset_generator.generate_dataset()
```

## Exercise 2: VSLAM and Navigation
**Problem**: Implement visual SLAM using Isaac ROS.

**Solution**:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np

class IsaacVSLAMNode(Node):
    def __init__(self):
        super().__init__('isaac_vsalm_node')

        # Subscribers for camera data
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            10
        )

        # Publisher for estimated pose
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/visual_slam/pose',
            10
        )

        # Initialize VSLAM components
        self.feature_detector = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.previous_keypoints = None
        self.previous_descriptors = None
        self.current_pose = np.eye(4)

    def image_callback(self, msg):
        # Convert ROS image to OpenCV
        image = self.ros_to_cv2(msg)

        # Detect features
        keypoints = self.feature_detector.detect(image)
        keypoints, descriptors = self.feature_detector.compute(image, keypoints)

        if self.previous_descriptors is not None:
            # Match features with previous frame
            matches = self.matcher.match(self.previous_descriptors, descriptors)

            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Extract matched points
            if len(matches) >= 10:  # Minimum matches required
                src_points = np.float32([self.previous_keypoints[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
                dst_points = np.float32([keypoints[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

                # Estimate motion using Essential matrix
                E, mask = cv2.findEssentialMat(src_points, dst_points, self.camera_matrix, threshold=1, prob=0.999)

                if E is not None:
                    # Recover pose
                    _, R, t, mask = cv2.recoverPose(E, src_points, dst_points, self.camera_matrix)

                    # Update current pose
                    transformation = np.eye(4)
                    transformation[:3, :3] = R
                    transformation[:3, 3] = t.flatten()
                    self.current_pose = self.current_pose @ transformation

                    # Publish pose
                    self.publish_pose()

        # Store current frame data for next iteration
        self.previous_keypoints = keypoints
        self.previous_descriptors = descriptors

    def camera_info_callback(self, msg):
        # Extract camera matrix from camera info
        self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def ros_to_cv2(self, ros_image):
        # Convert ROS image message to OpenCV image
        height = ros_image.height
        width = ros_image.width
        encoding = ros_image.encoding

        # This is a simplified conversion - in practice, you'd use cv_bridge
        image_data = np.frombuffer(ros_image.data, dtype=np.uint8)
        image = image_data.reshape((height, width, -1))

        return image

    def publish_pose(self):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"

        # Convert pose to message format
        pose_msg.pose.position.x = self.current_pose[0, 3]
        pose_msg.pose.position.y = self.current_pose[1, 3]
        pose_msg.pose.position.z = self.current_pose[2, 3]

        # Convert rotation matrix to quaternion
        qw = np.sqrt(1 + self.current_pose[0,0] + self.current_pose[1,1] + self.current_pose[2,2]) / 2
        qx = (self.current_pose[2,1] - self.current_pose[1,2]) / (4 * qw)
        qy = (self.current_pose[0,2] - self.current_pose[2,0]) / (4 * qw)
        qz = (self.current_pose[1,0] - self.current_pose[0,1]) / (4 * qw)

        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    vsalm_node = IsaacVSLAMNode()

    try:
        rclpy.spin(vsalm_node)
    except KeyboardInterrupt:
        pass
    finally:
        vsalm_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercise 3: Biped Locomotion
**Problem**: Implement a simple biped locomotion controller.

**Solution**:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np

class BipedLocomotionController(Node):
    def __init__(self):
        super().__init__('biped_locomotion_controller')

        # Publishers for joint commands
        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/joint_group_position_controller/commands',
            10
        )

        # Timer for locomotion control
        self.timer = self.create_timer(0.01, self.locomotion_callback)  # 100Hz

        # Walking gait parameters
        self.step_phase = 0.0
        self.step_frequency = 1.0  # Hz
        self.step_amplitude = 0.1  # meters
        self.step_offset = 0.0

        # Joint names for biped robot
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
        ]

    def locomotion_callback(self):
        # Update step phase
        self.step_phase += 2 * np.pi * self.step_frequency * 0.01
        if self.step_phase > 2 * np.pi:
            self.step_phase -= 2 * np.pi

        # Calculate joint positions for walking gait
        joint_positions = self.calculate_gait_pattern(self.step_phase)

        # Publish joint commands
        cmd_msg = Float64MultiArray()
        cmd_msg.data = joint_positions
        self.joint_cmd_pub.publish(cmd_msg)

    def calculate_gait_pattern(self, phase):
        # Simple walking gait pattern
        # This is a simplified version - real biped control is much more complex

        # Left leg pattern (opposite to right leg)
        left_hip = self.step_amplitude * np.sin(phase)
        left_knee = self.step_amplitude * 0.5 * np.sin(phase + np.pi/2)
        left_ankle = self.step_amplitude * 0.3 * np.sin(phase - np.pi/4)

        # Right leg pattern (180 degrees out of phase with left)
        right_hip = self.step_amplitude * np.sin(phase + np.pi)
        right_knee = self.step_amplitude * 0.5 * np.sin(phase + np.pi + np.pi/2)
        right_ankle = self.step_amplitude * 0.3 * np.sin(phase + np.pi - np.pi/4)

        return [left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle]

def main(args=None):
    rclpy.init(args=args)
    locomotion_controller = BipedLocomotionController()

    try:
        rclpy.spin(locomotion_controller)
    except KeyboardInterrupt:
        pass
    finally:
        locomotion_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercise 4: Perception Pipelines
**Problem**: Create a perception pipeline for object detection.

**Solution**:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from std_msgs.msg import Header
import cv2
import numpy as np
from cv_bridge import CvBridge

class PerceptionPipeline(Node):
    def __init__(self):
        super().__init__('perception_pipeline')

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/camera/depth/points',
            self.pointcloud_callback,
            10
        )

        # Publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/perception/detections',
            10
        )

        # Initialize components
        self.bridge = CvBridge()
        self.object_detector = self.initialize_detector()

    def initialize_detector(self):
        # Initialize a simple detector (in practice, this would be a deep learning model)
        # For this example, we'll use OpenCV's built-in HOG descriptor for people detection
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        return hog

    def image_callback(self, msg):
        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run object detection
        detections = self.detect_objects(cv_image)

        # Publish detections
        self.publish_detections(detections, msg.header)

    def detect_objects(self, image):
        # Run detection on the image
        # This is a simplified example using HOG for people detection
        (rects, weights) = self.object_detector.detectMultiScale(
            image, winStride=(4, 4), padding=(8, 8), scale=1.05
        )

        detections = []
        for (x, y, w, h) in rects:
            detection = {
                'bbox': (x, y, w, h),
                'confidence': float(weights[len(detections)] if len(weights) > len(detections) else 0.9)
            }
            detections.append(detection)

        return detections

    def pointcloud_callback(self, msg):
        # Process point cloud data for 3D object detection
        # This would typically involve clustering, segmentation, etc.
        self.get_logger().info(f'Received point cloud with {msg.height * msg.width} points')

    def publish_detections(self, detections, header):
        detection_array = Detection2DArray()
        detection_array.header = header

        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']

            vision_detection = Detection2D()
            vision_detection.header = header

            # Set bounding box
            vision_detection.bbox.center.x = bbox[0] + bbox[2] / 2
            vision_detection.bbox.center.y = bbox[1] + bbox[3] / 2
            vision_detection.bbox.size_x = bbox[2]
            vision_detection.bbox.size_y = bbox[3]

            # Add hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = "person"  # For this example
            hypothesis.score = confidence
            vision_detection.results.append(hypothesis)

            detection_array.detections.append(vision_detection)

        self.detection_pub.publish(detection_array)

def main(args=None):
    rclpy.init(args=args)
    perception_pipeline = PerceptionPipeline()

    try:
        rclpy.spin(perception_pipeline)
    except KeyboardInterrupt:
        pass
    finally:
        perception_pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```