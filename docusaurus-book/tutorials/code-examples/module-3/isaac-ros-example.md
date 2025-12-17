# Isaac ROS Example

This example demonstrates how to use Isaac ROS for perception and navigation tasks.

## Isaac ROS Bridge Setup

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge
import numpy as np

class IsaacPerceptionNode(Node):
    def __init__(self):
        super().__init__('isaac_perception_node')

        # Create subscribers for Isaac Sim sensors
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Create publisher for robot commands
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Object detection model (using Isaac ROS DetectNet)
        self.detector = self.initialize_detector()

        self.get_logger().info('Isaac Perception Node initialized')

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run object detection
            detections = self.detector.detect(cv_image)

            # Process detections for navigation
            navigation_commands = self.process_detections_for_navigation(detections)

            # Publish navigation commands
            if navigation_commands:
                self.cmd_vel_pub.publish(navigation_commands)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def odom_callback(self, msg):
        """Process odometry data"""
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        self.get_logger().info(f'Robot position: ({position.x}, {position.y}, {position.z})')

    def initialize_detector(self):
        """Initialize object detection model"""
        # In a real implementation, this would load a DetectNet model
        # For this example, we'll create a mock detector
        class MockDetector:
            def detect(self, image):
                # Mock detection - in reality, this would use Isaac ROS DetectNet
                # or other perception pipelines
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Simple blob detection as example
                detector = cv2.SimpleBlobDetector_create()
                keypoints = detector.detect(gray)
                return [{'bbox': (kp.pt[0], kp.pt[1], 30, 30), 'label': 'object', 'confidence': 0.8}
                        for kp in keypoints]

        return MockDetector()

    def process_detections_for_navigation(self, detections):
        """Process detections to generate navigation commands"""
        cmd = Twist()

        if not detections:
            # No objects detected, continue forward
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
        else:
            # Simple obstacle avoidance
            for detection in detections:
                bbox = detection['bbox']
                center_x = bbox[0] + bbox[2] / 2
                image_center = 320  # Assuming 640x480 image

                if center_x < image_center - 50:
                    # Object on the right, turn left
                    cmd.angular.z = 0.3
                elif center_x > image_center + 50:
                    # Object on the left, turn right
                    cmd.angular.z = -0.3
                else:
                    # Object in center, slow down or stop
                    cmd.linear.x = 0.1

        return cmd

def main(args=None):
    rclpy.init(args=args)
    perception_node = IsaacPerceptionNode()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        pass
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac Sim ROS Bridge Configuration

To use Isaac ROS with Isaac Sim, you need to configure the bridge properly:

```yaml
# config/isaac_ros_bridge.yaml
camera_config:
  image_topic: "/camera/color/image_raw"
  camera_info_topic: "/camera/color/camera_info"
  encoding: "bgr8"

lidar_config:
  scan_topic: "/scan"
  frame_id: "velodyne"

robot_config:
  cmd_vel_topic: "/cmd_vel"
  odom_topic: "/odom"
  base_frame: "base_link"
  odom_frame: "odom"
```

## Isaac Manipulation Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK, GetPositionFK
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class IsaacManipulationNode(Node):
    def __init__(self):
        super().__init__('isaac_manipulation_node')

        # Joint state subscriber
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Trajectory publisher for arm control
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )

        # Initialize with Isaac Sim robot
        self.current_joint_positions = {}
        self.get_logger().info('Isaac Manipulation Node initialized')

    def joint_state_callback(self, msg):
        """Update current joint positions"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]

    def move_to_pose(self, target_pose):
        """Move manipulator to target pose using inverse kinematics"""
        # This would typically call MoveIt's IK service
        joint_trajectory = self.calculate_ik_solution(target_pose)

        if joint_trajectory:
            self.trajectory_pub.publish(joint_trajectory)

    def calculate_ik_solution(self, target_pose):
        """Calculate inverse kinematics solution"""
        # In a real implementation, this would use MoveIt or other IK solvers
        # integrated with Isaac Sim
        trajectory = JointTrajectory()
        trajectory.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

        point = JointTrajectoryPoint()
        # Mock values - in reality, this would come from IK solver
        point.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        point.time_from_start.sec = 2
        point.time_from_start.nanosec = 0

        trajectory.points.append(point)
        return trajectory

    def pick_object(self, object_pose):
        """Execute pick operation on object"""
        # Approach object
        approach_pose = self.calculate_approach_pose(object_pose)
        self.move_to_pose(approach_pose)

        # Descend and grasp
        grasp_pose = self.calculate_grasp_pose(object_pose)
        self.move_to_pose(grasp_pose)

        # Close gripper
        self.close_gripper()

        # Lift object
        lift_pose = self.calculate_lift_pose(grasp_pose)
        self.move_to_pose(lift_pose)

    def calculate_approach_pose(self, object_pose):
        """Calculate approach pose for picking"""
        approach = object_pose.copy()
        approach.position.z += 0.2  # 20cm above object
        return approach

    def close_gripper(self):
        """Close the robot gripper"""
        # Publish gripper command
        # Implementation depends on gripper type
        pass
```

## Isaac Perception Pipeline Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Header
import numpy as np

class IsaacPerceptionPipeline(Node):
    def __init__(self):
        super().__init__('isaac_perception_pipeline')

        # Subscribe to Isaac Sim sensors
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.rgb_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/velodyne_points',
            self.pointcloud_callback,
            10
        )

        # Publish detections
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/detections',
            10
        )

        # Initialize perception models
        self.segmentation_model = self.load_segmentation_model()
        self.detection_model = self.load_detection_model()

        self.latest_depth = None
        self.get_logger().info('Isaac Perception Pipeline initialized')

    def rgb_callback(self, msg):
        """Process RGB image for object detection and segmentation"""
        import cv2
        from cv_bridge import CvBridge

        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run object detection
        detections = self.detection_model.detect(image)

        # Run semantic segmentation
        segmentation = self.segmentation_model.segment(image)

        # Combine with depth information for 3D understanding
        if self.latest_depth is not None:
            detections_3d = self.fuse_with_depth(detections, self.latest_depth)

        # Publish detections
        detection_msg = self.create_detection_message(detections_3d, msg.header)
        self.detection_pub.publish(detection_msg)

    def depth_callback(self, msg):
        """Process depth image"""
        from cv_bridge import CvBridge
        bridge = CvBridge()
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        self.latest_depth = depth_image

    def pointcloud_callback(self, msg):
        """Process point cloud data"""
        # Process point cloud for object detection and mapping
        # This would typically use PCL or similar libraries
        pass

    def load_detection_model(self):
        """Load object detection model"""
        # In Isaac ROS, this would typically use TensorRT optimized models
        # or DetectNet for object detection
        class MockDetectionModel:
            def detect(self, image):
                # Mock implementation
                # In reality, this would use Isaac ROS's detection capabilities
                import cv2
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Detect simple shapes as example
                contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                detections = []
                for contour in contours:
                    if cv2.contourArea(contour) > 100:  # Filter small contours
                        x, y, w, h = cv2.boundingRect(contour)
                        detections.append({
                            'bbox': [x, y, w, h],
                            'label': 'object',
                            'confidence': 0.8
                        })

                return detections

        return MockDetectionModel()

    def load_segmentation_model(self):
        """Load segmentation model"""
        # Mock segmentation model
        class MockSegmentationModel:
            def segment(self, image):
                # Mock segmentation
                return np.zeros_like(image[:, :, 0])  # Return empty segmentation

        return MockSegmentationModel()

    def fuse_with_depth(self, detections, depth_image):
        """Fuse 2D detections with depth information"""
        detections_3d = []

        for det in detections:
            bbox = det['bbox']
            # Get depth at center of bounding box
            center_x = int(bbox[0] + bbox[2] / 2)
            center_y = int(bbox[1] + bbox[3] / 2)

            if 0 <= center_x < depth_image.shape[1] and 0 <= center_y < depth_image.shape[0]:
                depth = depth_image[center_y, center_x]

                detection_3d = det.copy()
                detection_3d['depth'] = depth
                detection_3d['position_3d'] = self.pixel_to_3d(center_x, center_y, depth)
                detections_3d.append(detection_3d)

        return detections_3d

    def pixel_to_3d(self, u, v, depth):
        """Convert pixel coordinates + depth to 3D world coordinates"""
        # This requires camera intrinsic parameters
        # In Isaac Sim, these would come from camera_info topic
        fx, fy = 525.0, 525.0  # Mock focal lengths
        cx, cy = 319.5, 239.5  # Mock principal points

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return [x, y, z]

    def create_detection_message(self, detections_3d, header):
        """Create vision_msgs/Detection2DArray message"""
        from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

        detection_array = Detection2DArray()
        detection_array.header = header

        for det in detections_3d:
            detection = Detection2D()

            # Set bounding box
            detection.bbox.center.x = det['bbox'][0] + det['bbox'][2] / 2
            detection.bbox.center.y = det['bbox'][1] + det['bbox'][3] / 2
            detection.bbox.size_x = det['bbox'][2]
            detection.bbox.size_y = det['bbox'][3]

            # Set hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = det['label']
            hypothesis.score = det['confidence']

            detection.results.append(hypothesis)

            detection_array.detections.append(detection)

        return detection_array
```

These examples demonstrate how to use Isaac ROS for perception, navigation, and manipulation tasks, integrating with Isaac Sim's realistic sensor simulation and physics.