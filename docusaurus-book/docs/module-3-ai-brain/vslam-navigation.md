---
title: VSLAM and Navigation
sidebar_label: VSLAM and Navigation
description: Visual Simultaneous Localization and Mapping for robot navigation in Isaac Sim
sidebar_position: 2
---

# VSLAM and Navigation

## Overview

Visual Simultaneous Localization and Mapping (VSLAM) is a critical capability for autonomous robots that enables them to understand and navigate their environment. VSLAM allows robots to build a map of their surroundings while simultaneously determining their position within that map using visual sensors. This technology is essential for autonomous navigation in unknown environments.

## Learning Objectives

- Understand the principles of Visual SLAM
- Learn to implement VSLAM algorithms in robotics
- Configure and use Isaac ROS VSLAM packages
- Integrate VSLAM with navigation systems
- Evaluate VSLAM performance and limitations

## VSLAM Fundamentals

VSLAM combines visual odometry with mapping techniques to enable simultaneous localization and mapping:

1. **Feature Detection**: Identifying distinctive features in visual data
2. **Feature Matching**: Tracking features across frames
3. **Motion Estimation**: Calculating camera/robot motion
4. **Mapping**: Building a 3D map of the environment
5. **Loop Closure**: Recognizing previously visited locations

### VSLAM Pipeline

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import tf2_ros
from geometry_msgs.msg import TransformStamped

class VSLAMNode(Node):
    def __init__(self):
        super().__init__('vslam_node')

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, 'vslam_odom', 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'vslam_pose', 10)
        self.map_pub = self.create_publisher(MarkerArray, 'vslam_map', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_rect_color',
            self.image_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            10
        )

        # VSLAM components
        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Previous frame data
        self.prev_image = None
        self.prev_keypoints = None
        self.prev_descriptors = None

        # Camera intrinsics
        self.camera_matrix = None
        self.dist_coeffs = None

        # Robot pose tracking
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.frame_count = 0

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.get_logger().info('VSLAM Node Initialized')

    def camera_info_callback(self, msg):
        """Store camera intrinsic parameters"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """Process incoming camera images for VSLAM"""
        if self.camera_matrix is None:
            self.get_logger().warn('Camera matrix not received yet')
            return

        # Convert ROS image to OpenCV
        image = self.ros_image_to_cv2(msg)

        # Detect features in current frame
        current_keypoints, current_descriptors = self.detect_features(image)

        if self.prev_keypoints is not None and len(self.prev_keypoints) > 10:
            # Match features between frames
            matches = self.match_features(
                self.prev_descriptors,
                current_descriptors
            )

            if len(matches) >= 10:  # Need minimum matches for reliable pose estimation
                # Estimate motion between frames
                motion = self.estimate_motion(
                    self.prev_keypoints,
                    current_keypoints,
                    matches
                )

                if motion is not None:
                    # Update global pose
                    self.update_pose(motion)

                    # Publish odometry
                    self.publish_odometry(msg.header.stamp)

                    # Broadcast TF
                    self.broadcast_transform(msg.header.stamp)

        # Store current frame data for next iteration
        self.prev_image = image
        self.prev_keypoints = current_keypoints
        self.prev_descriptors = current_descriptors

        self.frame_count += 1

    def detect_features(self, image):
        """Detect and describe features in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints = self.feature_detector.detect(gray)
        keypoints, descriptors = self.feature_detector.compute(gray, keypoints)
        return keypoints, descriptors

    def match_features(self, prev_desc, curr_desc):
        """Match features between previous and current frames"""
        if prev_desc is None or curr_desc is None:
            return []

        # Use FLANN matcher for better performance
        matches = self.descriptor_matcher.knnMatch(prev_desc, curr_desc, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        return good_matches

    def estimate_motion(self, prev_kp, curr_kp, matches):
        """Estimate camera motion between frames"""
        if len(matches) < 8:  # Need minimum points for fundamental matrix
            return None

        # Extract matched points
        prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find essential matrix
        E, mask = cv2.findEssentialMat(
            prev_pts,
            curr_pts,
            self.camera_matrix,
            threshold=1.0,
            prob=0.999
        )

        if E is None or E.size == 0:
            return None

        # Recover pose
        _, R, t, _ = cv2.recoverPose(
            E,
            prev_pts,
            curr_pts,
            self.camera_matrix,
            distanceThresh=100
        )

        # Create transformation matrix
        motion = np.eye(4)
        motion[:3, :3] = R
        motion[:3, 3] = t.flatten()

        return motion

    def update_pose(self, motion):
        """Update the global pose with the estimated motion"""
        self.current_pose = self.current_pose @ np.linalg.inv(motion)

    def publish_odometry(self, stamp):
        """Publish odometry message"""
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = "vslam_frame"

        # Extract position and orientation
        position = self.current_pose[:3, 3]
        rotation_matrix = self.current_pose[:3, :3]
        r = R.from_matrix(rotation_matrix)
        quat = r.as_quat()

        # Set pose
        odom_msg.pose.pose.position.x = position[0]
        odom_msg.pose.pose.position.y = position[1]
        odom_msg.pose.pose.position.z = position[2]
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]

        # Set zero velocity for now (would come from differentiation in practice)
        odom_msg.twist.twist.linear.x = 0.0
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = 0.0

        self.odom_pub.publish(odom_msg)

        # Publish pose separately
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = "map"
        pose_msg.pose = odom_msg.pose.pose
        self.pose_pub.publish(pose_msg)

    def broadcast_transform(self, stamp):
        """Broadcast the transform from map to vslam_frame"""
        t = TransformStamped()

        t.header.stamp = stamp
        t.header.frame_id = "map"
        t.child_frame_id = "vslam_frame"

        # Extract position and orientation
        position = self.current_pose[:3, 3]
        rotation_matrix = self.current_pose[:3, :3]
        r = R.from_matrix(rotation_matrix)
        quat = r.as_quat()

        t.transform.translation.x = position[0]
        t.transform.translation.y = position[1]
        t.transform.translation.z = position[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

    def ros_image_to_cv2(self, ros_image):
        """Convert ROS Image message to OpenCV image"""
        # This is a simplified conversion
        # In practice, use cv_bridge
        dtype = np.uint8
        if ros_image.encoding == "rgb8":
            dtype = np.uint8
        elif ros_image.encoding == "rgba8":
            dtype = np.uint8
        elif ros_image.encoding == "mono8":
            dtype = np.uint8
        elif ros_image.encoding == "mono16":
            dtype = np.uint16

        img = np.frombuffer(ros_image.data, dtype=dtype).reshape(
            ros_image.height, ros_image.width, len(ros_image.encoding)
        )

        if ros_image.encoding.startswith("mono"):
            img = np.squeeze(img, axis=2)

        return img

def main(args=None):
    rclpy.init(args=args)
    vslam_node = VSLAMNode()

    try:
        rclpy.spin(vslam_node)
    except KeyboardInterrupt:
        pass
    finally:
        vslam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac ROS VSLAM Packages

Isaac ROS provides optimized VSLAM packages that leverage NVIDIA GPUs:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from stereo_msgs.msg import DisparityImage
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import MarkerArray
import message_filters
from cv_bridge import CvBridge
import numpy as np
import cv2

class IsaacROSNavigation(Node):
    def __init__(self):
        super().__init__('isaac_ros_navigation')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.local_map_pub = self.create_publisher(OccupancyGrid, 'local_costmap/costmap', 10)

        # Use message_filters to synchronize stereo images
        left_sub = message_filters.Subscriber(self, Image, '/camera/left/image_rect_color')
        right_sub = message_filters.Subscriber(self, Image, '/camera/right/image_rect_color')
        left_info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/left/camera_info')
        right_info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/right/camera_info')

        # Approximate time synchronizer for stereo processing
        ts = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub, left_info_sub, right_info_sub],
            queue_size=10,
            slop=0.1
        )
        ts.registerCallback(self.stereo_callback)

        # Navigation components
        self.cv_bridge = CvBridge()
        self.disparity_processor = self.initialize_disparity_processor()

        # Navigation parameters
        self.safe_distance = 0.5  # meters
        self.max_linear_speed = 0.3  # m/s
        self.min_turn_radius = 0.3  # meters

        # Global path planner
        self.global_planner = GlobalPathPlanner()

        # Local planner
        self.local_planner = LocalPlanner()

        # Robot state
        self.current_position = np.array([0.0, 0.0])
        self.current_orientation = 0.0  # radians

        self.get_logger().info('Isaac ROS Navigation Node Initialized')

    def initialize_disparity_processor(self):
        """Initialize stereo disparity processor"""
        # Create StereoBM or StereoSGBM matcher
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=9,
            P1=8 * 3 * 9**2,
            P2=32 * 3 * 9**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        return stereo

    def stereo_callback(self, left_img_msg, right_img_msg, left_info_msg, right_info_msg):
        """Process synchronized stereo images for depth estimation"""
        try:
            # Convert ROS images to OpenCV
            left_cv = self.cv_bridge.imgmsg_to_cv2(left_img_msg, desired_encoding='passthrough')
            right_cv = self.cv_bridge.imgmsg_to_cv2(right_img_msg, desired_encoding='passthrough')

            # Ensure images are grayscale for stereo processing
            if len(left_cv.shape) == 3:
                left_gray = cv2.cvtColor(left_cv, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_cv, cv2.COLOR_BGR2GRAY)
            else:
                left_gray = left_cv
                right_gray = right_cv

            # Compute disparity
            disparity = self.disparity_processor.compute(left_gray, right_gray).astype(np.float32) / 16.0

            # Convert disparity to depth
            depth_map = self.disparity_to_depth(disparity, left_info_msg)

            # Process depth map for navigation
            self.process_depth_for_navigation(depth_map)

        except Exception as e:
            self.get_logger().error(f'Stereo callback error: {e}')

    def disparity_to_depth(self, disparity, camera_info):
        """Convert disparity map to depth map using camera parameters"""
        # Extract focal length from camera matrix
        fx = camera_info.k[0]  # Focal length in x
        baseline = 0.1  # Baseline between stereo cameras in meters (adjust as needed)

        # Avoid division by zero
        disparity[disparity == 0] = 0.001

        # Calculate depth: depth = (baseline * focal_length) / disparity
        depth_map = (baseline * fx) / disparity

        # Apply depth limits
        depth_map[depth_map > 10.0] = 0  # Set far points to 0
        depth_map[depth_map < 0.1] = 0  # Set near points to 0

        return depth_map

    def process_depth_for_navigation(self, depth_map):
        """Process depth map for obstacle detection and navigation"""
        # Create occupancy grid from depth data
        occupancy_grid = self.create_occupancy_grid_from_depth(depth_map)

        # Publish local map
        self.local_map_pub.publish(occupancy_grid)

        # Analyze depth map for obstacles
        obstacle_map = self.analyze_obstacles(depth_map)

        # Plan safe trajectory
        safe_trajectory = self.plan_safe_trajectory(obstacle_map)

        # Execute navigation command
        self.execute_navigation(safe_trajectory)

    def create_occupancy_grid_from_depth(self, depth_map):
        """Create occupancy grid from depth map"""
        # Create a simple occupancy grid
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = "local_map"

        # Define grid parameters
        grid.info.resolution = 0.1  # 10cm per cell
        grid.info.width = 200  # 20m wide
        grid.info.height = 200  # 20m tall
        grid.info.origin.position.x = -10.0  # Center robot at (0,0)
        grid.info.origin.position.y = -10.0

        # Initialize grid data (values: -1 = unknown, 0 = free, 100 = occupied)
        grid.data = [-1] * (grid.info.width * grid.info.height)

        # Populate grid based on depth data
        # This is a simplified approach - in practice, you'd use more sophisticated mapping
        robot_pos = (100, 100)  # Grid center corresponds to robot position

        for y in range(depth_map.shape[0]):
            for x in range(depth_map.shape[1]):
                depth_value = depth_map[y, x]
                if depth_value > 0 and depth_value < self.safe_distance:
                    # Obstacle detected - mark as occupied
                    grid_x = int(x * grid.info.resolution)
                    grid_y = int(y * grid.info.resolution)

                    if 0 <= grid_x < grid.info.width and 0 <= grid_y < grid.info.height:
                        idx = grid_y * grid.info.width + grid_x
                        if 0 <= idx < len(grid.data):
                            grid.data[idx] = 100  # Occupied

        return grid

    def analyze_obstacles(self, depth_map):
        """Analyze depth map to identify obstacles and free space"""
        # Create a binary obstacle map
        obstacle_map = np.zeros_like(depth_map, dtype=np.uint8)

        # Mark areas closer than safe distance as obstacles
        obstacle_map[depth_map > 0] = 255  # Initially mark all valid depth readings
        obstacle_map[depth_map > self.safe_distance] = 0  # Unmark areas beyond safe distance

        return obstacle_map

    def plan_safe_trajectory(self, obstacle_map):
        """Plan a safe trajectory based on obstacle map"""
        # This is a simplified trajectory planner
        # In practice, use more sophisticated planners like DWA, TEB, or RRT*

        # Determine if path is clear ahead
        height, width = obstacle_map.shape
        center_x = width // 2

        # Check a corridor straight ahead
        ahead_region = obstacle_map[height//2:, center_x-20:center_x+20]

        if np.any(ahead_region == 255):  # Obstacle detected ahead
            # Need to turn or stop
            return {
                'linear_speed': 0.0,
                'angular_speed': 0.5,  # Turn to avoid obstacle
                'action': 'turn'
            }
        else:
            # Path ahead is clear
            return {
                'linear_speed': min(self.max_linear_speed, 0.2),
                'angular_speed': 0.0,
                'action': 'forward'
            }

    def execute_navigation(self, trajectory):
        """Execute the planned trajectory"""
        cmd_vel = Twist()
        cmd_vel.linear.x = trajectory['linear_speed']
        cmd_vel.angular.z = trajectory['angular_speed']

        self.cmd_vel_pub.publish(cmd_vel)

        self.get_logger().debug(f'Executing {trajectory["action"]}: '
                               f'linear={cmd_vel.linear.x:.2f}, '
                               f'angular={cmd_vel.angular.z:.2f}')

class GlobalPathPlanner:
    """Global path planner for navigation"""
    def __init__(self):
        pass

    def plan_path(self, start, goal, map):
        """Plan a global path from start to goal"""
        # In practice, implement A*, Dijkstra, or RRT*
        # This is a placeholder
        return []

class LocalPlanner:
    """Local trajectory planner for obstacle avoidance"""
    def __init__(self):
        pass

    def plan_local_trajectory(self, current_pose, global_path, obstacles):
        """Plan local trajectory considering obstacles"""
        # In practice, implement DWA, TEB, or similar
        # This is a placeholder
        return []
```

## VSLAM in Isaac Sim

Using VSLAM within Isaac Sim requires specific configurations:

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.sensor import Camera
from omni.isaac.range_sensor import _range_sensor
import numpy as np
import carb

class IsaacSimVSLAM:
    def __init__(self):
        # Initialize the world
        self.world = World(stage_units_in_meters=1.0)

        # Get assets root path
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return

        # Add a simple room environment
        add_reference_to_stage(
            usd_path=f"{assets_root_path}/Isaac/Environments/Simple_Room/simple_room.usda",
            prim_path="/World/defaultGroundPlane"
        )

        # Add a robot platform (represented as a cuboid for this example)
        self.robot = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Robot",
                name="robot",
                position=np.array([0, 0, 0.5]),
                size=0.3,
                mass=1.0
            )
        )

        # Add a camera for VSLAM
        self.camera = Camera(
            prim_path="/World/Camera",
            position=np.array([0.0, 0.0, 0.5]),
            frequency=30,
            resolution=(640, 480)
        )
        self.world.scene.add(self.camera)

        # Add some objects to create visual features
        self.objects = []
        positions = [[1, 1, 0.5], [-1, 1, 0.5], [1, -1, 0.5], [-1, -1, 0.5]]
        for i, pos in enumerate(positions):
            obj = self.world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Object{i}",
                    name=f"object_{i}",
                    position=np.array(pos),
                    size=0.2,
                    mass=0.5
                )
            )
            self.objects.append(obj)

        # VSLAM processing components
        self.feature_detector = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Pose tracking
        self.current_pose = np.eye(4)
        self.keyframe_poses = []
        self.map_points = []

        self.world.reset()

    def run_vslam_simulation(self):
        """Run the VSLAM simulation loop"""
        while True:
            # Step the physics simulation
            self.world.step(render=True)

            # Get camera image
            rgb_image = self.camera.get_rgb()

            if rgb_image is not None:
                # Process VSLAM
                self.process_vslam_frame(rgb_image)

            # Break if simulation ends
            if not self.world.is_playing():
                break

    def process_vslam_frame(self, image):
        """Process a single frame for VSLAM"""
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

        # Detect features
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)

        if descriptors is not None and len(descriptors) > 10:
            # If we have previous frame, try to match features
            if hasattr(self, 'prev_descriptors') and self.prev_descriptors is not None:
                matches = self.matcher.match(self.prev_descriptors, descriptors)

                # Sort matches by distance
                matches = sorted(matches, key=lambda x: x.distance)

                # Keep only good matches
                good_matches = matches[:50]  # Take top 50 matches

                if len(good_matches) >= 10:  # Need minimum matches for pose estimation
                    # Extract matched keypoints
                    prev_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    curr_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    # Estimate motion (simplified - would need camera intrinsics in practice)
                    if len(prev_pts) >= 8:
                        # Calculate essential matrix
                        E, mask = cv2.findEssentialMat(prev_pts, curr_pts, method=cv2.RANSAC, threshold=1.0)

                        if E is not None:
                            # Recover pose
                            _, R, t, _ = cv2.recoverPose(E, prev_pts, curr_pts)

                            # Create transformation matrix
                            motion = np.eye(4)
                            motion[:3, :3] = R
                            motion[:3, 3] = t.flatten()

                            # Update current pose
                            self.current_pose = self.current_pose @ np.linalg.inv(motion)

            # Store current frame data for next iteration
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors

            # Add current pose as keyframe periodically
            if len(self.keyframe_poses) == 0 or np.linalg.norm(
                self.current_pose[:3, 3] - self.keyframe_poses[-1][:3, 3]
            ) > 0.5:  # Add keyframe every 0.5m of movement
                self.keyframe_poses.append(self.current_pose.copy())

    def get_current_pose(self):
        """Return the current estimated pose"""
        return self.current_pose

    def get_trajectory(self):
        """Return the trajectory of keyframes"""
        return self.keyframe_poses

# Example usage in Isaac Sim
def example_isaac_sim_vslam():
    # This would be run within Isaac Sim
    vslam_sim = IsaacSimVSLAM()

    # Run for a certain number of steps
    for i in range(1000):
        vslam_sim.world.step(render=True)

        # Get camera image
        rgb_image = vslam_sim.camera.get_rgb()

        if rgb_image is not None:
            vslam_sim.process_vslam_frame(rgb_image)

        # Print current pose occasionally
        if i % 100 == 0:
            pose = vslam_sim.get_current_pose()
            print(f"Estimated pose at step {i}:")
            print(f"Position: [{pose[0,3]:.2f}, {pose[1,3]:.2f}, {pose[2,3]:.2f}]")

    # Print final trajectory
    trajectory = vslam_sim.get_trajectory()
    print(f"\nTrajectory with {len(trajectory)} keyframes")

    return vslam_sim

# Call the example (this would run in Isaac Sim)
# result = example_isaac_sim_vslam()
```

## Navigation with VSLAM

Integrating VSLAM with navigation systems:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray
import numpy as np
from scipy.spatial.distance import cdist
import tf2_ros
from geometry_msgs.msg import TransformStamped, Point
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Quaternion
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import tf_transformations

class VSLAMNavigationSystem(Node):
    def __init__(self):
        super().__init__('vslam_navigation_system')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.global_path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.local_path_pub = self.create_publisher(Path, 'local_plan', 10)
        self.viz_pub = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)

        # Subscribers
        self.vslam_pose_sub = self.create_subscription(
            PoseStamped,
            'vslam_pose',
            self.vslam_pose_callback,
            10
        )

        self.laser_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            'move_base_simple/goal',
            self.goal_callback,
            10
        )

        # Navigation parameters
        self.linear_speed = 0.2
        self.angular_speed = 0.3
        self.safe_distance = 0.5
        self.arrival_threshold = 0.3

        # Robot state
        self.current_pose = None
        self.has_active_goal = False
        self.goal_pose = None

        # Path planning
        self.global_path = []
        self.local_path = []

        # TF buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Timers
        self.nav_timer = self.create_timer(0.1, self.navigation_callback)

        self.get_logger().info('VSLAM Navigation System Initialized')

    def vslam_pose_callback(self, msg):
        """Update robot pose from VSLAM"""
        self.current_pose = msg.pose

    def laser_callback(self, msg):
        """Process laser scan for obstacle detection"""
        # Process laser data for local navigation
        pass

    def goal_callback(self, msg):
        """Receive navigation goal"""
        self.goal_pose = msg.pose
        self.has_active_goal = True
        self.get_logger().info(f'Received navigation goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')

        # Plan path to goal
        self.plan_global_path()

    def plan_global_path(self):
        """Simple path planning to goal (in practice, use A*, RRT*, etc.)"""
        if self.current_pose is None or self.goal_pose is None:
            return

        # Simple straight-line path (in practice, use proper path planning)
        start = np.array([self.current_pose.position.x, self.current_pose.position.y])
        goal = np.array([self.goal_pose.position.x, self.goal_pose.position.y])

        # Create a simple path (in practice, use proper path planner)
        direction = goal - start
        distance = np.linalg.norm(direction)

        if distance > 0:
            steps = max(int(distance / 0.1), 2)  # 10cm steps
            path = []

            for i in range(steps + 1):
                ratio = i / steps
                point = start + ratio * direction

                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = "map"
                pose_stamped.pose.position.x = point[0]
                pose_stamped.pose.position.y = point[1]
                pose_stamped.pose.position.z = 0.0

                # Set orientation towards goal
                yaw = np.arctan2(goal[1] - start[1], goal[0] - start[0])
                quat = tf_transformations.quaternion_from_euler(0, 0, yaw)
                pose_stamped.pose.orientation.x = quat[0]
                pose_stamped.pose.orientation.y = quat[1]
                pose_stamped.pose.orientation.z = quat[2]
                pose_stamped.pose.orientation.w = quat[3]

                path.append(pose_stamped)

            self.global_path = path

            # Publish global path
            path_msg = Path()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = "map"
            path_msg.poses = self.global_path
            self.global_path_pub.publish(path_msg)

    def navigation_callback(self):
        """Main navigation control loop"""
        if not self.has_active_goal or not self.current_pose:
            return

        if len(self.global_path) == 0:
            self.get_logger().warn('No global path available')
            return

        # Get robot position
        robot_pos = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y
        ])

        # Get goal position
        goal_pos = np.array([
            self.goal_pose.position.x,
            self.goal_pose.position.y
        ])

        # Check if reached goal
        distance_to_goal = np.linalg.norm(robot_pos - goal_pos)
        if distance_to_goal < self.arrival_threshold:
            self.get_logger().info('Reached goal!')
            self.stop_robot()
            self.has_active_goal = False
            return

        # Find next waypoint in path
        next_waypoint = self.get_next_waypoint(robot_pos)
        if next_waypoint is None:
            self.get_logger().warn('Could not find next waypoint')
            self.stop_robot()
            return

        # Calculate direction to waypoint
        direction = np.array([
            next_waypoint.pose.position.x - robot_pos[0],
            next_waypoint.pose.position.y - robot_pos[1]
        ])
        distance_to_waypoint = np.linalg.norm(direction)

        # Calculate robot's current orientation
        current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)

        # Calculate desired heading
        desired_yaw = np.arctan2(direction[1], direction[0])

        # Calculate heading error
        heading_error = self.normalize_angle(desired_yaw - current_yaw)

        # Simple proportional controller for navigation
        cmd_vel = Twist()

        # Move forward if aligned with goal direction
        if abs(heading_error) < 0.2:  # 0.2 radians = ~11 degrees
            cmd_vel.linear.x = min(self.linear_speed, 0.2)
        else:
            cmd_vel.linear.x = 0.0

        # Turn towards goal
        cmd_vel.angular.z = max(min(heading_error * 1.0, self.angular_speed), -self.angular_speed)

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

        # Log navigation status
        self.get_logger().debug(f'Heading error: {np.degrees(heading_error):.2f}°, '
                               f'Distance to waypoint: {distance_to_waypoint:.2f}m, '
                               f'Velocity: ({cmd_vel.linear.x:.2f}, {cmd_vel.angular.z:.2f})')

    def get_next_waypoint(self, robot_pos):
        """Find the next waypoint along the global path"""
        if len(self.global_path) == 0:
            return None

        # Find the closest waypoint
        min_dist = float('inf')
        closest_idx = 0

        for i, wp in enumerate(self.global_path):
            wp_pos = np.array([wp.pose.position.x, wp.pose.position.y])
            dist = np.linalg.norm(robot_pos - wp_pos)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Return the next waypoint after the closest one
        next_idx = min(closest_idx + 1, len(self.global_path) - 1)
        return self.global_path[next_idx]

    def get_yaw_from_quaternion(self, quat):
        """Extract yaw angle from quaternion"""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return yaw

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def stop_robot(self):
        """Stop the robot"""
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    nav_system = VSLAMNavigationSystem()

    try:
        rclpy.spin(nav_system)
    except KeyboardInterrupt:
        pass
    finally:
        nav_system.stop_robot()
        nav_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance and Limitations

VSLAM systems have several performance factors and limitations:

### Performance Factors:
1. **Computational Complexity**: Feature detection and matching require significant processing power
2. **Lighting Conditions**: Performance degrades in poor lighting or changing illumination
3. **Texture**: Featureless surfaces (white walls, sky) are difficult to track
4. **Motion Blur**: Fast movements can cause blur and tracking failures
5. **Scale Drift**: Long-term accumulation of errors can cause drift

### Limitations:
1. **Initialization**: Requires initial position or calibration
2. **Dynamic Objects**: Moving objects can interfere with mapping
3. **Loop Closure**: Recognizing previously visited locations can be challenging
4. **Scale Recovery**: Monocular systems cannot recover absolute scale without additional information

## Self-Check Questions


## Summary

Visual SLAM is a powerful technology that enables robots to navigate and map unknown environments using visual sensors. While VSLAM has limitations related to lighting, texture, and computational requirements, it provides valuable capabilities for autonomous navigation. The integration with Isaac Sim and ROS allows for effective testing and deployment of VSLAM-based navigation systems. Proper implementation requires careful consideration of performance factors and limitations to ensure reliable operation.