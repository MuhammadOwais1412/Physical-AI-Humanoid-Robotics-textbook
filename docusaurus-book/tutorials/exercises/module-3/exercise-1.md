# Module 3 Exercise: Perception Pipeline Implementation

## Objective

Implement a complete perception pipeline that integrates multiple sensors (camera and LIDAR) to detect and track objects in a simulated environment, and use the perception results for robot navigation.

## Requirements

1. Create a sensor fusion system that combines camera and LIDAR data
2. Implement object detection using deep learning
3. Create a tracking system for moving objects
4. Use perception results for obstacle avoidance in navigation
5. Visualize the perception pipeline results

## Steps

### 1. Setup the Perception System

Create a perception manager that handles multiple sensors:

```python
import numpy as np
import cv2
import torch
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

class PerceptionManager:
    def __init__(self):
        # Initialize sensor data buffers
        self.rgb_image = None
        self.depth_image = None
        self.lidar_data = None

        # Initialize perception modules
        self.object_detector = ObjectDetector()
        self.tracker = MultiObjectTracker()
        self.fusion_module = SensorFusionModule()
        self.nav_module = NavigationModule()

        # Initialize visualization
        self.vis = self.initialize_visualization()

    def initialize_visualization(self):
        """Initialize visualization for perception results"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Perception Pipeline", width=800, height=600)
        return vis

    def process_sensor_data(self, rgb_image, depth_image, lidar_data):
        """Process incoming sensor data through the pipeline"""
        # Store sensor data
        self.rgb_image = rgb_image
        self.depth_image = depth_image
        self.lidar_data = lidar_data

        # Step 1: Object detection from camera
        camera_detections = self.object_detector.detect(self.rgb_image)

        # Step 2: Object detection from LIDAR
        lidar_detections = self.process_lidar_data(self.lidar_data)

        # Step 3: Fuse detections from different sensors
        fused_detections = self.fusion_module.fuse_detections(
            camera_detections, lidar_detections
        )

        # Step 4: Track objects over time
        tracked_objects = self.tracker.update(fused_detections)

        # Step 5: Generate navigation-relevant information
        nav_info = self.nav_module.process_objects(tracked_objects)

        return {
            'detections': fused_detections,
            'tracks': tracked_objects,
            'navigation': nav_info
        }

    def process_lidar_data(self, lidar_data):
        """Process LIDAR data to detect objects"""
        # Convert LIDAR data to point cloud
        point_cloud = self.lidar_to_pointcloud(lidar_data)

        # Downsample point cloud
        downsampled = point_cloud.voxel_down_sample(voxel_size=0.1)

        # Segment ground plane
        plane_model, inliers = downsampled.segment_plane(
            distance_threshold=0.2,
            ransac_n=3,
            num_iterations=1000
        )

        # Extract objects (remove ground)
        object_points = downsampled.select_by_index(inliers, invert=True)

        # Cluster objects
        clusters = self.cluster_point_cloud(np.asarray(object_points.points))

        # Extract object information
        objects = []
        for cluster in clusters:
            centroid = np.mean(cluster, axis=0)
            size = np.std(cluster, axis=0)

            objects.append({
                'centroid': centroid,
                'size': size,
                'points': cluster,
                'sensor_type': 'lidar'
            })

        return objects

    def lidar_to_pointcloud(self, lidar_data):
        """Convert LIDAR range data to point cloud"""
        # Convert range data to Cartesian coordinates
        # This is a simplified example - actual implementation depends on LIDAR specs
        points = []

        for i, distance in enumerate(lidar_data):
            if distance > 0 and distance < 10:  # Valid range
                angle = i * (2 * np.pi / len(lidar_data))  # Assuming 360-degree LIDAR
                x = distance * np.cos(angle)
                y = distance * np.sin(angle)
                z = 0  # Assuming 2D LIDAR on ground level
                points.append([x, y, z])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        return pcd

    def cluster_point_cloud(self, points, eps=0.5, min_samples=10):
        """Cluster point cloud using DBSCAN"""
        from sklearn.cluster import DBSCAN

        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(points)

        clusters = []
        for label in set(labels):
            if label != -1:  # -1 is noise
                cluster_points = points[labels == label]
                clusters.append(cluster_points)

        return clusters
```

### 2. Object Detection Module

```python
class ObjectDetector:
    def __init__(self, model_path=None):
        # Initialize pre-trained model (e.g., YOLOv5, Faster R-CNN)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_path:
            # Load custom model
            self.model = torch.load(model_path)
        else:
            # Load pre-trained COCO model
            self.model = torch.hub.load(
                'ultralytics/yolov5',
                'yolov5s',
                pretrained=True,
                force_reload=True
            )

        self.model.to(self.device)
        self.model.eval()

        # Class names for filtering
        self.relevant_classes = ['person', 'car', 'bicycle', 'chair', 'cup', 'bottle']

    def detect(self, image):
        """Detect objects in image"""
        # Convert image to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Run inference
        results = self.model(image)

        # Process results
        detections = []
        for *xyxy, conf, cls in results.xyxy[0].tolist():
            class_name = self.model.names[int(cls)]

            if class_name in self.relevant_classes and conf > 0.5:
                detection = {
                    'bbox': [int(x) for x in xyxy],  # [x1, y1, x2, y2]
                    'confidence': conf,
                    'class': class_name,
                    'sensor_type': 'camera'
                }
                detections.append(detection)

        return detections
```

### 3. Multi-Object Tracking

```python
class MultiObjectTracker:
    def __init__(self, max_disappeared=10, max_distance=100):
        self.trackers = []
        self.next_id = 0
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections):
        """Update object tracks with new detections"""
        if len(self.trackers) == 0:
            # Create new trackers for all detections
            for det in detections:
                tracker = self.init_tracker(det)
                tracker['id'] = self.next_id
                self.next_id += 1
                self.trackers.append(tracker)
            return [t for t in self.trackers if t['disappeared'] == 0]

        # Calculate distance matrix
        D = np.zeros((len(self.trackers), len(detections)))
        for i, tracker in enumerate(self.trackers):
            for j, det in enumerate(detections):
                D[i, j] = self.calculate_distance(tracker, det)

        # Use Hungarian algorithm for assignment
        rows, cols = linear_sum_assignment(D)

        # Update assigned trackers
        used_trackers = set()
        used_detections = set()

        for (tracker_idx, det_idx) in zip(rows, cols):
            if D[tracker_idx, det_idx] < self.max_distance:
                self.update_tracker(self.trackers[tracker_idx], detections[det_idx])
                used_trackers.add(tracker_idx)
                used_detections.add(det_idx)
            else:
                self.trackers[tracker_idx]['disappeared'] += 1

        # Create new trackers for unassigned detections
        for det_idx in set(range(len(detections))) - used_detections:
            tracker = self.init_tracker(detections[det_idx])
            tracker['id'] = self.next_id
            self.next_id += 1
            self.trackers.append(tracker)

        # Mark unassigned trackers as disappeared
        for tracker_idx in set(range(len(self.trackers))) - used_trackers:
            if tracker_idx not in [r for r, c in zip(rows, cols) if D[r, c] < self.max_distance]:
                self.trackers[tracker_idx]['disappeared'] += 1

        # Remove old trackers
        self.trackers = [t for t in self.trackers if t['disappeared'] < self.max_disappeared]

        return [t for t in self.trackers if t['disappeared'] == 0]

    def init_tracker(self, detection):
        """Initialize a new tracker"""
        # Create Kalman filter for this object
        kf = KalmanFilter(dim_x=8, dim_z=4)  # Position and velocity

        if detection['sensor_type'] == 'camera':
            # Initialize from 2D bounding box
            bbox = detection['bbox']
            x, y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            kf.x = np.array([x, y, w, h, 0, 0, 0, 0])  # [x, y, w, h, vx, vy, vw, vh]
        else:  # LIDAR
            # Initialize from 3D position
            pos = detection['centroid']
            kf.x = np.array([pos[0], pos[1], pos[2], 0, 0, 0, 0, 0])  # [x, y, z, vx, vy, vz, 0, 0]

        # State transition matrix
        dt = 1/30  # Assuming 30 FPS
        kf.F = np.eye(8)
        kf.F[0, 4] = dt  # x = x + vx*dt
        kf.F[1, 5] = dt  # y = y + vy*dt
        kf.F[2, 6] = dt  # z = z + vz*dt

        # Measurement matrix
        kf.H = np.zeros((4, 8))
        kf.H[0, 0] = 1  # x
        kf.H[1, 1] = 1  # y
        kf.H[2, 2] = 1  # z
        kf.H[3, 3] = 1  # w

        # Covariance matrices
        kf.P *= 100
        kf.R = np.eye(4) * 5
        kf.Q = np.eye(8) * 0.1

        return {
            'kalman_filter': kf,
            'id': None,
            'disappeared': 0,
            'detection': detection
        }

    def update_tracker(self, tracker_info, detection):
        """Update tracker with new detection"""
        kf = tracker_info['kalman_filter']

        # Prepare measurement based on sensor type
        if detection['sensor_type'] == 'camera':
            bbox = detection['bbox']
            measurement = np.array([
                (bbox[0] + bbox[2]) / 2,  # center x
                (bbox[1] + bbox[3]) / 2,  # center y
                bbox[2] - bbox[0],        # width
                bbox[3] - bbox[1]         # height
            ])
        else:  # LIDAR
            pos = detection['centroid']
            measurement = np.array([pos[0], pos[1], pos[2], detection['size'][0]])

        # Update Kalman filter
        kf.update(measurement)
        tracker_info['disappeared'] = 0
        tracker_info['detection'] = detection

    def calculate_distance(self, tracker_info, detection):
        """Calculate distance between tracker prediction and detection"""
        kf = tracker_info['kalman_filter']

        # Predict next state
        predicted = kf.F @ kf.x

        # Calculate distance based on sensor type
        if detection['sensor_type'] == 'camera':
            det_center = np.array([
                (detection['bbox'][0] + detection['bbox'][2]) / 2,
                (detection['bbox'][1] + detection['bbox'][3]) / 2
            ])
            pred_center = predicted[:2]
        else:  # LIDAR
            det_center = detection['centroid'][:2]
            pred_center = predicted[:2]

        distance = np.sqrt(np.sum((det_center - pred_center) ** 2))
        return distance
```

### 4. Sensor Fusion Module

```python
class SensorFusionModule:
    def __init__(self):
        self.camera_to_lidar = self.get_camera_lidar_transform()  # Calibration

    def get_camera_lidar_transform(self):
        """Get transformation matrix between camera and LIDAR"""
        # This would come from sensor calibration
        # For this exercise, we'll use a placeholder
        return np.eye(4)

    def fuse_detections(self, camera_detections, lidar_detections):
        """Fuse detections from different sensors"""
        fused_objects = []

        # Transform LIDAR detections to camera frame
        lidar_in_camera = []
        for lidar_det in lidar_detections:
            transformed_pos = self.transform_point(
                lidar_det['centroid'], self.camera_to_lidar
            )
            lidar_det['centroid_camera'] = transformed_pos
            lidar_in_camera.append(lidar_det)

        # Associate camera and LIDAR detections
        associations = self.associate_detections(
            camera_detections, lidar_in_camera
        )

        # Create fused objects
        for cam_det, lidar_det in associations:
            if lidar_det is not None:
                # Combine information from both sensors
                fused_object = {
                    'position': lidar_det['centroid'],
                    'bbox_2d': cam_det['bbox'] if cam_det else None,
                    'confidence': (cam_det['confidence'] + 0.8) / 2 if cam_det else 0.8,
                    'class': cam_det['class'] if cam_det else 'unknown',
                    'size_3d': lidar_det['size'],
                    'sensor_fusion': True
                }
            else:
                # Camera-only detection
                fused_object = {
                    'position': self.estimate_3d_position(cam_det),
                    'bbox_2d': cam_det['bbox'],
                    'confidence': cam_det['confidence'],
                    'class': cam_det['class'],
                    'sensor_fusion': False
                }

            fused_objects.append(fused_object)

        return fused_objects

    def associate_detections(self, camera_dets, lidar_dets):
        """Associate camera and LIDAR detections"""
        associations = []
        used_lidar = set()

        for cam_det in camera_dets:
            # Project 2D bbox to 3D space
            cam_center_3d = self.bbox_to_3d_center(cam_det['bbox'])

            best_assoc = None
            best_distance = float('inf')

            for i, lidar_det in enumerate(lidar_dets):
                if i in used_lidar:
                    continue

                distance = np.linalg.norm(cam_center_3d - lidar_det['centroid'])
                if distance < best_distance and distance < 2.0:  # 2m threshold
                    best_distance = distance
                    best_assoc = i

            if best_assoc is not None:
                used_lidar.add(best_assoc)
                associations.append((cam_det, lidar_dets[best_assoc]))
            else:
                associations.append((cam_det, None))

        # Add unassociated LIDAR detections
        for i, lidar_det in enumerate(lidar_dets):
            if i not in used_lidar:
                associations.append((None, lidar_det))

        return associations

    def bbox_to_3d_center(self, bbox):
        """Estimate 3D center from 2D bounding box"""
        # This is a simplified estimation
        # In practice, you'd use depth information
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        # Estimate depth based on object class and size
        bbox_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        estimated_depth = max(1.0, 1000 / (bbox_size + 1))  # Rough estimation
        return np.array([x_center, y_center, estimated_depth])
```

### 5. Navigation Module

```python
class NavigationModule:
    def __init__(self):
        self.occupancy_grid = np.zeros((100, 100))  # 10x10m grid, 10cm resolution
        self.robot_pos = np.array([50, 50])  # Robot starts at center

    def process_objects(self, tracked_objects):
        """Process tracked objects for navigation"""
        obstacles = []
        free_space = []

        for obj in tracked_objects:
            if obj['disappeared'] == 0:  # Currently tracked
                pos = obj['kalman_filter'].x[:2]  # Get 2D position
                pos_grid = self.world_to_grid(pos)

                if self.is_valid_grid_pos(pos_grid):
                    # Mark as obstacle in occupancy grid
                    self.occupancy_grid[pos_grid[0], pos_grid[1]] = 1
                    obstacles.append(pos)

        # Plan path avoiding obstacles
        goal = np.array([90, 90])  # Goal at bottom-right
        path = self.plan_path(self.robot_pos, goal, obstacles)

        return {
            'obstacles': obstacles,
            'path': path,
            'occupancy_grid': self.occupancy_grid.copy()
        }

    def plan_path(self, start, goal, obstacles):
        """Simple path planning avoiding obstacles"""
        # This is a simplified implementation
        # In practice, use A*, RRT*, or other algorithms
        path = [start]

        current = start.copy()
        while np.linalg.norm(current - goal) > 2:  # Within 20cm of goal
            direction = goal - current
            direction = direction / np.linalg.norm(direction)  # Normalize

            # Move in direction of goal
            next_pos = current + direction * 5  # Move 50cm

            # Check for obstacles and adjust path if needed
            if self.check_collision(next_pos, obstacles):
                # Simple obstacle avoidance: move around
                next_pos = self.avoid_obstacle(current, direction, obstacles)

            path.append(next_pos.copy())
            current = next_pos

            if len(path) > 100:  # Safety limit
                break

        path.append(goal)
        return path

    def check_collision(self, pos, obstacles):
        """Check if position collides with obstacles"""
        pos_grid = self.world_to_grid(pos)
        if not self.is_valid_grid_pos(pos_grid):
            return True

        # Check nearby cells for obstacles
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_pos = [pos_grid[0] + dx, pos_grid[1] + dy]
                if (0 <= check_pos[0] < self.occupancy_grid.shape[0] and
                    0 <= check_pos[1] < self.occupancy_grid.shape[1]):
                    if self.occupancy_grid[check_pos[0], check_pos[1]] > 0.5:
                        return True

        return False

    def avoid_obstacle(self, current_pos, direction, obstacles):
        """Simple obstacle avoidance"""
        # Try to move perpendicular to the obstacle
        perpendicular = np.array([-direction[1], direction[0]])
        new_pos = current_pos + perpendicular * 3  # Move 30cm perpendicular

        # If that's also blocked, try the opposite direction
        if self.check_collision(new_pos, obstacles):
            new_pos = current_pos - perpendicular * 3

        return new_pos

    def world_to_grid(self, pos):
        """Convert world coordinates to grid coordinates"""
        return [int(pos[0]), int(pos[1])]

    def is_valid_grid_pos(self, pos):
        """Check if grid position is valid"""
        return (0 <= pos[0] < self.occupancy_grid.shape[0] and
                0 <= pos[1] < self.occupancy_grid.shape[1])
```

### 6. Main Execution Loop

```python
def main():
    # Initialize perception pipeline
    perception = PerceptionManager()

    # Simulate sensor data (in real implementation, get from ROS topics)
    import time
    import random

    for frame in range(100):  # Process 100 frames
        # Simulate sensor data
        rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth_image = np.random.rand(480, 640).astype(np.float32) * 10  # 0-10m
        lidar_data = np.random.rand(360).astype(np.float32) * 10     # 360-degree scan

        # Process perception pipeline
        results = perception.process_sensor_data(rgb_image, depth_image, lidar_data)

        # Print results
        print(f"Frame {frame}: Found {len(results['tracks'])} tracked objects")
        print(f"Navigation: {len(results['navigation']['path'])} path points")

        # Visualize results (optional)
        visualize_results(results)

        time.sleep(0.1)  # Simulate real-time processing

def visualize_results(results):
    """Visualize perception results"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot occupancy grid
    axes[0].imshow(results['navigation']['occupancy_grid'], cmap='gray', origin='lower')
    axes[0].set_title('Occupancy Grid')

    # Plot path
    path = np.array(results['navigation']['path'])
    if len(path) > 0:
        axes[0].plot(path[:, 1], path[:, 0], 'r-', linewidth=2, label='Planned Path')
        axes[0].plot(path[0, 1], path[0, 0], 'go', markersize=10, label='Start')
        axes[0].plot(path[-1, 1], path[-1, 0], 'ro', markersize=10, label='Goal')

    axes[0].legend()
    axes[0].grid(True)

    # Plot object tracks
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 100)

    for obj in results['tracks']:
        pos = obj['kalman_filter'].x[:2]
        axes[1].plot(pos[1], pos[0], 'bo', markersize=8)
        axes[1].text(pos[1], pos[0], f"ID:{obj['id']}", fontsize=8)

    axes[1].set_title('Object Tracks')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
```

## Expected Output

1. The system should detect objects from both camera and LIDAR data
2. Objects should be tracked consistently over time
3. Sensor data should be properly fused
4. Navigation path should avoid detected obstacles
5. The system should run in real-time (30 FPS or better)

## Evaluation Criteria

1. **Accuracy**: Correct detection and tracking of objects
2. **Robustness**: Handling of sensor noise and failures
3. **Efficiency**: Real-time performance
4. **Integration**: Proper fusion of different sensor modalities
5. **Navigation**: Safe path planning around obstacles

## Extension Challenges

1. Implement semantic segmentation for more detailed scene understanding
2. Add temporal consistency checks for more robust tracking
3. Integrate IMU data for ego-motion compensation
4. Implement learning-based tracking for better performance
5. Add human-robot interaction capabilities based on perception