# Module 2 Exercise: Digital Twin with Sensor Simulation

## Objective

Create a simple digital twin system that simulates a robot with basic sensors (camera and LIDAR) in a Unity or Gazebo environment, and visualize the sensor data in ROS 2.

## Requirements

1. Create a simulated robot with at least 2 sensors (camera and LIDAR)
2. Implement sensor data publishing to ROS 2 topics
3. Create a Unity or Gazebo world/environment
4. Implement a basic visualization of sensor data
5. Ensure real-time synchronization between physical simulation and digital twin

## Steps

### 1. Robot Model Setup

Create a simple robot model with sensors:

For Gazebo (in your URDF/Xacro):
```xml
<!-- Robot base -->
<link name="base_link">
  <visual>
    <geometry>
      <cylinder length="0.2" radius="0.3"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <cylinder length="0.2" radius="0.3"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="10"/>
    <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
  </inertial>
</link>

<!-- Camera sensor -->
<joint name="camera_joint" type="fixed">
  <parent link="base_link"/>
  <child link="camera_link"/>
  <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
</joint>

<link name="camera_link">
  <visual>
    <geometry>
      <box size="0.05 0.05 0.05"/>
    </geometry>
  </visual>
</link>

<gazebo reference="camera_link">
  <sensor type="camera" name="camera_sensor">
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_link</frame_name>
      <topic_name>camera/image_raw</topic_name>
    </plugin>
  </sensor>
</gazebo>

<!-- LIDAR sensor -->
<joint name="lidar_joint" type="fixed">
  <parent link="base_link"/>
  <child link="lidar_link"/>
  <origin xyz="0.15 0 0.2" rpy="0 0 0"/>
</joint>

<link name="lidar_link">
  <visual>
    <geometry>
      <cylinder length="0.05" radius="0.05"/>
    </geometry>
  </visual>
</link>

<gazebo reference="lidar_link">
  <sensor type="ray" name="lidar_sensor">
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>10.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_laser.so">
      <topic_name>scan</topic_name>
      <frame_name>lidar_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### 2. ROS 2 Node for Sensor Data Processing

Create a ROS 2 node to process and visualize sensor data:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import cv2
import numpy as np

class SensorDataProcessor(Node):
    def __init__(self):
        super().__init__('sensor_data_processor')

        # Create subscribers
        self.camera_subscriber = self.create_subscription(
            Image,
            'camera/image_raw',
            self.camera_callback,
            10
        )

        self.lidar_subscriber = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            10
        )

        # Initialize OpenCV bridge
        self.bridge = CvBridge()

        # Create publisher for processed image
        self.image_publisher = self.create_publisher(
            Image,
            'camera/processed_image',
            10
        )

        self.get_logger().info('Sensor Data Processor node initialized')

    def camera_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Process the image (e.g., edge detection)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Convert back to ROS Image message
        processed_msg = self.bridge.cv2_to_imgmsg(edges, encoding='mono8')
        processed_msg.header = msg.header

        # Publish processed image
        self.image_publisher.publish(processed_msg)

        # Display image
        cv2.imshow('Camera View', cv_image)
        cv2.imshow('Processed View', edges)
        cv2.waitKey(1)

    def lidar_callback(self, msg):
        # Process LIDAR data
        ranges = np.array(msg.ranges)

        # Filter out invalid ranges
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            self.get_logger().info(f'Min distance: {min_distance:.2f}m')

        # Create a simple visualization of LIDAR data
        self.visualize_lidar(msg)

    def visualize_lidar(self, scan_msg):
        # Create a polar plot of LIDAR data
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges))
        ranges = np.array(scan_msg.ranges)

        # Filter invalid ranges
        valid_idx = np.isfinite(ranges)
        angles = angles[valid_idx]
        ranges = ranges[valid_idx]

        if len(ranges) > 0:
            # Convert to Cartesian coordinates for visualization
            x = ranges * np.cos(angles)
            y = ranges * np.sin(angles)

            # Create a simple visualization
            img = np.zeros((400, 400, 3), dtype=np.uint8)
            for i in range(len(x)):
                px = int(200 + x[i] * 10)  # Scale factor of 10
                py = int(200 + y[i] * 10)  # Scale factor of 10
                if 0 <= px < 400 and 0 <= py < 400:
                    cv2.circle(img, (px, py), 2, (0, 255, 0), -1)

            cv2.imshow('LIDAR Scan', img)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    processor = SensorDataProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

### 3. Launch File

Create a launch file to start the simulation and processing node:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        # Launch Gazebo with your world
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('gazebo_ros'),
                    'launch',
                    'gazebo.launch.py'
                ])
            ]),
            launch_arguments={'world': 'path/to/your/world.sdf'}.items()
        ),

        # Launch your robot in Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-entity', 'robot', '-file', 'path/to/your/robot.urdf'],
            output='screen'
        ),

        # Launch the sensor data processor
        Node(
            package='your_package',
            executable='sensor_data_processor',
            name='sensor_data_processor',
            output='screen'
        )
    ])
```

### 4. Unity Implementation (Alternative)

If using Unity instead of Gazebo, implement the sensor simulation as follows:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class DigitalTwinSensors : MonoBehaviour
{
    public Camera cameraSensor;
    public Transform lidarSensor;
    public int lidarRays = 360;
    public float lidarRange = 10.0f;

    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        StartCoroutine(SendSensorData());
    }

    IEnumerator SendSensorData()
    {
        while (true)
        {
            // Send camera data
            SendCameraImage();

            // Send LIDAR data
            SendLIDARData();

            yield return new WaitForSeconds(0.1f); // 10Hz
        }
    }

    void SendCameraImage()
    {
        // Capture and send camera image
        Texture2D image = CaptureCameraImage(cameraSensor);
        ImageMsg rosImage = new ImageMsg();
        // ... populate ROS message
        ros.Send("camera/image_raw", rosImage);
    }

    void SendLIDARData()
    {
        float[] ranges = new float[lidarRays];

        for (int i = 0; i < lidarRays; i++)
        {
            float angle = (2 * Mathf.PI * i) / lidarRays;
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            RaycastHit hit;

            if (Physics.Raycast(lidarSensor.position, direction, out hit, lidarRange))
            {
                ranges[i] = hit.distance;
            }
            else
            {
                ranges[i] = lidarRange;
            }
        }

        LaserScanMsg laserScan = new LaserScanMsg();
        // ... populate laser scan message
        ros.Send("scan", laserScan);
    }

    Texture2D CaptureCameraImage(Camera cam)
    {
        // Implementation to capture camera image
        // Similar to the example shown in the lesson
        return null;
    }
}
```

## Expected Output

1. The robot should be visible in the simulation environment
2. Camera images should be published to `/camera/image_raw`
3. LIDAR scans should be published to `/scan`
4. Processed sensor data should be visualized
5. The digital twin should update in real-time with the simulation

## Evaluation Criteria

1. Correct sensor simulation and data publishing
2. Proper visualization of sensor data
3. Real-time synchronization between simulation and digital twin
4. Robust handling of sensor data
5. Clean and well-documented code

## Extension Challenges

1. Add more sensor types (IMU, force/torque sensors)
2. Implement sensor fusion algorithms
3. Add human models to the simulation
4. Implement basic navigation using sensor data
5. Add machine learning components for perception tasks