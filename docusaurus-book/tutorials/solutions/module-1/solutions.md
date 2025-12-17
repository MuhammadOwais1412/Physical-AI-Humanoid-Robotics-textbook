# Solutions for Module 1: The Robotic Nervous System (ROS 2)

## Exercise 1: ROS Nodes and Topics
**Problem**: Create a simple publisher and subscriber in ROS 2 using Python.

**Solution**:

### Publisher Node
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Node
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercise 2: Python ROS Control
**Problem**: Implement a simple action client for controlling a robot arm.

**Solution**:
```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory

class TrajectoryClient(Node):
    def __init__(self):
        super().__init__('trajectory_client')
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            'joint_trajectory_controller/follow_joint_trajectory'
        )

    def send_goal(self):
        goal_msg = FollowJointTrajectory.Goal()
        # Implementation details for trajectory
        # ...

        self._action_client.wait_for_server()
        return self._action_client.send_goal_async(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    client = TrajectoryClient()
    # Send trajectory goal
    rclpy.spin(client)

if __name__ == '__main__':
    main()
```

## Exercise 3: URDF for Humanoids
**Problem**: Create a basic URDF for a simple humanoid model.

**Solution**:
```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </visual>
  </link>

  <!-- Torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.15"/>
  </joint>

  <link name="torso">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.3"/>
      </geometry>
    </visual>
  </link>

  <!-- Head -->
  <joint name="head_joint" type="fixed">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.2"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.07"/>
      </geometry>
    </visual>
  </link>
</robot>
```