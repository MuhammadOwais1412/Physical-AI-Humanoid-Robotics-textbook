# Module 1 Exercise: Basic ROS 2 Publisher and Subscriber

## Objective

Create a simple ROS 2 publisher that publishes messages to a topic and a subscriber that receives and prints those messages.

## Requirements

1. Create a publisher node that publishes "Hello World" messages every 0.5 seconds
2. Create a subscriber node that receives and prints the messages
3. Use the `std_msgs/String` message type
4. The publisher should count the number of messages sent

## Steps

### 1. Create the Publisher Node

Create a Python file called `talker.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):

    def __init__(self):
        super().__init__('talker')
        self.publisher_ = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    talker = Talker()

    try:
        rclpy.spin(talker)
    except KeyboardInterrupt:
        pass
    finally:
        talker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. Create the Subscriber Node

Create a Python file called `listener.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):

    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    listener = Listener()

    try:
        rclpy.spin(listener)
    except KeyboardInterrupt:
        pass
    finally:
        listener.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 3. Package Structure

Create a proper ROS 2 package structure:

```
ros2_talker_listener/
├── src/
│   └── talker_listener/
│       ├── talker.py
│       └── listener.py
├── setup.py
├── setup.cfg
└── package.xml
```

### 4. Setup Files

Create a `setup.py` file:

```python
from setuptools import setup

package_name = 'talker_listener'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Simple talker/listener example',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = talker_listener.talker:main',
            'listener = talker_listener.listener:main',
        ],
    },
)
```

Create a `package.xml` file:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>talker_listener</name>
  <version>0.0.0</version>
  <description>Simple talker/listener example</description>
  <maintainer email="your_email@example.com">your_name</maintainer>
  <license>Apache License 2.0</license>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Running the Example

1. Source your ROS 2 environment:
   ```bash
   source /opt/ros/humble/setup.bash  # or your ROS 2 distribution
   ```

2. Navigate to your package directory and build:
   ```bash
   cd ros2_talker_listener
   colcon build --packages-select talker_listener
   source install/setup.bash
   ```

3. Run the publisher in one terminal:
   ```bash
   ros2 run talker_listener talker
   ```

4. Run the subscriber in another terminal:
   ```bash
   ros2 run talker_listener listener
   ```

## Expected Output

The publisher should output messages like:
```
[INFO] [1612345678.123456789] [talker]: Publishing: "Hello World: 0"
[INFO] [1612345678.623456789] [talker]: Publishing: "Hello World: 1"
```

The subscriber should output messages like:
```
[INFO] [1612345678.123456789] [listener]: I heard: "Hello World: 0"
[INFO] [1612345678.623456789] [listener]: I heard: "Hello World: 1"
```

## Extension Challenges

1. Modify the publisher to send different types of messages (integers, floats)
2. Add parameters to control the publishing rate
3. Create a service that allows changing the message content
4. Implement a more complex data structure as a custom message