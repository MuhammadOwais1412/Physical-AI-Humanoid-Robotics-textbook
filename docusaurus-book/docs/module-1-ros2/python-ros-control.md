---
sidebar_position: 3
title: 'Python ROS Control with rclpy'
---

# Python ROS Control with rclpy

This lesson covers how to control robots using Python and the rclpy client library, which provides Python bindings for ROS 2.

## Introduction to rclpy

rclpy is the Python client library for ROS 2. It provides the standard ROS 2 API in Python and allows you to create nodes, publish and subscribe to topics, and provide and use services.

## Basic Node Structure

Every ROS 2 Python node follows a similar structure:

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('node_name')
        # Initialize publishers, subscribers, services, etc.

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Publishers and Subscribers

### Creating a Publisher

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
```

### Creating a Subscriber

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
```

## Services and Clients

### Creating a Service Server

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response
```

### Creating a Client

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

## Robot Control Examples

### Joint State Publisher

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

class JointStatePublisher(Node):

    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Initialize joint names and positions
        self.joint_names = ['joint1', 'joint2', 'joint3']
        self.joint_positions = [0.0, 0.0, 0.0]

    def timer_callback(self):
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        self.publisher_.publish(msg)
```

## Best Practices

1. **Use meaningful node names**: Choose descriptive names that reflect the node's purpose
2. **Handle parameters**: Use `declare_parameter()` to make nodes configurable
3. **Log appropriately**: Use `self.get_logger().info/warn/error()` for debugging
4. **Clean up resources**: Always call `destroy_node()` when shutting down
5. **Use QoS profiles**: Configure Quality of Service settings for your specific needs

## Error Handling

Always include proper error handling in your ROS 2 nodes:

```python
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    node.get_logger().info('Keyboard interrupt received, shutting down')
finally:
    node.destroy_node()
    rclpy.shutdown()
```

Python with rclpy provides a powerful and accessible way to develop robotic applications, making it ideal for rapid prototyping and educational purposes.