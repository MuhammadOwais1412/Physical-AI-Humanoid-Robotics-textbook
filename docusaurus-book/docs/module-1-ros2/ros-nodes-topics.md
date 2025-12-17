---
sidebar_position: 2
title: 'ROS 2 Nodes, Topics, and Services'
---

# ROS 2 Nodes, Topics, and Services

This lesson covers the fundamental communication patterns in ROS 2: nodes, topics, and services.

## Nodes

A node is an executable that uses ROS 2 to communicate with other nodes. Nodes are designed to perform a specific task and can be combined with other nodes to perform more complex tasks.

### Creating a Node

In Python, you create a node by defining a class that inherits from `Node`:

```python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
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

## Topics

Topics are named buses over which nodes exchange messages. Publishers write data to topics and subscribers receive data from topics. This is a many-to-many relationship.

### Topic Communication

- Publishers send messages to a topic
- Subscribers receive messages from a topic
- Multiple publishers and subscribers can use the same topic
- Communication is asynchronous

## Services

Services provide a request/reply communication pattern. A client sends a request to a service server, which processes the request and sends back a response.

### Service Communication

- Synchronous communication
- Request/Response pattern
- One-to-one relationship (one client, one server)

## Actions

Actions are a more advanced communication pattern that supports long-running tasks with feedback and goal preemption.

## Practical Example

Let's look at how to implement a simple publisher and subscriber:

```python
# publisher_member_function.py
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
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
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

This communication model forms the backbone of ROS 2 applications and enables the modular design that makes ROS 2 so powerful for robotics development.