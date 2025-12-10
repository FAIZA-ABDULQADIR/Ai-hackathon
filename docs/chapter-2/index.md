---
title: 'Chapter 2: ROS 2 Fundamentals - Nodes, Topics, Services, and Actions'
sidebar_position: 3
description: 'Understanding the core concepts of ROS 2: nodes, topics, services, and actions'
keywords: [ros2, robotics, nodes, topics, services, actions, middleware, communication]
tags: [ros2, fundamentals, communication, robotics]
---

# Chapter 2: ROS 2 Fundamentals - Nodes, Topics, Services, and Actions

## Introduction to ROS 2 Architecture

ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms. ROS 2 addresses many of the limitations of ROS 1 by providing improved security, real-time support, and better cross-platform compatibility.

### Why ROS 2?

ROS 2 was developed to address key limitations of ROS 1:

- **Real-time Support**: Better integration with real-time systems
- **Security**: Built-in security features and authentication
- **Cross-Platform**: Improved support for Windows, macOS, and various Linux distributions
- **Middleware**: Uses DDS (Data Distribution Service) for communication
- **Professional Development**: Better suited for production and commercial applications

## Core Concepts of ROS 2

### Nodes

A node is a process that performs computation. ROS 2 is designed to be a distributed system of nodes working together. Each node can perform specific tasks and communicate with other nodes.

**Key characteristics of nodes:**
- Each node is typically responsible for a specific task
- Nodes are designed to be modular and reusable
- Multiple nodes can run simultaneously
- Nodes can be written in different programming languages (C++, Python, etc.)

### Topics and Publishing/Subscription Model

Topics enable asynchronous communication between nodes using a publish/subscribe pattern:

```
┌─────────────┐publish    ┌─────────────┐
│  Publisher  │──────────▶│   Topic     │
│    Node     │    msg    │(message bus)│
└─────────────┘           └─────────────┘
                           ▲
                           │
                    subscribe
                           │
                    ┌─────────────┐
                    │ Subscriber  │
                    │    Node     │
                    └─────────────┘
```

- **Publisher**: Node that sends messages to a topic
- **Subscriber**: Node that receives messages from a topic
- **Message**: Data structure sent between nodes
- **Topic**: Named bus over which nodes exchange messages

### Services

Services provide synchronous request/response communication:

```
┌─────────────┐ request  ┌─────────────┐
│  Client     │─────────▶│   Service   │
│    Node     │    ◀─────│    Server   │
└─────────────┘ response │    Node     │
                        └─────────────┘
```

- **Service Server**: Node that provides a service
- **Service Client**: Node that uses a service
- **Service Interface**: Defines request/response structure

### Actions

Actions are for long-running tasks that may provide feedback:

```
┌─────────────┐ goal     ┌─────────────┐
│  Action     │─────────▶│   Action    │
│  Client     │    ◀─────│   Server    │
│    Node     │ feedback │    Node     │
└─────────────┘    │     └─────────────┘
                   │
            ┌─────────────┐
            │   Result    │
            │   (final)   │
            └─────────────┘
```

- **Action Server**: Node that executes long-running tasks
- **Action Client**: Node that requests and monitors tasks
- **Action Interface**: Defines goal, feedback, and result structures

## Working with Nodes

### Creating a Node in Python (rclpy)

```python
import rclpy
from rclpy.node import Node

class MyRobotNode(Node):
    def __init__(self):
        super().__init__('my_robot_node')
        self.get_logger().info('My Robot Node has started')

def main(args=None):
    rclpy.init(args=args)
    node = MyRobotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Node in C++

```cpp
#include "rclcpp/rclcpp.hpp"

class MyRobotNode : public rclcpp::Node
{
public:
    MyRobotNode() : Node("my_robot_node")
    {
        RCLCPP_INFO(this->get_logger(), "My Robot Node has started");
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MyRobotNode>());
    rclcpp::shutdown();
    return 0;
}
```

## Topics: Publisher and Subscriber

### Publisher Example (Python)

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TalkerNode(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher_ = self.create_publisher(String, 'chatter', 10)
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
    talker = TalkerNode()
    rclpy.spin(talker)
    talker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Example (Python)

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ListenerNode(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    listener = ListenerNode()
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Services: Client and Server

### Service Server Example (Python)

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class AddTwoIntsServer(Node):
    def __init__(self):
        super().__init__('add_two_ints_server')
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {request.a} + {request.b} = {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    add_two_ints_server = AddTwoIntsServer()
    rclpy.spin(add_two_ints_server)
    add_two_ints_server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client Example (Python)

```python
import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class AddTwoIntsClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main():
    rclpy.init()

    client = AddTwoIntsClient()
    response = client.send_request(int(sys.argv[1]), int(sys.argv[2]))

    if response is not None:
        client.get_logger().info(
            f'Result of add_two_ints: {response.sum}')
    else:
        client.get_logger().info('Service call failed')

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Actions

Actions are used for long-running tasks that may provide feedback and can be canceled:

### Action Server Example (Python)

```python
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            callback_group=rclpy.callback_groups.ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Publishing feedback: {feedback_msg.sequence}')

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Returning result: {result.sequence}')

        return result

def main(args=None):
    rclpy.init(args=args)
    fibonacci_action_server = FibonacciActionServer()
    rclpy.spin(fibonacci_action_server)
    fibonacci_action_server.destroy()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## URDF (Unified Robot Description Format)

URDF is an XML format for representing a robot model in ROS. It describes the robot's physical and visual properties.

### Basic URDF Example

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Arm link -->
  <link name="arm_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
  </link>

  <!-- Joint connecting base and arm -->
  <joint name="base_to_arm" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
</robot>
```

## ROS 2 Launch Files

Launch files allow you to start multiple nodes with a single command:

```xml
<launch>
  <node pkg="my_package" exec="talker" name="talker_node"/>
  <node pkg="my_package" exec="listener" name="listener_node"/>
</launch>
```

Or using Python launch files:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='demo_nodes_py',
            executable='talker',
            name='talker_node'
        ),
        Node(
            package='demo_nodes_py',
            executable='listener',
            name='listener_node'
        )
    ])
```

## ROS 2 Parameters

Parameters allow configuration of nodes:

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)

        # Get parameter values
        robot_name = self.get_parameter('robot_name').value
        max_velocity = self.get_parameter('max_velocity').value

        self.get_logger().info(f'Robot: {robot_name}, Max velocity: {max_velocity}')

def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quality of Service (QoS) Settings

QoS settings allow fine-tuning of communication behavior:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Create a QoS profile
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)

# Use in publisher
publisher = node.create_publisher(String, 'topic', qos_profile)
```

## Best Practices

### 1. Node Design
- Keep nodes focused on a single responsibility
- Use appropriate naming conventions
- Handle errors gracefully
- Implement proper cleanup

### 2. Message Design
- Use appropriate message types
- Consider message size and frequency
- Design for future extensibility

### 3. Resource Management
- Properly clean up resources
- Handle node lifecycle appropriately
- Use appropriate callback groups for threading

### 4. Testing
- Test nodes in isolation
- Verify communication patterns
- Test error conditions

## Summary

ROS 2 provides a comprehensive framework for robot software development with:

- **Nodes**: The basic computational units
- **Topics**: Asynchronous publish/subscribe communication
- **Services**: Synchronous request/response communication
- **Actions**: Long-running tasks with feedback
- **URDF**: Robot description format
- **Launch files**: Multi-node orchestration
- **Parameters**: Configuration management
- **QoS**: Communication behavior tuning

Understanding these fundamental concepts is essential for developing robust robotic applications. Each concept serves a specific purpose in the ROS 2 ecosystem and choosing the right communication pattern is crucial for effective robot software architecture.

---

## Exercises

1. **Node Creation**: Create a simple node that publishes the current time to a topic every second.

2. **Service Implementation**: Implement a service that converts temperatures between Celsius and Fahrenheit.

3. **URDF Modeling**: Create a URDF for a simple differential drive robot with two wheels.

4. **Communication Pattern**: Design a robot system using appropriate communication patterns (nodes, topics, services) for sensor data processing and navigation.

## References and Citations

1. ROS 2 Documentation. (2023). ROS 2 Concepts. https://docs.ros.org/en/humble/Concepts.html
2. Open Robotics. (2023). ROS 2 Design. https://design.ros2.org/
3. Quigley, M., Gerkey, B., & Smart, W. D. (2015). Programming robots with ROS: a practical introduction to the robot operating system. O'Reilly Media.
4. DDS (Data Distribution Service) specification. (2023). Object Management Group. https://www.omg.org/spec/DDS/
5. ROS 2 Documentation. (2023). Tutorials. https://docs.ros.org/en/humble/Tutorials.html
6. ROS 2 Documentation. (2023). Quality of Service. https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service-Settings.html

## Accessibility Features

This content includes:
- Proper heading hierarchy (H1, H2, H3) for screen readers
- Clear, descriptive list items
- Semantic structure for assistive technologies
- Code examples with syntax highlighting
- Diagrams and visual representations using ASCII art
