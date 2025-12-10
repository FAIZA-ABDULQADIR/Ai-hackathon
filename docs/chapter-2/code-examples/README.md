# Chapter 2 Code Examples

This directory contains code examples for Chapter 2: ROS 2 Fundamentals.

## Examples

### service-server.py
A ROS 2 service server that provides an 'add_two_ints' service to add two integers.

### service-client.py
A ROS 2 service client that calls the 'add_two_ints' service to add two integers.

## Running the Examples

1. Make sure you have ROS 2 installed (e.g., Humble Hawksbills)
2. Source your ROS 2 environment: `source /opt/ros/humble/setup.bash`
3. Run the service server: `python3 service-server.py`
4. In another terminal, run the client: `python3 service-client.py 5 7`

You should see the client send a request to the server and receive the sum of the two numbers.