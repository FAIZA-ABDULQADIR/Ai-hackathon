# Chapter 1 Code Examples

This directory contains code examples for Chapter 1: Physical AI Foundations.

## Examples

### simple-publisher.py
A basic ROS 2 publisher node that sends "Hello World" messages to a topic.

### simple-subscriber.py
A basic ROS 2 subscriber node that listens to messages from a topic and logs them.

## Running the Examples

1. Make sure you have ROS 2 installed (e.g., Humble Hawksbill)
2. Source your ROS 2 environment: `source /opt/ros/humble/setup.bash`
3. Run the publisher: `python3 simple-publisher.py`
4. In another terminal, run the subscriber: `python3 simple-subscriber.py`

You should see the publisher sending messages and the subscriber receiving them.