#!/usr/bin/env python3

"""
Isaac Sim Nav2 Controller

This script demonstrates Nav2 integration with Isaac Sim,
including waypoint navigation and path execution.
"""

import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
import tf_transformations
import time
import math


class IsaacSimNav2Controller:
    """
    A Nav2 controller for Isaac Sim navigation
    """
    def __init__(self):
        # Initialize ROS 2
        rclpy.init()

        # Create navigator
        self.navigator = BasicNavigator()

        print("Waiting for Nav2 to become active...")
        # Wait for Nav2 to be active
        self.navigator.waitUntilNav2Active()
        print("Nav2 is now active!")

    def create_pose_stamped(self, position_x, position_y, rotation_z, rotation_w):
        """Create a PoseStamped message"""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        pose.pose.position.x = position_x
        pose.pose.position.y = position_y
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = rotation_z
        pose.pose.orientation.w = rotation_w

        return pose

    def navigate_to_waypoints(self, waypoints):
        """Navigate to a sequence of waypoints"""
        print(f"Starting navigation to {len(waypoints)} waypoints...")

        # Go through each waypoint
        for i, waypoint in enumerate(waypoints):
            print(f"\nNavigating to waypoint {i+1}/{len(waypoints)}")
            print(f"Target position: ({waypoint.pose.position.x:.2f}, {waypoint.pose.position.y:.2f})")

            # Send navigation goal
            self.navigator.goToPose(waypoint)

            # Monitor progress
            start_time = time.time()
            while not self.navigator.isTaskComplete():
                feedback = self.navigator.getFeedback()
                if feedback:
                    elapsed_time = time.time() - start_time
                    print(f"  Distance remaining: {feedback.distance_remaining:.2f}m, "
                          f"Time elapsed: {elapsed_time:.1f}s", end='\r')

                    # Check for navigation failure
                    if feedback.navigation_time > 60:  # Timeout after 60 seconds
                        print(f"\nNavigation timeout to waypoint {i+1}, cancelling goal...")
                        self.navigator.cancelTask()
                        break

                time.sleep(0.5)  # Small delay to prevent excessive feedback

            # Check result
            result = self.navigator.getResult()
            elapsed_time = time.time() - start_time

            if result == 1:  # Cancelled
                print(f"\nNavigation to waypoint {i+1} was cancelled.")
            elif result == 2:  # Failed
                print(f"\nNavigation to waypoint {i+1} failed after {elapsed_time:.1f}s.")
            else:  # Succeeded (3)
                print(f"\nSuccessfully reached waypoint {i+1} in {elapsed_time:.1f}s.")

            # Small delay between waypoints
            time.sleep(1)

        print(f"\nCompleted navigation to all {len(waypoints)} waypoints.")

    def run_square_navigation(self):
        """Run a square navigation pattern"""
        print("\n=== Running Square Navigation Demo ===")

        # Define a square pattern of waypoints
        waypoints = [
            self.create_pose_stamped(1.0, 0.0, 0.0, 1.0),      # Move 1m along x-axis
            self.create_pose_stamped(1.0, 1.0, 0.707, 0.707),  # Move 1m along y-axis, rotate 90 degrees
            self.create_pose_stamped(0.0, 1.0, 1.0, 0.0),      # Move back 1m along x-axis, rotate 90 degrees
            self.create_pose_stamped(0.0, 0.0, 0.707, 0.707),  # Return to start, rotate 90 degrees
        ]

        self.navigate_to_waypoints(waypoints)

    def run_circular_navigation(self):
        """Run a circular navigation pattern"""
        print("\n=== Running Circular Navigation Demo ===")

        # Define waypoints for a circular path
        waypoints = []
        radius = 2.0
        num_points = 8

        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)

            # Calculate orientation to face the next point
            next_angle = 2 * math.pi * (i + 1) / num_points
            quat = tf_transformations.quaternion_from_euler(0, 0, next_angle)

            waypoints.append(self.create_pose_stamped(x, y, quat[2], quat[3]))

        self.navigate_to_waypoints(waypoints)

    def run_dynamic_navigation(self):
        """Run navigation with simulated dynamic obstacles"""
        print("\n=== Running Dynamic Navigation Demo ===")

        # Navigate to a goal position
        goal_pose = self.create_pose_stamped(5.0, 0.0, 0.0, 1.0)

        print("Navigating to goal position (5.0, 0.0)...")
        print("This demo simulates dynamic obstacle handling.")

        # Send navigation goal
        self.navigator.goToPose(goal_pose)

        # Monitor navigation while simulating dynamic obstacles
        start_time = time.time()
        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            if feedback:
                elapsed_time = time.time() - start_time
                print(f"  Distance to goal: {feedback.distance_remaining:.2f}m, "
                      f"Time elapsed: {elapsed_time:.1f}s", end='\r')

                # Simulate dynamic obstacle detection and handling
                # In real implementation, this would integrate with Isaac Sim's
                # obstacle detection capabilities
                if feedback.distance_remaining < 1.5 and elapsed_time > 10:
                    print(f"\n  Potential obstacle detected near goal, continuing...")
                    # In a real system, this might trigger obstacle avoidance behaviors

            time.sleep(0.5)

        result = self.navigator.getResult()
        elapsed_time = time.time() - start_time

        if result == 3:  # Succeeded
            print(f"\nSuccessfully navigated to goal in {elapsed_time:.1f}s.")
        else:
            print(f"\nNavigation completed with result: {result} in {elapsed_time:.1f}s.")

    def run_navigation_demo(self):
        """Run a complete navigation demonstration"""
        print("Starting Isaac Sim Nav2 Navigation Demo...")

        # Run different navigation patterns
        self.run_square_navigation()
        self.run_circular_navigation()
        self.run_dynamic_navigation()

        print("\nNavigation demo completed successfully!")

    def cleanup(self):
        """Clean up ROS 2 resources"""
        rclpy.shutdown()


def main():
    """Main function to run the Nav2 controller"""
    print("Initializing Isaac Sim Nav2 Controller...")

    # Create Nav2 controller instance
    nav_controller = IsaacSimNav2Controller()

    try:
        # Run the complete navigation demo
        nav_controller.run_navigation_demo()

    except KeyboardInterrupt:
        print("\nNavigation demo interrupted by user.")
    except Exception as e:
        print(f"\nError during navigation: {e}")
    finally:
        # Clean up
        nav_controller.cleanup()
        print("Nav2 controller cleanup completed.")


if __name__ == "__main__":
    main()