#!/usr/bin/env python3

"""
Advanced Humanoid Locomotion Controller

This script demonstrates an advanced locomotion controller for humanoid robots
using the Zero Moment Point (ZMP) and Linear Inverted Pendulum Model (LIPM) approaches.
"""

import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


class HumanoidLocomotionController:
    """
    Advanced locomotion controller for humanoid robots
    """
    def __init__(self, robot_mass=75.0, robot_height=1.7, control_frequency=200):
        self.robot_mass = robot_mass
        self.robot_height = robot_height
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency

        # Zero Moment Point (ZMP) controller parameters
        self.zmp_kp = 100.0
        self.zmp_kd = 20.0

        # Linear Inverted Pendulum Model (LIPM) parameters
        self.pendulum_height = 0.9  # Height of CoM above ground
        self.gravity = 9.81
        self.omega = np.sqrt(self.gravity / self.pendulum_height)

        # Gait parameters
        self.step_length = 0.3
        self.step_height = 0.1
        self.step_duration = 1.0
        self.double_support_ratio = 0.2

        # Current state
        self.com_position = np.array([0.0, 0.0, self.pendulum_height])
        self.com_velocity = np.array([0.0, 0.0, 0.0])
        self.com_acceleration = np.array([0.0, 0.0, 0.0])

        # Support foot tracking
        self.left_foot_position = np.array([-0.1, 0.1, 0.0])
        self.right_foot_position = np.array([-0.1, -0.1, 0.0])
        self.support_foot = "left"  # Start with left foot support

        # Walking state
        self.step_phase = 0.0
        self.step_count = 0
        self.walking_velocity = np.array([0.3, 0.0])  # Desired walking velocity

    def update_state(self, com_pos, com_vel, left_foot_pos, right_foot_pos):
        """Update the controller with current robot state"""
        self.com_position = com_pos
        self.com_velocity = com_vel
        self.left_foot_position = left_foot_pos
        self.right_foot_position = right_foot_pos

    def compute_zmp(self, com_pos, com_vel, com_acc):
        """Compute Zero Moment Point from CoM state"""
        zmp_x = com_pos[0] - (com_pos[2] / self.gravity) * com_acc[0]
        zmp_y = com_pos[1] - (com_pos[2] / self.gravity) * com_acc[1]
        return np.array([zmp_x, zmp_y])

    def compute_desired_zmp(self, time):
        """Compute desired ZMP trajectory for walking"""
        # Generate a stable ZMP trajectory following the walking pattern
        step_period = self.step_duration
        phase = (time % step_period) / step_period

        # Desired ZMP follows the support foot with some anticipation
        if self.support_foot == "left":
            support_pos = self.left_foot_position[:2]
        else:
            support_pos = self.right_foot_position[:2]

        # Smooth transition between steps
        transition_width = 0.1  # 10% of step duration
        if phase < transition_width:
            # Transition from previous support foot
            if self.step_count % 2 == 0:
                prev_support = self.right_foot_position[:2]  # Previous was right
            else:
                prev_support = self.left_foot_position[:2]   # Previous was left
            alpha = phase / transition_width
            desired_zmp = (1 - alpha) * prev_support + alpha * support_pos
        else:
            desired_zmp = support_pos

        return desired_zmp

    def compute_com_trajectory(self, time):
        """Compute desired CoM trajectory using LIPM"""
        # Using Linear Inverted Pendulum Model to generate CoM trajectory
        # that will result in the desired ZMP

        # Desired ZMP
        desired_zmp = self.compute_desired_zmp(time)

        # Current ZMP
        current_zmp = self.compute_zmp(self.com_position, self.com_velocity, self.com_acceleration)

        # ZMP error
        zmp_error = desired_zmp - current_zmp

        # Compute CoM acceleration to correct ZMP error
        com_acc_correction = self.gravity / self.pendulum_height * zmp_error

        # Desired CoM trajectory with smooth transitions
        t_in_step = time % self.step_duration
        step_ratio = t_in_step / self.step_duration

        # Forward progression
        desired_x = self.walking_velocity[0] * time
        desired_y = 0.0  # For now, walking straight

        # Lateral adjustment based on support foot
        if self.support_foot == "left":
            desired_y = 0.1  # Left foot slightly lateral
        else:
            desired_y = -0.1  # Right foot slightly lateral

        # Vertical movement for step height
        if self.support_foot == "left":
            # Left foot swing trajectory
            if self.step_count % 2 == 0:  # Swing phase for left foot
                swing_phase = (step_ratio - self.double_support_ratio) / (1 - 2 * self.double_support_ratio)
                if 0 <= swing_phase <= 1:
                    desired_z = self.pendulum_height + self.step_height * np.sin(np.pi * swing_phase)
                else:
                    desired_z = self.pendulum_height
            else:
                desired_z = self.pendulum_height
        else:
            # Right foot swing trajectory
            if self.step_count % 2 == 1:  # Swing phase for right foot
                swing_phase = (step_ratio - self.double_support_ratio) / (1 - 2 * self.double_support_ratio)
                if 0 <= swing_phase <= 1:
                    desired_z = self.pendulum_height + self.step_height * np.sin(np.pi * swing_phase)
                else:
                    desired_z = self.pendulum_height
            else:
                desired_z = self.pendulum_height

        desired_com = np.array([desired_x, desired_y, desired_z])

        # Compute desired velocity and acceleration
        dt = 0.01  # Small time step for numerical differentiation
        next_time = time + dt
        next_desired_com = self.compute_com_trajectory_for_time(next_time)

        desired_vel = (next_desired_com - desired_com) / dt
        desired_acc = (desired_vel - self.com_velocity) / dt + com_acc_correction

        return desired_com, desired_vel, desired_acc

    def compute_com_trajectory_for_time(self, time):
        """Helper function to compute CoM position at a given time"""
        t_in_step = time % self.step_duration
        step_ratio = t_in_step / self.step_duration

        # Forward progression
        desired_x = self.walking_velocity[0] * time
        desired_y = 0.0

        # Lateral adjustment based on support foot
        if self.support_foot == "left":
            desired_y = 0.1
        else:
            desired_y = -0.1

        # Vertical movement
        desired_z = self.pendulum_height

        return np.array([desired_x, desired_y, desired_z])

    def compute_foot_trajectory(self, time):
        """Compute desired foot trajectories for walking"""
        # Left foot trajectory
        left_foot_pos = self.left_foot_position.copy()

        # Right foot trajectory
        right_foot_pos = self.right_foot_position.copy()

        # Determine swing foot based on step phase
        t_in_step = time % self.step_duration
        step_ratio = t_in_step / self.step_duration

        # Switch support foot at mid-stance
        if step_ratio > 0.5 and self.step_count != int(time / self.step_duration):
            self.step_count = int(time / self.step_duration)
            # Switch support foot
            if self.support_foot == "left":
                self.support_foot = "right"
            else:
                self.support_foot = "left"

        # Compute swing foot trajectory
        if self.support_foot == "left":
            # Right foot is swing foot
            swing_ratio = (step_ratio - self.double_support_ratio) / (1 - 2 * self.double_support_ratio)
            if 0 <= swing_ratio <= 1:
                # Swing motion
                right_foot_pos[0] = self.left_foot_position[0] + self.step_length * swing_ratio
                right_foot_pos[1] = -0.1  # Return to nominal position
                right_foot_pos[2] = self.step_height * np.sin(np.pi * swing_ratio)  # Arc motion
        else:
            # Left foot is swing foot
            swing_ratio = (step_ratio - self.double_support_ratio) / (1 - 2 * self.double_support_ratio)
            if 0 <= swing_ratio <= 1:
                # Swing motion
                left_foot_pos[0] = self.right_foot_position[0] + self.step_length * swing_ratio
                left_foot_pos[1] = 0.1  # Return to nominal position
                left_foot_pos[2] = self.step_height * np.sin(np.pi * swing_ratio)  # Arc motion

        return left_foot_pos, right_foot_pos

    def compute_joint_commands(self, time):
        """Compute joint commands for the entire robot to achieve desired trajectories"""
        # Compute desired CoM and foot trajectories
        desired_com, desired_com_vel, desired_com_acc = self.compute_com_trajectory(time)
        desired_left_foot, desired_right_foot = self.compute_foot_trajectory(time)

        # Inverse kinematics and dynamics would be computed here
        # For this example, we'll return placeholder joint commands
        num_joints = 28  # Example: 28 DOF humanoid
        joint_commands = np.zeros(num_joints)

        # This would typically involve:
        # 1. Inverse kinematics to achieve desired CoM and foot positions
        # 2. Whole-body control to distribute forces appropriately
        # 3. Joint-space control for final commands

        # Return placeholder values with some realistic structure
        # Lower body joints (legs) would be computed based on CoM and foot control
        joint_commands[:6] = desired_left_foot  # Left leg position control
        joint_commands[6:12] = desired_right_foot  # Right leg position control
        joint_commands[12:18] = desired_com[:3] * 0.1  # CoM control affecting upper body
        joint_commands[18:] = 0.0  # Arms and other joints (simplified)

        return joint_commands

    def visualize_locomotion(self, duration=5.0):
        """Visualize the locomotion patterns"""
        times = np.arange(0, duration, 0.02)  # 50 Hz visualization
        com_x = []
        com_y = []
        left_foot_x = []
        left_foot_y = []
        right_foot_x = []
        right_foot_y = []
        support_foot = []

        for t in times:
            # Update support foot state
            step_ratio = (t % self.step_duration) / self.step_duration
            if step_ratio > 0.5:
                current_step_count = int(t / self.step_duration)
                if current_step_count != self.step_count:
                    self.step_count = current_step_count
                    if self.support_foot == "left":
                        self.support_foot = "right"
                    else:
                        self.support_foot = "left"

            # Compute trajectories
            desired_com, _, _ = self.compute_com_trajectory(t)
            left_foot_pos, right_foot_pos = self.compute_foot_trajectory(t)

            # Store for plotting
            com_x.append(desired_com[0])
            com_y.append(desired_com[1])
            left_foot_x.append(left_foot_pos[0])
            left_foot_y.append(left_foot_pos[1])
            right_foot_x.append(right_foot_pos[0])
            right_foot_y.append(right_foot_pos[1])
            support_foot.append(1 if self.support_foot == "left" else 2)

        # Create visualization
        plt.figure(figsize=(12, 8))

        # Plot CoM and foot trajectories
        plt.subplot(2, 2, 1)
        plt.plot(com_x, com_y, 'r-', label='CoM trajectory', linewidth=2)
        plt.plot(left_foot_x, left_foot_y, 'b--', label='Left foot trajectory')
        plt.plot(right_foot_x, right_foot_y, 'g--', label='Right foot trajectory')
        plt.scatter(com_x[0], com_y[0], color='red', s=100, label='Start', zorder=5)
        plt.scatter(com_x[-1], com_y[-1], color='black', s=100, label='End', zorder=5)
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Locomotion Trajectories')
        plt.legend()
        plt.grid(True)

        # Plot X positions over time
        plt.subplot(2, 2, 2)
        plt.plot(times, com_x, 'r-', label='CoM X', linewidth=2)
        plt.plot(times, left_foot_x, 'b--', label='Left foot X')
        plt.plot(times, right_foot_x, 'g--', label='Right foot X')
        plt.xlabel('Time (s)')
        plt.ylabel('X Position (m)')
        plt.title('X Position vs Time')
        plt.legend()
        plt.grid(True)

        # Plot Y positions over time
        plt.subplot(2, 2, 3)
        plt.plot(times, com_y, 'r-', label='CoM Y', linewidth=2)
        plt.plot(times, left_foot_y, 'b--', label='Left foot Y')
        plt.plot(times, right_foot_y, 'g--', label='Right foot Y')
        plt.xlabel('Time (s)')
        plt.ylabel('Y Position (m)')
        plt.title('Y Position vs Time')
        plt.legend()
        plt.grid(True)

        # Plot support foot over time
        plt.subplot(2, 2, 4)
        plt.step(times, support_foot, where='post', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Support Foot')
        plt.title('Support Foot (1=Left, 2=Right)')
        plt.yticks([1, 2], ['Left', 'Right'])
        plt.grid(True)

        plt.tight_layout()
        plt.show()


def main():
    print("Initializing Humanoid Locomotion Controller...")

    # Initialize locomotion controller
    controller = HumanoidLocomotionController()

    print("Computing locomotion trajectories...")

    # Example: Compute joint commands for a short duration
    for t in np.arange(0, 2.0, 0.01):  # 2 seconds of walking
        joint_commands = controller.compute_joint_commands(t)

        if int(t * 100) % 100 == 0:  # Print every second
            print(f"Time: {t:.2f}s, Joint commands (first 6): {joint_commands[:6]}")

    print("Locomotion control computation completed.")

    # Visualize the locomotion (optional)
    print("Generating locomotion visualization...")
    controller.visualize_locomotion(duration=3.0)


if __name__ == "__main__":
    main()