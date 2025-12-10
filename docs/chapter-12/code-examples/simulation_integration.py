#!/usr/bin/env python3
"""
Simulation Integration Example for Capstone Project
Chapter 12 - Physical AI & Humanoid Robotics Textbook
Demonstrates integration between Gazebo, Unity, and Isaac Sim
"""

import numpy as np
import time
import threading
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class RobotState:
    """Data class to represent robot state"""
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    position: np.ndarray  # x, y, z
    orientation: np.ndarray  # quaternion [x, y, z, w]
    timestamp: float

class SimulationSynchronizer:
    """Synchronizes state between multiple simulation environments"""

    def __init__(self):
        self.gazebo_state = RobotState(
            joint_positions=np.zeros(28),
            joint_velocities=np.zeros(28),
            joint_torques=np.zeros(28),
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            timestamp=time.time()
        )

        self.unity_state = RobotState(
            joint_positions=np.zeros(28),
            joint_velocities=np.zeros(28),
            joint_torques=np.zeros(28),
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            timestamp=time.time()
        )

        self.isaac_sim_state = RobotState(
            joint_positions=np.zeros(28),
            joint_velocities=np.zeros(28),
            joint_torques=np.zeros(28),
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            timestamp=time.time()
        )

        self.tolerance = 0.01  # 1cm tolerance for synchronization
        self.sync_thread = None
        self.running = False

    def update_gazebo_state(self, joint_positions: np.ndarray, position: np.ndarray, orientation: np.ndarray):
        """Update Gazebo simulation state"""
        self.gazebo_state.joint_positions = joint_positions
        self.gazebo_state.position = position
        self.gazebo_state.orientation = orientation
        self.gazebo_state.timestamp = time.time()

        # Synchronize with other simulators
        self.synchronize_states()

    def update_unity_state(self, joint_positions: np.ndarray, position: np.ndarray, orientation: np.ndarray):
        """Update Unity digital twin state"""
        self.unity_state.joint_positions = joint_positions
        self.unity_state.position = position
        self.unity_state.orientation = orientation
        self.unity_state.timestamp = time.time()

        # Synchronize with other simulators
        self.synchronize_states()

    def update_isaac_sim_state(self, joint_positions: np.ndarray, position: np.ndarray, orientation: np.ndarray):
        """Update Isaac Sim state"""
        self.isaac_sim_state.joint_positions = joint_positions
        self.isaac_sim_state.position = position
        self.isaac_sim_state.orientation = orientation
        self.isaac_sim_state.timestamp = time.time()

        # Synchronize with other simulators
        self.synchronize_states()

    def synchronize_states(self):
        """Synchronize states between all simulators"""
        # Calculate average state to use as reference
        avg_joint_positions = (self.gazebo_state.joint_positions +
                              self.unity_state.joint_positions +
                              self.isaac_sim_state.joint_positions) / 3.0

        avg_position = (self.gazebo_state.position +
                       self.unity_state.position +
                       self.isaac_sim_state.position) / 3.0

        avg_orientation = self.average_quaternions([
            self.gazebo_state.orientation,
            self.unity_state.orientation,
            self.isaac_sim_state.orientation
        ])

        # Update all simulators to match average (within tolerance)
        self.gazebo_state.joint_positions = avg_joint_positions
        self.gazebo_state.position = avg_position
        self.gazebo_state.orientation = avg_orientation

        self.unity_state.joint_positions = avg_joint_positions
        self.unity_state.position = avg_position
        self.unity_state.orientation = avg_orientation

        self.isaac_sim_state.joint_positions = avg_joint_positions
        self.isaac_sim_state.position = avg_position
        self.isaac_sim_state.orientation = avg_orientation

    def average_quaternions(self, quaternions: List[np.ndarray]) -> np.ndarray:
        """Average a list of quaternions"""
        # Simple averaging (for demonstration - in practice, use proper quaternion averaging)
        avg_quat = np.mean(quaternions, axis=0)
        # Normalize the quaternion
        norm = np.linalg.norm(avg_quat)
        if norm > 0:
            avg_quat = avg_quat / norm
        return avg_quat

    def check_synchronization(self) -> Dict[str, float]:
        """Check synchronization between simulators"""
        sync_errors = {}

        # Joint position errors
        gazebo_unity_joint_error = np.mean(np.abs(
            self.gazebo_state.joint_positions - self.unity_state.joint_positions
        ))
        sync_errors['gazebo_unity_joint'] = gazebo_unity_joint_error

        gazebo_isaac_joint_error = np.mean(np.abs(
            self.gazebo_state.joint_positions - self.isaac_sim_state.joint_positions
        ))
        sync_errors['gazebo_isaac_joint'] = gazebo_isaac_joint_error

        unity_isaac_joint_error = np.mean(np.abs(
            self.unity_state.joint_positions - self.isaac_sim_state.joint_positions
        ))
        sync_errors['unity_isaac_joint'] = unity_isaac_joint_error

        # Position errors
        gazebo_unity_pos_error = np.linalg.norm(
            self.gazebo_state.position - self.unity_state.position
        )
        sync_errors['gazebo_unity_pos'] = gazebo_unity_pos_error

        gazebo_isaac_pos_error = np.linalg.norm(
            self.gazebo_state.position - self.isaac_sim_state.position
        )
        sync_errors['gazebo_isaac_pos'] = gazebo_isaac_pos_error

        unity_isaac_pos_error = np.linalg.norm(
            self.unity_state.position - self.isaac_sim_state.position
        )
        sync_errors['unity_isaac_pos'] = unity_isaac_pos_error

        return sync_errors

    def start_synchronization_monitor(self):
        """Start monitoring synchronization in a separate thread"""
        self.running = True
        self.sync_thread = threading.Thread(target=self._monitor_loop)
        self.sync_thread.start()

    def stop_synchronization_monitor(self):
        """Stop the synchronization monitoring"""
        self.running = False
        if self.sync_thread:
            self.sync_thread.join()

    def _monitor_loop(self):
        """Internal monitoring loop"""
        while self.running:
            sync_errors = self.check_synchronization()

            # Check if any error exceeds tolerance
            max_error = max(sync_errors.values())
            if max_error > self.tolerance:
                print(f"Synchronization error detected: {max_error:.4f}")
                print(f"Errors: {sync_errors}")

            time.sleep(0.1)  # Check every 100ms

class GazeboInterface:
    """Interface to Gazebo simulation"""

    def __init__(self):
        self.connected = False
        self.robot_model = "humanoid_robot"
        self.world_name = "integrated_world"

    def connect(self):
        """Connect to Gazebo simulation"""
        print(f"Connecting to Gazebo simulation with robot: {self.robot_model}")
        self.connected = True
        return True

    def spawn_robot(self, position: np.ndarray, orientation: np.ndarray):
        """Spawn robot in Gazebo world"""
        if self.connected:
            print(f"Spawning robot at position: {position}, orientation: {orientation}")
            return True
        return False

    def set_joint_positions(self, joint_positions: np.ndarray):
        """Set joint positions in Gazebo"""
        if self.connected:
            print(f"Setting {len(joint_positions)} joint positions in Gazebo")
            return True
        return False

    def get_robot_state(self) -> RobotState:
        """Get current robot state from Gazebo"""
        return RobotState(
            joint_positions=np.random.random(28) * 2 - 1,  # Random positions between -1 and 1
            joint_velocities=np.random.random(28) * 0.1,
            joint_torques=np.random.random(28) * 10,
            position=np.random.random(3) * 2 - 1,  # Random position between -1 and 1
            orientation=np.array([0, 0, 0, 1]),  # Identity quaternion
            timestamp=time.time()
        )

class UnityInterface:
    """Interface to Unity digital twin"""

    def __init__(self):
        self.connected = False
        self.digital_twin_port = 10000
        self.ros_tcp_endpoint = "127.0.0.1:10000"

    def connect(self):
        """Connect to Unity digital twin"""
        print(f"Connecting to Unity digital twin at {self.ros_tcp_endpoint}")
        self.connected = True
        return True

    def load_robot_model(self, model_path: str):
        """Load robot model in Unity"""
        if self.connected:
            print(f"Loading robot model from: {model_path}")
            return True
        return False

    def update_robot_transform(self, position: np.ndarray, orientation: np.ndarray):
        """Update robot transform in Unity"""
        if self.connected:
            print(f"Updating robot transform in Unity: pos={position}, rot={orientation}")
            return True
        return False

    def update_joint_positions(self, joint_positions: np.ndarray):
        """Update joint positions in Unity digital twin"""
        if self.connected:
            print(f"Updating {len(joint_positions)} joint positions in Unity")
            return True
        return False

    def get_robot_state(self) -> RobotState:
        """Get current robot state from Unity"""
        return RobotState(
            joint_positions=np.random.random(28) * 2 - 1,  # Random positions between -1 and 1
            joint_velocities=np.random.random(28) * 0.1,
            joint_torques=np.random.random(28) * 10,
            position=np.random.random(3) * 2 - 1,  # Random position between -1 and 1
            orientation=np.array([0, 0, 0, 1]),  # Identity quaternion
            timestamp=time.time()
        )

class IsaacSimInterface:
    """Interface to Isaac Sim"""

    def __init__(self):
        self.connected = False
        self.simulation_context = None
        self.robot_articulation = None

    def connect(self):
        """Connect to Isaac Sim"""
        print("Connecting to Isaac Sim")
        self.connected = True
        return True

    def load_robot(self, usd_path: str):
        """Load robot in Isaac Sim"""
        if self.connected:
            print(f"Loading robot from USD: {usd_path}")
            return True
        return False

    def set_robot_state(self, joint_positions: np.ndarray, position: np.ndarray, orientation: np.ndarray):
        """Set robot state in Isaac Sim"""
        if self.connected:
            print(f"Setting robot state in Isaac Sim: joints={len(joint_positions)}, pos={position}")
            return True
        return False

    def get_robot_state(self) -> RobotState:
        """Get current robot state from Isaac Sim"""
        return RobotState(
            joint_positions=np.random.random(28) * 2 - 1,  # Random positions between -1 and 1
            joint_velocities=np.random.random(28) * 0.1,
            joint_torques=np.random.random(28) * 10,
            position=np.random.random(3) * 2 - 1,  # Random position between -1 and 1
            orientation=np.array([0, 0, 0, 1]),  # Identity quaternion
            timestamp=time.time()
        )

class SimulationManager:
    """Manages the integration of all simulation environments"""

    def __init__(self):
        self.gazebo = GazeboInterface()
        self.unity = UnityInterface()
        self.isaac_sim = IsaacSimInterface()
        self.synchronizer = SimulationSynchronizer()

        # Connect to all simulators
        self.connect_all()

    def connect_all(self):
        """Connect to all simulation environments"""
        print("Connecting to all simulation environments...")

        gazebo_connected = self.gazebo.connect()
        unity_connected = self.unity.connect()
        isaac_sim_connected = self.isaac_sim.connect()

        if all([gazebo_connected, unity_connected, isaac_sim_connected]):
            print("All simulation environments connected successfully!")
            return True
        else:
            print("Failed to connect to all simulation environments")
            return False

    def spawn_robot_in_all(self, position: np.ndarray, orientation: np.ndarray):
        """Spawn robot in all simulation environments"""
        print(f"Spawning robot in all simulators at position: {position}")

        # Spawn in Gazebo
        gazebo_spawned = self.gazebo.spawn_robot(position, orientation)

        # For Unity and Isaac Sim, we'll just update their state to match
        self.unity.update_robot_transform(position, orientation)
        self.isaac_sim.set_robot_state(np.zeros(28), position, orientation)

        # Update synchronizer
        self.synchronizer.update_gazebo_state(np.zeros(28), position, orientation)

        return all([gazebo_spawned])

    def execute_motion_sequence(self, joint_trajectories: List[np.ndarray], duration: float = 5.0):
        """Execute a motion sequence across all simulators"""
        print(f"Executing motion sequence over {duration} seconds...")

        num_steps = int(duration / 0.01)  # 100Hz control
        step_time = duration / num_steps

        for i in range(num_steps):
            # Get current trajectory point
            if i < len(joint_trajectories):
                joint_positions = joint_trajectories[i]
            else:
                joint_positions = joint_trajectories[-1]  # Hold final position

            # Update all simulators
            self.gazebo.set_joint_positions(joint_positions)
            self.unity.update_joint_positions(joint_positions)
            self.isaac_sim.set_robot_state(joint_positions,
                                         self.synchronizer.gazebo_state.position,
                                         self.synchronizer.gazebo_state.orientation)

            # Update synchronizer
            self.synchronizer.update_gazebo_state(joint_positions,
                                                self.synchronizer.gazebo_state.position,
                                                self.synchronizer.gazebo_state.orientation)

            time.sleep(step_time)

            # Print progress
            if i % 100 == 0:  # Print every 100 steps
                print(f"Progress: {i}/{num_steps} steps completed")

    def visualize_synchronization(self):
        """Visualize the synchronization between simulators"""
        # Create a 3D plot showing robot positions in all simulators
        fig = plt.figure(figsize=(12, 5))

        # Plot 1: Robot positions in 3D
        ax1 = fig.add_subplot(121, projection='3d')

        # Get current positions
        gazebo_pos = self.synchronizer.gazebo_state.position
        unity_pos = self.synchronizer.unity_state.position
        isaac_pos = self.synchronizer.isaac_sim_state.position

        # Plot positions
        ax1.scatter([gazebo_pos[0]], [gazebo_pos[1]], [gazebo_pos[2]],
                   c='red', label='Gazebo', s=100)
        ax1.scatter([unity_pos[0]], [unity_pos[1]], [unity_pos[2]],
                   c='blue', label='Unity', s=100)
        ax1.scatter([isaac_pos[0]], [isaac_pos[1]], [isaac_pos[2]],
                   c='green', label='Isaac Sim', s=100)

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Robot Positions in Different Simulators')
        ax1.legend()

        # Plot 2: Joint position comparison
        ax2 = fig.add_subplot(122)

        joint_indices = range(min(10, len(self.synchronizer.gazebo_state.joint_positions)))  # Show first 10 joints
        ax2.plot(joint_indices, self.synchronizer.gazebo_state.joint_positions[:len(joint_indices)],
                'ro-', label='Gazebo', markersize=4)
        ax2.plot(joint_indices, self.synchronizer.unity_state.joint_positions[:len(joint_indices)],
                'bo-', label='Unity', markersize=4)
        ax2.plot(joint_indices, self.synchronizer.isaac_sim_state.joint_positions[:len(joint_indices)],
                'go-', label='Isaac Sim', markersize=4)

        ax2.set_xlabel('Joint Index')
        ax2.set_ylabel('Joint Position (rad)')
        ax2.set_title('Joint Position Comparison (First 10 Joints)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def run_integration_test(self):
        """Run a comprehensive integration test"""
        print("Starting simulation integration test...")

        # 1. Spawn robot in all simulators
        initial_pos = np.array([0.0, 0.0, 0.5])
        initial_rot = np.array([0.0, 0.0, 0.0, 1.0])
        self.spawn_robot_in_all(initial_pos, initial_rot)

        # 2. Generate a simple motion trajectory
        print("Generating motion trajectory...")
        num_points = 500
        joint_trajectories = []

        for i in range(num_points):
            t = i / num_points * 2 * np.pi  # Full cycle

            # Create a simple periodic motion
            joint_positions = np.zeros(28)
            for j in range(28):
                joint_positions[j] = 0.5 * np.sin(t + j * 0.1)  # Different phase for each joint

            joint_trajectories.append(joint_positions)

        # 3. Execute motion sequence
        print("Executing motion sequence...")
        self.execute_motion_sequence(joint_trajectories, duration=10.0)

        # 4. Check synchronization
        print("Checking synchronization...")
        sync_errors = self.synchronizer.check_synchronization()
        print("Synchronization errors:")
        for key, error in sync_errors.items():
            print(f"  {key}: {error:.6f}")

        # 5. Visualize results
        print("Visualizing synchronization...")
        self.visualize_synchronization()

        print("Integration test completed!")

def main():
    """Main function to demonstrate simulation integration"""
    print("Simulation Integration Example - Capstone Project Chapter 12")
    print("=" * 60)

    # Create simulation manager
    sim_manager = SimulationManager()

    # Run integration test
    sim_manager.run_integration_test()

    print("\nSimulation integration demonstration completed!")

if __name__ == "__main__":
    main()