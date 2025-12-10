#!/usr/bin/env python3

"""
Isaac Sim Perception Pipeline Example

This script demonstrates a complete perception pipeline in Isaac Sim,
including robot setup, sensor configuration, data collection, and
basic computer vision processing.
"""

import omni
import carb
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.sensor import Camera, LidarRtx
from omni.isaac.core.utils.prims import get_prim_at_path
import cv2
from scipy.spatial.transform import Rotation as R


class IsaacSimPerceptionPipeline:
    """
    A complete perception pipeline for Isaac Sim
    """
    def __init__(self):
        # Initialize Isaac Sim world
        self.world = World(stage_units_in_meters=1.0)
        self.camera = None
        self.lidar = None
        self.robot = None

    def setup_environment(self):
        """Set up the simulation environment with robot and sensors"""
        # Get assets root path
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets. Please check your installation.")
            return False

        # Add a simple room environment
        add_reference_to_stage(
            usd_path=f"{assets_root_path}/Isaac/Environments/Simple_Room.usd",
            prim_path="/World/Room"
        )

        # Add a Carter robot (common mobile robot platform)
        robot_path = f"{assets_root_path}/Isaac/Robots/Carter/carter_navigate.usd"
        add_reference_to_stage(
            usd_path=robot_path,
            prim_path="/World/Carter"
        )

        # Position the robot
        self.world.scene.add_default_ground_plane()

        return True

    def setup_sensors(self):
        """Configure perception sensors on the robot"""
        # Add RGB camera
        self.camera = self.world.scene.add(
            Camera(
                prim_path="/World/Carter/Carter_Camera/Camera",
                frequency=30,
                resolution=(640, 480)
            )
        )

        # Add LiDAR sensor
        self.lidar = self.world.scene.add(
            LidarRtx(
                prim_path="/World/Carter/Lidar",
                translation=np.array([0.0, 0.0, 0.3]),  # Position 30cm above ground
                config="Carter_Lidar",
                rotation=(0, 0, 0)
            )
        )

        # Set camera parameters
        self.camera.set_focal_length(24.0)
        self.camera.set_horizontal_aperture(20.955)
        self.camera.set_vertical_aperture(15.29)

    def process_camera_data(self, rgb_image, depth_image):
        """
        Process camera data for basic computer vision tasks

        Args:
            rgb_image: RGB image from the camera
            depth_image: Depth image from the camera

        Returns:
            Processed data including object detection results
        """
        # Convert to OpenCV format
        rgb_cv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Basic edge detection
        gray = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area
        valid_contours = [c for c in contours if cv2.contourArea(c) > 100]

        # Draw contours on the image
        result_image = rgb_cv.copy()
        cv2.drawContours(result_image, valid_contours, -1, (0, 255, 0), 2)

        # Calculate object centers
        object_centers = []
        for contour in valid_contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                object_centers.append((cx, cy))

        return {
            'object_centers': object_centers,
            'contour_count': len(valid_contours),
            'processed_image': result_image
        }

    def process_lidar_data(self, lidar_data):
        """
        Process LiDAR data for basic obstacle detection

        Args:
            lidar_data: Raw LiDAR measurements

        Returns:
            Processed data including obstacle information
        """
        # Convert to numpy array if needed
        if not isinstance(lidar_data, np.ndarray):
            lidar_data = np.array(lidar_data)

        # Filter out invalid measurements (NaN or infinity)
        valid_points = lidar_data[~np.isnan(lidar_data).any(axis=1)]
        valid_points = valid_points[~np.isinf(valid_points).any(axis=1)]

        # Calculate distances from origin
        distances = np.linalg.norm(valid_points, axis=1)

        # Detect obstacles within 2 meters
        obstacle_indices = np.where(distances < 2.0)[0]
        obstacle_points = valid_points[obstacle_indices]

        # Calculate basic statistics
        if len(obstacle_points) > 0:
            obstacle_centers = np.mean(obstacle_points, axis=0)
            obstacle_count = len(obstacle_points)
        else:
            obstacle_centers = np.array([0.0, 0.0, 0.0])
            obstacle_count = 0

        return {
            'obstacle_count': obstacle_count,
            'obstacle_centers': obstacle_centers,
            'valid_points_count': len(valid_points)
        }

    def run_perception_pipeline(self, simulation_steps=1000):
        """
        Run the complete perception pipeline

        Args:
            simulation_steps: Number of simulation steps to run
        """
        print("Starting perception pipeline...")

        # Initialize the world
        self.world.reset()

        for step in range(simulation_steps):
            # Step the simulation
            self.world.step(render=True)

            # Process at 1Hz (every 60 steps assuming 60Hz simulation)
            if step % 60 == 0:
                # Get camera data
                try:
                    rgb_image = self.camera.get_rgb()
                    depth_image = self.camera.get_depth()

                    # Process camera data
                    camera_results = self.process_camera_data(rgb_image, depth_image)

                    # Get LiDAR data
                    lidar_data = self.lidar.get_linear_depth_data()

                    # Process LiDAR data
                    lidar_results = self.process_lidar_data(lidar_data)

                    # Print results
                    print(f"Step {step}:")
                    print(f"  Camera - Objects detected: {camera_results['contour_count']}")
                    print(f"  LiDAR - Obstacles detected: {lidar_results['obstacle_count']}")
                    print(f"  LiDAR - Valid points: {lidar_results['valid_points_count']}")

                    # Visualize results (optional)
                    if step % 180 == 0:  # Every 3 seconds
                        cv2.imshow("Processed Camera View", camera_results['processed_image'])
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                except Exception as e:
                    print(f"Error processing sensor data: {e}")
                    continue

        print("Perception pipeline completed.")

    def cleanup(self):
        """Clean up resources"""
        cv2.destroyAllWindows()
        self.world.clear()


def main():
    """Main function to run the perception pipeline"""
    print("Initializing Isaac Sim Perception Pipeline...")

    # Create perception pipeline instance
    pipeline = IsaacSimPerceptionPipeline()

    # Setup environment
    if not pipeline.setup_environment():
        print("Failed to setup environment. Exiting.")
        return

    # Setup sensors
    pipeline.setup_sensors()

    # Run the perception pipeline
    try:
        pipeline.run_perception_pipeline(simulation_steps=1800)  # Run for 30 seconds at 60Hz
    except KeyboardInterrupt:
        print("Pipeline interrupted by user.")
    finally:
        # Clean up
        pipeline.cleanup()
        print("Pipeline cleanup completed.")


if __name__ == "__main__":
    main()