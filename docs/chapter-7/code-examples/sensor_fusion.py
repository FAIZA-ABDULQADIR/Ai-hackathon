#!/usr/bin/env python3

"""
Isaac Sim Sensor Fusion Example

This script demonstrates sensor fusion in Isaac Sim,
combining camera and LiDAR data for enhanced perception.
"""

import omni
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.sensor import Camera, LidarRtx
import cv2
from scipy.spatial.transform import Rotation as R


class IsaacSimSensorFusion:
    """
    A sensor fusion pipeline combining camera and LiDAR data in Isaac Sim
    """
    def __init__(self):
        # Initialize Isaac Sim world
        self.world = World(stage_units_in_meters=1.0)
        self.camera = None
        self.lidar = None

        # Define camera intrinsics (simplified)
        self.camera_intrinsics = np.array([
            [320, 0, 320],    # fx, 0, cx
            [0, 320, 240],    # 0, fy, cy
            [0, 0, 1]         # 0, 0, 1
        ])

        # Define transformation from LiDAR to camera frame
        # This would be calibrated in a real system
        self.lidar_to_camera = np.eye(4)  # Identity for simplicity
        self.lidar_to_camera[0, 3] = 0.1  # 10cm offset in x
        self.lidar_to_camera[1, 3] = 0.0  # No offset in y
        self.lidar_to_camera[2, 3] = 0.2  # 20cm offset in z

    def setup_sensors(self):
        """Set up multiple sensors for fusion"""
        # Get assets root path
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("Could not find Isaac Sim assets. Please check your installation.")
            return False

        # Add a simple room environment
        add_reference_to_stage(
            usd_path=f"{assets_root_path}/Isaac/Environments/Simple_Room.usd",
            prim_path="/World/Room"
        )

        # Add camera sensor
        self.camera = self.world.scene.add(
            Camera(
                prim_path="/World/Camera",
                frequency=30,
                resolution=(640, 480)
            )
        )

        # Set camera position
        self.camera.set_world_pose(position=np.array([2.0, 0.0, 1.5]))

        # Add LiDAR sensor
        self.lidar = self.world.scene.add(
            LidarRtx(
                prim_path="/World/Lidar",
                translation=np.array([2.1, 0.0, 1.7]),  # Slightly offset from camera
                config="Carter_Lidar",
                rotation=(0, 0, 0)
            )
        )

        return True

    def project_lidar_to_camera(self, lidar_points):
        """
        Project LiDAR points to camera image coordinates

        Args:
            lidar_points: 3D points from LiDAR in LiDAR coordinate frame

        Returns:
            Projected 2D points in camera image coordinates
        """
        if lidar_points.size == 0:
            return np.array([]), np.array([])

        # Convert to homogeneous coordinates
        points_homo = np.hstack([lidar_points, np.ones((lidar_points.shape[0], 1))])

        # Transform from LiDAR frame to camera frame
        points_cam = points_homo @ self.lidar_to_camera.T
        points_cam = points_cam[:, :3]  # Remove homogeneous coordinate

        # Filter points that are in front of the camera
        valid_indices = points_cam[:, 2] > 0.1  # More than 0.1m in front
        points_cam = points_cam[valid_indices]

        if points_cam.size == 0:
            return np.array([]), np.array([])

        # Project to image coordinates
        points_img = points_cam @ self.camera_intrinsics.T
        points_img = points_img[:, :2] / points_img[:, 2:3]  # Perspective division

        # Filter points within image bounds
        valid_x = (points_img[:, 0] >= 0) & (points_img[:, 0] < 640)
        valid_y = (points_img[:, 1] >= 0) & (points_img[:, 1] < 480)
        valid_indices = valid_x & valid_y

        points_img = points_img[valid_indices]
        points_cam = points_cam[valid_indices]

        return points_img, points_cam

    def create_fused_visualization(self, rgb_image, projected_points, camera_frame_points):
        """
        Create a visualization combining camera and LiDAR data
        """
        # Make a copy of the RGB image for visualization
        fused_image = rgb_image.copy()

        # Draw LiDAR points on the image
        for i, (u, v) in enumerate(projected_points):
            if 0 <= u < rgb_image.shape[1] and 0 <= v < rgb_image.shape[0]:
                # Color based on depth (distance from camera)
                depth = camera_frame_points[i, 2]
                color = self.depth_to_color(depth)

                # Draw point with size based on depth (closer = larger)
                radius = max(1, min(5, int(5 / max(0.1, depth))))
                cv2.circle(fused_image, (int(u), int(v)), radius, color, -1)

        return fused_image

    def depth_to_color(self, depth):
        """
        Convert depth value to RGB color (blue for close, red for far)
        """
        min_depth, max_depth = 0.1, 20.0
        normalized_depth = (depth - min_depth) / (max_depth - min_depth)
        normalized_depth = max(0, min(1, normalized_depth))  # Clamp to [0, 1]

        # Create color gradient: blue (close) -> green -> red (far)
        if normalized_depth < 0.5:
            # Blue to green
            blue = int(255 * (1 - 2 * normalized_depth))
            green = int(255 * 2 * normalized_depth)
            red = 0
        else:
            # Green to red
            green = int(255 * (1 - 2 * (normalized_depth - 0.5)))
            red = int(255 * 2 * (normalized_depth - 0.5))
            blue = 0

        return (blue, green, red)

    def generate_mock_lidar_data(self, num_points=1000):
        """
        Generate mock LiDAR data for demonstration
        In a real implementation, this would come from the LiDAR sensor
        """
        # Create mock LiDAR points around the sensor
        # This simulates a simple scene with some objects
        points = []

        # Add some random points to simulate environment
        for _ in range(num_points):
            # Random spherical coordinates
            r = np.random.uniform(1.0, 10.0)  # Distance 1-10m
            theta = np.random.uniform(-0.5, 0.5)  # Azimuth
            phi = np.random.uniform(-0.2, 0.2)   # Elevation

            # Convert to Cartesian
            x = r * np.cos(phi) * np.cos(theta)
            y = r * np.cos(phi) * np.sin(theta)
            z = r * np.sin(phi)

            points.append([x, y, z])

        # Add some clustered points to simulate objects
        for _ in range(100):
            # Object cluster at [5, 0, 1]
            x = 5 + np.random.normal(0, 0.5)
            y = np.random.normal(0, 0.5)
            z = 1 + np.random.normal(0, 0.5)
            points.append([x, y, z])

        return np.array(points)

    def run_fusion_pipeline(self, num_frames=180):
        """Run the complete sensor fusion pipeline"""
        print("Starting Isaac Sim Sensor Fusion Pipeline...")

        # Initialize the world
        self.world.reset()

        for frame in range(num_frames):
            # Step the simulation
            self.world.step(render=True)

            # Process at 3Hz (every 20 steps assuming 60Hz simulation)
            if frame % 20 == 0:
                try:
                    # Get camera image
                    rgb_image = self.camera.get_rgb()

                    # Get LiDAR data (mock for this example)
                    lidar_points = self.generate_mock_lidar_data()

                    # Project LiDAR points to camera image coordinates
                    projected_points, camera_frame_points = self.project_lidar_to_camera(lidar_points)

                    # Create fused visualization
                    if len(projected_points) > 0:
                        fused_image = self.create_fused_visualization(
                            rgb_image, projected_points, camera_frame_points
                        )
                    else:
                        fused_image = rgb_image.copy()

                    # Create a side-by-side comparison
                    comparison = np.hstack((rgb_image, fused_image))

                    # Add labels
                    cv2.putText(
                        comparison,
                        "Camera Only",
                        (50, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )
                    cv2.putText(
                        comparison,
                        "Camera + LiDAR Fusion",
                        (640 + 50, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )

                    # Display result
                    cv2.imshow("Isaac Sim Sensor Fusion", comparison)

                    # Print fusion info
                    print(f"Frame {frame}: LiDAR points projected: {len(projected_points)}")

                    # Break on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                except Exception as e:
                    print(f"Error processing frame {frame}: {e}")
                    continue

        print("Sensor fusion pipeline completed.")

    def cleanup(self):
        """Clean up resources"""
        cv2.destroyAllWindows()
        self.world.clear()


def main():
    """Main function to run the sensor fusion pipeline"""
    print("Initializing Isaac Sim Sensor Fusion Pipeline...")

    # Create fusion instance
    fusion_system = IsaacSimSensorFusion()

    # Setup sensors
    if not fusion_system.setup_sensors():
        print("Failed to setup sensors. Exiting.")
        return

    # Run the fusion pipeline
    try:
        fusion_system.run_fusion_pipeline(num_frames=180)  # Run for 3 seconds at 60Hz
    except KeyboardInterrupt:
        print("Pipeline interrupted by user.")
    finally:
        # Clean up
        fusion_system.cleanup()
        print("Pipeline cleanup completed.")


if __name__ == "__main__":
    main()