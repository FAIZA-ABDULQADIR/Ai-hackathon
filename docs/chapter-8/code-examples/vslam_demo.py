#!/usr/bin/env python3

"""
Isaac Sim VSLAM Demo

This script demonstrates Visual SLAM in Isaac Sim,
including feature detection, camera pose estimation, and map building.
"""

import omni
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.sensor import Camera
import cv2
from scipy.spatial.transform import Rotation as R
import time


class IsaacSimVSLAMDemo:
    """
    A VSLAM demonstration for Isaac Sim
    """
    def __init__(self):
        # Initialize Isaac Sim world
        self.world = World(stage_units_in_meters=1.0)
        self.camera = None
        self.previous_frame = None
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.map_points = []  # 3D points in the map
        self.keyframes = []   # Keyframe poses
        self.frame_count = 0

        # Camera intrinsic parameters (simplified)
        self.camera_matrix = np.array([
            [320, 0, 320],
            [0, 320, 240],
            [0, 0, 1]
        ])

        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=500)

    def setup_vslam_environment(self):
        """Setup environment for VSLAM testing"""
        # Get assets root path
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("Could not find Isaac Sim assets. Please check your installation.")
            return False

        # Add a complex environment
        add_reference_to_stage(
            usd_path=f"{assets_root_path}/Isaac/Environments/Simple_Room.usd",
            prim_path="/World/Room"
        )

        # Add camera sensor
        self.camera = self.world.scene.add(
            Camera(
                prim_path="/World/VSLAMCamera",
                frequency=30,
                resolution=(640, 480)
            )
        )

        # Set initial camera pose
        self.camera.set_world_pose(
            position=np.array([0.0, 0.0, 1.5]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0])  # No rotation initially
        )

        return True

    def extract_features(self, image):
        """Extract features from image using ORB"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect and compute features
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if keypoints is not None:
            # Convert keypoints to numpy array
            points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        else:
            points = np.array([], dtype=np.float32)

        return points, descriptors

    def estimate_camera_motion(self, current_frame, previous_frame):
        """Estimate camera motion between frames"""
        if previous_frame is None:
            return np.eye(4)  # No motion for first frame

        # Extract features from both frames
        curr_points, curr_desc = self.extract_features(current_frame)
        prev_points, prev_desc = self.extract_features(previous_frame)

        if len(curr_points) < 10 or len(prev_points) < 10:
            return np.eye(4)  # Not enough features

        # Match features using brute force matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Convert descriptors to the right format
        if curr_desc is not None and prev_desc is not None and len(curr_desc) > 0 and len(prev_desc) > 0:
            matches = bf.match(prev_desc, curr_desc)

            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Use only good matches
            good_matches = matches[:min(50, len(matches))]

            if len(good_matches) >= 10:  # Need at least 10 points for pose estimation
                # Get matched points
                src_points = np.float32([prev_points[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
                dst_points = np.float32([curr_points[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)

                # Estimate essential matrix (assuming calibrated camera)
                essential_matrix, mask = cv2.findEssentialMat(
                    src_points, dst_points, self.camera_matrix,
                    method=cv2.RANSAC, threshold=1.0
                )

                if essential_matrix is not None:
                    # Recover pose
                    _, rotation, translation, mask = cv2.recoverPose(
                        essential_matrix, src_points, dst_points, self.camera_matrix
                    )

                    # Create transformation matrix
                    transform = np.eye(4)
                    transform[:3, :3] = rotation
                    transform[:3, 3] = translation.flatten()

                    return transform

        return np.eye(4)  # Default: no motion

    def update_map(self, current_frame):
        """Update the map with new observations"""
        # Extract features
        points, _ = self.extract_features(current_frame)

        # Store the current camera pose as a keyframe
        self.keyframes.append(self.current_pose.copy())

        # Store some points in the map (in camera coordinate frame)
        if len(points) > 0:
            # Convert image points to 3D using depth (simplified)
            # In real implementation, this would use triangulation with previous views
            for pt in points[:20]:  # Only store first 20 points to avoid too many
                # This is a simplified approach - real implementation would triangulate
                # Create a 3D point in camera frame (depth is assumed for demo)
                point_3d = np.array([pt[0]/320.0 - 1.0, pt[1]/240.0 - 1.0, 1.0])

                # Transform to world frame
                world_point = self.current_pose @ np.append(point_3d, 1)
                self.map_points.append(world_point[:3])

    def run_vslam_pipeline(self, num_frames=600):
        """Run the complete VSLAM pipeline"""
        print("Starting Isaac Sim VSLAM Pipeline...")
        print("This demo simulates VSLAM by processing camera images and estimating motion.")

        # Initialize the world
        self.world.reset()

        for frame in range(num_frames):
            # Step the simulation
            self.world.step(render=True)

            # Process at 10Hz (every 6 steps assuming 60Hz simulation)
            if frame % 6 == 0:
                try:
                    # Get camera image
                    current_frame = self.camera.get_rgb()

                    # Estimate camera motion
                    motion_transform = self.estimate_camera_motion(current_frame, self.previous_frame)

                    # Update current pose
                    self.current_pose = self.current_pose @ motion_transform

                    # Update the map
                    self.update_map(current_frame)

                    # Print pose information
                    position = self.current_pose[:3, 3]
                    print(f"Frame {frame}: Position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")

                    # Store current frame for next iteration
                    self.previous_frame = current_frame.copy()

                except Exception as e:
                    print(f"Error processing frame {frame}: {e}")
                    continue

        print(f"VSLAM pipeline completed.")
        print(f"Final position: {self.current_pose[:3, 3]}")
        print(f"Map contains {len(self.map_points)} points and {len(self.keyframes)} keyframes.")

    def cleanup(self):
        """Clean up resources"""
        self.world.clear()


def main():
    """Main function to run the VSLAM demo"""
    print("Initializing Isaac Sim VSLAM Demo...")

    # Create VSLAM demo instance
    vslam_demo = IsaacSimVSLAMDemo()

    # Setup environment
    if not vslam_demo.setup_vslam_environment():
        print("Failed to setup VSLAM environment. Exiting.")
        return

    # Run the VSLAM pipeline
    try:
        vslam_demo.run_vslam_pipeline(num_frames=600)  # Run for 10 seconds at 60Hz
    except KeyboardInterrupt:
        print("VSLAM demo interrupted by user.")
    finally:
        # Clean up
        vslam_demo.cleanup()
        print("VSLAM demo cleanup completed.")


if __name__ == "__main__":
    main()