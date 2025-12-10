#!/usr/bin/env python3

"""
Isaac Sim Semantic Segmentation Example

This script demonstrates semantic segmentation in Isaac Sim,
including segmentation visualization and class mapping.
"""

import omni
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.sensor import Camera
import cv2
import random


class IsaacSimSemanticSegmentation:
    """
    A semantic segmentation pipeline for Isaac Sim
    """
    def __init__(self):
        # Initialize Isaac Sim world
        self.world = World(stage_units_in_meters=1.0)
        self.camera = None
        self.segmentation_map = None

        # Define color map for different classes
        self.color_map = np.array([
            [128, 64, 128],    # Road
            [244, 35, 232],    # Person
            [70, 70, 70],      # Car
            [102, 102, 156],   # Wall
            [190, 153, 153],   # Fence
            [153, 153, 153],   # Pole
            [250, 170, 30],    # Traffic light
            [220, 220, 0],     # Traffic sign
            [107, 142, 35],    # Vegetation
            [152, 251, 152],   # Terrain
            [70, 130, 180],    # Sky
            [220, 20, 60],     # Pedestrian
            [255, 0, 0],       # Rider
            [0, 0, 142],       # Truck
            [0, 0, 70],        # Bus
            [0, 60, 100],      # Train
            [0, 80, 100],      # Motorcycle
            [0, 0, 230],       # Bicycle
        ], dtype=np.uint8)

        # Class names corresponding to the color map
        self.class_names = [
            "Road", "Person", "Car", "Wall", "Fence", "Pole",
            "Traffic Light", "Traffic Sign", "Vegetation", "Terrain",
            "Sky", "Pedestrian", "Rider", "Truck", "Bus", "Train",
            "Motorcycle", "Bicycle"
        ]

    def setup_scene(self):
        """Set up the simulation environment for segmentation"""
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
                prim_path="/World/SegmentationCamera",
                frequency=30,
                resolution=(640, 480)
            )
        )

        # Set camera position
        self.camera.set_world_pose(position=np.array([2.0, 0.0, 1.5]))

        return True

    def generate_mock_segmentation(self, height, width):
        """Generate mock segmentation map for demonstration"""
        # Create a mock segmentation map with different regions
        segmentation = np.zeros((height, width), dtype=np.uint8)

        # Create some mock regions (in a real implementation, this would come from Isaac Sim)
        # Create a "road" region at the bottom
        road_height = height // 3
        segmentation[height - road_height:, :] = 0  # Road class

        # Create a "wall" region in the middle
        wall_start = height // 3
        wall_end = 2 * height // 3
        segmentation[wall_start:wall_end, :] = 3  # Wall class

        # Create some "object" regions
        for _ in range(5):
            x = random.randint(0, width - 50)
            y = random.randint(0, height - 50)
            size = random.randint(20, 50)
            class_id = random.randint(1, len(self.class_names) - 1)  # Avoid road class

            segmentation[y:y+size, x:x+size] = class_id

        return segmentation

    def apply_segmentation_color_map(self, segmentation):
        """Apply color map to segmentation mask"""
        # Create RGB image from segmentation
        height, width = segmentation.shape
        colored_segmentation = np.zeros((height, width, 3), dtype=np.uint8)

        for class_id in np.unique(segmentation):
            mask = segmentation == class_id
            if class_id < len(self.color_map):
                colored_segmentation[mask] = self.color_map[class_id]
            else:
                # Use default color for unknown classes
                colored_segmentation[mask] = [255, 255, 255]

        return colored_segmentation

    def blend_segmentation(self, rgb_image, segmentation_overlay, alpha=0.5):
        """Blend RGB image with segmentation overlay"""
        return cv2.addWeighted(rgb_image, 1 - alpha, segmentation_overlay, alpha, 0)

    def run_segmentation_pipeline(self, num_frames=240):
        """Run the complete semantic segmentation pipeline"""
        print("Starting Isaac Sim Semantic Segmentation Pipeline...")

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

                    # Generate mock segmentation (in real implementation, this would come from Isaac Sim)
                    height, width = rgb_image.shape[:2]
                    segmentation = self.generate_mock_segmentation(height, width)

                    # Apply color map to segmentation
                    colored_segmentation = self.apply_segmentation_color_map(segmentation)

                    # Blend original image with segmentation
                    blended_image = self.blend_segmentation(rgb_image, colored_segmentation, alpha=0.4)

                    # Create a side-by-side comparison
                    comparison = np.hstack((rgb_image, blended_image))

                    # Display result
                    cv2.imshow("Isaac Sim Semantic Segmentation", comparison)

                    # Print segmentation info
                    unique_classes = np.unique(segmentation)
                    class_counts = [np.sum(segmentation == cls) for cls in unique_classes]

                    print(f"Frame {frame}: Segmentation classes detected: {len(unique_classes)}")
                    for cls, count in zip(unique_classes, class_counts):
                        if cls < len(self.class_names):
                            print(f"  {self.class_names[cls]}: {count} pixels")
                        else:
                            print(f"  Unknown class {cls}: {count} pixels")

                    # Break on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                except Exception as e:
                    print(f"Error processing frame {frame}: {e}")
                    continue

        print("Semantic segmentation pipeline completed.")

    def cleanup(self):
        """Clean up resources"""
        cv2.destroyAllWindows()
        self.world.clear()


def main():
    """Main function to run the semantic segmentation pipeline"""
    print("Initializing Isaac Sim Semantic Segmentation Pipeline...")

    # Create segmentation instance
    segmenter = IsaacSimSemanticSegmentation()

    # Setup scene
    if not segmenter.setup_scene():
        print("Failed to setup scene. Exiting.")
        return

    # Run the segmentation pipeline
    try:
        segmenter.run_segmentation_pipeline(num_frames=240)  # Run for 4 seconds at 60Hz
    except KeyboardInterrupt:
        print("Pipeline interrupted by user.")
    finally:
        # Clean up
        segmenter.cleanup()
        print("Pipeline cleanup completed.")


if __name__ == "__main__":
    main()