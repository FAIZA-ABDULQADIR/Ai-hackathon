#!/usr/bin/env python3

"""
Isaac Sim Object Detection Example

This script demonstrates a complete object detection pipeline in Isaac Sim,
including scene setup, camera configuration, object detection model integration,
and result visualization.
"""

import omni
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.sensor import Camera
import cv2
import torch
import torchvision.transforms as T
import random
import string


class IsaacSimObjectDetection:
    """
    A complete object detection pipeline for Isaac Sim
    """
    def __init__(self):
        # Initialize Isaac Sim world
        self.world = World(stage_units_in_meters=1.0)
        self.camera = None
        self.detection_model = None
        self.class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus",
            "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
            "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        ]

    def setup_scene(self):
        """Set up the simulation environment with objects to detect"""
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

        # Add objects to detect
        # In a real implementation, we would add specific objects with known prim paths
        # For this example, we'll simulate detection results

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

        return True

    def load_detection_model(self):
        """Load a pre-trained object detection model (using a mock for this example)"""
        # In a real implementation, this would load an actual model
        # For this example, we'll create a mock detection function
        print("Mock detection model loaded")
        return True

    def mock_detect_objects(self, image):
        """Mock object detection function that simulates detection results"""
        # Generate random detections to simulate model output
        num_detections = random.randint(1, 5)  # 1-5 random detections
        detections = []

        for _ in range(num_detections):
            # Generate random bounding box
            x1 = random.randint(0, image.shape[1] - 100)
            y1 = random.randint(0, image.shape[0] - 100)
            width = random.randint(50, 150)
            height = random.randint(50, 150)

            x2 = min(x1 + width, image.shape[1])
            y2 = min(y1 + height, image.shape[0])

            # Random class and confidence
            class_id = random.randint(0, len(self.class_names) - 1)
            confidence = random.uniform(0.5, 0.99)

            detections.append([x1, y1, x2, y2, confidence, class_id])

        return np.array(detections)

    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on the image"""
        result_image = image.copy()

        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label
            label = f"{self.class_names[int(cls)]}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            # Draw label background
            cv2.rectangle(
                result_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1
            )

            # Draw label text
            cv2.putText(
                result_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )

        return result_image

    def run_detection_pipeline(self, num_frames=300):
        """Run the complete object detection pipeline"""
        print("Starting Isaac Sim Object Detection Pipeline...")

        # Initialize the world
        self.world.reset()

        # Load detection model (mock)
        self.load_detection_model()

        for frame in range(num_frames):
            # Step the simulation
            self.world.step(render=True)

            # Process at 3Hz (every 20 steps assuming 60Hz simulation)
            if frame % 20 == 0:
                try:
                    # Get camera image
                    rgb_image = self.camera.get_rgb()

                    # Run object detection (mock)
                    detections = self.mock_detect_objects(rgb_image)

                    # Draw detections on image
                    result_image = self.draw_detections(rgb_image, detections)

                    # Display result
                    cv2.imshow("Isaac Sim Object Detection", result_image)

                    # Print detection info
                    print(f"Frame {frame}: Detected {len(detections)} objects")
                    for i, detection in enumerate(detections):
                        x1, y1, x2, y2, conf, cls = detection
                        print(f"  Object {i+1}: {self.class_names[int(cls)]} ({conf:.2f})")

                    # Break on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                except Exception as e:
                    print(f"Error processing frame {frame}: {e}")
                    continue

        print("Object detection pipeline completed.")

    def cleanup(self):
        """Clean up resources"""
        cv2.destroyAllWindows()
        self.world.clear()


def main():
    """Main function to run the object detection pipeline"""
    print("Initializing Isaac Sim Object Detection Pipeline...")

    # Create object detection instance
    detector = IsaacSimObjectDetection()

    # Setup scene
    if not detector.setup_scene():
        print("Failed to setup scene. Exiting.")
        return

    # Run the detection pipeline
    try:
        detector.run_detection_pipeline(num_frames=300)  # Run for 5 seconds at 60Hz
    except KeyboardInterrupt:
        print("Pipeline interrupted by user.")
    finally:
        # Clean up
        detector.cleanup()
        print("Pipeline cleanup completed.")


if __name__ == "__main__":
    main()