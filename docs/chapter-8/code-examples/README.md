# Isaac Sim Navigation Examples

This directory contains examples for implementing navigation and path planning in Isaac Sim, including VSLAM, Nav2 integration, and path planning algorithms.

## Examples Included

1. **vslam_demo.py** - Visual SLAM implementation with camera pose estimation
2. **nav2_controller.py** - Nav2 integration with waypoint navigation
3. **path_planning.py** - Path planning algorithms (A* implementation)
4. **validate-examples.sh** - Validation script for testing the examples

## Requirements
- NVIDIA Isaac Sim
- ROS 2 (Humble Hawksbill recommended)
- Python 3.8+
- OpenCV, NumPy, SciPy, Matplotlib

## Usage
```bash
# Run VSLAM demo
python vslam_demo.py

# Run Nav2 controller demo
python nav2_controller.py

# Run path planning demo
python path_planning.py
```