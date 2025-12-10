from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Launch Gazebo with a robot model and sensors."""
    return LaunchDescription([
        # Launch Gazebo with the custom world
        ExecuteProcess(
            cmd=[
                'gazebo',
                '--verbose',
                '-s', 'libgazebo_ros_init.so',
                '-s', 'libgazebo_ros_factory.so',
                PathJoinSubstitution([
                    get_package_share_directory('my_robot_description'),
                    'worlds',
                    'simple_room.world'
                ])
            ],
            output='screen'
        ),

        # Spawn the robot in Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'robot_with_sensors',
                '-file', PathJoinSubstitution([
                    get_package_share_directory('my_robot_description'),
                    'urdf',
                    'robot_with_sensors.urdf'
                ]),
                '-x', '0',
                '-y', '0',
                '-z', '0.1'
            ],
            output='screen'
        ),
    ])