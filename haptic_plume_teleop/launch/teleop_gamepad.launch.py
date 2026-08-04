#!/usr/bin/env python3
"""
Bring up gamepad teleop: the joy driver and the axis mapper.

Starts joy_node to read the device and teleop_mapper_node to turn its axes into
world-frame velocity commands on /hp/cmd_vel. Needs drone.launch.py running to
have anything to fly.

The keyboard fallback is deliberately NOT a launch file: teleop_twist_keyboard
reads raw stdin, and ros2 launch gives child processes no TTY, so it dies with
"termios.error: Inappropriate ioctl for device". Run it by hand instead:

    ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args \\
      -r /cmd_vel:=/hp/cmd_vel -p stamped:=true -p frame_id:=map -p speed:=1.0

:author: premmm
:date: August 4, 2026
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    Generate the launch description.

    Returns
    -------
    LaunchDescription
        The declared arguments, the joy driver, and the axis mapper.

    """
    # 1. Constants for paths to different files and folders
    package_name_teleop = 'haptic_plume_teleop'

    default_params_file = PathJoinSubstitution(
        [FindPackageShare(package_name_teleop), 'config', 'axis_map_gamepad.yaml']
    )

    # 2. Launch configuration variables (alphabetical)
    device_id = LaunchConfiguration('device_id')
    params_file = LaunchConfiguration('params_file')

    # 3. Declare the launch arguments
    declare_device_id_cmd = DeclareLaunchArgument(
        name='device_id',
        default_value='0',
        description='Joystick device index, i.e. the N in /dev/input/jsN'
    )

    declare_params_file_cmd = DeclareLaunchArgument(
        name='params_file',
        default_value=default_params_file,
        description='Full path to the axis-map parameter file'
    )

    # 4. Nodes and includes
    # Read the raw device. deadzone is 0.0 on purpose: the axis map owns the
    # deadband, and applying it twice would rescale the stick range twice.
    # autorepeat_rate keeps /joy alive while the sticks are still, so the
    # mapper's joy_timeout only fires if this driver actually dies.
    start_joy_cmd = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[{
            'device_id': device_id,
            'deadzone': 0.0,
            'autorepeat_rate': 20.0,
        }]
    )

    # Turn device axes into world-frame velocity commands, deadman-gated
    start_teleop_mapper_cmd = Node(
        package=package_name_teleop,
        executable='teleop_mapper_node',
        name='teleop_mapper',
        output='screen',
        parameters=[params_file]
    )

    # 5. Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_device_id_cmd)
    ld.add_action(declare_params_file_cmd)

    # Add the actions
    ld.add_action(start_joy_cmd)
    ld.add_action(start_teleop_mapper_cmd)

    return ld
