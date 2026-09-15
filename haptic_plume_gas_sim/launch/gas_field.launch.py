#!/usr/bin/env python3
"""
Bring up the ground-truth gas field server.

Starts gas_field_server, which samples the analytic plume at the drone's
position and publishes the lagged, noisy measurement the estimator consumes.
It needs /hp/odom, so run drone.launch.py alongside it (or B7's sim_teleop
launch, which includes both).

Every node parameter comes from params_file alone. Do not add launch arguments
that re-pass single parameters: a value passed after the file silently wins
over it, which is exactly how leaks_on_pipes: true in gas_params.yaml was once
ignored. To change a parameter for one run, point params_file at a copy.

There is deliberately no use_sim_time argument: this project runs on wall
clock (decision D3) with no /clock bridge.

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
        The declared arguments and the gas field server node.

    """
    # 1. Constants for paths to different files and folders
    package_name_gas_sim = 'haptic_plume_gas_sim'

    default_params_file = PathJoinSubstitution(
        [FindPackageShare(package_name_gas_sim), 'config', 'gas_params.yaml']
    )

    # 2. Launch configuration variables
    params_file = LaunchConfiguration('params_file')

    # 3. Declare the launch arguments
    declare_params_file_cmd = DeclareLaunchArgument(
        name='params_file',
        default_value=default_params_file,
        description='Full path to the gas_field_server parameter file'
    )

    # 4. Nodes and includes
    # The truth side of the measurement chain: field, sensor lag, noise, markers.
    # obstacles_file is left empty in the YAML, so the node resolves it to
    # haptic_plume_description's obstacles.yaml -- the SAME file the kinematics
    # clamp reads, so a leak can only sit on a pipe the drone collides with (D1).
    start_gas_field_server_cmd = Node(
        package=package_name_gas_sim,
        executable='gas_field_server',
        name='gas_field_server',
        output='screen',
        parameters=[params_file]
    )

    # 5. Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_params_file_cmd)

    # Add the actions
    ld.add_action(start_gas_field_server_cmd)

    return ld
