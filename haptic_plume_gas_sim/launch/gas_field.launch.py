#!/usr/bin/env python3
"""
Bring up the ground-truth gas field server.

Starts gas_field_server, which samples the analytic plume at the drone's
position and publishes the lagged, noisy measurement the estimator consumes.
It needs /hp/odom, so run drone.launch.py alongside it (or B7's sim_teleop
launch, which includes both).

There is deliberately no use_sim_time argument: this project runs on wall
clock (decision D3) with no /clock bridge.

:author: premmm
:date: August 4, 2026
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
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
    package_name_description = 'haptic_plume_description'

    default_params_file = PathJoinSubstitution(
        [FindPackageShare(package_name_gas_sim), 'config', 'gas_params.yaml']
    )
    # The SAME file the kinematics clamp reads, so a leak can only ever be
    # placed on a pipe the drone actually collides with (decision D1).
    default_obstacles_file = PathJoinSubstitution(
        [FindPackageShare(package_name_description), 'config', 'obstacles.yaml']
    )

    # 2. Launch configuration variables (alphabetical)
    leaks_on_pipes = LaunchConfiguration('leaks_on_pipes')
    obstacles_file = LaunchConfiguration('obstacles_file')
    params_file = LaunchConfiguration('params_file')

    # 3. Declare the launch arguments
    declare_leaks_on_pipes_cmd = DeclareLaunchArgument(
        name='leaks_on_pipes',
        default_value='false',
        description='Place leaks on the pipe surfaces instead of anywhere in the room'
    )

    declare_obstacles_file_cmd = DeclareLaunchArgument(
        name='obstacles_file',
        default_value=default_obstacles_file,
        description='Full path to the pipe geometry leaks are placed on'
    )

    declare_params_file_cmd = DeclareLaunchArgument(
        name='params_file',
        default_value=default_params_file,
        description='Full path to the gas_field_server parameter file'
    )

    # 4. Nodes and includes
    # The truth side of the measurement chain: field, sensor lag, noise, markers
    start_gas_field_server_cmd = Node(
        package=package_name_gas_sim,
        executable='gas_field_server',
        name='gas_field_server',
        output='screen',
        parameters=[params_file,
                    {'leaks_on_pipes': ParameterValue(leaks_on_pipes,
                                                      value_type=bool),
                     'obstacles_file': obstacles_file}]
    )

    # 5. Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_leaks_on_pipes_cmd)
    ld.add_action(declare_obstacles_file_cmd)
    ld.add_action(declare_params_file_cmd)

    # Add the actions
    ld.add_action(start_gas_field_server_cmd)

    return ld
