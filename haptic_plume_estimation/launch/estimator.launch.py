#!/usr/bin/env python3
"""
Bring up the MultiPLE plume estimator on the live gas measurement.

Runs plume_estimator alone. It needs only /hp/gas/measurement, which
gas_field_server already publishes, so this composes with sim_teleop.launch.py
instead of duplicating any of it.

:author: premmm
:date: August 28, 2026
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    Generate the launch description for the plume estimator.

    Returns
    -------
    LaunchDescription
        The declared arguments and the plume estimator node.

    """
    # 1. Constants for paths to different files and folders
    package_name_estimation = 'haptic_plume_estimation'

    # 2. Set the path to different files and folders
    default_params_file = PathJoinSubstitution(
        [FindPackageShare(package_name_estimation), 'config',
         'estimator_params.yaml'])

    # 3. Launch configuration variables
    params_file = LaunchConfiguration('params_file')
    pf_seed = LaunchConfiguration('pf_seed')

    # 4. Declare the launch arguments
    declare_params_file_cmd = DeclareLaunchArgument(
        name='params_file',
        default_value=default_params_file,
        description='Full path to the estimator parameter YAML')

    declare_pf_seed_cmd = DeclareLaunchArgument(
        name='pf_seed',
        default_value='0',
        description='Particle filter Generator seed; overrides the YAML for D7 replay')

    # 5. Nodes and includes
    # The node name MUST stay 'plume_estimator': it is the top-level key in
    # estimator_params.yaml, and a mismatch makes ROS ignore the entire file
    # silently. use_sim_time is never set -- decision D3 is wall clock.
    start_plume_estimator_cmd = Node(
        package=package_name_estimation,
        executable='plume_estimator_node',
        name='plume_estimator',
        output='screen',
        parameters=[params_file,
                    {'pf_seed': ParameterValue(pf_seed, value_type=int)}])

    # 6. Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_pf_seed_cmd)

    # Add any actions
    ld.add_action(start_plume_estimator_cmd)

    return ld
