#!/usr/bin/env python3
"""
Bring up the drone motion authority.

Starts drone_kinematics_node (the pose authority, decision D1) alongside the static
map -> odom identity transform. frames.md names the launch file as the publisher of
that transform, which is why it lives here rather than inside the node.

There is deliberately no use_sim_time argument: this project runs on wall clock
(decision D3) with no /clock bridge.

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
        The declared arguments, the kinematics node, and the static map -> odom TF.

    """
    # 1. Constants for paths to different files and folders
    package_name_drone = 'haptic_plume_drone'
    package_name_description = 'haptic_plume_description'

    default_params_file = PathJoinSubstitution(
        [FindPackageShare(package_name_drone), 'config', 'drone_params.yaml']
    )
    default_obstacles_file = PathJoinSubstitution(
        [FindPackageShare(package_name_description), 'config', 'obstacles.yaml']
    )

    # 2. Launch configuration variables (alphabetical)
    obstacles_file = LaunchConfiguration('obstacles_file')
    params_file = LaunchConfiguration('params_file')

    # 3. Declare the launch arguments
    declare_obstacles_file_cmd = DeclareLaunchArgument(
        name='obstacles_file',
        default_value=default_obstacles_file,
        description='Full path to the analytic world geometry read by the clamp'
    )

    declare_params_file_cmd = DeclareLaunchArgument(
        name='params_file',
        default_value=default_params_file,
        description='Full path to the drone_kinematics_node parameter file'
    )

    # 4. Nodes and includes
    # The motion authority: integrates /hp/cmd_vel and owns odom -> base_link
    start_drone_kinematics_cmd = Node(
        package=package_name_drone,
        executable='drone_kinematics_node',
        name='drone_kinematics',
        output='screen',
        parameters=[params_file, {'obstacles_file': obstacles_file}]
    )

    # map -> odom is identity: a kinematic drone has zero drift and this node IS
    # ground truth. Omitting the translation and rotation arguments gives identity.
    start_map_to_odom_tf_cmd = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_static_tf',
        output='screen',
        arguments=[
            '--frame-id', 'map',
            '--child-frame-id', 'odom'
        ]
    )

    # 5. Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_obstacles_file_cmd)
    ld.add_action(declare_params_file_cmd)

    # Add the actions
    ld.add_action(start_drone_kinematics_cmd)
    ld.add_action(start_map_to_odom_tf_cmd)

    return ld
