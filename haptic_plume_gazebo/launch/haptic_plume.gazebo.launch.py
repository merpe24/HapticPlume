#!/usr/bin/env python3
"""
Launch Gazebo as the renderer for the HapticPlume simulation.

Gazebo draws the world and hosts the FPV camera; it never moves the drone. Decision D1
makes drone_kinematics_node the motion authority, and the PoseMirror system plugin
teleports the model onto /hp/odom every simulation step. Nothing here starts that node
- run haptic_plume_drone/drone.launch.py alongside this, or use the bringup file that
includes both.

Spawning reads /robot_description, so robot_state_publisher must already be running
with use_gazebo:=true. That include lives in the bringup file, not here.

There is deliberately no use_sim_time argument and no /clock bridge: this project runs
on wall clock (decision D3).

:author: premmm
:date: August 5, 2026
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription
)
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    Generate the launch description.

    Returns
    -------
    LaunchDescription
        The declared arguments, the Gazebo server and client, the ROS <-> gz bridges,
        and the model spawner.

    """
    # 1. Constants for paths to different files and folders
    package_name_gazebo = 'haptic_plume_gazebo'
    default_world_file = 'pipeline.world'

    # 2. Set the path to different files and folders
    pkg_share_gazebo = FindPackageShare(package=package_name_gazebo).find(package_name_gazebo)
    pkg_ros_gz_sim = FindPackageShare(package='ros_gz_sim').find('ros_gz_sim')

    ros_gz_bridge_config_file = os.path.join(pkg_share_gazebo, 'config', 'ros_gz_bridge.yaml')


    # 3. Launch configuration variables (alphabetical)
    headless = LaunchConfiguration('headless')
    pitch = LaunchConfiguration('pitch')
    robot_name = LaunchConfiguration('robot_name')
    roll = LaunchConfiguration('roll')
    world_file = LaunchConfiguration('world_file')
    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')
    yaw = LaunchConfiguration('yaw')
    z = LaunchConfiguration('z')

    world_path = PathJoinSubstitution([pkg_share_gazebo, 'worlds', world_file])

    # 4. Declare the launch arguments
    declare_headless_cmd = DeclareLaunchArgument(
        name='headless',
        default_value='false',
        choices=['true', 'false'],
        description='Whether to suppress the Gazebo GUI (server only)'
    )

    declare_robot_name_cmd = DeclareLaunchArgument(
        name='robot_name',
        default_value='drone',
        description='Name given to the spawned model in Gazebo'
    )

    declare_world_file_cmd = DeclareLaunchArgument(
        name='world_file',
        default_value=default_world_file,
        description='World file in haptic_plume_gazebo/worlds. pipeline.world is the '
                    '20x20 m validated experiment world; lab.world is the 6x8 m lab '
                    'replica and is NOT an experiment world'
    )

    # Spawn pose. These must match start_position in haptic_plume_drone's params file,
    # or the drone visibly jumps the moment the first /hp/odom message arrives.
    declare_x_cmd = DeclareLaunchArgument(
        name='x', default_value='1.0', description='Spawn x position [m], map frame')

    declare_y_cmd = DeclareLaunchArgument(
        name='y', default_value='1.0', description='Spawn y position [m], map frame')

    declare_z_cmd = DeclareLaunchArgument(
        name='z', default_value='1.0', description='Spawn z position [m], map frame')

    declare_roll_cmd = DeclareLaunchArgument(
        name='roll', default_value='0.0', description='Spawn roll [rad]')

    declare_pitch_cmd = DeclareLaunchArgument(
        name='pitch', default_value='0.0', description='Spawn pitch [rad]')

    declare_yaw_cmd = DeclareLaunchArgument(
        name='yaw', default_value='0.0', description='Spawn yaw [rad]')


    # Start the Gazebo server (physics and sensors, no window)
    start_gazebo_server_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments=[('gz_args', [' -r -s -v 4 ', world_path])])

    # Start the Gazebo client (the GUI) unless running headless
    start_gazebo_client_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': ['-g ']}.items(),
        condition=UnlessCondition(headless))

    # Bridge ROS topics and Gazebo messages: the pose feed out, camera_info back
    start_gazebo_ros_bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': ros_gz_bridge_config_file}],
        output='screen')

    # Includes optimizations to minimize latency and bandwidth when streaming images
    start_gazebo_ros_image_bridge_cmd = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=['/fpv/image'],
        remappings=[('/fpv/image', '/hp/fpv/image_raw')],
        output='screen')

    # Spawn the drone from the URDF that robot_state_publisher is already publishing
    start_gazebo_ros_spawner_cmd = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-topic', '/robot_description',
            '-name', robot_name,
            '-allow_renaming', 'true',
            '-x', x,
            '-y', y,
            '-z', z,
            '-R', roll,
            '-P', pitch,
            '-Y', yaw
        ])

    # 6. Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_headless_cmd)
    ld.add_action(declare_robot_name_cmd)
    ld.add_action(declare_world_file_cmd)
    ld.add_action(declare_x_cmd)
    ld.add_action(declare_y_cmd)
    ld.add_action(declare_z_cmd)
    ld.add_action(declare_roll_cmd)
    ld.add_action(declare_pitch_cmd)
    ld.add_action(declare_yaw_cmd)

    # Add the actions
    ld.add_action(start_gazebo_server_cmd)
    ld.add_action(start_gazebo_client_cmd)
    ld.add_action(start_gazebo_ros_bridge_cmd)
    ld.add_action(start_gazebo_ros_image_bridge_cmd)
    ld.add_action(start_gazebo_ros_spawner_cmd)

    return ld
