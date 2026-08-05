#!/usr/bin/env python3
"""
Bring up the whole HapticPlume simulation for teleoperated flight.

Include order matters and follows the data flow: robot_state_publisher publishes the
URDF, drone_kinematics_node becomes the pose authority, the gas server starts
sampling /hp/odom, Gazebo spawns the model and mirrors that pose, and teleop last.
Gazebo's `create` blocks until /robot_description appears, so it cannot race ahead of
the description; and pipeline.world has zero gravity, so a model that spawns before
the first /hp/odom simply waits at its spawn pose instead of falling.

Per-node parameter files are deliberately NOT exposed here. drone.launch.py,
gas_field.launch.py and teleop_gamepad.launch.py each declare their own params_file
with different meanings, and hoisting one name to cover three would guarantee a
mix-up. Override them on the child launch if you need to.

There is no use_sim_time argument and no /clock bridge: wall clock only (D3).

:author: premmm
:date: August 5, 2026
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    Generate the launch description.

    Returns
    -------
    LaunchDescription
        The declared arguments and the five child launch files that make up the
        Phase B system.

    """
    # 1. Constants for paths to different files and folders
    package_name_description = 'haptic_plume_description'
    package_name_drone = 'haptic_plume_drone'
    package_name_gas_sim = 'haptic_plume_gas_sim'
    package_name_gazebo = 'haptic_plume_gazebo'
    package_name_teleop = 'haptic_plume_teleop'

    # 2. Set the path to different files and folders
    description_launch = PathJoinSubstitution(
        [FindPackageShare(package_name_description),
         'launch', 'robot_state_publisher.launch.py']
    )
    drone_launch = PathJoinSubstitution(
        [FindPackageShare(package_name_drone), 'launch', 'drone.launch.py']
    )
    gas_launch = PathJoinSubstitution(
        [FindPackageShare(package_name_gas_sim), 'launch', 'gas_field.launch.py']
    )
    gazebo_launch = PathJoinSubstitution(
        [FindPackageShare(package_name_gazebo),
         'launch', 'haptic_plume.gazebo.launch.py']
    )
    teleop_launch = PathJoinSubstitution(
        [FindPackageShare(package_name_teleop), 'launch', 'teleop_gamepad.launch.py']
    )

    default_obstacles_file = PathJoinSubstitution(
        [FindPackageShare(package_name_description), 'config', 'obstacles.yaml']
    )

    # 3. Launch configuration variables (alphabetical)
    headless = LaunchConfiguration('headless')
    obstacles_file = LaunchConfiguration('obstacles_file')
    prefix = LaunchConfiguration('prefix')
    use_rviz = LaunchConfiguration('use_rviz')
    use_teleop = LaunchConfiguration('use_teleop')
    world_file = LaunchConfiguration('world_file')
    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')
    z = LaunchConfiguration('z')

    # 4. Declare the launch arguments
    declare_headless_cmd = DeclareLaunchArgument(
        name='headless',
        default_value='false',
        choices=['true', 'false'],
        description='Whether to suppress the Gazebo GUI (server only)'
    )

    # obstacles_file and world_file are a MATCHED PAIR: the analytic capsules the
    # kinematics clamp reads must describe the same pipes the world draws. Changing
    # one without the other is the silent-divergence failure frames.md 8 warns about.
    declare_obstacles_file_cmd = DeclareLaunchArgument(
        name='obstacles_file',
        default_value=default_obstacles_file,
        description='Analytic world geometry read by the collision clamp. Must match '
                    'the pipes drawn in world_file'
    )

    declare_prefix_cmd = DeclareLaunchArgument(
        name='prefix',
        default_value='',
        description='Prefix applied to all link and joint names (multi-robot support)'
    )

    declare_use_rviz_cmd = DeclareLaunchArgument(
        name='use_rviz',
        default_value='true',
        choices=['true', 'false'],
        description='Whether to start RViz'
    )

    # Teleop is separable so the smoke test can run without a gamepad plugged in
    declare_use_teleop_cmd = DeclareLaunchArgument(
        name='use_teleop',
        default_value='true',
        choices=['true', 'false'],
        description='Whether to start the gamepad driver and axis mapper'
    )

    declare_world_file_cmd = DeclareLaunchArgument(
        name='world_file',
        default_value='pipeline.world',
        description='World file in haptic_plume_gazebo/worlds. pipeline.world is the '
                    '20x20 m validated experiment world'
    )

    # Spawn pose. Must match start_position in haptic_plume_drone/config/drone_params.yaml
    declare_x_cmd = DeclareLaunchArgument(
        name='x', default_value='1.0', description='Spawn x position [m], map frame')

    declare_y_cmd = DeclareLaunchArgument(
        name='y', default_value='1.0', description='Spawn y position [m], map frame')

    declare_z_cmd = DeclareLaunchArgument(
        name='z', default_value='1.0', description='Spawn z position [m], map frame')

    # 5. Nodes and includes
    # Publish the drone's TF tree. use_gazebo is forced true: the spawned model needs
    # the camera sensor and the PoseMirror plugin tag, and forgetting this flag makes
    # the drone spawn as an inert body with no plugin and no camera.
    description_cmd = GroupAction(
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(description_launch),
                launch_arguments={
                    'prefix': prefix,
                    'use_gazebo': 'true',
                    'use_rviz': use_rviz
                }.items())
        ],
        scoped=True)

    # The motion authority (D1) and the static map -> odom transform
    drone_cmd = GroupAction(
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(drone_launch),
                launch_arguments={
                    'obstacles_file': obstacles_file
                }.items())
        ],
        scoped=True)

    # Ground-truth gas field and the lagged, noisy sensor
    gas_cmd = GroupAction(
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(gas_launch))
        ],
        scoped=True)

    # Gazebo: renders the world, hosts the FPV camera, mirrors the pose
    gazebo_cmd = GroupAction(
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(gazebo_launch),
                launch_arguments={
                    'headless': headless,
                    'world_file': world_file,
                    'x': x,
                    'y': y,
                    'z': z
                }.items())
        ],
        scoped=True)

    # Gamepad input -> world-frame /hp/cmd_vel, deadman-gated
    teleop_cmd = GroupAction(
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(teleop_launch))
        ],
        scoped=True,
        condition=IfCondition(use_teleop))

    # 6. Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_headless_cmd)
    ld.add_action(declare_obstacles_file_cmd)
    ld.add_action(declare_prefix_cmd)
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(declare_use_teleop_cmd)
    ld.add_action(declare_world_file_cmd)
    ld.add_action(declare_x_cmd)
    ld.add_action(declare_y_cmd)
    ld.add_action(declare_z_cmd)

    # Add the actions, in data-flow order
    ld.add_action(description_cmd)
    ld.add_action(drone_cmd)
    ld.add_action(gas_cmd)
    ld.add_action(gazebo_cmd)
    ld.add_action(teleop_cmd)

    return ld
