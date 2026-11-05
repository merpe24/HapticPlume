#!/usr/bin/env python3
"""
Bring up the full HapticPlume system: sim, estimator and the visual display.

Composes the existing launch files instead of repeating them: sim_teleop.launch.py
for the world, drone and gas, estimator.launch.py for the particle filter, plus
viz_gate_node for what the operator sees and experiment_director_node, which owns
/hp/condition and starts and ends trials. Teleop defaults to OFF here, because a
running teleop_mapper publishes /hp/cmd_vel at 100 Hz and out-votes fly_plume.py.

Nothing is drawn or felt until a trial is started with /hp/start_trial.

:author: premmm
:date: October 22, 2026
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
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
        The declared arguments, the sim and estimator includes, the display gate
        and the stand-in condition sender.

    """
    # 1. Constants for paths to different files and folders
    package_name_bringup = 'haptic_plume_bringup'
    package_name_description = 'haptic_plume_description'
    package_name_estimation = 'haptic_plume_estimation'
    package_name_experiment = 'haptic_plume_experiment'

    # 2. Set the path to different files and folders
    sim_launch = PathJoinSubstitution(
        [FindPackageShare(package_name_bringup), 'launch', 'sim_teleop.launch.py']
    )
    estimator_launch = PathJoinSubstitution(
        [FindPackageShare(package_name_estimation), 'launch', 'estimator.launch.py']
    )

    # 3. Launch configuration variables (alphabetical)
    rviz_config = LaunchConfiguration('rviz_config')
    use_teleop = LaunchConfiguration('use_teleop')

    # 4. Declare the launch arguments
    declare_rviz_config_cmd = DeclareLaunchArgument(
        name='rviz_config',
        default_value='haptic_plume_description.rviz',
        choices=['haptic_plume_description.rviz', 'operator.rviz'],
        description='RViz view: the developer view (with ground truth) or '
                    'participant view (operator.rviz, no ground truth)'
    )

    declare_use_teleop_cmd = DeclareLaunchArgument(
        name='use_teleop',
        default_value='false',
        choices=['true', 'false'],
        description='Whether to start the gamepad. Off so fly_plume.py is not out-voted'
    )

    # 5. Nodes and includes
    # World, drone, gas server, Gazebo and RViz
    sim_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch),
        launch_arguments={
            'use_teleop': use_teleop,
            'rviz_config_file': PathJoinSubstitution(
                [FindPackageShare(package_name_description), 'rviz', rviz_config]),
        }.items()
    )

    # The particle filter
    estimator_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(estimator_launch)
    )

    # What the operator sees, gated by the condition
    viz_gate_cmd = Node(
        package=package_name_experiment,
        executable='viz_gate_node',
        name='viz_gate',
        output='screen'
    )

    # Owns /hp/condition: resets everything, then enables one condition per trial
    director_cmd = Node(
        package=package_name_experiment,
        executable='experiment_director_node',
        name='experiment_director',
        output='screen'
    )

    # 6. Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_rviz_config_cmd)
    ld.add_action(declare_use_teleop_cmd)

    # Add the actions, in data-flow order
    ld.add_action(sim_cmd)
    ld.add_action(estimator_cmd)
    ld.add_action(viz_gate_cmd)
    ld.add_action(director_cmd)

    return ld
