#!/usr/bin/env python3
"""
Launch Rviz visualization for HapticPlume drone.

This launch file setup the complete visualization environment for the drone,
including robot state publisher and Rviz2.
It handles loading and processing of URDF/XACRO files and controller configurations.

:author: premmm
:date: 1 Aug, 2026
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    """Generate the launch description
    
    Returns:
        LaunchDescription: Complete launch description
    """

    # 1. Constants for paths to different files and folders
    package_name_description = 'haptic_plume_description'
    default_urdf_model_path = PathJoinSubstitution(
        [FindPackageShare(package_name_description), 'urdf', 'robots', 'drone.urdf.xacro']
    )
    default_rviz_config_path = PathJoinSubstitution(
        [FindPackageShare(package_name_description), 'rviz', 'haptic_plume_description.rviz']
    )


    # 2. Launch configuration variables (alphabetical)
    prefix = LaunchConfiguration('prefix')
    rviz_config_file = LaunchConfiguration('rviz_config_file')
    urdf_model = LaunchConfiguration('urdf_model')
    use_gazebo = LaunchConfiguration('use_gazebo')
    use_rviz = LaunchConfiguration('use_rviz')


    # 3. Declare the launch arguments
    declare_prefix_cmd = DeclareLaunchArgument(
        name='prefix',
        default_value='',
        description='Prefix applied to all link and joint names (multi-robot support)'
    )

    declare_rviz_config_file_cmd = DeclareLaunchArgument(
        name='rviz_config_file',
        default_value=default_rviz_config_path,
        description='Full path to Rviz config file to use'
    )

    declare_urdf_model_cmd = DeclareLaunchArgument(
        name='urdf_model',
        default_value=default_urdf_model_path,
        description='Full path to the drone xacro description'
    )

    declare_use_gazebo_cmd = DeclareLaunchArgument(
        name='use_gazebo',
        default_value='false',
        choices=['true', 'false'],
        description='Whether to include Gazebo-only tags (gz camera sensor in the URDF)'
    )

    declare_use_rviz_cmd = DeclareLaunchArgument(
        name='use_rviz',
        default_value='true',
        choices=['true', 'false'],
        description='Whether to start RViz'
    )


    # 4. Nodes and includes
    # Publish the drone's TF tree form the expanded xacro
    start_robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': ParameterValue(
                Command(['xacro', ' ', urdf_model,
                         ' ', 'prefix:=', prefix,
                         ' ', 'use_gazebo:=', use_gazebo]),
                value_type=str
            )
        }]
    )

    # Visualize the drone and its frames
    start_rviz_cmd = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )


    #5. Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_prefix_cmd)
    ld.add_action(declare_rviz_config_file_cmd)
    ld.add_action(declare_urdf_model_cmd)
    ld.add_action(declare_use_gazebo_cmd)
    ld.add_action(declare_use_rviz_cmd)

    # Add any acitons
    ld.add_action(start_robot_state_publisher_cmd)
    ld.add_action(start_rviz_cmd)

    return ld