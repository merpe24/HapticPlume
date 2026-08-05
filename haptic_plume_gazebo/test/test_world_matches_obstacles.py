#!/usr/bin/env python3
"""
Assert the pipes drawn in each world match the analytic capsules the clamp reads.

obstacles.yaml (haptic_plume_description) is the collision authority: both
drone_kinematics_node's clamp and Phase C's Parametric Risk Field read it through
libobstacle_field.so, while Gazebo draws pipeline.world from an independent copy of
the same numbers. Nothing else keeps the two in step, so editing one and not the
other yields a world where the force field pushes off a pipe that is not where the
operator sees it. frames.md 8 names this test as the fix for that gap.

:author: premmm
:date: August 6, 2026
"""

import math
import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import pytest
import yaml

# Position tolerance [m]. Both files are hand-written decimal literals and should
# agree exactly; this only absorbs float round-trip through the SDF text.
TOLERANCE = 1e-6

# Each entry pairs a world with the obstacle YAML whose capsules it draws.
# lab.world joins this list when it lands, paired with config/obstacles_lab.yaml.
WORLD_OBSTACLE_PAIRS = [
    ('pipeline.world', 'obstacles.yaml'),
]


def cylinder_axis(roll, pitch, yaw):
    """
    Return the world-frame unit vector along a cylinder's local +z axis.

    SDF poses are roll-pitch-yaw about fixed axes, so R = Rz(yaw) Ry(pitch) Rx(roll),
    and the axis is that matrix applied to [0, 0, 1].

    :param roll: Rotation about x [rad]
    :type roll: float
    :param pitch: Rotation about y [rad]
    :type pitch: float
    :param yaw: Rotation about z [rad]
    :type yaw: float
    :return: Unit vector along the cylinder axis, in world coordinates
    :rtype: tuple
    """
    return (
        math.cos(yaw) * math.sin(pitch) * math.cos(roll) + math.sin(yaw) * math.sin(roll),
        math.sin(yaw) * math.sin(pitch) * math.cos(roll) - math.cos(yaw) * math.sin(roll),
        math.cos(pitch) * math.cos(roll),
    )


def read_world_pipes(world_path):
    """
    Extract every cylinder model from a world file as capsule endpoints.

    Models whose geometry is not a cylinder (floor, walls) are skipped.

    :param world_path: Full path to the SDF world file
    :type world_path: str
    :return: Model name -> (p0, p1, radius), endpoints in world coordinates
    :rtype: dict
    """
    root = ET.parse(world_path).getroot()

    pipes = {}
    for model in root.iter('model'):
        cylinder = model.find('.//geometry/cylinder')
        if cylinder is None:
            continue

        pose = [float(value) for value in model.find('pose').text.split()]
        centre, rpy = pose[:3], pose[3:]
        radius = float(cylinder.find('radius').text)
        length = float(cylinder.find('length').text)

        # The drawn length is |p1 - p0|; the capsule's hemispherical caps extend a
        # further `radius` past each end and are deliberately not drawn.
        axis = cylinder_axis(*rpy)
        half = [0.5 * length * component for component in axis]

        pipes[model.get('name')] = (
            tuple(c - h for c, h in zip(centre, half)),
            tuple(c + h for c, h in zip(centre, half)),
            radius,
        )

    return pipes


@pytest.mark.parametrize('world_file,obstacles_file', WORLD_OBSTACLE_PAIRS)
def test_world_matches_obstacles(world_file, obstacles_file):
    """Every capsule in the obstacle YAML is drawn in the same place in the world."""
    world_path = os.path.join(
        get_package_share_directory('haptic_plume_gazebo'), 'worlds', world_file)
    obstacles_path = os.path.join(
        get_package_share_directory('haptic_plume_description'), 'config', obstacles_file)

    with open(obstacles_path) as handle:
        obstacles = yaml.safe_load(handle)

    pipes = read_world_pipes(world_path)

    for segment in obstacles['segments']:
        name = segment['name']
        assert name in pipes, \
            f'{world_file} draws no cylinder model named {name}'

        drawn_p0, drawn_p1, drawn_radius = pipes[name]

        assert drawn_radius == pytest.approx(segment['radius'], abs=TOLERANCE), \
            f'{name}: world radius {drawn_radius} != yaml radius {segment["radius"]}'

        # A capsule is symmetric, so the world may draw the cylinder either way round
        expected = sorted([tuple(segment['p0']), tuple(segment['p1'])])
        drawn = sorted([drawn_p0, drawn_p1])

        for expected_point, drawn_point in zip(expected, drawn):
            assert drawn_point == pytest.approx(expected_point, abs=TOLERANCE), \
                f'{name}: world endpoint {drawn_point} != yaml endpoint {expected_point}'
