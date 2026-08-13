#!/usr/bin/env python3
"""
T1: headless smoke test of the whole Phase B simulation.

Brings up sim_teleop.launch.py with no GUI, no RViz and no gamepad, then asserts the
chain is alive end to end: every node present, /hp/odom at 100 Hz,
/hp/gas/measurement at 20 Hz carrying usable values, TF resolving
map -> odom -> base_link, and a clean exit. The rate bands are deliberately wide --
T1 answers "is the chain alive", not "what is the jitter"; the tight numbers are
measured by hand at each phase checkpoint.

:author: premmm
:date: August 13, 2026
"""

import math
import subprocess
import time
import unittest

from haptic_plume_interfaces.msg import GasConcentration

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution

from launch_ros.substitutions import FindPackageShare

import launch_testing.actions
import launch_testing.asserts

from nav_msgs.msg import Odometry

import pytest

import rclpy
from rclpy.time import Time

import tf2_ros

STARTUP_TIMEOUT = 90.0
TF_TIMEOUT = 15.0
EXIT_SIGTERM = -15
GZ_REAP_TIMEOUT = 10.0

RATE_WINDOW = 5.0

# Measured at the B6 checkpoint: /hp/odom 100.005 Hz, /hp/gas/measurement 20.000 Hz.
# The bands are wide on purpose -- a tight band flakes under load and turns a smoke
# test into a timing benchmark.
ODOM_RATE_BAND = (90.0, 110.0)
GAS_RATE_BAND = (18.0, 22.0)

# Node names, not executable names: drone.launch.py renames the kinematics node.
EXPECTED_NODES = (
    'drone_kinematics',
    'gas_field_server',
    'map_to_odom_static_tf',
    'robot_state_publisher',
)


@pytest.mark.launch_test
def generate_test_description():
    """Bring up the Phase B system headless, with no RViz and no gamepad."""
    if subprocess.run(['pgrep', '-f', 'gz sim -r -s'],
                      capture_output=True, check=False).returncode == 0:
        raise RuntimeError(
            'a gz server is already running: T1 would start a second one and every '
            'gz query would then pick between them non-deterministically')

    sim_teleop_launch = PathJoinSubstitution(
        [FindPackageShare('haptic_plume_bringup'), 'launch', 'sim_teleop.launch.py']
    )

    start_sim_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_teleop_launch),
        launch_arguments={
            'headless': 'true',
            'use_rviz': 'false',
            'use_teleop': 'false'
        }.items())

    return LaunchDescription([start_sim_cmd, launch_testing.actions.ReadyToTest()])


def spin_for(node, seconds):
    """Spin the node for a wall-clock duration [s]."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)


def wait_until(node, predicate, timeout):
    """Spin until predicate() holds or timeout [s] expires; return whether it held."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        rclpy.spin_once(node, timeout_sec=0.1)
    return predicate()


def arrival_rate(stamps):
    """Return the mean arrival rate [Hz] from a list of monotonic receive times."""
    if len(stamps) < 2:
        return 0.0
    return (len(stamps) - 1) / (stamps[-1] - stamps[0])


class TestSimSmoke(unittest.TestCase):
    """Assertions made while the simulation is running."""

    @classmethod
    def setUpClass(cls):
        """Wait for the system to come up, then sample every topic in one window."""
        rclpy.init()
        cls.node = rclpy.create_node('t1_smoke_probe')
        cls.tf_buffer = tf2_ros.Buffer()
        cls.tf_listener = tf2_ros.TransformListener(cls.tf_buffer, cls.node)

        cls.odom_stamps = []
        cls.gas_stamps = []
        cls.gas_msgs = []

        def on_gas(msg):
            cls.gas_stamps.append(time.monotonic())
            cls.gas_msgs.append(msg)

        cls.node.create_subscription(
            Odometry, '/hp/odom',
            lambda _: cls.odom_stamps.append(time.monotonic()), 10)
        cls.node.create_subscription(
            GasConcentration, '/hp/gas/measurement', on_gas, 10)

        wait_until(
            cls.node,
            lambda: set(EXPECTED_NODES).issubset(cls.node.get_node_names()),
            STARTUP_TIMEOUT)
        cls.node_names = cls.node.get_node_names()

        # Discard whatever arrived while the system was still coming up, then
        # measure both rates over one clean window.
        cls.odom_stamps.clear()
        cls.gas_stamps.clear()
        cls.gas_msgs.clear()
        spin_for(cls.node, RATE_WINDOW)

    @classmethod
    def tearDownClass(cls):
        """Tear down the probe node."""
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_expected_nodes_are_up(self):
        """Every node sim_teleop.launch.py starts is present."""
        missing = sorted(set(EXPECTED_NODES) - set(self.node_names))
        self.assertFalse(missing, f'nodes never appeared: {missing}')

    def test_odom_publishes_at_its_configured_rate(self):
        """The motion authority ticks at 100 Hz (D1)."""
        rate = arrival_rate(self.odom_stamps)
        low, high = ODOM_RATE_BAND
        self.assertTrue(
            low <= rate <= high,
            f'/hp/odom arrived at {rate:.1f} Hz, expected {low}-{high} Hz')

    def test_measurement_publishes_at_its_configured_rate(self):
        """The gas server ticks at 20 Hz -- the rate its lag was discretized at."""
        rate = arrival_rate(self.gas_stamps)
        low, high = GAS_RATE_BAND
        self.assertTrue(
            low <= rate <= high,
            f'/hp/gas/measurement arrived at {rate:.1f} Hz, expected {low}-{high} Hz')

    def test_measurement_values_are_usable(self):
        """Concentration is finite and >= 0, as GasConcentration.msg promises."""
        self.assertTrue(self.gas_msgs, '/hp/gas/measurement never published')
        for msg in self.gas_msgs:
            self.assertTrue(
                math.isfinite(msg.concentration),
                f'non-finite concentration {msg.concentration}')
            self.assertGreaterEqual(msg.concentration, 0.0)

    def test_tf_resolves_map_to_base_link(self):
        """TF resolves map -> odom -> base_link, no gaps (frames.md 2)."""
        self.assertTrue(
            wait_until(
                self.node,
                lambda: self.tf_buffer.can_transform('map', 'base_link', Time()),
                TF_TIMEOUT),
            'TF never resolved map -> base_link')


@launch_testing.post_shutdown_test()
class TestCleanShutdown(unittest.TestCase):
    """Assertions made after the launch has been torn down."""

    def test_no_gz_server_is_left_behind(self):

        subprocess.run(['pkill', '-INT', '-f', 'gz sim -r -s'], check=False)
        deadline = time.monotonic() + GZ_REAP_TIMEOUT
        while time.monotonic() < deadline:
            if subprocess.run(['pgrep', '-f', 'gz sim -r -s'],
                              capture_output=True, check=False).returncode != 0:
                return
            time.sleep(0.5)
        self.fail('a gz server survived SIGINT after the launch was torn down')

    def test_every_process_exited_cleanly(self, proc_info):

        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[
                launch_testing.asserts.EXIT_OK,
                launch_testing.asserts.EXIT_SIGINT,
                EXIT_SIGTERM,
            ])
