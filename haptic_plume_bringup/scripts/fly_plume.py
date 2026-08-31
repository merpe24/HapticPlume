#!/usr/bin/env python3
"""
Scripted plume-tracking flight, for checking the estimator without a human.

Pins a scenario, resets the belief, then flies a boustrophedon up the plume
axis and reports what the filter does with it. This exists because a failing
live run has three candidate causes -- stale config, bad flying, broken
estimator -- and only a repeatable flight separates them.

Subscription Topics:
    /hp/odom (nav_msgs/Odometry): Drone pose, the closed-loop feedback
    /hp/gas/true_concentration (haptic_plume_interfaces/GasConcentration): Truth; proves contact
    /hp/gas/measurement_compensated (haptic_plume_interfaces/GasConcentration): Eq. (28), PF input
    /hp/belief/summary (haptic_plume_interfaces/BeliefSummary): sigma, n_eff and status per update
    /hp/events/source_consumed (haptic_plume_interfaces/TaskEvent): Declaration, ends the run

Publishing Topics:
    /hp/cmd_vel (geometry_msgs/TwistStamped): World-frame velocity at ~200 Hz

Services called:
    /hp/set_scenario (haptic_plume_interfaces/SetScenario): Pins the world and returns ground truth
    /hp/reset_estimator (haptic_plume_interfaces/ResetEstimator): Clears the belief before the run

Usage:
    python3 fly_plume.py [seed] [duration_s]

READ THIS BEFORE BELIEVING A RESULT:
  * teleop_mapper publishes /hp/cmd_vel at 100 Hz even with the deadman
    released, and it WILL out-vote this script -- B3's accel limiter never
    builds speed on alternating commands, so the drone barely moves. Suspend
    it first (reversible):   kill -STOP $(pgrep -f teleop_mapper_node)
    and afterwards:          kill -CONT $(pgrep -f teleop_mapper_node)
  * Node parameters are read once at construction and there is no parameter
    callback, so editing a YAML changes nothing until the node is restarted.
    Verify with `ros2 param get /plume_estimator tau_rec` before trusting any
    observation.

:author: premmm
:date: August 30, 2026
"""

import math
import sys
import time

from ament_index_python.packages import get_package_share_directory

from geometry_msgs.msg import TwistStamped

from haptic_plume_interfaces.msg import BeliefSummary, GasConcentration, TaskEvent
from haptic_plume_interfaces.srv import ResetEstimator, SetScenario

from nav_msgs.msg import Odometry

import numpy as np

import rclpy
from rclpy.node import Node

import yaml

SPEED = 1.0                 # [m/s] matches the Phase A agent, so times compare
REACH = 0.30                # [m] waypoint acceptance radius
ENTRY = 7.0                 # [m] downwind start; beyond this main_run blocks seed 3
CROSS = 2.0                 # [m] sweep half-width; the plume is only ~+/-1.5 m wide
ADVANCE = 1.0               # [m] upwind step per pass
# Pipe radius 0.30 + drone collision radius 0.377 + margin. Under this the
# B3 clamp pins the drone against the capsule and slides it, which reads as
# "one axis frozen" and is easy to mistake for a dead publisher.
KEEP_OUT = 0.80
STATUS = {0: 'INIT', 1: 'TRACKING', 2: 'CONVERGING', 3: 'RESET', 4: 'DIVERGED'}


def load_segments():
    """Read the capsule field from its authority, so this cannot drift from D1."""
    path = f'{get_package_share_directory("haptic_plume_description")}/config/obstacles.yaml'
    with open(path) as handle:
        return yaml.safe_load(handle).get('segments', [])


def dodge(point, segments):
    """Push a waypoint out of any capsule's footprint, staying on its near side."""
    for seg in segments:
        p0, p1 = np.array(seg['p0'], float), np.array(seg['p1'], float)
        if abs(p0[0] - p1[0]) > 1e-6 or abs(p0[2] - 1.0) > 0.5:
            continue                       # only the vertical-plane run matters at z=1
        lo, hi = sorted((p0[1], p1[1]))
        keep = seg['radius'] + KEEP_OUT - 0.30
        if lo - keep < point[1] < hi + keep:
            if point[0] > p0[0]:
                point[0] = max(point[0], p0[0] + keep)
            else:
                point[0] = min(point[0], p0[0] - keep)
    point[0] = float(np.clip(point[0], 0.5, 19.5))
    point[1] = float(np.clip(point[1], 0.5, 19.5))
    point[2] = 1.0
    return point


def waypoints(source, theta, segments):
    """Boustrophedon: sweep across the plume axis, marching upwind."""
    along = np.array([math.cos(theta), math.sin(theta), 0.0])
    cross = np.array([-math.sin(theta), math.cos(theta), 0.0])
    out = [dodge((source + along * ENTRY).copy(), segments)]
    side, dist = 1.0, ENTRY
    while dist > 0.4:
        for _ in range(2):
            out.append(dodge((source + along * dist + cross * (CROSS * side)).copy(),
                             segments))
            side = -side
        dist -= ADVANCE
    return out


class Flyer(Node):
    """Closed-loop waypoint follower that logs the belief alongside the truth."""

    def __init__(self):
        super().__init__('fly_plume')
        self.pub = self.create_publisher(TwistStamped, '/hp/cmd_vel', 10)
        self.pos = None
        self.true_c = 0.0
        self.comp_c = 0.0
        self.belief = None
        self.declared = None
        self.create_subscription(Odometry, '/hp/odom', self._on_odom, 10)
        self.create_subscription(
            GasConcentration, '/hp/gas/true_concentration',
            lambda m: setattr(self, 'true_c', m.concentration), 20)
        self.create_subscription(
            GasConcentration, '/hp/gas/measurement_compensated',
            lambda m: setattr(self, 'comp_c', m.concentration), 20)
        self.create_subscription(
            BeliefSummary, '/hp/belief/summary',
            lambda m: setattr(self, 'belief', m), 10)
        self.create_subscription(
            TaskEvent, '/hp/events/source_consumed',
            lambda m: setattr(self, 'declared', m), 10)

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        self.pos = np.array([p.x, p.y, p.z])

    def send(self, velocity):
        """Publish a world-frame command. frame_id is 'map' -- see frames.md 6.1."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.twist.linear.x = float(velocity[0])
        msg.twist.linear.y = float(velocity[1])
        self.pub.publish(msg)

    def call(self, srv_type, name, request):
        """Blocking service call; returns None rather than hanging forever."""
        client = self.create_client(srv_type, name)
        if not client.wait_for_service(timeout_sec=5.0):
            print(f'  service {name} unavailable', flush=True)
            return None
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        return future.result()


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 240.0

    rclpy.init()
    node = Flyer()

    request = SetScenario.Request()
    request.n_sources, request.seed = 1, seed
    scenario = node.call(SetScenario, '/hp/set_scenario', request)
    if scenario is None or not scenario.success:
        print('could not pin the scenario; is gas_field_server up?')
        rclpy.shutdown()
        return
    truth = scenario.true_source_positions[0]
    source = np.array([truth.x, truth.y, truth.z])
    theta = scenario.wind_direction
    node.call(ResetEstimator, '/hp/reset_estimator', ResetEstimator.Request())

    if node.count_publishers('/hp/cmd_vel') > 1:
        print('WARNING: another node publishes /hp/cmd_vel and will out-vote this '
              'script.\n         kill -STOP $(pgrep -f teleop_mapper_node)  '
              '(kill -CONT to restore)\n', flush=True)

    while node.pos is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    route = waypoints(source, theta, load_segments())
    print(f'seed {seed}: source ({source[0]:.2f}, {source[1]:.2f}), '
          f'theta {math.degrees(theta):.1f} deg, {len(route)} waypoints, '
          f'{budget:.0f} s budget', flush=True)
    print('  t     x      y    true_c   comp_c    sigma   n_eff  status     '
          'est_xy      err', flush=True)

    index, start, last_log = 0, time.time(), -5.0
    while rclpy.ok() and time.time() - start < budget and index < len(route):
        rclpy.spin_once(node, timeout_sec=0.0)
        delta = route[index] - node.pos
        delta[2] = 0.0
        distance = float(np.linalg.norm(delta))
        if distance < REACH:
            index += 1
            continue
        # Ease in near the waypoint so the accel limiter is not fighting a step.
        node.send(delta / distance * min(SPEED, distance * 2.0))

        elapsed = time.time() - start
        if elapsed - last_log >= 5.0:
            last_log = elapsed
            summary = node.belief
            if summary is not None:
                estimate = np.array(summary.mean[0:3])
                print(f'{elapsed:5.0f} {node.pos[0]:6.2f} {node.pos[1]:6.2f} '
                      f'{node.true_c:8.4f} {node.comp_c:+8.4f} '
                      f'{summary.sigma_position_norm:8.3f} {summary.n_eff:7.1f} '
                      f'{STATUS.get(summary.status, "?"):<10} '
                      f'({estimate[0]:5.2f},{estimate[1]:5.2f}) '
                      f'{np.linalg.norm(estimate - source):6.2f}', flush=True)
            if node.declared is not None:
                point = node.declared.position
                error = np.linalg.norm(np.array([point.x, point.y, point.z]) - source)
                print(f'\n*** DECLARED at t={elapsed:.0f} s: ({point.x:.2f}, '
                      f'{point.y:.2f})  error {error:.2f} m  '
                      f'[{node.declared.detail}]', flush=True)
                break
        time.sleep(0.005)

    for _ in range(60):                    # bleed the command to zero before exiting
        node.send([0.0, 0.0])
        time.sleep(0.005)
    print(f'done at t={time.time() - start:.0f} s, waypoint {index}/{len(route)}',
          flush=True)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
