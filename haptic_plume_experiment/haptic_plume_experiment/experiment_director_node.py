#!/usr/bin/env python3
"""
Run the experiment's trials: reset everything, then switch one condition on.

Publishing Topics:
    /hp/condition (haptic_plume_interfaces/ConditionConfig): Trial condition and flags
    /hp/events/trial (haptic_plume_interfaces/TaskEvent): TRIAL_START / TRIAL_END, ids and seeds

Services:
    /hp/start_trial (haptic_plume_interfaces/StartTrial): Reset everything, then enable a condition
    /hp/end_trial (haptic_plume_interfaces/EndTrial): Switch everything off, log the end

Services called:
    /hp/reset_drone (std_srvs/Trigger): Drone back to its fixed start, zero velocity
    /hp/set_scenario (haptic_plume_interfaces/SetScenario): New gas world, one leak, from the seed
    /hp/reset_estimator (haptic_plume_interfaces/ResetEstimator): Fresh belief, consumed cleared

ROS Parameters:
    bag_root (str): Folder that gets one bag folder per trial (default ~/hp_bags/trials)
    source_dir (str): The git repo, for the commit hash in trial.yaml
        (default ~/Projects/ros2_ws/src/haptic_plume)

:author: premmm
:date: October 26, 2026
"""

from datetime import datetime
import os
import signal
import subprocess
import time

from haptic_plume_interfaces.msg import ConditionConfig, TaskEvent
from haptic_plume_interfaces.srv import EndTrial, ResetEstimator, SetScenario, StartTrial

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from std_srvs.srv import Trigger

import yaml

SERVICE_TIMEOUT = 5.0   # [s] a reset step slower than this counts as failed
RECORDER_TIMEOUT = 5.0  # [s] how long to wait for the bag recorder to connect

VISUAL_CONDITIONS = (ConditionConfig.VISUAL_ONLY, ConditionConfig.VISUAL_HAPTIC)
HAPTIC_CONDITIONS = (ConditionConfig.HAPTIC_ONLY, ConditionConfig.VISUAL_HAPTIC)


def git_state(source_dir):
    """Return (commit hash, True if there are uncommitted edits) for source_dir."""
    def git(*args):
        return subprocess.run(['git', '-C', source_dir, *args],
                              capture_output=True, text=True).stdout.strip()
    return git('rev-parse', 'HEAD'), git('status', '--porcelain') != ''


def make_idle():
    """Return a condition with every cue and display off: zero force, empty RViz."""
    config = ConditionConfig()
    config.trial_id = 'idle'
    return config


def make_condition(condition, trial_id, scenario_seed, pf_seed):
    """Return one trial's ConditionConfig; the only place a condition maps to flags."""
    if condition not in VISUAL_CONDITIONS + HAPTIC_CONDITIONS:
        raise ValueError(f'unknown condition {condition}')
    haptic = condition in HAPTIC_CONDITIONS
    visual = condition in VISUAL_CONDITIONS

    config = ConditionConfig()
    config.condition = condition
    config.prf_enabled = config.well_enabled = config.heaviness_enabled = haptic
    config.prf_gain = config.well_gain = config.heaviness_gain = 1.0
    config.show_belief_markers = config.show_particles = visual
    config.show_gauge = config.show_wind_arrow = visual
    config.trial_id = trial_id
    config.scenario_seed = scenario_seed
    config.pf_seed = pf_seed
    return config


class ExperimentDirector(Node):
    """Own the trial lifecycle; the only publisher on /hp/condition."""

    def __init__(self):
        """Publish IDLE once, so nothing is felt or drawn until a trial starts."""
        super().__init__('experiment_director')
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher_condition = self.create_publisher(
            ConditionConfig, '/hp/condition', latched_qos)
        self.publisher_condition.publish(make_idle())
        self.get_logger().info('experiment director up, IDLE')

        # Clients get their own group so their replies can
        # arrive while start_trial_callback is still waiting on them
        self.client_group = MutuallyExclusiveCallbackGroup()
        self.client_reset_drone = self.create_client(
            Trigger, '/hp/reset_drone', callback_group=self.client_group)
        self.client_set_scenario = self.create_client(
            SetScenario, '/hp/set_scenario', callback_group=self.client_group)
        self.client_reset_estimator = self.create_client(
            ResetEstimator, '/hp/reset_estimator', callback_group=self.client_group)
        self.service_start = self.create_service(
            StartTrial, '/hp/start_trial', self.start_trial_callback)
        self.service_end = self.create_service(
            EndTrial, '/hp/end_trial', self.end_trial_callback)
        self.publisher_events = self.create_publisher(TaskEvent, '/hp/events/trial', 10)
        self.trial_id = None    # the running trial, None while IDLE

        self.declare_parameter('bag_root', '~/hp_bags/trials')
        self.bag_root = os.path.expanduser(
            self.get_parameter('bag_root').get_parameter_value().string_value)
        self.declare_parameter('source_dir', '~/Projects/ros2_ws/src/haptic_plume')
        self.source_dir = os.path.expanduser(
            self.get_parameter('source_dir').get_parameter_value().string_value)
        self.recorder = None    # the running 'ros2 bag record', None when not recording

    def call(self, client, request):
        """Call one reset service and wait; raise if it is down, slow or refuses."""
        if not client.wait_for_service(timeout_sec=SERVICE_TIMEOUT):
            raise RuntimeError(f'{client.srv_name} is not available')
        response = client.call(request, timeout_sec=SERVICE_TIMEOUT)
        if response is None or not response.success:
            raise RuntimeError(f'{client.srv_name} failed')
        return response

    def start_bag(self, trial_id):
        """Start recording every topic except the camera; wait until it listens."""
        self.stop_bag()
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.bag_path = os.path.join(self.bag_root, f'{trial_id}_{stamp}')
        os.makedirs(self.bag_root, exist_ok=True)
        self.recorder = subprocess.Popen(
            ['ros2', 'bag', 'record', '-s', 'mcap', '-o', self.bag_path,
             '--all-topics', '--exclude-regex', 'fpv',
             '--disable-keyboard-controls'],
            stdin=subprocess.DEVNULL, start_new_session=True)
        deadline = time.monotonic() + RECORDER_TIMEOUT
        while self.publisher_events.get_subscription_count() == 0:
            if time.monotonic() > deadline:
                raise RuntimeError('bag recorder did not connect')
            time.sleep(0.1)

    def stop_bag(self):
        """Ctrl-C the recorder so it closes the mcap file; safe when not recording."""
        if self.recorder is None:
            return
        self.recorder.send_signal(signal.SIGINT)
        try:
            self.recorder.wait(timeout=4.0)
        except subprocess.TimeoutExpired:
            self.get_logger().error('bag recorder hung, killing it')
            self.recorder.kill()
        self.recorder = None

    def start_trial_callback(self, request, response):
        """Zero everything, reset the world in order, then enable the condition."""
        self.publisher_condition.publish(make_idle())
        try:
            self.start_bag(request.trial_id)
            config = make_condition(request.condition, request.trial_id,
                                    request.scenario_seed, request.pf_seed)
            self.call(self.client_reset_drone, Trigger.Request())
            scenario = self.call(self.client_set_scenario, SetScenario.Request(
                n_sources=1, seed=request.scenario_seed))
            self.call(self.client_reset_estimator, ResetEstimator.Request(
                clear_consumed=True, pf_seed=request.pf_seed))
        except (ValueError, RuntimeError) as error:
            self.stop_bag()
            response.success = False
            response.message = f'trial not started, staying IDLE: {error}'
            self.get_logger().error(response.message)
            return response

        commit, dirty = git_state(self.source_dir)
        self.trial_info = {
            'trial_id': request.trial_id,
            'condition': int(request.condition),
            'scenario_seed': int(request.scenario_seed),
            'pf_seed': int(request.pf_seed),
            'true_leak': [[p.x, p.y, p.z] for p in scenario.true_source_positions],
            'git_commit': commit,
            'git_dirty': dirty,
            'start_time': datetime.now().isoformat(timespec='seconds'),
            'bag': self.bag_path,
        }

        self.publisher_condition.publish(config)
        self.trial_id = request.trial_id
        self.publish_event(
            TaskEvent.TRIAL_START,
            f'trial {request.trial_id} condition {request.condition} '
            f'scenario_seed {request.scenario_seed} pf_seed {request.pf_seed} '
            f'git {commit[:7]}{"-dirty" if dirty else ""}')
        response.success = True
        response.message = f'trial {request.trial_id} started'
        self.get_logger().info(response.message)
        return response

    def end_trial_callback(self, request, response):
        """Switch every cue and display off and log the end of the trail."""
        self.publisher_condition.publish(make_idle())
        if self.trial_id is None:
            response.success = False
            response.message = 'no trial running, already IDLE'
            return response

        self.publish_event(TaskEvent.TRIAL_END,
                           f'trial {self.trial_id} reason {request.reason}')
        time.sleep(0.5)     # let the recorder write TRIAL_END before it stops
        self.stop_bag()
        self.trial_info['end_time'] = datetime.now().isoformat(timespec='seconds')
        self.trial_info['end_reason'] = request.reason
        with open(os.path.join(self.bag_path, 'trial.yaml'), 'w') as file:
            yaml.safe_dump(self.trial_info, file, sort_keys=False)

        response.success = True
        response.message = f'trial {self.trial_id} ended: {request.reason}'
        self.get_logger().info(response.message)
        self.trial_id = None
        return response

    def publish_event(self, event_type, detail):
        """Log a trial boundary on /hp/events/trial."""
        event = TaskEvent()
        event.header.stamp = self.get_clock().now().to_msg()
        event.header.frame_id = 'map'
        event.event_type = event_type
        event.detail = detail
        self.publisher_events.publish(event)


def main(args=None):
    """Spin the director until Ctrl-C."""
    rclpy.init(args=args)
    node = ExperimentDirector()
    try:
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        node.stop_bag()
        rclpy.try_shutdown()
