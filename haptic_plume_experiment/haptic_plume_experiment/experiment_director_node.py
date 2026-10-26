#!/usr/bin/env python3
"""
Run the experiment's trials: reset everything, then switch one condition on.

Publishing Topics:
    /hp/condition (haptic_plume_interfaces/ConditionCondfig): Trial condition and flags

Services:
    /hp/start_trial (haptci_plume_interfaces/StartTrial): Reset everything, then enable a condition

Services called:
    /hp/reset_drone (std_srvs/Trigger): Drone back to its fixed start, zero velocity
    /hp/set_scenario (haptic_plume_interfaces/SetScenario): New gas world, one leak, from the seed
    /hp/reset_estimator (haptic_plume_interfaces/ResetEstimator): Fresh belief, consumed set cleared
    
:author: premmm
:date: October 26, 2026
"""

from haptic_plume_interfaces.msg import ConditionConfig
from haptic_plume_interfaces.srv import ResetEstimator, SetScenario, StartTrial

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from std_srvs.srv import Trigger

SERVICE_TIMEOUT = 5.0   # [s] a reset step slower than this counts as failed

VISUAL_CONDITIONS = (ConditionConfig.VISUAL_ONLY, ConditionConfig.VISUAL_HAPTIC)
HAPTIC_CONDITIONS = (ConditionConfig.HAPTIC_ONLY, ConditionConfig.VISUAL_HAPTIC)

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

    def call(self, client, request):
        """Call one reset service and wait; raise if it is down, slow or refuses."""
        if not client.wait_for_service(timeout_sec=SERVICE_TIMEOUT):
            raise RuntimeError(f'{client.srv_name} is not available')
        response = client.call(request, timeout_sec=SERVICE_TIMEOUT)
        if response is None or not response.success:
            raise RuntimeError(f'{client.srv_name} failed')
        return response

    def start_trial_callback(self, request, response):
        """Zero everything, reset the world in order, then enable the condition."""
        self.publisher_condition.publish(make_idle())
        try:
            config = make_condition(request.condition, request.trial_id,
                                    request.scenario_seed, request.pf_seed)
            self.call(self.client_reset_drone, Trigger.Request())
            self.call(self.client_set_scenario, SetScenario.Request(
                n_sources=1, seed=request.scenario_seed))
            self.call(self.client_reset_estimator, ResetEstimator.Request(
                clear_consumed=True, pf_seed=request.pf_seed))
        except (ValueError, RuntimeError) as error:
            response.success = False
            response.message = f'trial not started, staying IDLE: {error}'
            self.get_logger().error(response.message)
            return response

        self.publisher_condition.publish(config)
        response.success = True
        response.message = f'trial {request.trial_id} started'
        self.get_logger().info(response.message)
        return response


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
        rclpy.try_shutdown()
