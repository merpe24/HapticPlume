#!/usr/bin/env python3
"""
Run the experiment's trials: reset everything, then switch one condition on.

Publishing Topics:
    /hp/condition (haptic_plume_interfaces/ConditionCondfig): Trial condition and flags
    
:author: premmm
:date: October 26, 2026
"""

from haptic_plume_interfaces.msg import ConditionConfig

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

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


def main(args=None):
    """Spin the director until Ctrl-C."""
    rclpy.init(args=args)
    node = ExperimentDirector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        