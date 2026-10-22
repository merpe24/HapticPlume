#!/usr/bin/env python3
"""
Draw the particle filter's belief in RViz for the operator.

Subscription Topics:
    /hp/belief/summary (haptic_plume_interfaces/BeliefSummary): Belief and its best guess, 5 Hz
    /hp/belief/particles (haptic_plume_interfaces/ParticleSet): Every particle, 1 Hz
    /hp/gas/measurement_compensated (haptic_plume_interfaces/GasConcentration): Lag-corrected gas
    /hp/condition (haptic_plume_interfaces/ConditionConfig): Trial condition and flags

Publishing Topics:
    /hp/display/belief (visualization_msgs/Marker): Ball on best guess, ring sized by spread
    /hp/display/particles (visualization_msgs/Marker): A dot for every particle
    /hp/display/gauge (visualization_msgs/Marker): Bar on the drone, taller = more gas
    /hp/display/wind (visualization_msgs/Marker): Arrow pointing downwind

:author: premmm
:date: October 14, 2026
"""

import math

from builtin_interfaces.msg import Duration

from geometry_msgs.msg import Point

from haptic_plume_interfaces.msg import (
    BeliefSummary, ConditionConfig, GasConcentration, ParticleSet)

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from std_msgs.msg import ColorRGBA

from visualization_msgs.msg import Marker

GAUGE_FULL = 0.30           # [kg/m^3] max concentration
GAUGE_HEIGHT = 2.0          # [m] max bar height
WIND_ARROW_LENGTH = 3.0     # [m]


def status_colour(status):
    """Pick the ring colour from the filter's status."""
    colour = ColorRGBA()
    colour.a = 1.0
    if status == BeliefSummary.DIVERGED:
        colour.r = colour.g = colour.b = 0.6   # grey: diverge
    elif status == BeliefSummary.CONVERGING:
        colour.r = colour.g = 1.0              # yellow: converge
    else:
        colour.r = colour.b = 1.0              # magenta: searching
    return colour


class VizGate(Node):
    """Turn belief messages into RViz markers."""

    def __init__(self):
        """Set up one subscriber and one publisher."""
        super().__init__('viz_gate')
        self.publisher_belief = self.create_publisher(
            Marker, '/hp/display/belief', 10)
        self.subscription_summary = self.create_subscription(
            BeliefSummary, '/hp/belief/summary', self.summary_callback, 10)

        cloud_qos = QoSProfile(depth=1,
                               reliability=ReliabilityPolicy.BEST_EFFORT)
        self.publisher_particles = self.create_publisher(
            Marker, '/hp/display/particles', 10)
        self.subscription_particles = self.create_subscription(
            ParticleSet, '/hp/belief/particles', self.particles_callback, cloud_qos)

        self.publisher_gauge = self.create_publisher(
            Marker, '/hp/display/gauge', 10)
        self.subscription_gauge = self.create_subscription(
            GasConcentration, '/hp/gas/measurement_compensated', self.gauge_callback, 10)

        self.publisher_wind = self.create_publisher(
            Marker, '/hp/display/wind', 10)

        self.condition = None
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.subscription_condition = self.create_subscription(
            ConditionConfig, '/hp/condition', self.condition_callback, latched_qos)

    def condition_callback(self, condition):
        """Remember the trial's condition; every layer checks it before drawing."""
        if condition != self.condition:
            self.get_logger().info(
                f'condition {condition.condition}, trial "{condition.trial_id}"')
        self.condition = condition

    def visual_on(self):
        """Return True if a condition has arrived and it is not haptic-only."""
        return (self.condition is not None
                and self.condition.condition != ConditionConfig.HAPTIC_ONLY)

    def summary_callback(self, summary):
        """Put a red ball on the filter's best guess of the leak."""
        if not self.visual_on():
            return
        if self.condition.show_wind_arrow:
            self.publish_wind_arrow(summary)
        if not self.condition.show_belief_markers:
            return

        ball = Marker()
        ball.header.frame_id = 'map'
        ball.header.stamp = summary.header.stamp
        ball.ns = 'belief'
        ball.id = 0
        ball.type = Marker.SPHERE
        ball.action = Marker.ADD
        ball.pose.position = summary.haptic_target
        ball.pose.orientation.w = 1.0
        ball.scale.x = ball.scale.y = ball.scale.z = 0.4
        ball.color.r = 1.0
        ball.color.a = 1.0
        ball.lifetime = Duration(sec=1)
        self.publisher_belief.publish(ball)

        ring = Marker()
        ring.header.frame_id = 'map'
        ring.header.stamp = summary.header.stamp
        ring.ns = 'confidence'
        ring.id = 0
        ring.type = Marker.LINE_STRIP
        ring.action = Marker.ADD
        ring.pose.orientation.w = 1.0
        ring.scale.x = 0.1
        ring.color = status_colour(summary.status)
        radius = summary.sigma_position_norm
        centre = summary.haptic_target
        for k in range(65):
            angle = 2.0 * math.pi * k / 64
            point = Point()
            point.x = centre.x + radius * math.cos(angle)
            point.y = centre.y + radius * math.sin(angle)
            point.z = centre.z + 0.05
            ring.points.append(point)
        ring.lifetime = Duration(sec=1)
        self.publisher_belief.publish(ring)

    def publish_wind_arrow(self, summary):
        """Draw an arrow from the believed leak along the believed airflow."""
        theta = summary.mean[3]
        start = Point()
        start.x = summary.haptic_target.x
        start.y = summary.haptic_target.y
        start.z = summary.haptic_target.z + 0.1
        end = Point()
        end.x = start.x + WIND_ARROW_LENGTH * math.cos(theta)
        end.y = start.y + WIND_ARROW_LENGTH * math.sin(theta)
        end.z = start.z

        arrow = Marker()
        arrow.header.frame_id = 'map'
        arrow.header.stamp = summary.header.stamp
        arrow.ns = 'wind'
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose.orientation.w = 1.0
        arrow.points = [start, end]
        arrow.scale.x = 0.12
        arrow.scale.y = 0.3
        arrow.scale.z = 0.4
        arrow.color.g = 0.8
        arrow.color.b = 1.0
        arrow.color.a = 1.0
        arrow.lifetime = Duration(sec=1)
        self.publisher_wind.publish(arrow)

    def particles_callback(self, cloud):
        """Put one small green dot on every particle's guess of the leak."""
        if not self.visual_on() or not self.condition.show_particles:
            return

        dots = Marker()
        dots.header.frame_id = 'map'
        dots.header.stamp = cloud.header.stamp
        dots.ns = 'particles'
        dots.id = 0
        dots.type = Marker.POINTS
        dots.action = Marker.ADD
        dots.pose.orientation.w = 1.0
        dots.scale.x = dots.scale.y = 0.05
        dots.color.g = 1.0
        dots.color.a = 0.7
        for i in range(cloud.n_particles):
            start = i * cloud.n_params
            dot = Point()
            dot.x = float(cloud.particles[start])
            dot.y = float(cloud.particles[start + 1])
            dot.z = float(cloud.particles[start + 2])
            dots.points.append(dot)
        dots.lifetime = Duration(sec=2)
        self.publisher_particles.publish(dots)

    def gauge_callback(self, sample):
        """Orange bar on the drone, taller where there is more gas."""
        if not self.visual_on() or not self.condition.show_gauge:
            return

        level = min(max(sample.concentration, 0.0), GAUGE_FULL) / GAUGE_FULL
        height = max(level * GAUGE_HEIGHT, 0.01)
        top = sample.position.z + 0.3 + height

        bar = Marker()
        bar.header.frame_id = 'map'
        bar.header.stamp = sample.header.stamp
        bar.ns = 'gauge'
        bar.id = 0
        bar.type = Marker.CYLINDER
        bar.action = Marker.ADD
        bar.pose.position.x = sample.position.x
        bar.pose.position.y = sample.position.y
        bar.pose.position.z = top - height / 2.0
        bar.pose.orientation.w = 1.0
        bar.scale.x = bar.scale.y = 0.15
        bar.scale.z = height
        bar.color.r = 1.0
        bar.color.g = 0.5
        bar.color.a = 1.0
        bar.lifetime = Duration(sec=1)
        self.publisher_gauge.publish(bar)

        label = Marker()
        label.header = bar.header
        label.ns = 'gauge_value'
        label.id = 0
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = sample.position.x
        label.pose.position.y = sample.position.y
        label.pose.position.z = top + 0.3
        label.pose.orientation.w = 1.0
        label.scale.z = 0.3
        label.color.r = label.color.g = label.color.b = 1.0
        label.color.a = 1.0
        label.text = f'{sample.concentration:.3f}'
        label.lifetime = Duration(sec=1)
        self.publisher_gauge.publish(label)


def main(args=None):
    """Start the node and keep it running until Ctrl-C."""
    rclpy.init(args=args)
    node = VizGate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
