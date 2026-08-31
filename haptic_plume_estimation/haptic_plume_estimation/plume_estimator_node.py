#!/usr/bin/env python3
"""
MultiPLE estimator: Phase A's particle filter, live on ROS.

This node is a shell. Every line of algorithm already exists and is tested in
consume.py, particle_filter.py and sensor_compensation.py; the node owns no
state of its own beyond a decimation counter. Anything else it invents is
state that an offline bag replay cannot reconstruct, and D7 dies with it.

RATE COUPLING -- the design point this node turns on. gas_field_server builds
its lag with dt = 1 / publish_rate (20 Hz), and inverting Eq. (24) requires the
same discretization that generated it. So the compensator runs on EVERY sample
at 20 Hz, and the filter is fed every 4th compensated sample at the 5 Hz Phase
A validated. Decimate, never average: averaging four samples cuts the noise by
2x while q = 0.1 stays tuned to the per-sample sigma_noise, and a quieter
signal under an unchanged likelihood scale is an over-confident filter -- a
confident wrong well, which is exactly what risk R1 is about.

No /hp/odom subscription: GasConcentration already carries the sample position,
so measurement and pose arrive married in one message. That is why the field
exists; do not add a message_filters sync against odom.

Subscription Topics:
    /hp/gas/measurement (haptic_plume_interfaces/GasConcentration): Lagged, noisy sample

Publishing Topics:
    /hp/gas/measurement_compensated (haptic_plume_interfaces/GasConcentration): Eq. (28), 20 Hz
    /hp/belief/summary (haptic_plume_interfaces/BeliefSummary): Belief mean, sigma and status, 5 Hz
    /hp/belief/particles (haptic_plume_interfaces/ParticleSet): Raw cloud for RViz, 1 Hz
    /hp/sources/consumed (haptic_plume_interfaces/SourceEstimateArray): Consumed set, latched
    /hp/events/source_consumed (haptic_plume_interfaces/TaskEvent): One event per consume

Services:
    /hp/reset_estimator (haptic_plume_interfaces/ResetEstimator): Reset mid- or between trials

ROS Parameters:
    sensor_rate [Hz]: Arrival rate of /hp/gas/measurement; must equal the gas server's
    pf_decimation [-]: Feed the filter every Nth sample; 4 gives Phase A's 5 Hz
    rate_tolerance [-]: Fractional period mismatch that triggers the rate warning
    particles_rate [Hz]: Publish rate of the visualization cloud
    n_particles [-]: Particle count; 1500 is the validated value, not the dataclass default
    q [kg/m^3]: Likelihood scale of Eq. (10)
    sigma_c [m]: Consume threshold of Eq. (11); 0.5, not the paper's over-consuming 0.7
    d_cluster [m]: Agglomerative merge distance of Algorithm 1
    max_sources [-]: Refuse to keep consuming past this many
    pf_seed [-]: Particle filter Generator seed, logged at startup for D7 replay
    tau_rise [s]: Sensor rise time constant, Eq. (24)
    tau_rec [s]: Sensor recovery time constant
    gamma [rad/s]: Compensator low-pass coefficient, Eq. (27)
    map_frame: Frame stamped on every published message

:author: premmm
:date: August 28, 2026
"""

from geometry_msgs.msg import Point

from haptic_plume_estimation.clustering import cluster_labels

from haptic_plume_estimation.consume import ConsumedSet, should_consume
from haptic_plume_estimation.particle_filter import (
    PlumeFilterConfig,
    PlumeParticleFilter,
    STATUS_CONVERGING,
    STATUS_DIVERGED,
    STATUS_INIT,
    STATUS_RESET,
    STATUS_TRACKING,
)
from haptic_plume_estimation.sensor_compensation import InverseFeedforwardCompensator

from haptic_plume_interfaces.msg import (
    BeliefSummary,
    GasConcentration,
    ParticleSet,
    SourceEstimate,
    SourceEstimateArray,
    TaskEvent,
)
from haptic_plume_interfaces.srv import ResetEstimator

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

# The filter reports status as a string; BeliefSummary carries a uint8. One
# explicit table, so a renamed constant fails loudly instead of silently
# publishing INIT forever.
STATUS_TO_MSG = {
    STATUS_INIT: BeliefSummary.INIT,
    STATUS_TRACKING: BeliefSummary.TRACKING,
    STATUS_CONVERGING: BeliefSummary.CONVERGING,
    STATUS_RESET: BeliefSummary.RESET,
    STATUS_DIVERGED: BeliefSummary.DIVERGED,
}

# Rolling window for the arrival-rate check, in samples. Two seconds at 20 Hz:
# long enough that one late message does not trip it, short enough to notice a
# genuine rate change before a whole trial is logged against a bad estimate.
RATE_WINDOW = 40


class PlumeEstimator(Node):
    """Runs the MultiPLE filter on the live gas measurement and publishes the belief."""

    def __init__(self):
        """Declare parameters, build the estimator chain, and wire the interface."""
        super().__init__('plume_estimator')

        # Parameters
        # Every default mirrors config/estimator_params.yaml, because `ros2 run`
        # without --params-file never reads that file -- and the C2 checkpoint
        # starts the node exactly that way. If the two disagree the node behaves
        # differently depending on how it was launched.
        self.declare_parameter('sensor_rate', 20.0)
        self.declare_parameter('pf_decimation', 4)
        self.declare_parameter('rate_tolerance', 0.05)
        self.declare_parameter('particles_rate', 1.0)
        self.declare_parameter('n_particles', 1500)
        self.declare_parameter('q', 0.1)
        self.declare_parameter('sigma_c', 0.5)
        self.declare_parameter('d_cluster', 1.8)
        self.declare_parameter('max_sources', 6)
        self.declare_parameter('pf_seed', 0)
        self.declare_parameter('tau_rise', 3.0)
        self.declare_parameter('tau_rec', 8.0)
        self.declare_parameter('gamma', 5.0)
        self.declare_parameter('map_frame', 'map')

        self.sensor_rate = self.get_parameter('sensor_rate').value
        self.pf_decimation = self.get_parameter('pf_decimation').value
        self.rate_tolerance = self.get_parameter('rate_tolerance').value
        self.particles_rate = self.get_parameter('particles_rate').value
        self.sigma_c = self.get_parameter('sigma_c').value
        self.d_cluster = self.get_parameter('d_cluster').value
        self.max_sources = self.get_parameter('max_sources').value
        self.map_frame = self.get_parameter('map_frame').value
        self.pf_seed = self.get_parameter('pf_seed').value

        # Estimator chain
        # PlumeFilterConfig's dataclass defaults are the PAPER's: n_particles
        # 3500, sigma_c 0.7. Phase A never ran those. sigma_c 0.7 over-consumes
        # on this noisier chain -- 2 true leaks, 3.0 found, 17% success at 240 s.
        # Pass every value explicitly; a bare PlumeFilterConfig() is a silent
        # regression, not a crash.
        self.filter_config = PlumeFilterConfig(
            n_particles=self.get_parameter('n_particles').value,
            q=self.get_parameter('q').value,
            sigma_c=self.sigma_c,
            d_cluster=self.d_cluster)
        self.pf = PlumeParticleFilter(self.filter_config,
                                      np.random.default_rng(self.pf_seed))
        self.consumed = ConsumedSet()
        self.consume_sigmas = []

        # The compensator inverts the sensor's own discretization, so it is
        # built with the SENSOR's dt and update() is always called with that
        # nominal step -- never a measured wall-clock delta. Same discipline as
        # B3's fixed-dt integrator and B5's nominal-dt lag.
        self.sensor_dt = 1.0 / self.sensor_rate
        self.compensator = InverseFeedforwardCompensator(
            self.get_parameter('tau_rise').value,
            self.get_parameter('tau_rec').value,
            self.sensor_dt,
            self.get_parameter('gamma').value)

        # State. The two counters are the ONLY algorithm state the node owns;
        # everything else lives in the library so a bag replay can rebuild it.
        self.sample_count = 0
        self.update_count = 0
        self.particles_every = max(
            1, round(self.sensor_rate / self.pf_decimation / self.particles_rate))
        self.last_arrival = None
        self.arrival_periods = []

        # Publishers
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        cloud_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.publisher_compensated = self.create_publisher(
            GasConcentration, '/hp/gas/measurement_compensated', 10)
        self.publisher_summary = self.create_publisher(
            BeliefSummary, '/hp/belief/summary', 10)
        self.publisher_particles = self.create_publisher(
            ParticleSet, '/hp/belief/particles', cloud_qos)
        self.publisher_consumed = self.create_publisher(
            SourceEstimateArray, '/hp/sources/consumed', latched_qos)
        self.publisher_event = self.create_publisher(
            TaskEvent, '/hp/events/source_consumed', 10)

        # Subscribers
        self.subscription_measurement = self.create_subscription(
            GasConcentration,
            '/hp/gas/measurement',
            self.measurement_callback,
            10)

        # Services
        self.service_reset = self.create_service(
            ResetEstimator, '/hp/reset_estimator', self.reset_callback)

        # Latch the empty set once, so a subscriber that starts late gets an
        # explicit "nothing consumed yet" instead of silence.
        self.publish_consumed()

        # D7: to replay headless trial s, pass pf_seed = s + 10000 --
        # eval_estimator.py:214 derives its filter stream that way.
        self.get_logger().info(
            f'plume_estimator up: pf_seed={self.pf_seed}, '
            f'{self.filter_config.n_particles} particles, sigma_c={self.sigma_c} m, '
            f'compensating at {self.sensor_rate} Hz, filtering at '
            f'{self.sensor_rate / self.pf_decimation} Hz')

    def measurement_callback(self, msg):
        """Compensate every sample, feed the filter every Nth, and test for a consume."""
        self.check_arrival_rate()

        # Tick order mirrors eval_estimator.run_trial exactly, for D7:
        # compensate -> decimate -> subtract consumed -> update -> consume test.
        z_ff = self.compensator.update(msg.concentration)
        self.publish_compensated(msg, z_ff)

        self.sample_count += 1
        if self.sample_count % self.pf_decimation:
            return

        position = np.array([msg.position.x, msg.position.y, msg.position.z])
        m_hat = float(self.consumed.concentration(*position))
        self.pf.update(position, z_ff, m_hat=m_hat)
        self.update_count += 1

        if (should_consume(self.pf.sigma_position_norm, self.sigma_c)
                and len(self.consumed) < self.max_sources):
            self.do_consume(msg)

        # pf.status is read AFTER the consume block on purpose: pf.reset() has
        # run by then, so the status always describes the same belief as the
        # mean and sigma published beside it. Reporting the pre-consume
        # CONVERGING next to a fresh uniform prior is exactly the "confident
        # well over a reset belief" force pop that risk R5 is about.
        self.publish_summary(msg)
        if self.update_count % self.particles_every == 0:
            self.publish_particles(msg)

    def do_consume(self, msg):
        """Turn a converged belief into a consumed source. The order is load-bearing."""
        alpha = self.pf.mmse

        # 1. The event goes out BEFORE pf.reset(). The composer's fade-out
        #    starts on this event, and it must never meet a reset belief
        #    without having been told why.
        event = TaskEvent()
        event.header.stamp = msg.header.stamp
        event.header.frame_id = self.map_frame
        event.event_type = TaskEvent.SOURCE_CONSUMED
        event.position = Point(x=float(alpha[0]), y=float(alpha[1]), z=float(alpha[2]))
        event.detail = f'sigma_position_norm={self.pf.sigma_position_norm:.3f} m'
        self.publisher_event.publish(event)

        self.consumed.push(alpha)
        self.consume_sigmas.append(self.pf.sigma_position_norm)

        # merge() collapses duplicates but does not report which entry went
        # where, so the per-source sigma would be lost. Regroup with the same
        # cluster_labels call agglomerative_merge makes -- same labels, same
        # first-appearance order -- and keep the tightest belief that fed each
        # surviving source.
        labels = cluster_labels(self.consumed.alphas, self.d_cluster)
        self.consume_sigmas = [
            min(sigma for sigma, label in zip(self.consume_sigmas, labels)
                if label == index)
            for index in range(labels.max() + 1)]
        self.consumed.merge(self.d_cluster)

        self.publish_consumed()
        self.pf.reset()

        self.get_logger().info(
            f'consumed source {len(self.consumed)} at '
            f'({alpha[0]:.2f}, {alpha[1]:.2f}) m, Q={alpha[4]:.4f} kg/s')

    def check_arrival_rate(self):
        """Warn if the sensor's real period disagrees with the dt the compensator inverts."""
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_arrival is not None:
            self.arrival_periods.append(now - self.last_arrival)
            if len(self.arrival_periods) > RATE_WINDOW:
                self.arrival_periods.pop(0)
            if len(self.arrival_periods) == RATE_WINDOW:
                observed = sum(self.arrival_periods) / RATE_WINDOW
                error = abs(observed - self.sensor_dt) / self.sensor_dt
                if error > self.rate_tolerance:
                    self.get_logger().warn(
                        f'measurement period {observed * 1e3:.1f} ms differs by '
                        f'{error * 100:.0f}% from the {self.sensor_dt * 1e3:.1f} ms the '
                        f'compensator inverts. Eq. (24) cancels only at the step that '
                        f'generated the lag; the residual displaces the estimated source '
                        f'DOWNWIND and reads as a plume-model error. Check sensor_rate '
                        f'against gas_field_server publish_rate.',
                        throttle_duration_sec=10.0)
        self.last_arrival = now

    def publish_compensated(self, msg, z_ff):
        """Republish one sample with Eq. (28) applied, at the full sensor rate."""
        out = GasConcentration()
        out.header = msg.header
        out.concentration = float(z_ff)
        out.position = msg.position
        self.publisher_compensated.publish(out)

    def publish_summary(self, msg):
        """Publish the belief: mmse for every logged metric, mne_position for the well."""
        summary = BeliefSummary()
        summary.header.stamp = msg.header.stamp
        summary.header.frame_id = self.map_frame
        summary.mean = self.pf.mmse.tolist()
        summary.sigma = self.pf.sigma.tolist()
        summary.sigma_position_norm = self.pf.sigma_position_norm
        summary.n_eff = float(self.pf.n_eff)
        summary.status = STATUS_TO_MSG[self.pf.status]
        summary.n_consumed = len(self.consumed)
        target = self.pf.mne_position
        summary.haptic_target = Point(x=float(target[0]), y=float(target[1]),
                                      z=float(target[2]))
        summary.well_confidence = self.well_confidence()
        self.publisher_summary.publish(summary)

    def well_confidence(self):
        """Map belief tightness onto [0, 1]. PROVISIONAL -- see the field's comment."""
        # C0 measured the != DIVERGED gate as on from t = 0 and wrong 66-87% of
        # rendered time, with 40-78% of hunts never becoming usable at all. This
        # ratio is a placeholder that at least falls to 0 on a diffuse belief;
        # risk R1's third gate must replace it before C4 renders a narrow well.
        # Any replacement has to be dimensionless or window-normalized and
        # validated on >=2-source worlds -- both earlier attempts died there.
        if self.pf.status == STATUS_DIVERGED:
            return 0.0
        return float(np.clip(self.sigma_c / max(self.pf.sigma_position_norm, 1e-6),
                             0.0, 1.0))

    def publish_particles(self, msg):
        """Publish the raw cloud for RViz: float32, best effort, dropped freely."""
        cloud = ParticleSet()
        cloud.header.stamp = msg.header.stamp
        cloud.header.frame_id = self.map_frame
        cloud.n_particles = self.pf.particles.shape[0]
        cloud.n_params = self.pf.particles.shape[1]
        cloud.particles = self.pf.particles.astype(np.float32).ravel().tolist()
        cloud.weights = self.pf.weights.astype(np.float32).tolist()
        self.publisher_particles.publish(cloud)

    def publish_consumed(self):
        """Latch the consumed set. alpha order is PLUME_PARAM_NAMES: theta is 3, Q is 4."""
        array = SourceEstimateArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.header.frame_id = self.map_frame
        for index, alpha in enumerate(self.consumed.alphas):
            source = SourceEstimate()
            source.position = Point(x=float(alpha[0]), y=float(alpha[1]),
                                    z=float(alpha[2]))
            source.wind_direction = float(alpha[3])
            source.emission_rate = float(alpha[4])
            source.wind_speed = float(alpha[5])
            source.d_y = float(alpha[6])
            source.d_z = float(alpha[7])
            source.sigma_position_norm = self.consume_sigmas[index]
            source.id = index
            array.sources.append(source)
        self.publisher_consumed.publish(array)

    def reset_callback(self, request, response):
        """Reset the filter, and the compensator's lag state along with it."""
        if request.pf_seed >= 0:
            # Rebuild rather than reseed in place: pf.reset() deliberately keeps
            # its generator, and a fresh object also zeroes n_updates so a new
            # trial reports INIT instead of RESET.
            self.pf = PlumeParticleFilter(self.filter_config,
                                          np.random.default_rng(request.pf_seed))
            self.pf_seed = request.pf_seed
        else:
            self.pf.reset()

        # run_trial builds a fresh compensator per trial, so it never had to
        # reset one; a long-lived node does. Without this, the first samples of
        # trial N+1 are compensated against trial N's z_prev and the belief
        # opens on a lag that is not there.
        self.compensator.reset()
        self.sample_count = 0
        self.update_count = 0

        if request.clear_consumed:
            self.consumed.clear()
            self.consume_sigmas.clear()
            self.publish_consumed()

        response.success = True
        response.message = (f'reset: pf_seed={self.pf_seed}, '
                            f'{len(self.consumed)} consumed source(s) retained')
        self.get_logger().info(response.message)
        return response


def main(args=None):
    """Initialize and run the plume estimator."""
    rclpy.init(args=args)
    try:
        node = PlumeEstimator()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
