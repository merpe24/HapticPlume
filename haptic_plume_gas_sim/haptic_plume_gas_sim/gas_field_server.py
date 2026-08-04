#!/usr/bin/env python3
"""
Ground-truth gas field server: the Phase A truth chain, live on ROS.

This node holds one Scenario and, once per tick:
    - samples the true concentration at the drone's current position
    - runs it through the validated fluctuation -> lag -> noise sensor chain
    - publishes both the measurement and the ground truth
    - draws the field and the airflow arrows for RViz

sensor_model.py is imported, never modified: it stays ROS-free. That separation
is Phase A's rule and is what keeps the GADEN swap (Phase F) open and limits the
inverse-crime coupling of risk R8.

RATE COUPLING -- the one real design point of this node.
FirstOrderSensor discretizes paper Eq. (24) by backward Euler at a nominal dt,
and haptic_plume_estimation.sensor_compensation inverts that SAME
discretization. They only cancel if they use the same step, so the lag here is
built with dt = 1 / publish_rate, and update() is always called with that
nominal dt rather than a measured wall-clock delta -- the same reason
drone_kinematics_node integrates at a fixed dt.

    ==> Phase C consequence: the compensator must run at the SENSOR's rate
        (publish_rate, 20 Hz) even though the particle filter updates at 5 Hz.
        Compensate every sample at 20 Hz, then decimate to feed the filter.
        Compensating at 5 Hz a signal generated at 20 Hz leaves residual lag,
        and residual lag looks exactly like a source displaced downwind.

GROUND TRUTH WARNING.
/hp/gas/true_concentration and /hp/gas/field_markers are ground truth. They are
debug and logging channels only, and nothing on them may reach the operator
during a trial: decision D6 makes viz_gate_node the only gate, and the
operator's RViz subscribes to /hp/display/* alone.

Subscription Topics:
    /hp/odom (nav_msgs/Odometry): Drone pose; the point the sensor samples

Publishing Topics:
    /hp/gas/measurement (haptic_plume_interfaces/GasConcentration): Lagged, noisy sample
    /hp/gas/true_concentration (haptic_plume_interfaces/GasConcentration): Ground truth
    /hp/gas/field_markers (visualization_msgs/MarkerArray): Field + airflow, latched

Services:
    /hp/set_scenario (haptic_plume_interfaces/SetScenario): Rebuild the world from a seed

:author: premmm
:date: August 4, 2026
"""

import math

from geometry_msgs.msg import Point

from haptic_plume_core.scenario import random_scenario

from haptic_plume_gas_sim.sensor_model import FirstOrderSensor, GasField, GasSensor

from haptic_plume_interfaces.msg import GasConcentration
from haptic_plume_interfaces.srv import SetScenario

from nav_msgs.msg import Odometry

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from std_msgs.msg import ColorRGBA

from visualization_msgs.msg import Marker, MarkerArray

# Marker namespaces, so RViz can toggle the two layers independently.
MARKER_NS_FIELD = 'gas_field'
MARKER_NS_AIRFLOW = 'airflow'

# Colours are fixed here, not exposed as parameters: they carry no information.
FIELD_COLOR = (1.0, 0.35, 0.0)      # orange gas cloud
AIRFLOW_COLOR = (0.2, 0.6, 1.0)     # blue airflow arrows


class GasFieldServer(Node):
    """Samples the true gas field at the drone and publishes a lagged, noisy reading."""

    def __init__(self):
        """Declare parameters, build the first scenario, and wire the interface."""
        super().__init__('gas_field_server')

        # --- Parameters -----------------------------------------------------
        # The sensor constants are the ones the Phase A suite was validated
        # with (EvalConfig, eval_estimator.py:86-90). Changing them here
        # silently invalidates that baseline.
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('n_sources', 1)
        self.declare_parameter('seed', 42)
        self.declare_parameter('tau_rise', 3.0)
        self.declare_parameter('tau_rec', 8.0)
        self.declare_parameter('sigma_noise', 2.0e-3)
        self.declare_parameter('sigma_fluct', 0.2)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('marker_altitude', 1.0)
        self.declare_parameter('marker_grid_step', 0.25)
        self.declare_parameter('marker_c0', 2.0e-3)
        self.declare_parameter('marker_alpha_gain', 0.25)
        self.declare_parameter('marker_alpha_min', 0.02)
        self.declare_parameter('marker_airflow_length', 3.0)

        self.publish_rate = self.get_parameter('publish_rate').value
        self.tau_rise = self.get_parameter('tau_rise').value
        self.tau_rec = self.get_parameter('tau_rec').value
        self.sigma_noise = self.get_parameter('sigma_noise').value
        self.sigma_fluct = self.get_parameter('sigma_fluct').value
        self.map_frame = self.get_parameter('map_frame').value
        self.marker_altitude = self.get_parameter('marker_altitude').value
        self.marker_grid_step = self.get_parameter('marker_grid_step').value
        self.marker_c0 = self.get_parameter('marker_c0').value
        self.marker_alpha_gain = self.get_parameter('marker_alpha_gain').value
        self.marker_alpha_min = self.get_parameter('marker_alpha_min').value
        self.marker_airflow_length = self.get_parameter('marker_airflow_length').value

        if self.publish_rate <= 0.0:
            raise ValueError('publish_rate must be positive')

        # The lag's nominal dt MUST equal the publish period -- see the module
        # docstring. Deriving it here is what stops the two drifting apart.
        self.dt = 1.0 / self.publish_rate

        # --- State ----------------------------------------------------------
        # Filled by _apply_scenario below; the timer refuses to sample until an
        # odom message has actually arrived.
        self.scenario = None
        self.field = None
        self.sensor = None
        self.latest_odom = None

        # --- Publishers -----------------------------------------------------
        self.publisher_measurement = self.create_publisher(
            GasConcentration, '/hp/gas/measurement', 10)

        self.publisher_true = self.create_publisher(
            GasConcentration, '/hp/gas/true_concentration', 10)

        # Latched: the markers change only when the scenario does, so RViz must
        # be able to start late and still receive the current field.
        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher_markers = self.create_publisher(
            MarkerArray, '/hp/gas/field_markers', latched_qos)

        # --- Subscriber -----------------------------------------------------
        # The drone is the pose authority (decision D1); we only read it.
        self.subscription_odom = self.create_subscription(
            Odometry,
            '/hp/odom',
            self.odom_callback,
            10)

        # --- Service --------------------------------------------------------
        self.service_set_scenario = self.create_service(
            SetScenario, '/hp/set_scenario', self.set_scenario_callback)

        # --- Build the startup scenario and start ticking --------------------
        self._apply_scenario(
            self.get_parameter('n_sources').value,
            self.get_parameter('seed').value)

        self.timer = self.create_timer(self.dt, self.sensor_callback)

        self.get_logger().info(
            f'gas_field_server up at {self.publish_rate:.1f} Hz (sensor dt {self.dt:.4f} s)')

    def _apply_scenario(self, n_sources, seed):
        """
        Rebuild the world, the sensor, and the markers from one seed.

        Called from the constructor and from the /hp/set_scenario callback. The
        node spins single-threaded, so this swap can never interleave with a
        timer tick -- that is what makes it atomic without a lock.

        :param n_sources: number of leaks to place, >= 1
        :param seed: integer seed; the same seed always yields the same world
        :return: the scenario now in force
        :rtype: haptic_plume_core.scenario.Scenario
        """
        scenario = random_scenario(int(n_sources), int(seed))

        # Seeding the sensor stream from the scenario seed is exactly what the
        # Phase A trial loop does (eval_estimator.py:146), so a live run of
        # seed s meets the same noise realisation as the headless study did.
        # A fresh FirstOrderSensor also starts at z = 0, which is the per-trial
        # reset Phase E needs.
        sensor = GasSensor(
            FirstOrderSensor(self.tau_rise, self.tau_rec, self.dt),
            np.random.default_rng(int(seed)),
            sigma_noise=self.sigma_noise,
            sigma_fluct=self.sigma_fluct)

        self.scenario = scenario
        self.field = GasField(scenario.alphas)
        self.sensor = sensor

        self.publish_field_markers()

        self.get_logger().info(
            f'scenario {scenario.name}: {scenario.n_sources} source(s), '
            f'theta {scenario.alphas[0, 3]:.3f} rad, v {scenario.alphas[0, 5]:.3f} m/s')
        return scenario

    def odom_callback(self, msg):
        """Store the latest drone pose; the timer decides when to sample it."""
        self.latest_odom = msg

    def sensor_callback(self):
        """Sample the field at the drone's position and publish one measurement."""
        if self.latest_odom is None:
            self.get_logger().warn(
                'no /hp/odom received yet -- not sampling',
                throttle_duration_sec=5.0)
            return

        position = self.latest_odom.pose.pose.position

        # The odom pose is expressed in the odom frame, and map -> odom is the
        # identity transform (frames.md 1), so these numbers are already map
        # coordinates. If that ever stops being identity, transform here.
        c_true = float(self.field.concentration(position.x, position.y, position.z))

        # Nominal dt on purpose: a measured wall-clock delta would break the
        # exact round trip with the Phase C compensator (module docstring).
        c_measured = self.sensor.measure(c_true)

        stamp = self.get_clock().now().to_msg()
        self.publisher_measurement.publish(self._gas_msg(stamp, position, c_measured))
        self.publisher_true.publish(self._gas_msg(stamp, position, c_true))

    def set_scenario_callback(self, request, response):
        """Regenerate the world from the requested seed and report the ground truth."""
        try:
            scenario = self._apply_scenario(request.n_sources, request.seed)
        except (ValueError, RuntimeError) as exc:
            # random_scenario raises ValueError for n_sources < 1 and
            # RuntimeError when rejection sampling cannot separate the sources.
            # The previous scenario is left untouched, so the node keeps running.
            response.success = False
            response.message = str(exc)
            self.get_logger().error(f'set_scenario rejected: {exc}')
            return response

        response.success = True
        response.message = (
            f'scenario {scenario.name} active with {scenario.n_sources} source(s)')
        response.scenario_name = scenario.name

        # random_scenario defaults to shared_wind=True, so every plume carries
        # the same theta and v and row 0 speaks for the whole field.
        response.wind_direction = float(scenario.alphas[0, 3])
        response.wind_speed = float(scenario.alphas[0, 5])
        response.true_source_positions = [
            Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
            for p in scenario.source_positions
        ]
        return response

    def publish_field_markers(self):
        """
        Draw the true field and the airflow arrows for RViz.

        GROUND TRUTH -- debug only, never routed to the operator (decision D6).

        This is also the R4 acceptance check (frames.md 4.1): the cubes must
        trail in the direction each arrow points, with nothing on the far side.
        """
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()

        # A scenario with fewer sources than the last would otherwise leave
        # orphaned arrows on screen, and RViz keeps them forever.
        clear = Marker()
        clear.header.frame_id = self.map_frame
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        markers.markers.append(self._field_marker(stamp))
        markers.markers.extend(self._airflow_markers(stamp))

        self.publisher_markers.publish(markers)

    def _field_marker(self, stamp):
        """
        Build the CUBE_LIST of the concentration field at the operating altitude.

        :param stamp: header stamp shared by every marker in this array
        :return: one CUBE_LIST marker, transparent where there is no gas
        :rtype: visualization_msgs.msg.Marker
        """
        area = self.scenario.search_area
        x_axis = np.arange(area[0, 0], area[0, 1] + 1e-9, self.marker_grid_step)
        y_axis = np.arange(area[1, 0], area[1, 1] + 1e-9, self.marker_grid_step)
        grid_x, grid_y = np.meshgrid(x_axis, y_axis, indexing='ij')
        grid_z = np.full_like(grid_x, self.marker_altitude)

        concentration = np.asarray(self.field.concentration(grid_x, grid_y, grid_z))

        # Decision D5's log law, with globally fixed constants. Per-scenario
        # normalisation is forbidden: it would make the visual cue depend on
        # which trial the operator is flying, which is a confound.
        alpha = np.clip(
            self.marker_alpha_gain * np.log1p(concentration / self.marker_c0), 0.0, 1.0)
        visible = alpha >= self.marker_alpha_min

        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = stamp
        marker.ns = MARKER_NS_FIELD
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0     # RViz warns on an all-zero quaternion
        marker.scale.x = self.marker_grid_step
        marker.scale.y = self.marker_grid_step
        marker.scale.z = self.marker_grid_step

        for x, y, a in zip(grid_x[visible], grid_y[visible], alpha[visible]):
            marker.points.append(
                Point(x=float(x), y=float(y), z=float(self.marker_altitude)))
            marker.colors.append(ColorRGBA(
                r=FIELD_COLOR[0], g=FIELD_COLOR[1], b=FIELD_COLOR[2], a=float(a)))

        return marker

    def _airflow_markers(self, stamp):
        """
        Build one airflow arrow per source, for the R4 sign check.

        :param stamp: header stamp shared by every marker in this array
        :return: list of ARROW markers, one per true source
        :rtype: list
        """
        markers = []
        for index, alpha in enumerate(self.scenario.alphas):
            x_s, y_s, z_s = float(alpha[0]), float(alpha[1]), float(alpha[2])
            theta = float(alpha[3])

            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = stamp
            marker.ns = MARKER_NS_AIRFLOW
            marker.id = index
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            # For a two-point arrow, scale is (shaft diameter, head diameter,
            # head length) -- NOT the arrow length, which the points give.
            marker.scale.x = 0.08
            marker.scale.y = 0.16
            marker.scale.z = 0.25
            marker.color = ColorRGBA(
                r=AIRFLOW_COLOR[0], g=AIRFLOW_COLOR[1], b=AIRFLOW_COLOR[2], a=1.0)

            # frames.md 4: theta is the direction the airflow blows TOWARD, so
            # the arrow and the plume must trail the SAME way. If they oppose,
            # a sign is flipped -- stop and fix it before going any further.
            marker.points.append(Point(x=x_s, y=y_s, z=z_s))
            marker.points.append(Point(
                x=x_s + self.marker_airflow_length * math.cos(theta),
                y=y_s + self.marker_airflow_length * math.sin(theta),
                z=z_s))

            markers.append(marker)

        return markers

    def _gas_msg(self, stamp, position, concentration):
        """
        Fill one GasConcentration message.

        :param stamp: sample time
        :param position: geometry_msgs Point where the sample was taken [m]
        :param concentration: the value to report [kg/m^3]
        :return: the populated message
        :rtype: haptic_plume_interfaces.msg.GasConcentration
        """
        msg = GasConcentration()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        msg.concentration = float(concentration)
        msg.position.x = float(position.x)
        msg.position.y = float(position.y)
        msg.position.z = float(position.z)
        return msg


def main(args=None):
    """Initialize and run the gas field server."""
    rclpy.init(args=args)
    try:
        node = GasFieldServer()
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
