"""
Chemical-sensor model — the truth side of the measurement chain.

Turns the analytic ground-truth gas field into what the drone's sensor
actually reports. Three effects are modelled, in the order they physically
occur:

    1. Turbulence fluctuation — multiplicative lognormal (unit mean) applied
       to the true concentration. The Gaussian plume model is time-averaged,
       so a real instantaneous reading is intermittent around it.
    2. Sensor dynamics — first-order lag with SEPARATE rise and recovery time
       constants, Goodell et al. (RAS 2026) Eq. (24). Real MOX/MPS gas sensors
       recover much more slowly than they rise.
    3. Electronics noise — additive white Gaussian, applied to the lagged
       output (this is the nu[k] of paper Eq. 12).

haptic_plume_estimation undoes step 2 with the inverse-feedforward compensator
(paper Eqs. 25-28) and must never import this module: keeping the two sides
apart is what limits the inverse-crime coupling of risk R8 and keeps GADEN
swappable in Phase F.

Zero ROS dependencies — importable from plain pytest (Phase A).

Units:
    concentration [kg/m^3] · dt [s] · tau_rise, tau_rec [s] · b [-]

:author: premmm
:date: July 29, 2026
"""

from haptic_plume_core.plume_model import total_concentration

import numpy as np


class FirstOrderSensor:
    """
    First-order sensor lag with split rise/recovery time constants.

    Implements paper Eq. (24), G(s) = Z(s)/Z_true(s) = b / (tau s + 1), with
    tau = tau_rise while the output is rising and tau = tau_rec while it is
    recovering. Discretized by backward Euler, which is what the compensator
    in haptic_plume_estimation.sensor_compensation inverts exactly:

        tau (z[k] - z[k-1]) / dt + z[k] = b z_true[k]
        =>  z[k] = (b z_true[k] + a z[k-1]) / (1 + a),   a = tau / dt

    The rise/recovery branch is decided before solving: from the line above,
    z[k] - z[k-1] = (b z_true[k] - z[k-1]) / (1 + a), so the sign of the
    output derivative is the sign of (b z_true[k] - z[k-1]). The compensator
    reaches the same decision from z alone, so the two branch identically and
    the round trip stays exact.
    """

    def __init__(self, tau_rise, tau_rec, dt, b=1.0, z_init=0.0):
        """
        Construct the lag with its time constants and nominal sample time.

        :param tau_rise: rise time constant [s], used while the output climbs
        :param tau_rec: recovery time constant [s], used while it decays
        :param dt: nominal sample time [s]; may be overridden per update()
        :param b: static gain of the sensor [-]
        :param z_init: initial sensor output [kg/m^3]
        """
        if tau_rise <= 0.0 or tau_rec <= 0.0:
            raise ValueError('tau_rise and tau_rec must be positive')
        if dt <= 0.0:
            raise ValueError('dt must be positive')
        if b <= 0.0:
            raise ValueError('b must be positive')
        self.tau_rise = float(tau_rise)
        self.tau_rec = float(tau_rec)
        self.dt = float(dt)
        self.b = float(b)
        self._z_init = float(z_init)
        self.z_prev = float(z_init)

    def reset(self, z_init=None):
        """
        Return the lag to its initial output, for a per-trial reset.

        :param z_init: output to reset to [kg/m^3]; None reuses the
            constructor value
        :return: None
        :rtype: NoneType
        """
        self.z_prev = self._z_init if z_init is None else float(z_init)

    def tau_for(self, z_true):
        """
        Return the time constant this sample will use (rise or recovery).

        :param z_true: true concentration at this sample [kg/m^3]
        :return: tau_rise if the output is about to climb, else tau_rec
        :rtype: float
        """
        rising = self.b * float(z_true) - self.z_prev > 0.0
        return self.tau_rise if rising else self.tau_rec

    def update(self, z_true, dt=None):
        """
        Advance the lag by one sample and return the sensor output.

        :param z_true: true concentration at this sample [kg/m^3]
        :param dt: sample time [s]; None uses the constructor value
        :return: lagged sensor output z[k] [kg/m^3]
        :rtype: float
        """
        step = self.dt if dt is None else float(dt)
        if step <= 0.0:
            raise ValueError('dt must be positive')
        a = self.tau_for(z_true) / step
        z = (self.b * float(z_true) + a * self.z_prev) / (1.0 + a)
        self.z_prev = z
        return z


class GasField:
    """
    Ground-truth gas field: the superposition of the scenario's plumes.

    Thin wrapper over haptic_plume_core so that every truth-side evaluation
    goes through one object that a scenario reset can swap wholesale.
    """

    def __init__(self, alphas):
        """
        Construct the field from the scenario's plume parameters.

        :param alphas: (L, 8) array of plume parameter vectors, columns
            ordered as haptic_plume_core.plume_model.PLUME_PARAM_NAMES
        """
        self.alphas = np.atleast_2d(np.asarray(alphas, dtype=float))
        if self.alphas.shape[1] != 8:
            raise ValueError('alphas must have 8 columns (see PLUME_PARAM_NAMES)')

    @property
    def n_sources(self):
        """Return the number of true sources in the field."""
        return int(self.alphas.shape[0])

    def concentration(self, x_w, y_w, z_w):
        """
        Evaluate the true total concentration at world-frame point(s).

        :param x_w: world x coordinate(s) [m]
        :param y_w: world y coordinate(s) [m]
        :param z_w: world z coordinate(s) [m]
        :return: true concentration [kg/m^3]
        :rtype: numpy.ndarray or float
        """
        return total_concentration(x_w, y_w, z_w, self.alphas)


class GasSensor:
    """
    Full truth-to-measurement chain: fluctuation, then lag, then noise.

    One instance per simulated sensor. All randomness comes from the injected
    numpy Generator, so a seeded scenario replays identically (risk R13: we
    guarantee deterministic scenarios, not deterministic human trajectories).
    """

    def __init__(self, lag, rng, sigma_noise=0.0, sigma_fluct=0.0,
                 clamp_negative=True):
        """
        Construct the sensor around a lag model and a random generator.

        :param lag: FirstOrderSensor instance holding the sensor dynamics
        :param rng: numpy.random.Generator supplying both noise streams
        :param sigma_noise: std of the additive white Gaussian noise nu[k]
            on the lagged output [kg/m^3]
        :param sigma_fluct: shape parameter of the unit-mean lognormal
            turbulence fluctuation [-]; 0 disables it
        :param clamp_negative: clamp the reported measurement at zero, as a
            real sensor cannot report negative concentration
        """
        if sigma_noise < 0.0 or sigma_fluct < 0.0:
            raise ValueError('noise parameters must be non-negative')
        self.lag = lag
        self.rng = rng
        self.sigma_noise = float(sigma_noise)
        self.sigma_fluct = float(sigma_fluct)
        self.clamp_negative = bool(clamp_negative)

    def reset(self, z_init=None):
        """
        Reset the sensor dynamics; the generator is deliberately untouched.

        :param z_init: output to reset the lag to [kg/m^3], or None
        :return: None
        :rtype: NoneType
        """
        self.lag.reset(z_init)

    def measure(self, c_true, dt=None):
        """
        Produce one measurement from a true concentration sample.

        :param c_true: true (time-averaged) concentration [kg/m^3]
        :param dt: sample time [s]; None uses the lag's nominal value
        :return: measurement z[k] as the estimator will see it [kg/m^3]
        :rtype: float
        """
        c = float(c_true)
        if self.sigma_fluct > 0.0:
            # Unit-mean lognormal: exp(N(-s^2/2, s)) has expectation 1, so the
            # fluctuation adds intermittency without biasing the time average.
            c *= float(self.rng.lognormal(-0.5 * self.sigma_fluct**2,
                                          self.sigma_fluct))
        z = self.lag.update(c, dt)
        if self.sigma_noise > 0.0:
            z += float(self.rng.normal(0.0, self.sigma_noise))
        if self.clamp_negative:
            z = max(z, 0.0)
        return z
