"""
Inverse-feedforward compensation of the chemical sensor's dynamics.

Implements Goodell et al. (RAS 2026) Section 3.9, Eqs. (25)-(28). A slow gas
sensor smears each reading across several seconds, which violates the
conditional-independence assumption the particle filter's likelihood relies
on. The fix is to run the measurement through an inverse sensor model:

    G(s)        = b / (tau s + 1)                            (Eq. 24)
    G^-1(s)     = (tau s + 1) / b                            (Eq. 26)
    G^-1_lp(s)  = (tau s + 1) gamma / (b (s + gamma))        (Eq. 27)

The bare inverse differentiates, so it amplifies sensor noise; the low-pass
term gamma/(s + gamma) is what makes it usable on a real signal. Larger gamma
compensates more completely but passes more noise through — that trade-off is
the only tuning knob here.

Discretizing Eq. (27) by backward Euler gives paper Eq. (28), written here as

    z_ff[k] = ( z[k] gamma (1 + tau/dt)
                - z[k-1] gamma tau/dt
                + z_ff[k-1] b/dt ) / ( b (gamma + 1/dt) )

which is Eq. (28) with numerator and denominator both scaled by tau.

The result z_ff is the observation used in paper Eq. (12), i.e. what the
particle filter and the haptic heaviness cue (decision D5) both consume.

This module never imports haptic_plume_gas_sim: the estimator only ever sees
the measurement stream, exactly as it will on hardware.

Units:
    concentration [kg/m^3] · dt [s] · tau_rise, tau_rec [s] · gamma [rad/s]

:author: premmm
:date: July 29, 2026
"""

import numpy as np


class InverseFeedforwardCompensator:
    """
    Undo a first-order sensor lag with a low-pass-limited inverse model.

    The compensator must be configured with the SAME tau_rise, tau_rec and b
    as the physical sensor it inverts. It picks the rise/recovery branch from
    the measured signal alone (tau_rise while z climbs, tau_rec while it
    decays), which is the same decision the plant makes, so the two branch
    together and the compensation stays exact in the noise-free case.

    Composed with a backward-Euler first-order lag, this compensator returns
    exactly the low-pass-filtered true signal:

        z_ff[k] = (gamma dt z_true[k] + z_ff[k-1]) / (1 + gamma dt)

    so gamma -> infinity recovers z_true itself. test_sensor_compensation.py
    checks that identity against haptic_plume_gas_sim's forward model.
    """

    def __init__(self, tau_rise, tau_rec, dt, gamma, b=1.0, z_init=0.0):
        """
        Construct the compensator from the sensor model it has to invert.

        :param tau_rise: sensor rise time constant [s]
        :param tau_rec: sensor recovery time constant [s]
        :param dt: nominal measurement sample time [s]
        :param gamma: low-pass coefficient of Eq. (27) [rad/s]; higher means
            more complete compensation and more amplified noise
        :param b: static gain of the sensor [-]
        :param z_init: initial measurement and compensated output [kg/m^3]
        """
        if tau_rise <= 0.0 or tau_rec <= 0.0:
            raise ValueError('tau_rise and tau_rec must be positive')
        if dt <= 0.0:
            raise ValueError('dt must be positive')
        if gamma <= 0.0:
            raise ValueError('gamma must be positive')
        if b <= 0.0:
            raise ValueError('b must be positive')
        self.tau_rise = float(tau_rise)
        self.tau_rec = float(tau_rec)
        self.dt = float(dt)
        self.gamma = float(gamma)
        self.b = float(b)
        self._z_init = float(z_init)
        self.z_prev = float(z_init)
        self.z_ff_prev = float(z_init)

    def reset(self, z_init=None):
        """
        Clear both filter states, for a per-trial or per-source reset.

        :param z_init: measurement and output to reset to [kg/m^3]; None
            reuses the constructor value
        :return: None
        :rtype: NoneType
        """
        value = self._z_init if z_init is None else float(z_init)
        self.z_prev = value
        self.z_ff_prev = value

    def tau_for(self, z):
        """
        Return the time constant for this sample, from the measured slope.

        :param z: current measurement z[k] [kg/m^3]
        :return: tau_rise if the measurement is rising, else tau_rec
        :rtype: float
        """
        return self.tau_rise if float(z) - self.z_prev > 0.0 else self.tau_rec

    def update(self, z, dt=None):
        """
        Compensate one measurement — paper Eq. (28).

        :param z: raw measurement z[k] from the gas sensor [kg/m^3]
        :param dt: sample time [s]; None uses the constructor value
        :return: compensated measurement z_ff[k] [kg/m^3]
        :rtype: float
        """
        step = self.dt if dt is None else float(dt)
        if step <= 0.0:
            raise ValueError('dt must be positive')
        z = float(z)
        tau = self.tau_for(z)
        numerator = (z * self.gamma * (1.0 + tau / step)
                     - self.z_prev * self.gamma * tau / step
                     + self.z_ff_prev * self.b / step)
        denominator = self.b * (self.gamma + 1.0 / step)
        z_ff = numerator / denominator
        self.z_prev = z
        self.z_ff_prev = z_ff
        return z_ff


def compensate_series(measurements, tau_rise, tau_rec, dt, gamma, b=1.0,
                      z_init=0.0):
    """
    Compensate a whole measurement series in one call (offline bag replay).

    Decision D7 requires the belief to be reproducible bit-exactly from a
    logged (pose, measurement) stream; this is the entry point that replay
    tooling uses so it cannot drift from the live compensator.

    :param measurements: sequence of raw measurements z[k] [kg/m^3]
    :param tau_rise: sensor rise time constant [s]
    :param tau_rec: sensor recovery time constant [s]
    :param dt: sample time [s], scalar or one value per measurement
    :param gamma: low-pass coefficient of Eq. (27) [rad/s]
    :param b: static gain of the sensor [-]
    :param z_init: initial measurement and compensated output [kg/m^3]
    :return: compensated series z_ff[k], same length as the input
    :rtype: numpy.ndarray
    """
    z = np.atleast_1d(np.asarray(measurements, dtype=float))
    steps = np.broadcast_to(np.asarray(dt, dtype=float), z.shape)
    compensator = InverseFeedforwardCompensator(
        tau_rise, tau_rec, float(steps[0]), gamma, b=b, z_init=z_init)
    return np.array([compensator.update(zk, step)
                     for zk, step in zip(z, steps)])
