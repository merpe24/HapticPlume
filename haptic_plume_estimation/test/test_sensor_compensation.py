"""
Hand-computed unit tests for haptic_plume_estimation.sensor_compensation.

The key property proved here is analytic, not empirical: a backward-Euler
first-order lag followed by the Eq. (28) compensator is EXACTLY the
backward-Euler low-pass gamma/(s + gamma) applied to the true signal,

    z_ff[k] = (gamma dt z_true[k] + z_ff[k-1]) / (1 + gamma dt),

for every sample, including across rise/recovery branch switches. The plant is
re-implemented inline in three lines as a hand reference, so this file stays a
pure unit test; the cross-package round trip against the real
haptic_plume_gas_sim sensor lives in haptic_plume_experiment.

:author: premmm
:date: July 29, 2026
"""

from haptic_plume_estimation.sensor_compensation import (
    compensate_series,
    InverseFeedforwardCompensator,
)

import numpy as np

import pytest

TAU_RISE = 2.0
TAU_REC = 4.0
DT = 1.0
GAMMA = 1.0


def make_compensator(gamma=GAMMA, dt=DT):
    """Build the reference compensator: tau=(2, 4) s, b=1, gamma=1 rad/s."""
    return InverseFeedforwardCompensator(
        tau_rise=TAU_RISE, tau_rec=TAU_REC, dt=dt, gamma=gamma, b=1.0)


def lag_step(z_prev, z_true, dt=DT, b=1.0):
    """
    Advance a hand-written backward-Euler first-order lag by one sample.

    Deliberately a local copy of haptic_plume_gas_sim's recursion so this test
    file does not couple the estimator to the simulator (CLAUDE.md structural
    rule): z[k] = (b z_true + a z[k-1]) / (1 + a) with a = tau / dt, and tau
    chosen by the sign of (b z_true - z[k-1]).
    """
    tau = TAU_RISE if b * z_true - z_prev > 0.0 else TAU_REC
    a = tau / dt
    return (b * z_true + a * z_prev) / (1.0 + a)


def low_pass_step(y_prev, z_true, gamma=GAMMA, dt=DT):
    """Advance the backward-Euler low-pass gamma/(s + gamma) by one sample."""
    return (gamma * dt * z_true + y_prev) / (1.0 + gamma * dt)


def test_first_sample_hand_value():
    """
    z[1]=1 from rest, tau_rise=2, dt=1, gamma=1, b=1.

    numerator   = 1 * 1 * (1 + 2/1) - 0 + 0 = 3
    denominator = 1 * (1 + 1/1)             = 2   ->  z_ff = 1.5
    """
    assert make_compensator().update(1.0) == pytest.approx(1.5, rel=1e-12)


def test_compensation_of_a_step_is_the_low_pass_of_the_step():
    """
    The exact identity, checked on a unit step, with hand values.

    Plant: z = 1/3 then 5/9. Compensated: 0.5 then 0.75, which is exactly
    (z_true + z_ff_prev)/2, the low-pass recursion at gamma dt = 1.
    """
    compensator = make_compensator()
    z_plant = lag_step(0.0, 1.0)
    assert z_plant == pytest.approx(1.0 / 3.0, rel=1e-12)
    assert compensator.update(z_plant) == pytest.approx(0.5, rel=1e-12)

    z_plant = lag_step(z_plant, 1.0)
    assert z_plant == pytest.approx(5.0 / 9.0, rel=1e-12)
    assert compensator.update(z_plant) == pytest.approx(0.75, rel=1e-12)


def test_round_trip_identity_holds_across_branch_switches():
    """
    Rising and falling inputs both invert exactly.

    The plant selects its time constant from (b z_true - z_prev) and the
    compensator from (z - z_prev); those always agree in sign, so a signal
    that repeatedly rises and decays never desynchronizes the two branches.
    """
    rng = np.random.default_rng(11)
    z_true_series = np.concatenate([
        np.full(15, 1.0),          # rise
        np.zeros(15),              # recovery
        rng.uniform(0.0, 2.0, 40),  # ragged, branch-switching every sample
    ])
    compensator = make_compensator()
    z_plant = 0.0
    y_lp = 0.0
    for z_true in z_true_series:
        z_plant = lag_step(z_plant, z_true)
        y_lp = low_pass_step(y_lp, z_true)
        assert compensator.update(z_plant) == pytest.approx(y_lp, rel=1e-10)


def test_large_gamma_recovers_the_true_signal():
    """As gamma grows the low-pass vanishes and z_ff -> z_true."""
    z_true_series = [0.0, 1.0, 1.0, 0.4, 0.4, 0.0, 2.0]
    errors = {}
    for gamma in (1.0, 100.0):
        compensator = make_compensator(gamma=gamma)
        z_plant = 0.0
        worst = 0.0
        for z_true in z_true_series:
            z_plant = lag_step(z_plant, z_true)
            worst = max(worst, abs(compensator.update(z_plant) - z_true))
        errors[gamma] = worst
    assert errors[100.0] < 0.05
    assert errors[100.0] < errors[1.0]


def test_uncompensated_lag_is_the_error_being_removed():
    """
    The compensator is worth having: it beats the raw measurement.

    Guards against a sign or branch bug that silently turns Eq. (28) into an
    expensive pass-through.
    """
    z_true_series = np.concatenate([np.full(10, 1.0), np.zeros(10)])
    compensator = make_compensator(gamma=20.0)
    z_plant = 0.0
    raw_error = 0.0
    compensated_error = 0.0
    for z_true in z_true_series:
        z_plant = lag_step(z_plant, z_true)
        raw_error += abs(z_plant - z_true)
        compensated_error += abs(compensator.update(z_plant) - z_true)
    assert compensated_error < 0.2 * raw_error


def test_higher_gamma_amplifies_measurement_noise():
    """
    The stated trade-off is real: gamma buys accuracy with noise.

    This is why Eq. (27) exists at all instead of the bare inverse Eq. (26),
    and it is the knob to turn if the heaviness cue (D5) feels gritty. Run at
    a realistic sample time (20 Hz against seconds-long time constants), where
    the sample-to-sample gain on white noise approaches gamma (1 + tau/dt) /
    (gamma + 1/dt) and so grows steeply with gamma.
    """
    rng = np.random.default_rng(5)
    noise = rng.normal(0.0, 1e-3, 400)
    spreads = []
    for gamma in (1.0, 50.0):
        compensator = make_compensator(gamma=gamma, dt=0.05)
        out = [compensator.update(n) for n in noise]
        spreads.append(float(np.std(out)))
    assert spreads[1] > 5.0 * spreads[0]


def test_reset_clears_both_states():
    """reset() makes the next sample behave like the very first one."""
    compensator = make_compensator()
    compensator.update(1.0)
    compensator.update(0.2)
    compensator.reset()
    assert compensator.z_prev == 0.0
    assert compensator.z_ff_prev == 0.0
    assert compensator.update(1.0) == pytest.approx(1.5, rel=1e-12)


def test_branch_selection_reported_matches_the_measured_slope():
    """tau_for() uses the measurement's own slope, not the true signal's."""
    compensator = make_compensator()
    assert compensator.tau_for(1.0) == TAU_RISE
    compensator.update(1.0)
    assert compensator.tau_for(0.5) == TAU_REC
    assert compensator.tau_for(1.0) == TAU_REC   # equal is not rising


def test_per_call_dt_overrides_the_nominal():
    """
    dt=2 with tau_rise=2, gamma=1: numerator 1*(1+1)=2, denominator 1.5.

    Measurement timestamps drive the real filter, so the override has to work.
    """
    assert make_compensator().update(1.0, dt=2.0) == pytest.approx(
        2.0 / 1.5, rel=1e-12)


def test_compensate_series_matches_stepwise_updates():
    """The offline replay entry point (D7) equals the live filter exactly."""
    measurements = [0.0, 0.3, 0.9, 1.2, 0.7, 0.1, 0.0, 0.05]
    compensator = make_compensator()
    stepwise = [compensator.update(z) for z in measurements]
    series = compensate_series(measurements, TAU_RISE, TAU_REC, DT, GAMMA)
    np.testing.assert_allclose(series, stepwise, rtol=1e-12)


def test_invalid_parameters_are_rejected():
    """Time constants, sample time, low-pass coefficient and gain are positive."""
    with pytest.raises(ValueError):
        InverseFeedforwardCompensator(0.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        InverseFeedforwardCompensator(1.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        InverseFeedforwardCompensator(1.0, 1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        InverseFeedforwardCompensator(1.0, 1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        InverseFeedforwardCompensator(1.0, 1.0, 1.0, 1.0, b=0.0)
    with pytest.raises(ValueError):
        make_compensator().update(1.0, dt=0.0)
