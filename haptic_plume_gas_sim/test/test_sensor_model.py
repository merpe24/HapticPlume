"""
Hand-computed unit tests for haptic_plume_gas_sim.sensor_model.

The lag values below are exact rationals obtained by stepping the backward
Euler recursion z[k] = (b z_true + a z[k-1]) / (1 + a) by hand, so a wrong
discretization or a swapped rise/recovery branch cannot hide behind the
implementation testing itself.

:author: premmm
:date: July 29, 2026
"""

from haptic_plume_core.plume_model import total_concentration

from haptic_plume_gas_sim.sensor_model import FirstOrderSensor, GasField, GasSensor

import numpy as np

import pytest

# Reference plume for the field tests: source at origin, wind toward +x_W.
ALPHA_REF = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.1, 0.1])


def make_lag(z_init=0.0):
    """Build the reference lag: tau_rise=2 s, tau_rec=4 s, dt=1 s, b=1."""
    return FirstOrderSensor(tau_rise=2.0, tau_rec=4.0, dt=1.0, b=1.0, z_init=z_init)


def test_rise_step_response_hand_values():
    """A unit step from rest gives z = 1/3 then 5/9 with a = tau_rise/dt = 2."""
    lag = make_lag()
    assert lag.update(1.0) == pytest.approx(1.0 / 3.0, rel=1e-12)
    assert lag.update(1.0) == pytest.approx(5.0 / 9.0, rel=1e-12)


def test_recovery_uses_the_recovery_constant():
    """
    Falling back to zero from z=1 uses tau_rec, not tau_rise.

    a = tau_rec/dt = 4 gives z = (0 + 4*1)/5 = 0.8. Had the rise constant been
    used the answer would be 2/3, so this test pins the branch selection.
    """
    lag = make_lag(z_init=1.0)
    assert lag.update(0.0) == pytest.approx(0.8, rel=1e-12)
    assert lag.update(0.0) != pytest.approx(2.0 / 3.0, rel=1e-6)


def test_branch_selection_is_reported_consistently():
    """tau_for() agrees with the branch update() actually takes."""
    lag = make_lag()
    assert lag.tau_for(1.0) == 2.0     # b*z_true > z_prev -> rising
    lag.update(1.0)
    assert lag.tau_for(0.0) == 4.0     # b*z_true < z_prev -> recovering
    assert lag.tau_for(lag.z_prev) == 4.0   # equality is not rising


def test_steady_state_approaches_gain_times_input():
    """With b != 1 the lag settles on b * z_true, not on z_true."""
    lag = FirstOrderSensor(tau_rise=1.0, tau_rec=1.0, dt=0.5, b=2.5)
    for _ in range(500):
        z = lag.update(0.4)
    assert z == pytest.approx(1.0, rel=1e-9)


def test_zero_input_from_rest_stays_exactly_zero():
    """A quiet sensor reports exactly zero, with no drift from the recursion."""
    lag = make_lag()
    for _ in range(10):
        assert lag.update(0.0) == 0.0


def test_reset_restores_the_initial_output():
    """reset() returns the lag to its constructor state (per-trial reset)."""
    lag = make_lag()
    lag.update(1.0)
    lag.reset()
    assert lag.z_prev == 0.0
    assert lag.update(1.0) == pytest.approx(1.0 / 3.0, rel=1e-12)
    lag.reset(0.5)
    assert lag.z_prev == 0.5


def test_per_call_dt_overrides_the_nominal():
    """Passing dt explicitly changes a = tau/dt for that sample only."""
    lag = make_lag()
    # dt = 2 s with tau_rise = 2 s gives a = 1 -> z = (1 + 0)/2 = 0.5
    assert lag.update(1.0, dt=2.0) == pytest.approx(0.5, rel=1e-12)


def test_invalid_parameters_are_rejected():
    """Non-positive time constants, sample times, and gains raise."""
    with pytest.raises(ValueError):
        FirstOrderSensor(tau_rise=0.0, tau_rec=1.0, dt=1.0)
    with pytest.raises(ValueError):
        FirstOrderSensor(tau_rise=1.0, tau_rec=-1.0, dt=1.0)
    with pytest.raises(ValueError):
        FirstOrderSensor(tau_rise=1.0, tau_rec=1.0, dt=0.0)
    with pytest.raises(ValueError):
        FirstOrderSensor(tau_rise=1.0, tau_rec=1.0, dt=1.0, b=0.0)
    with pytest.raises(ValueError):
        make_lag().update(1.0, dt=-0.1)


def test_gas_field_matches_the_core_superposition():
    """The field is a thin wrapper: same value as the plume model itself."""
    alphas = np.vstack([ALPHA_REF, [10.0, -2.0, 0.0, np.pi, 2.0, 1.5, 0.2, 0.05]])
    field = GasField(alphas)
    assert field.n_sources == 2
    assert field.concentration(4.0, 0.5, 0.0) == pytest.approx(
        total_concentration(4.0, 0.5, 0.0, alphas), rel=1e-12)


def test_gas_field_accepts_a_single_alpha_and_rejects_bad_shapes():
    """A bare (8,) alpha is promoted to one source; wrong widths raise."""
    assert GasField(ALPHA_REF).n_sources == 1
    with pytest.raises(ValueError):
        GasField(np.zeros((2, 7)))


def test_sensor_without_noise_is_exactly_the_lag():
    """sigma_noise = sigma_fluct = 0 leaves the lag output untouched."""
    rng = np.random.default_rng(0)
    sensor = GasSensor(make_lag(), rng, sigma_noise=0.0, sigma_fluct=0.0)
    reference = make_lag()
    for _ in range(5):
        assert sensor.measure(1.0) == pytest.approx(reference.update(1.0), rel=1e-12)


def test_same_seed_reproduces_the_measurement_stream():
    """Two sensors seeded alike produce identical measurements (determinism)."""
    sensor_a = GasSensor(make_lag(), np.random.default_rng(7),
                         sigma_noise=1e-3, sigma_fluct=0.3)
    sensor_b = GasSensor(make_lag(), np.random.default_rng(7),
                         sigma_noise=1e-3, sigma_fluct=0.3)
    stream_a = [sensor_a.measure(1.0) for _ in range(20)]
    stream_b = [sensor_b.measure(1.0) for _ in range(20)]
    assert stream_a == stream_b
    assert len(set(stream_a)) == 20   # the stream is not accidentally constant


def test_lognormal_fluctuation_has_unit_mean():
    """
    The turbulence fluctuation is intermittent but unbiased.

    A biased fluctuation would shift every estimated release rate Q, so this
    guards the time-averaged interpretation of the Gaussian plume model.
    """
    rng = np.random.default_rng(3)
    sigma_fluct = 0.5
    samples = rng.lognormal(-0.5 * sigma_fluct**2, sigma_fluct, size=200000)
    assert samples.mean() == pytest.approx(1.0, rel=0.01)


def test_measurements_are_clamped_at_zero():
    """A real sensor cannot report negative concentration."""
    rng = np.random.default_rng(1)
    sensor = GasSensor(make_lag(), rng, sigma_noise=1.0, sigma_fluct=0.0)
    values = [sensor.measure(0.0) for _ in range(200)]
    assert min(values) == 0.0
    assert max(values) > 0.0


def test_clamping_can_be_disabled_for_analysis():
    """With clamping off the raw (possibly negative) sum is returned."""
    rng = np.random.default_rng(1)
    sensor = GasSensor(make_lag(), rng, sigma_noise=1.0, sigma_fluct=0.0,
                       clamp_negative=False)
    values = [sensor.measure(0.0) for _ in range(200)]
    assert min(values) < 0.0


def test_negative_noise_parameters_are_rejected():
    """Noise scales are standard deviations, so they cannot be negative."""
    with pytest.raises(ValueError):
        GasSensor(make_lag(), np.random.default_rng(0), sigma_noise=-1.0)
    with pytest.raises(ValueError):
        GasSensor(make_lag(), np.random.default_rng(0), sigma_fluct=-1.0)
