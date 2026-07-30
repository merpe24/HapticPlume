"""
Tests for haptic_plume_experiment.eval_estimator and the cross-package chain.

This package is the one place allowed to import both the simulator and the
estimator, so it is also the only place the two halves of the sensor model can
be checked against each other. That round trip is the first test below.

The rest keep the Phase A harness honest: the metric it reports, the
determinism the study depends on, and the risk R1 result itself — a hovering
pilot must make the belief say DIVERGED rather than invent a source.

:author: premmm
:date: July 29, 2026
"""

from haptic_plume_core.scenario import random_scenario

from haptic_plume_estimation.particle_filter import STATUS_DIVERGED

from haptic_plume_estimation.sensor_compensation import (
    InverseFeedforwardCompensator,
)

from haptic_plume_experiment.eval_estimator import (
    EvalConfig,
    mean_pair_distance,
    PLANNERS,
    run_non_informative_study,
    run_suite,
    run_trial,
    summarize,
)

from haptic_plume_gas_sim.sensor_model import FirstOrderSensor, GasSensor

import numpy as np

import pytest

TAU_RISE = 3.0
TAU_REC = 8.0
DT = 0.2
GAMMA = 5.0


def small_config(**kwargs):
    """Build a deliberately tiny configuration so the harness tests stay quick."""
    defaults = {'max_steps': 250, 'n_particles': 400, 'mi_particles': 100,
                'plan_every': 10}
    defaults.update(kwargs)
    return EvalConfig(**defaults)


def test_lag_round_trip_across_the_two_packages():
    """
    The real simulator lag and the real compensator invert each other.

    Composed, they are exactly the backward-Euler low-pass gamma/(s + gamma)
    applied to the true concentration, so with no noise the compensated signal
    is a first-order filter of the truth and nothing else. This is the check
    that the two packages were configured from the same sensor model — the
    estimator-side unit test can only compare against its own hand copy.
    """
    rng = np.random.default_rng(0)
    truth = np.concatenate([
        np.zeros(10), np.full(30, 0.08), np.full(30, 0.01),
        rng.uniform(0.0, 0.15, 60),
    ])
    sensor = GasSensor(FirstOrderSensor(TAU_RISE, TAU_REC, DT), rng,
                       sigma_noise=0.0, sigma_fluct=0.0)
    compensator = InverseFeedforwardCompensator(TAU_RISE, TAU_REC, DT, GAMMA)

    low_pass = 0.0
    for c_true in truth:
        low_pass = (GAMMA * DT * c_true + low_pass) / (1.0 + GAMMA * DT)
        assert compensator.update(sensor.measure(c_true)) == pytest.approx(
            low_pass, rel=1e-9, abs=1e-15)


def test_mean_pair_distance_hand_values():
    """Eq. (29) pairs sources by minimum total distance, not by input order."""
    truth = np.array([[0.0, 0.0, 1.0], [10.0, 0.0, 1.0]])
    estimates = np.array([[10.5, 0.0, 1.0], [0.5, 0.0, 1.0]])
    assert mean_pair_distance(truth, estimates) == pytest.approx(0.5)


def test_mean_pair_distance_with_unequal_counts():
    """
    Only min(L, M) pairs exist; the count error is scored separately.

    Here one true source is missed entirely, and the reported distance is the
    quality of the single estimate that was made.
    """
    truth = np.array([[0.0, 0.0, 1.0], [10.0, 0.0, 1.0]])
    assert mean_pair_distance(truth, np.array([[0.25, 0.0, 1.0]])) == \
        pytest.approx(0.25)
    assert mean_pair_distance(truth, np.empty((0, 3))) is None


def test_trial_is_reproducible_from_its_seed():
    """
    Decision D7: the same seed replays the same belief, exactly.

    Everything stochastic in the loop — turbulence, sensor noise, the prior
    draw, resampling — comes from a seeded generator, so a study can be rerun
    from its log rather than from memory.
    """
    scenario = random_scenario(1, seed=3)
    config = small_config()
    first = run_trial(scenario, config, seed=3)
    second = run_trial(scenario, config, seed=3)
    assert first.n_found == second.n_found
    assert first.diverged_fraction == second.diverged_fraction
    assert first.final_status == second.final_status
    # assert_equal, not ==: the error is NaN when nothing was consumed, and
    # NaN != NaN would make this test pass for the wrong reason.
    np.testing.assert_equal(first.localization_error_m,
                            second.localization_error_m)
    np.testing.assert_array_equal(first.consumed_positions,
                                  second.consumed_positions)


def test_infotaxis_localizes_a_single_leak():
    """
    The end-to-end chain works: noisy lagged sensor in, located source out.

    Kept to one scenario and a short flight so this stays a unit test; the
    statistical claim lives in the eval report, not here.
    """
    scenario = random_scenario(1, seed=2)
    config = small_config(max_steps=900, n_particles=1000, mi_particles=250)
    result = run_trial(scenario, config, seed=2)
    assert result.n_found == 1
    assert result.localization_error_pct < 10.0
    assert np.isfinite(result.localization_time_s)


def test_hovering_off_plume_reports_divergence_and_finds_nothing():
    """
    Risk R1, stated as a test: no information must not become a false source.

    The composer keys the belief-driven haptic well off this status, so a
    hovering pilot feeling a confident pull toward an invented leak is the
    exact failure this guards.
    """
    scenario = random_scenario(1, seed=4)
    scenario.start_position = np.array([1.0, 19.0, 1.0])
    result = run_trial(scenario, small_config(max_steps=400), seed=4,
                       planner='hover')
    assert result.n_found == 0
    assert result.final_status == STATUS_DIVERGED
    # The detector only speaks after diverged_patience updates, so the
    # fraction is bounded by (max_steps - patience) / max_steps.
    assert result.diverged_fraction > 0.4


def test_every_planner_runs_and_reports_a_result():
    """The R1 study needs all four trajectory types to complete a trial."""
    scenario = random_scenario(1, seed=6)
    config = small_config(max_steps=120)
    for planner in PLANNERS:
        result = run_trial(scenario, config, seed=6, planner=planner)
        assert result.planner == planner
        assert result.steps == 120
        assert result.n_found >= 0


def test_unknown_planner_is_rejected():
    """A typo in a study script should fail immediately, not run silently."""
    with pytest.raises(ValueError):
        run_trial(random_scenario(1, seed=0), small_config(),
                  planner='chemotaxis')


def test_summarize_scores_success_on_the_source_count():
    """A run that reports the wrong number of leaks is not a success."""
    scenario = random_scenario(1, seed=8)
    results = [run_trial(scenario, small_config(max_steps=120), seed=8,
                         planner='hover')]
    stats = summarize(results)
    assert stats['n_trials'] == 1
    assert stats['success_rate'] == 0.0
    assert summarize([])['n_trials'] == 0


def test_suite_and_study_helpers_return_one_row_per_trial():
    """The drivers wire scenarios to trials without dropping any."""
    config = small_config(max_steps=100)
    suite = run_suite([1], n_trials=2, base_seed=0, config=config)
    assert len(suite) == 2
    assert {r.seed for r in suite} == {1000, 1001}

    study = run_non_informative_study(n_sources=1, n_trials=1, base_seed=0,
                                      config=config)
    assert set(study) == set(PLANNERS)
    assert all(len(rows) == 1 for rows in study.values())
