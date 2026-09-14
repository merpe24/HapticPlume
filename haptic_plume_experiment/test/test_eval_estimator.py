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

import os

from ament_index_python.packages import get_package_share_directory

from haptic_plume_core.scenario import random_scenario

from haptic_plume_estimation.particle_filter import STATUS_DIVERGED

from haptic_plume_estimation.sensor_compensation import (
    InverseFeedforwardCompensator,
)

from haptic_plume_experiment.eval_estimator import (
    _clamp_to_free_space,
    _cross_axis_spread,
    DRONE_RADIUS,
    EvalConfig,
    mean_pair_distance,
    PLANNERS,
    run_gate_study,
    run_non_informative_study,
    run_suite,
    run_trial,
    summarize,
)

from haptic_plume_gas_sim.sensor_model import FirstOrderSensor, GasSensor

import numpy as np

import pytest

import yaml

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


def test_cross_axis_spread_is_measured_perpendicular_to_the_wind():
    """
    A path straight down the wind axis has zero cross-axis spread.

    The risk R1 gate rests entirely on this projection being on the right
    axis. Getting it 90 degrees wrong would invert the gate — it would block
    exactly the informative paths and pass the degenerate ones — and no other
    test in the suite would notice. Hand-computed on theta = 0, where the
    wind runs along +x, so cross-axis is plain y.
    """
    along_wind = [np.array([float(x), 5.0, 1.0]) for x in range(10)]
    assert _cross_axis_spread(along_wind, 0.0) == pytest.approx(0.0)

    # The same points read against a wind blowing along +y are pure spread.
    assert _cross_axis_spread(along_wind, np.pi / 2.0) == pytest.approx(
        float(np.std(np.arange(10.0))))

    crosswind = [np.array([3.0, float(y), 1.0]) for y in range(10)]
    assert _cross_axis_spread(crosswind, 0.0) == pytest.approx(
        float(np.std(np.arange(10.0))))


def test_cross_axis_spread_handles_a_degenerate_window():
    """Fewer than two samples is 0.0, not a NaN reaching the gate."""
    assert _cross_axis_spread([], 0.0) == 0.0
    assert _cross_axis_spread([np.array([1.0, 1.0, 1.0])], 0.0) == 0.0


def test_path_window_is_scoped_to_the_last_filter_reset():
    """
    Each consume reports flying done since the previous one, not since launch.

    A live gate cannot judge source two on the flying done for source one, so
    run_trial clears its position window whenever the filter resets. Without
    that clear the recorded spreads rise monotonically, because the window
    only ever grows. This is the scoping, not a claim that the metric gates
    anything — it does not, see format_observability_table.
    """
    config = small_config(max_steps=900)
    result = run_trial(random_scenario(2, 4), config, seed=4,
                       planner='infotaxis')
    spreads = result.path_cross_at_consume
    if spreads.size < 3:
        pytest.skip('too few consumes in this short trial to judge scoping')
    # A cumulative window is non-decreasing by construction; a scoped one is
    # not. One decrease is enough to prove the window was cleared.
    assert np.any(np.diff(spreads) < 0.0)


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


# The two capsules of haptic_plume_description/config/obstacles.yaml, inlined so
# the clamp tests need no installed share directory.
PIPES = [
    (np.array([10.0, 5.0, 1.0]), np.array([10.0, 15.0, 1.0]), 0.30),
    (np.array([10.0, 5.0, 0.0]), np.array([10.0, 5.0, 1.0]), 0.30),
]
ROOM = np.array([[0.0, 20.0], [0.0, 20.0]])
MARGIN = 0.05


def clamp(point):
    """Run the harness clamp on a copy of point, with the live drone's numbers."""
    return _clamp_to_free_space(np.array(point, dtype=float), PIPES,
                                DRONE_RADIUS, MARGIN, ROOM)


def test_default_config_leaves_every_live_extra_off():
    """Every Phase A number rests on these defaults, so the live extras are opt-in."""
    config = EvalConfig()
    assert config.pf_decimation == 1
    assert config.sensor_dt == config.dt
    assert config.compensator_tau_rec is None
    assert config.drone_radius is None
    assert not config.stop_at_declaration
    assert config.n_twins == 0


def test_live_config_matches_the_node_parameter_file():
    """
    EvalConfig.live() is the chain estimator_params.yaml configures.

    Read from the installed YAML rather than restated, because a headless study
    that drifts from the node's parameters measures a system nobody flies --
    the 2026-08-30 rate bug was exactly that.
    """
    path = os.path.join(get_package_share_directory('haptic_plume_estimation'),
                        'config', 'estimator_params.yaml')
    with open(path) as handle:
        params = yaml.safe_load(handle)['plume_estimator']['ros__parameters']
    config = EvalConfig.live()
    assert 1.0 / config.sensor_dt == pytest.approx(params['sensor_rate'])
    assert config.pf_decimation == params['pf_decimation']
    assert config.n_particles == params['n_particles']
    assert config.q == pytest.approx(params['q'])
    assert config.sigma_c == pytest.approx(params['sigma_c'])
    assert config.d_cluster == pytest.approx(params['d_cluster'])
    assert config.max_sources == params['max_sources']
    assert config.tau_rise == pytest.approx(params['tau_rise'])
    assert config.compensator_tau_rec == pytest.approx(params['tau_rec'])
    assert config.gamma == pytest.approx(params['gamma'])
    assert config.n_twins == params['n_twins']
    assert config.agreement_m == pytest.approx(params['agreement_m'])
    assert config.drone_radius == DRONE_RADIUS
    assert config.stop_at_declaration


def test_clamp_slides_along_a_pipe_instead_of_stopping():
    """
    A drone pushed into main_run keeps its position along the pipe.

    Only the component into the pipe is removed, which is what lets the drone
    slide instead of stopping dead, and what obstacle_field.cpp does.
    """
    out = clamp([9.9, 10.0, 1.0])
    assert out[0] == pytest.approx(10.0 - 0.30 - DRONE_RADIUS - MARGIN)
    assert out[1] == pytest.approx(10.0)
    assert out[2] == pytest.approx(1.0)


def test_clamp_leaves_free_space_untouched():
    """A drone nowhere near a pipe or a wall does not move."""
    point = [4.0, 12.0, 1.0]
    np.testing.assert_array_equal(clamp(point), np.array(point))


def test_clamp_clears_both_capsules_where_they_meet():
    """main_run and riser share an endpoint; the result must be outside both."""
    for point in ([10.1, 5.05, 1.0], [10.05, 4.8, 1.0], [9.8, 5.2, 1.0]):
        out = clamp(point)
        for p0, p1, radius in PIPES:
            axis = p1 - p0
            t = np.clip((out - p0) @ axis / (axis @ axis), 0.0, 1.0)
            gap = np.linalg.norm(out - (p0 + t * axis))
            assert gap >= radius + DRONE_RADIUS + MARGIN - 1e-9


def test_clamp_keeps_the_body_inside_the_room():
    """The walls are inset by the drone radius, as the drone node insets them."""
    out = clamp([0.1, 19.95, 1.0])
    assert out[0] == pytest.approx(DRONE_RADIUS)
    assert out[1] == pytest.approx(20.0 - DRONE_RADIUS)


def test_drone_radius_without_obstacles_is_rejected():
    """Asking for the clamp without saying what to clamp against fails loudly."""
    with pytest.raises(ValueError):
        run_trial(random_scenario(1, seed=0),
                  small_config(drone_radius=DRONE_RADIUS))


def test_gate_study_flies_every_planner_with_the_gate_off_and_on():
    """The gate study pairs each planner's gate-off run with a gate-on run."""
    study = run_gate_study(n_trials=1, config=small_config(max_steps=40,
                                                           n_twins=1))
    assert set(study) == {(name, label) for name in PLANNERS
                          for label in ('off', 'on')}
    assert all(len(rows) == 1 for rows in study.values())


def test_live_trial_ends_at_its_declaration():
    """
    stop_at_declaration ends the run on the consume, as the operator's press does.

    Seed 10 declares within 5 s under the live configuration (measured
    2026-09-13), so this stays a unit test. steps then counts only the filter
    updates actually run.
    """
    config = EvalConfig.live()
    result = run_trial(random_scenario(1, seed=10, pipe_segments=PIPES), config,
                       seed=10, obstacles=PIPES)
    assert result.n_found == 1
    assert result.steps < config.max_steps
    assert result.localization_time_s == pytest.approx(result.steps * config.dt)


def test_withheld_declaration_resets_and_the_search_goes_on():
    """
    A declaration the twins disagree on is refused, counted, and searched past.

    Seed 23 withholds once and then declares under the live configuration
    (measured 2026-09-13).
    """
    config = EvalConfig.live()
    result = run_trial(random_scenario(1, seed=23, pipe_segments=PIPES), config,
                       seed=23, obstacles=PIPES)
    assert result.declarations_withheld == 1
    assert result.n_found == 1
