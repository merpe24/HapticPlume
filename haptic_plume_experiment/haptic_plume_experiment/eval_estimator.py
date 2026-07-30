"""
Headless evaluation of the MultiPLE estimator — the Phase A definition of done.

Runs the whole zero-ROS chain end to end on seeded scenarios:

    scenario -> GasField -> GasSensor (lognormal fluctuation, first-order lag,
    white noise) -> InverseFeedforwardCompensator -> PlumeParticleFilter
    -> ConsumedSet (+ agglomerative merge) -> MI waypoint suggestion -> motion

and reports the three numbers Goodell et al. (RAS 2026) score on: did the run
find the correct number of sources, how far off were they (paper Eqs. 29-30,
normalized by the search-area diagonal), and how long did it take.

This is the only module allowed to import both haptic_plume_gas_sim and
haptic_plume_estimation; keeping that pairing here is what stops the estimator
from ever depending on the simulator (CLAUDE.md structural rule, risk R8).

The second study is the one this project actually needs. A human pilot is not
an infotaxis agent, and the estimator's quality is partly a DEPENDENT variable
of how they fly (risk R1). --study non-informative replays the same scenarios
along trajectories a person plausibly produces — parking off-plume, parking in
the plume, a straight transect, a run down the plume axis — and reports what
the belief does. Those numbers decide how much the haptic well can be trusted
to say anything at all.

Usage::

    python3 -m haptic_plume_experiment.eval_estimator --study suite
    python3 -m haptic_plume_experiment.eval_estimator --study non-informative
    python3 -m haptic_plume_experiment.eval_estimator --study both --quick

:author: premmm
:date: July 29, 2026
"""

import argparse
from dataclasses import dataclass, field
from itertools import permutations

from haptic_plume_core.scenario import random_scenario

from haptic_plume_estimation.consume import ConsumedSet, should_consume
from haptic_plume_estimation.info_gain import best_waypoint, candidate_waypoints
from haptic_plume_estimation.particle_filter import (
    PlumeFilterConfig,
    PlumeParticleFilter,
    PROJECT_Q,
    STATUS_DIVERGED,
)
from haptic_plume_estimation.sensor_compensation import (
    InverseFeedforwardCompensator,
)

from haptic_plume_gas_sim.sensor_model import FirstOrderSensor, GasField, GasSensor

import numpy as np


@dataclass
class EvalConfig:
    """
    Everything the headless trial loop needs, in one seedable object.

    Rates follow the node graph in CLAUDE.md: the estimator runs at ~5 Hz and
    the waypoint suggester at ~2 Hz, so dt = 0.2 s and a replan every ten
    steps reproduce the timing the live system will have.
    """

    dt: float = 0.2                 # estimator period [s] (5 Hz)
    speed: float = 1.0              # drone ground speed [m/s]
    max_steps: int = 1200           # 240 s of flight
    n_particles: int = 1500
    q: float = PROJECT_Q            # likelihood scale in kg/m^3, Eq. (10)
    # Consume threshold, Eq. (11) [m]. The paper's 0.5 -> 0.7 is too permissive
    # against this project's noisier measurement chain: the Phase A sweep at
    # 0.7 consumed phantom sources once the real ones were gone (2 true leaks,
    # 3.0 found on average, 17% success). 0.5 keeps success near the paper's
    # 74% without pushing the localization error up the way 0.4 does.
    sigma_c: float = 0.5
    d_cluster: float = 1.8          # agglomerative merge distance [m]
    max_sources: int = 6            # refuse to keep consuming forever
    plan_every: int = 10            # replan period [steps] (2 Hz)
    plan_step: float = 2.0          # candidate ring radius [m]
    n_directions: int = 8
    mi_particles: int = 400         # particles subsampled for the quadrature
    tau_rise: float = 3.0           # sensor rise time constant [s]
    tau_rec: float = 8.0            # sensor recovery time constant [s]
    gamma: float = 5.0              # compensator low-pass coefficient [rad/s]
    sigma_noise: float = 2.0e-3     # white sensor noise [kg/m^3]
    sigma_fluct: float = 0.2        # lognormal turbulence fluctuation [-]

    def filter_config(self):
        """
        Derive the particle-filter configuration from this evaluation config.

        :return: the matching PlumeFilterConfig
        :rtype: PlumeFilterConfig
        """
        return PlumeFilterConfig(n_particles=self.n_particles, q=self.q,
                                 sigma_c=self.sigma_c,
                                 d_cluster=self.d_cluster)


@dataclass
class TrialResult:
    """Everything one trial reports; one row of the summary tables."""

    scenario: str
    planner: str
    seed: int
    n_true: int
    n_found: int
    localization_error_m: float = float('nan')
    localization_error_pct: float = float('nan')
    localization_time_s: float = float('nan')
    diverged_fraction: float = 0.0
    final_status: str = ''
    final_sigma_norm: float = float('nan')
    path_length_m: float = 0.0
    steps: int = 0
    consumed_positions: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3)))

    @property
    def correct_count(self):
        """Report whether the run found exactly the number of leaks present."""
        return self.n_found == self.n_true


def run_trial(scenario, config=None, seed=0, planner='infotaxis'):
    """
    Fly one scenario headless and return its metrics.

    :param scenario: haptic_plume_core.scenario.Scenario to fly
    :param config: EvalConfig; None uses the defaults
    :param seed: seed for the sensor noise and the particle filter
    :param planner: name from PLANNERS — 'infotaxis' is the paper's agent, the
        rest are the non-informative human-like paths of the risk R1 study
    :return: the trial's metrics
    :rtype: TrialResult
    """
    config = config if config is not None else EvalConfig()
    if planner not in PLANNERS:
        raise ValueError(f'unknown planner {planner!r}; have {sorted(PLANNERS)}')

    sensor_rng = np.random.default_rng(seed)
    filter_rng = np.random.default_rng(seed + 10_000)

    field_truth = GasField(scenario.alphas)
    sensor = GasSensor(
        FirstOrderSensor(config.tau_rise, config.tau_rec, config.dt),
        sensor_rng, sigma_noise=config.sigma_noise,
        sigma_fluct=config.sigma_fluct)
    compensator = InverseFeedforwardCompensator(
        config.tau_rise, config.tau_rec, config.dt, config.gamma)
    pf = PlumeParticleFilter(config.filter_config(), filter_rng)
    consumed = ConsumedSet()

    position = scenario.start_position.copy()
    target = position.copy()
    plan_state = {}
    n_diverged = 0
    path_length = 0.0
    last_consume_step = None

    for step in range(config.max_steps):
        # --- sense: truth -> lagged, noisy measurement -> compensated ---
        c_true = float(field_truth.concentration(*position))
        z_ff = compensator.update(sensor.measure(c_true))

        # --- believe: subtract what is already consumed, then update ---
        m_hat = float(consumed.concentration(*position))
        status = pf.update(position, z_ff, m_hat=m_hat)
        n_diverged += status == STATUS_DIVERGED

        # --- consume: a tight enough belief becomes a localized source ---
        if (should_consume(pf.sigma_position_norm, config.sigma_c)
                and len(consumed) < config.max_sources):
            consumed.push(pf.mmse)
            consumed.merge(config.d_cluster)
            last_consume_step = step
            pf.reset()

        # --- plan and move: kinematic x_dot = v, paper Eq. (20) ---
        if step % config.plan_every == 0:
            target = PLANNERS[planner](position, pf, scenario, config,
                                       step, plan_state)
        moved = _step_towards(position, target, config.speed * config.dt,
                              scenario.search_area)
        path_length += moved

    return _summarize(scenario, planner, seed, config, consumed, pf,
                      n_diverged, path_length, last_consume_step)


def _summarize(scenario, planner, seed, config, consumed, pf, n_diverged,
               path_length, last_consume_step):
    """
    Turn the end state of a trial into a TrialResult.

    :param scenario: the scenario that was flown
    :param planner: planner name used
    :param seed: the trial's seed
    :param config: EvalConfig used
    :param consumed: ConsumedSet at the end of the run
    :param pf: the particle filter at the end of the run
    :param n_diverged: number of updates that reported DIVERGED
    :param path_length: distance flown [m]
    :param last_consume_step: step index of the final consume, or None
    :return: the assembled result
    :rtype: TrialResult
    """
    result = TrialResult(
        scenario=scenario.name,
        planner=planner,
        seed=seed,
        n_true=scenario.n_sources,
        n_found=len(consumed),
        diverged_fraction=n_diverged / config.max_steps,
        final_status=pf.status,
        final_sigma_norm=pf.sigma_position_norm,
        path_length_m=path_length,
        steps=config.max_steps,
        consumed_positions=consumed.positions,
    )
    error = mean_pair_distance(scenario.source_positions, consumed.positions)
    if error is not None:
        result.localization_error_m = error
        result.localization_error_pct = 100.0 * error / scenario.diagonal
    if last_consume_step is not None:
        result.localization_time_s = (last_consume_step + 1) * config.dt
    return result


def mean_pair_distance(true_positions, estimated_positions):
    """
    Average distance over the best pairing of true and estimated sources.

    This is theta(A-hat, A_true) of paper Eq. (29): sources are matched by
    minimizing total distance, and only min(L, M) pairs exist when the counts
    disagree — an under- or over-count is already penalized by the success
    criterion, so it is not double-counted here.

    :param true_positions: (L, 3) true source positions [m]
    :param estimated_positions: (M, 3) estimated source positions [m]
    :return: mean paired distance [m], or None if either set is empty
    :rtype: float or NoneType
    """
    true_positions = np.atleast_2d(np.asarray(true_positions, dtype=float))
    estimated = np.atleast_2d(np.asarray(estimated_positions, dtype=float))
    if true_positions.size == 0 or estimated.size == 0:
        return None
    if true_positions.shape[0] <= estimated.shape[0]:
        small, large = true_positions, estimated
    else:
        small, large = estimated, true_positions
    n = small.shape[0]
    best = float('inf')
    for choice in permutations(range(large.shape[0]), n):
        distances = np.linalg.norm(small - large[list(choice)], axis=1)
        best = min(best, float(distances.mean()))
    return best


def _step_towards(position, target, max_distance, search_area):
    """
    Advance the kinematic drone toward a target, in place — paper Eq. (20).

    :param position: (3,) current position, modified in place [m]
    :param target: (3,) desired position [m]
    :param max_distance: how far the drone may travel this step [m]
    :param search_area: (2, 2) flyable extent, clipped against [m]
    :return: distance actually travelled [m]
    :rtype: float
    """
    before = position.copy()
    delta = np.asarray(target, dtype=float) - position
    distance = float(np.linalg.norm(delta))
    if distance > 1e-12:
        position += delta * (min(max_distance, distance) / distance)
    position[0] = np.clip(position[0], search_area[0, 0], search_area[0, 1])
    position[1] = np.clip(position[1], search_area[1, 0], search_area[1, 1])
    return float(np.linalg.norm(position - before))


# --------------------------------------------------------------------------
# Planners. Each returns the next target position given the current state.
# 'infotaxis' is the paper's agent; the others are the human-like trajectories
# of the risk R1 study and deliberately ignore the belief entirely.
# --------------------------------------------------------------------------

def _plan_infotaxis(position, pf, scenario, config, step, state):
    """Maximize mutual information over a ring of candidates — Eq. (21)."""
    del step, state
    indices = np.linspace(0, pf.config.n_particles - 1,
                          min(config.mi_particles, pf.config.n_particles),
                          dtype=int)
    particles = pf.particles[indices]
    weights = np.full(indices.size, 1.0 / indices.size)
    candidates = candidate_waypoints(
        position, config.plan_step, n_directions=config.n_directions,
        include_hold=True, bounds=scenario.search_area)
    _, waypoint, _ = best_waypoint(particles, weights, candidates,
                                   pf.likelihood_sigma, n_grid=128)
    return waypoint


def _plan_hold(position, pf, scenario, config, step, state):
    """Stay put: the pilot who stops to look at the camera feed."""
    del pf, scenario, config, step
    return state.setdefault('anchor', position.copy())


def _plan_transect(position, pf, scenario, config, step, state):
    """
    Fly a straight line across the search area, reversing at the edges.

    A plausible untrained sweep: it covers ground but has no reason to linger
    where the belief is ambiguous.
    """
    del pf, config, step
    heading = state.setdefault('heading', np.array([1.0, 0.35, 0.0]))
    heading = heading / np.linalg.norm(heading)
    target = position + heading * 5.0
    inside_x = scenario.search_area[0, 0] < target[0] < scenario.search_area[0, 1]
    inside_y = scenario.search_area[1, 0] < target[1] < scenario.search_area[1, 1]
    if not (inside_x and inside_y):
        state['heading'] = -state['heading']
        heading = state['heading'] / np.linalg.norm(state['heading'])
        target = position + heading * 5.0
    return target


def _plan_downwind(position, pf, scenario, config, step, state):
    """
    Ride the plume axis downwind from the first source, then turn around.

    The intuitive human strategy — follow the smell — and the one that gives
    the filter a single streamline to fit, which it can do far too confidently.
    """
    del pf, config, step
    theta = float(scenario.alphas[0, 3])
    axis = np.array([np.cos(theta), np.sin(theta), 0.0])
    origin = scenario.source_positions[0]
    along = state.get('along', 3.0)
    state['along'] = along + 1.0 if along < 12.0 else 3.0
    target = origin + axis * state['along']
    target[0] = np.clip(target[0], *scenario.search_area[0])
    target[1] = np.clip(target[1], *scenario.search_area[1])
    del position
    return target


PLANNERS = {
    'infotaxis': _plan_infotaxis,
    'hover': _plan_hold,
    'transect': _plan_transect,
    'downwind': _plan_downwind,
}


def run_suite(n_sources_list=(1, 2, 3), n_trials=10, base_seed=0, config=None,
              planner='infotaxis'):
    """
    Run the seeded scenario suite the Phase A definition of done is scored on.

    :param n_sources_list: how many leaks to place, one entry per group
    :param n_trials: trials per group
    :param base_seed: first seed; each trial uses base_seed + index
    :param config: EvalConfig; None uses the defaults
    :param planner: planner name to fly with
    :return: list of TrialResult
    :rtype: list
    """
    config = config if config is not None else EvalConfig()
    results = []
    for n_sources in n_sources_list:
        for trial in range(n_trials):
            seed = base_seed + 1000 * n_sources + trial
            scenario = random_scenario(n_sources, seed)
            results.append(run_trial(scenario, config, seed=seed,
                                     planner=planner))
    return results


def run_non_informative_study(n_sources=2, n_trials=5, base_seed=0,
                              config=None):
    """
    Compare informative and non-informative trajectories — the risk R1 study.

    Each scenario is flown by every planner, so the only thing that changes
    between rows is the path the belief was fed.

    :param n_sources: leaks per scenario
    :param n_trials: scenarios to run
    :param base_seed: first seed
    :param config: EvalConfig; None uses the defaults
    :return: dict mapping planner name to its list of TrialResult
    :rtype: dict
    """
    config = config if config is not None else EvalConfig()
    results = {name: [] for name in PLANNERS}
    for trial in range(n_trials):
        seed = base_seed + trial
        scenario = random_scenario(n_sources, seed)
        for name in PLANNERS:
            results[name].append(
                run_trial(scenario, config, seed=seed, planner=name))
    return results


def summarize(results):
    """
    Aggregate a list of trials into the paper's three headline numbers.

    :param results: list of TrialResult
    :return: dict of aggregate metrics
    :rtype: dict
    """
    if not results:
        return {'n_trials': 0}
    successes = [r for r in results if r.correct_count]
    errors = [r.localization_error_pct for r in successes
              if np.isfinite(r.localization_error_pct)]
    times = [r.localization_time_s for r in successes
             if np.isfinite(r.localization_time_s)]
    return {
        'n_trials': len(results),
        'success_rate': len(successes) / len(results),
        'mean_error_pct': float(np.mean(errors)) if errors else float('nan'),
        'mean_time_s': float(np.mean(times)) if times else float('nan'),
        'mean_found': float(np.mean([r.n_found for r in results])),
        'diverged_fraction': float(np.mean([r.diverged_fraction
                                            for r in results])),
    }


def format_suite_table(results_by_group):
    """
    Render the scenario-suite results as a fixed-width table.

    :param results_by_group: dict mapping a group label to its TrialResult list
    :return: the printable table
    :rtype: str
    """
    lines = [
        'sources  trials  success  mean err %  mean time s  mean found  '
        'diverged',
        '-------  ------  -------  ----------  -----------  ----------  '
        '--------',
    ]
    for label, results in results_by_group.items():
        s = summarize(results)
        lines.append(
            f'{label:>7}  {s["n_trials"]:>6}  {s["success_rate"]:>7.2f}  '
            f'{s["mean_error_pct"]:>10.2f}  {s["mean_time_s"]:>11.1f}  '
            f'{s["mean_found"]:>10.2f}  {s["diverged_fraction"]:>8.2f}')
    return '\n'.join(lines)


def format_study_table(results_by_planner):
    """
    Render the non-informative-path study as a fixed-width table.

    :param results_by_planner: dict mapping planner name to TrialResult list
    :return: the printable table
    :rtype: str
    """
    lines = [
        'planner     trials  success  mean err %  mean found  diverged  '
        'path m',
        '----------  ------  -------  ----------  ----------  --------  '
        '------',
    ]
    for name, results in results_by_planner.items():
        s = summarize(results)
        path = float(np.mean([r.path_length_m for r in results]))
        lines.append(
            f'{name:<10}  {s["n_trials"]:>6}  {s["success_rate"]:>7.2f}  '
            f'{s["mean_error_pct"]:>10.2f}  {s["mean_found"]:>10.2f}  '
            f'{s["diverged_fraction"]:>8.2f}  {path:>6.1f}')
    return '\n'.join(lines)


def main(args=None):
    """Entry point: run the requested study and print its table."""
    parser = argparse.ArgumentParser(
        description='Headless evaluation of the MultiPLE estimator (Phase A)')
    parser.add_argument('--study', default='both',
                        choices=('suite', 'non-informative', 'both'),
                        help='which study to run')
    parser.add_argument('--trials', type=int, default=10,
                        help='trials per scenario group')
    parser.add_argument('--seed', type=int, default=0, help='base seed')
    parser.add_argument('--sources', type=int, nargs='+', default=[1, 2, 3],
                        help='source counts for the suite')
    parser.add_argument('--steps', type=int, default=EvalConfig.max_steps,
                        help='simulation steps per trial')
    parser.add_argument('--quick', action='store_true',
                        help='small, fast run for smoke-testing the harness')
    parsed = parser.parse_args(args)

    config = EvalConfig(max_steps=parsed.steps)
    trials = parsed.trials
    sources = parsed.sources
    if parsed.quick:
        config = EvalConfig(max_steps=200, n_particles=600, mi_particles=200)
        trials, sources = 2, [1, 2]

    if parsed.study in ('suite', 'both'):
        grouped = {}
        for n_sources in sources:
            grouped[str(n_sources)] = run_suite(
                [n_sources], n_trials=trials, base_seed=parsed.seed,
                config=config)
        print('\nScenario suite (infotaxis agent, seeded 20 m x 20 m worlds)')
        print(format_suite_table(grouped))
        print('\nPaper reference (Gaussian-plume environment, infotaxis): '
              'success ~74%, mean localization error 1.21%.')

    if parsed.study in ('non-informative', 'both'):
        study = run_non_informative_study(
            n_sources=min(sources), n_trials=trials, base_seed=parsed.seed,
            config=config)
        print('\nNon-informative path study (risk R1): same worlds, '
              'different trajectories')
        print(format_study_table(study))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
