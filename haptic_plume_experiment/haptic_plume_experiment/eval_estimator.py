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
    STATUS_CONVERGING,
    STATUS_DIVERGED,
)
from haptic_plume_estimation.sensor_compensation import (
    InverseFeedforwardCompensator,
)

from haptic_plume_gas_sim.sensor_model import FirstOrderSensor, GasField, GasSensor

import numpy as np

WELL_BEARING_TOL_DEG = 10.0
WELL_GATES = ('tracking', 'converging')

# The tolerance above is a judgement call, not a measurement, and the Phase C0
# conclusions turn on it: 10 deg is 1.7 m of lateral error at a 10 m standoff,
# but an operator pulled 30 deg off still flies broadly the right way and
# still feeds the filter useful readings. Scored at several widths so the
# conclusion can be checked against the choice that produced it.
WELL_TOLERANCES_DEG = (10.0, 20.0, 30.0, 45.0)

# What the well is aimed at. 'belief' is the filter's guess at the source
# (mne_position). 'waypoint' is the next place worth looking, i.e. the MI
# waypoint the suggester will publish -- which is what D4 says the well should
# track until the belief has converged. Measuring only 'belief' measures half
# the design.
WELL_TARGETS = ('belief', 'waypoint')


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
class WellHunts:
    """
    What the haptic well did during one trial, under one gating rule.

    One entry per hunt: the filter chases one leak at a time, and a hunt runs
    from the trial start or the last filter reset to the next consume. Kept
    raw rather than pre-averaged for the same reason path_cross_at_consume is:
    the risk R1 attempt that averaged over whole trials ended up measuring the
    consume rate instead of the thing it asked about.

    Read the timings through had_target or they lie. Measured 2026-08-14:
    with one leak, 75% of hunts happen after that leak has already been found,
    and 26% do with three. Those hunts drag every average toward "wrong"
    without saying anything about the well, and because the contamination
    falls as leaks are added it reverses the ordering between source counts
    rather than merely offsetting it.
    """

    # [s] into the hunt when a pull first appeared at all
    on_times: np.ndarray = field(default_factory=lambda: np.empty(0))
    # [s] into the hunt when that pull settled onto a real leak; NaN if the
    # hunt ended with the operator still being pulled somewhere wrong
    true_times: np.ndarray = field(default_factory=lambda: np.empty(0))
    # [-] share of pull-on steps pointing further than the tolerance away
    misleading_fractions: np.ndarray = field(
        default_factory=lambda: np.empty(0))
    # True where a leak was still unfound, i.e. the pull had something it
    # could have been right about. False marks the filter chasing residual
    # noise after the last real leak was consumed.
    had_target: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=bool))
    # [s] of pull actually rendered during the hunt. Not the same as the span
    # from t_on to the end: the gating status flickers, so a hunt can span a
    # minute and contain a second of force. A cue that is on for under a
    # second at a time is a force pop (risk R5), not guidance.
    on_durations: np.ndarray = field(default_factory=lambda: np.empty(0))


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
    # Cross-axis path spread at each consume, kept raw rather than
    # pre-averaged: the threshold that separates an informative path from a
    # run down the plume axis is read off the distributions, not from a mean.
    path_cross_at_consume: np.ndarray = field(
        default_factory=lambda: np.empty(0))

    # Haptic-well timings keyed by gating rule (WELL_GATES). Distinct from
    # localization_time_s above, which is when the ESTIMATOR finished — the
    # operator's experience starts and ends at different moments.
    well_hunts: dict = field(default_factory=dict)

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
    path_cross = []
    visited = []
    trackers = [_WellTracker(well_target, gate, config.dt, tol)
                for well_target in WELL_TARGETS
                for gate in WELL_GATES
                for tol in WELL_TOLERANCES_DEG]
    unfound = _unfound_sources(scenario.source_positions, consumed.positions,
                               config.d_cluster)

    for step in range(config.max_steps):
        visited.append(position.copy())
        # --- sense: truth -> lagged, noisy measurement -> compensated ---
        c_true = float(field_truth.concentration(*position))
        z_ff = compensator.update(sensor.measure(c_true))

        # --- believe: subtract what is already consumed, then update ---
        m_hat = float(consumed.concentration(*position))
        status = pf.update(position, z_ff, m_hat=m_hat)
        n_diverged += status == STATUS_DIVERGED

        # --- score the well: what would the operator's hand be feeling? ---
        # Read before the consume block, exactly as path_cross is: after
        # pf.reset() this measures a fresh uniform prior. `target` is last
        # step's waypoint, which is exactly what a live composer renders --
        # the suggester publishes at 2 Hz and the composer holds the last one.
        bearing_error = {
            'belief': _well_bearing_error(position, pf.mne_position, unfound),
            'waypoint': _well_bearing_error(position, target, unfound),
        }
        for tracker in trackers:
            tracker.step(step, status, bearing_error[tracker.target])

        # --- consume: a tight enough belief becomes a localized source ---
        if (should_consume(pf.sigma_position_norm, config.sigma_c)
                and len(consumed) < config.max_sources):
            # Read before pf.reset() below, or this measures a fresh prior.
            path_cross.append(_cross_axis_spread(visited, pf.mmse[3]))
            consumed.push(pf.mmse)
            consumed.merge(config.d_cluster)
            last_consume_step = step
            pf.reset()
            # Scope the path window to this belief. Without the clear, source
            # two onward is judged on flying done for source one, and the gate
            # waves everything through. Mirrors pf.reset() exactly.
            visited.clear()

            # Same scoping for the well: a consume ends one hunt, and the
            # retarget that follows starts the next. Close before recomputing
            # unfound below — the pre-consume value is what says whether this
            # hunt had a real leak to be right about.
            for tracker in trackers:
                tracker.close(step, len(unfound) > 0)
            unfound = _unfound_sources(scenario.source_positions,
                                       consumed.positions, config.d_cluster)

        # --- plan and move: kinematic x_dot = v, paper Eq. (20) ---
        if step % config.plan_every == 0:
            target = PLANNERS[planner](position, pf, scenario, config,
                                       step, plan_state)
        moved = _step_towards(position, target, config.speed * config.dt,
                              scenario.search_area)
        path_length += moved

    # The trial almost always ends mid-hunt; close_if_open skips the empty
    # hunt left behind when the very last step happened to consume.
    for tracker in trackers:
        tracker.close_if_open(config.max_steps - 1, len(unfound) > 0)

    return _summarize(scenario, planner, seed, config, consumed, pf,
                      n_diverged, path_length, last_consume_step, path_cross,
                      trackers)


def _summarize(scenario, planner, seed, config, consumed, pf, n_diverged,
               path_length, last_consume_step, path_cross, trackers):
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
    :param path_cross: cross-axis path spreads recorded at each consume [m]
    :param trackers: the well trackers, already closed
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
        path_cross_at_consume=np.asarray(path_cross, dtype=float),
        well_hunts={t.key: t.hunts() for t in trackers},
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


def _run_grouped(sources, n_trials, base_seed, config):
    """Run the suite once per source count, keyed by count for the tables."""
    return {str(n): run_suite([n], n_trials=n_trials, base_seed=base_seed,
                              config=config)
            for n in sources}


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


def format_observability_table(results_by_planner):
    """
    Render the cross-axis path spread at consume, one row per planner.

    INSTRUMENTATION ONLY — this is not a gate, and the numbers below are why.

    It was built as the risk R1 gate: block the haptic well until the drone
    has flown far enough sideways to the wind that the source position is
    actually observable. Over a whole trial it looked decisive (infotaxis
    4.99 m against downwind 1.29 m). Scoped correctly to the window since the
    last filter reset -- which is the only scoping a live gate can use, since
    source two must not be judged on flying done for source one -- the
    separation inverts:

        1 source : infotaxis p50 0.36 m, downwind p50 0.91 m
        2 sources: infotaxis p50 0.53 m, downwind p50 1.38 m

    The confound is consume rate. infotaxis consumes ~3.2 times per trial and
    downwind ~1.1, so infotaxis windows are short and accumulate less spread.
    The metric reads time-since-last-consume, not flight diversity, and the
    whole-trial version was reading total path length. Kept because the
    measurement is what exposed this, and the next candidate (normalize by
    window length, or restrict to in-plume samples) needs the same plumbing.

    :param results_by_planner: dict mapping planner name to TrialResult list
    :return: the printable table
    :rtype: str
    """
    lines = [
        'planner     n cons   p10 m   p50 m   p90 m',
        '----------  ------  ------  ------  ------',
    ]
    for name, results in results_by_planner.items():
        cross = _stack([r.path_cross_at_consume for r in results])
        if cross.size == 0:
            lines.append(f'{name:<10}  {0:>6}  (never consumed)')
            continue
        p10, p50, p90 = np.percentile(cross, (10, 50, 90))
        lines.append(
            f'{name:<10}  {cross.size:>6}  {p10:>6.2f}  {p50:>6.2f}  '
            f'{p90:>6.2f}')
    return '\n'.join(lines)


def format_well_table(results_by_group):
    """
    Render what the haptic well did, per source count and gating rule.

    The columns are the whole point of Phase C0:

        live      hunts with a leak still to find; every timing below is over
                  these alone
        phantom   hunts the filter ran after the last real leak was already
                  consumed. The operator still feels a pull, but nothing it
                  could point at remains, so scoring one says nothing about
                  the well. Counted rather than hidden: a high number is a
                  real usability problem in its own right, and the reason the
                  system needs an operator "found it" declaration (O3)
        t_on      when a pull first appears in a hunt
        felt      seconds of pull actually rendered in the hunt; far
                  below the span from t_on onward means it flickers
        t_usable  when that pull settles onto a leak and stays there
        wrong %   share of pull-on time spent pointing more than
                  WELL_BEARING_TOL_DEG away from every unfound leak
        never %   share of live hunts where the pull never became usable

    t_usable - t_on is how long the operator is pulled at empty floor, which
    is the number nothing in this project had ever measured.

    :param results_by_group: dict mapping a group label to its TrialResult list
    :return: the printable table
    :rtype: str
    """
    lines = [
        'sources  gate          live  phantom  t_on s  felt s  t_usable s  '
        'wrong %  never %',
        '-------  -----------  -----  -------  ------  ------  ----------  '
        '-------  -------',
    ]
    for label, results in results_by_group.items():
        for gate in WELL_GATES:
            key = ('belief', gate, WELL_BEARING_TOL_DEG)
            lines.append(f'{label:>7}  {gate:<11}  '
                         + _well_row([r.well_hunts[key] for r in results]))
    return '\n'.join(lines)


def format_well_sweep_table(results_by_group):
    """
    Render the well against both targets, over a sweep of tolerances.

    Two questions in one table, both raised by the first Phase C0 run:

    1. Is "usable" being decided by the arbitrary 10 deg bar? Every row is
       the same flight scored at a different width, so a conclusion that
       survives all four widths is about the well and not about the bar.
    2. Does aiming the well at the next place worth looking beat aiming it at
       the current guess? D4 already specifies the former during search, and
       the first run only ever measured the latter.

    Fixed at the 'tracking' gate throughout. The 'converging' gate renders
    0.6-0.8 s of force per hunt (measured 2026-08-14), i.e. about four scored
    steps, so every percentage under it rests on almost no samples and
    never % is decided by a single step. It is also unusable as a cue for
    exactly that reason: CONVERGING spans a belief spread of 0.5-1.0 m, the
    filter crosses that band in well under a second, and the spread wobbles
    across the threshold, so the gate opens and shuts repeatedly. That is a
    force pop (risk R5), not guidance.

    A caveat on the 'waypoint' rows. An information-maximizing waypoint is not
    trying to point at a leak, so bearing-to-leak scores it on a job it was
    never given and probably reads low. The number that would settle it is
    time-to-find under an operator who actually follows the cue, which needs a
    well-following planner in PLANNERS rather than another observer.

    :param results_by_group: dict mapping a group label to its TrialResult list
    :return: the printable table
    :rtype: str
    """
    lines = [
        'sources  target    tol deg   live  phantom  t_on s  felt s  '
        't_usable s  wrong %  never %',
        '-------  --------  -------  -----  -------  ------  ------  '
        '----------  -------  -------',
    ]
    for label, results in results_by_group.items():
        for well_target in WELL_TARGETS:
            for tol in WELL_TOLERANCES_DEG:
                key = (well_target, 'tracking', tol)
                lines.append(
                    f'{label:>7}  {well_target:<8}  {tol:>7.0f}  '
                    + _well_row([r.well_hunts[key] for r in results]))
    return '\n'.join(lines)


def _well_row(hunts):
    """
    Render one row of well statistics, over live hunts only.

    :param hunts: list of WellHunts, one per trial
    :return: the formatted row body, without its leading label columns
    :rtype: str
    """
    on = _stack([h.on_times[h.had_target] for h in hunts])
    felt = _stack([h.on_durations[h.had_target] for h in hunts])
    settled = _stack([h.true_times[h.had_target] for h in hunts])
    wrong = _stack([h.misleading_fractions[h.had_target] for h in hunts])
    n_all = sum(h.had_target.size for h in hunts)
    n_live = sum(int(h.had_target.sum()) for h in hunts)
    never = (100.0 * float(np.mean(~np.isfinite(settled)))
             if settled.size else float('nan'))
    return (f'{n_live:>5}  {n_all - n_live:>7}  {_finite_mean(on):>6.1f}  '
            f'{_finite_mean(felt):>6.1f}  {_finite_mean(settled):>10.1f}  '
            f'{100.0 * _finite_mean(wrong):>7.1f}  {never:>7.1f}')


def _cross_axis_spread(positions, theta):
    """
    Measure the flown path's spread perpendicular to the believed wind axis.

    A property of the TRAJECTORY, not of the belief. Candidate risk R1 gate;
    it does not separate the paths once scoped correctly. See
    format_observability_table for the measured numbers and CLAUDE.md for the
    two negative results this and the belief-shape ratio produced.

    Uses the BELIEVED theta, not the true one, because a live gate only ever
    has the belief. Callers must clear the position window whenever the filter
    resets, or source two onward is judged on flying done for source one.

    :param positions: list of (3,) world positions since the last filter reset
    :param theta: believed wind direction [rad]
    :return: cross-axis spread of the path [m]
    :rtype: float
    """
    if len(positions) < 2:
        return 0.0
    xy = np.asarray(positions, dtype=float)[:, 0:2]
    across = np.array([-np.sin(theta), np.cos(theta)])
    return float(np.std(xy @ across))


def _well_bearing_error(position, target, unfound_positions):
    """
    Bearing error from the rendered pull to the nearest leak still to find.

    Horizontal only: the drone flies at fixed altitude (frames.md section 5)
    and the well is rendered in the horizontal plane (section 6).

    The minimum is taken over every leak not yet found, not over the one the
    belief happens to be chasing. O4 makes which leak comes first emergent,
    so the operator cannot be sent after a specific one and does not care
    which of them a correct pull leads to.

    :param position: (3,) drone position [m]
    :param target: (3,) haptic well target, i.e. the MNE position [m]
    :param unfound_positions: (K, 3) true leaks with no consumed estimate yet
    :return: smallest absolute bearing error [deg], NaN when undefined
    :rtype: float
    """
    if len(unfound_positions) == 0:
        return float('nan')
    here = np.asarray(position, dtype=float)[0:2]
    to_target = np.asarray(target, dtype=float)[0:2] - here
    truths = np.atleast_2d(np.asarray(unfound_positions, dtype=float))
    delta = truths[:, 0:2] - here
    # A pull with no direction, or a leak the drone is sitting on top of, has
    # no bearing to be right or wrong about.
    if (np.linalg.norm(to_target) < 1e-9
            or float(np.min(np.linalg.norm(delta, axis=1))) < 1e-9):
        return float('nan')
    bearing = np.arctan2(to_target[1], to_target[0])
    truth = np.arctan2(delta[:, 1], delta[:, 0])
    error = np.arctan2(np.sin(truth - bearing), np.cos(truth - bearing))
    return float(np.degrees(np.min(np.abs(error))))


def _unfound_sources(true_positions, consumed_positions, d_match):
    """
    Return the true leaks that no consumed estimate has landed near yet.

    :param true_positions: (L, 3) true source positions [m]
    :param consumed_positions: (M, 3) consumed estimate positions [m]
    :param d_match: how close an estimate must be to count as found [m]
    :return: (K, 3) the still-unfound true positions
    :rtype: numpy.ndarray
    """
    truths = np.atleast_2d(np.asarray(true_positions, dtype=float))
    estimates = np.asarray(consumed_positions, dtype=float)
    if estimates.size == 0:
        return truths
    estimates = np.atleast_2d(estimates)
    distances = np.linalg.norm(truths[:, None, :] - estimates[None, :, :],
                               axis=2)
    return truths[distances.min(axis=1) > d_match]


def _finite_mean(values):
    """Mean over the finite entries only; NaN when there are none."""
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float('nan')


class _WellTracker:
    """
    Per-hunt bookkeeping for one (target, gate, tolerance) combination.

    A pure observer: it reads the belief and never writes to it, so a whole
    grid of trackers can run side by side and none of them can change a
    trial's outcome. That is what makes the tolerance sweep free — the
    tolerance decides which steps get scored, not what the drone flew.

    A hunt is one belief's lifetime — trial start or the last pf.reset() to
    the next consume — because that is the only window a live composer could
    ever scope to. Source two must not be credited with flying done for
    source one.
    """

    def __init__(self, target, gate, dt, tol_deg=WELL_BEARING_TOL_DEG):
        """
        Build a tracker for one target, gating rule and tolerance.

        :param target: name from WELL_TARGETS
        :param gate: name from WELL_GATES
        :param dt: simulation step period [s]
        :param tol_deg: bearing error a pull may carry and still be usable
        """
        if target not in WELL_TARGETS:
            raise ValueError(f'unknown well target {target!r}; '
                             f'have {WELL_TARGETS}')
        self.target = target
        if gate not in WELL_GATES:
            raise ValueError(f'unknown well gate {gate!r}; have {WELL_GATES}')
        self.gate = gate
        self.dt = dt
        self.tol_deg = tol_deg
        self.on_times = []
        self.true_times = []
        self.misleading_fractions = []
        self.had_target = []
        self.on_durations = []
        self._start_hunt(0)

    @property
    def key(self):
        """Identify this tracker in TrialResult.well_hunts."""
        return (self.target, self.gate, self.tol_deg)

    def is_on(self, status):
        """
        Report whether this rule would render a pull at all.

        :param status: the filter's status this step
        :return: True when the well is rendered
        :rtype: bool
        """
        if self.gate == 'converging':
            return status == STATUS_CONVERGING
        return status != STATUS_DIVERGED

    def step(self, step, status, bearing_error_deg):
        """
        Score one simulation step.

        :param step: step index within the trial
        :param status: the filter's status this step
        :param bearing_error_deg: bearing error of the rendered pull [deg]
        """
        if not self.is_on(status):
            return
        self._n_on += 1
        self._last_on = step
        if self._first_on is None:
            self._first_on = step
        # NaN counts as misleading: a pull with no leak to point at is not a
        # pull that points at a leak. Whole hunts of it are excluded later by
        # had_target rather than here, so the count stays meaningful either
        # way this is read.
        if not (bearing_error_deg <= self.tol_deg):
            self._n_bad += 1
            self._last_bad = step

    def close(self, step, had_target):
        """
        End the current hunt and record its numbers.

        The settle time is measured backwards from the end of the hunt, not
        taken at the first moment the bearing happened to be right: a diffuse
        belief sweeps past the correct bearing by luck. A hunt whose last
        rendered pull was still wrong never became usable at all, and is
        recorded as NaN rather than as the hunt's length.

        :param step: step index the hunt ended on
        :param had_target: whether a leak was still unfound during this hunt
        """
        self.on_times.append(self._t(self._first_on))
        self.true_times.append(self._settle_time())
        self.misleading_fractions.append(
            self._n_bad / self._n_on if self._n_on else float('nan'))
        self.had_target.append(bool(had_target))
        self.on_durations.append(self._n_on * self.dt)
        self._start_hunt(step + 1)

    def close_if_open(self, step, had_target):
        """Close a partial hunt at the end of a trial; skip an empty one."""
        if step >= self._hunt_start:
            self.close(step, had_target)

    def hunts(self):
        """
        Freeze what was recorded into arrays.

        :return: the per-hunt timings
        :rtype: WellHunts
        """
        return WellHunts(
            on_times=np.asarray(self.on_times, dtype=float),
            true_times=np.asarray(self.true_times, dtype=float),
            misleading_fractions=np.asarray(self.misleading_fractions,
                                            dtype=float),
            had_target=np.asarray(self.had_target, dtype=bool),
            on_durations=np.asarray(self.on_durations, dtype=float))

    def _start_hunt(self, step):
        """Begin a fresh hunt, discarding the previous one's counters."""
        self._hunt_start = step
        self._first_on = None
        self._last_on = None
        self._last_bad = None
        self._n_on = 0
        self._n_bad = 0

    def _t(self, step):
        """Convert an absolute step index to seconds into this hunt."""
        if step is None:
            return float('nan')
        return (step - self._hunt_start) * self.dt

    def _settle_time(self):
        """Seconds in after which no rendered pull was ever misleading."""
        if self._first_on is None:
            return float('nan')                 # nothing was ever rendered
        if self._last_bad is None:
            return self._t(self._first_on)      # right from the first tick
        if self._last_bad == self._last_on:
            return float('nan')                 # still wrong when it ended
        return self._t(self._last_bad + 1)


def _stack(arrays):
    """
    Concatenate a list of 1-D arrays, tolerating an empty list.

    :param arrays: list of 1-D numpy arrays
    :return: the concatenation, or an empty array
    :rtype: numpy.ndarray
    """
    arrays = [a for a in arrays if a.size]
    return np.concatenate(arrays) if arrays else np.empty(0)


def main(args=None):
    """Entry point: run the requested study and print its table."""
    parser = argparse.ArgumentParser(
        description='Headless evaluation of the MultiPLE estimator (Phase A)')
    parser.add_argument('--study', default='both',
                        choices=('suite', 'non-informative', 'well', 'both'),
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
        grouped = _run_grouped(sources, trials, parsed.seed, config)
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
        print('\nObservability at consume (cross-axis path spread since the '
              'last reset) — INSTRUMENTATION, not a working gate')
        print(format_observability_table(study))

    if parsed.study == 'well':
        grouped = _run_grouped(sources, trials, parsed.seed, config)
        print('\nHaptic well observability (Phase C0): when does a pull '
              'appear, and when does it start telling the truth?')
        print(format_well_table(grouped))
        print(f'\nBearing tolerance {WELL_BEARING_TOL_DEG:.0f} deg. Compare '
              't_usable against the mean time s column of --study suite, '
              'which is when the ESTIMATOR finished, not the operator.')
        print('\nSame flights, scored against both well targets over a sweep '
              'of tolerances (tracking gate)')
        print(format_well_sweep_table(grouped))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
