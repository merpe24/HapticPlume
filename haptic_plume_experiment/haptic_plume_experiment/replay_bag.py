"""
Replay a trial bag through the estimator library and check it is bit-exact (D7).

The raw /hp/gas/measurement samples go through the same compensator, filter,
twins and consume test the live plume_estimator_node runs, in the same order,
and every output is compared with == against what the node published.
"""

import argparse
import os

from ament_index_python.packages import get_package_share_directory
from haptic_plume_estimation.agreement_gate import AgreementGate, twin_seeds
from haptic_plume_estimation.consume import ConsumedSet, should_consume
from haptic_plume_estimation.particle_filter import PlumeFilterConfig, PlumeParticleFilter
from haptic_plume_estimation.sensor_compensation import InverseFeedforwardCompensator
from haptic_plume_experiment.bag_io import read_bag, read_trial_yaml
from haptic_plume_interfaces.msg import TaskEvent
import numpy as np
import yaml

# How many compensated values in a row a candidate start must reproduce with ==.
N_CHECK = 12

# The first post-reset sample can arrive just after TRIAL_START, so search past it.
SEARCH_PAST_START_S = 1.0

RAW = '/hp/gas/measurement'
COMPENSATED = '/hp/gas/measurement_compensated'
SUMMARY = '/hp/belief/summary'
CONSUMED = '/hp/events/source_consumed'
TRIAL = '/hp/events/trial'


def load_params():
    """Return the installed estimator_params.yaml, the file the live node loads."""
    path = os.path.join(get_package_share_directory('haptic_plume_estimation'),
                        'config', 'estimator_params.yaml')
    with open(path) as handle:
        return yaml.safe_load(handle)['plume_estimator']['ros__parameters']


def make_compensator(params):
    """Build the compensator at the SENSOR's dt, as the node does."""
    return InverseFeedforwardCompensator(params['tau_rise'], params['tau_rec'],
                                         1.0 / params['sensor_rate'], params['gamma'])


def reproduces(raw, compensated, index, params):
    """Return True if a fresh compensator from raw[index] reproduces N_CHECK outputs."""
    compensator = make_compensator(params)
    for stamp, msg in raw[index:index + N_CHECK]:
        if compensator.update(msg.concentration) != compensated.get(stamp):
            return False
    return True


def find_start(raw, compensated, summary_stamps, t_trial, params):
    """
    Find the raw sample index where the director's reset_estimator took effect.

    A candidate is an index from which a fresh compensator reproduces the live
    compensated stream exactly. A raw 0.0 leaves a fresh compensator unchanged,
    so the sample after the true reset can pass too; the true one is the one
    whose decimation phase puts filter updates on the belief summary stamps.

    :param raw: [(stamp_s, GasConcentration)] from /hp/gas/measurement
    :param compensated: {stamp_s: concentration} from measurement_compensated
    :param summary_stamps: set of /hp/belief/summary stamps
    :param t_trial: TRIAL_START stamp [s]
    :param params: estimator parameters, as load_params returns them
    :return: ((on_phase, index) chosen, every candidate); ties go to the latest
    :rtype: tuple
    """
    decimation = params['pf_decimation']
    candidates = []
    for index, (stamp, _) in enumerate(raw):
        if stamp > t_trial + SEARCH_PAST_START_S or index + N_CHECK > len(raw):
            break
        if reproduces(raw, compensated, index, params):
            update_stamps = [raw[j][0] for j in range(index + decimation - 1, len(raw),
                                                      decimation)]
            on_phase = sum(update in summary_stamps for update in update_stamps)
            candidates.append((on_phase, index))
    if not candidates:
        raise ValueError('no reset found: different estimator_params.yaml or pf_seed?')
    return max(candidates), candidates


class Replay:
    """The node's estimator state, as reset_estimator leaves it at a trial start."""

    def __init__(self, params, pf_seed):
        """Build a fresh filter, twins, consumed set and compensator from pf_seed."""
        self.params = params
        config = PlumeFilterConfig(n_particles=params['n_particles'], q=params['q'],
                                   sigma_c=params['sigma_c'],
                                   d_cluster=params['d_cluster'])
        self.pf = PlumeParticleFilter(config, np.random.default_rng(pf_seed))
        self.gate = None
        if params['n_twins'] > 0:
            self.gate = AgreementGate(config, twin_seeds(pf_seed, params['n_twins']),
                                      params['agreement_m'])
        self.consumed = ConsumedSet()
        self.compensator = make_compensator(params)
        self.sample_count = 0
        self.withheld = 0

    def step(self, msg):
        """
        Process one raw sample in measurement_callback's exact order.

        :param msg: one GasConcentration from /hp/gas/measurement
        :return: (z_ff, summary, declared); summary is None between filter
            updates, declared is the [x, y, z] position or None
        :rtype: tuple
        """
        z_ff = self.compensator.update(msg.concentration)
        self.sample_count += 1
        if self.sample_count % self.params['pf_decimation']:
            return z_ff, None, None

        position = np.array([msg.position.x, msg.position.y, msg.position.z])
        m_hat = float(self.consumed.concentration(*position))
        self.pf.update(position, z_ff, m_hat=m_hat)
        if self.gate is not None:
            self.gate.update(position, z_ff, m_hat=m_hat)

        declared = None
        if (should_consume(self.pf.sigma_position_norm, self.params['sigma_c'])
                and len(self.consumed) < self.params['max_sources']):
            spread = self.gate.spread(self.pf) if self.gate is not None else 0.0
            if self.gate is None or spread <= self.params['agreement_m']:
                alpha = self.pf.mmse
                declared = alpha[:3].tolist()
                self.consumed.push(alpha)
                self.consumed.merge(self.params['d_cluster'])
                self.pf.reset()
                if self.gate is not None:
                    self.gate.reset()
            else:
                self.withheld += 1
                self.pf.reset()
                self.gate.reset()

        summary = (self.pf.mmse.tolist(), self.pf.sigma_position_norm,
                   float(self.pf.n_eff), len(self.consumed),
                   self.pf.mne_position.tolist())
        return z_ff, summary, declared


def live_summary(msg):
    """Return a BeliefSummary's compared fields, in Replay.step's summary order."""
    target = msg.haptic_target
    return (list(msg.mean), msg.sigma_position_norm, msg.n_eff, msg.n_consumed,
            [target.x, target.y, target.z])


def replay_bag(path, pf_seed=None):
    """
    Replay one trial bag and compare every output with the live node's.

    Every live message from the reset on must be matched with ==, so a replay
    that produces fewer outputs than the node cannot pass.

    :param path: bag folder holding the mcap and trial.yaml
    :param pf_seed: override trial.yaml's seed; a wrong one must fail
    :return: {stream: (matched, live_total)}, the reset index, withheld count
    :rtype: tuple
    """
    params = load_params()
    if pf_seed is None:
        pf_seed = read_trial_yaml(path)['pf_seed']
    bag = read_bag(path, [RAW, COMPENSATED, SUMMARY, CONSUMED, TRIAL])
    raw = bag[RAW]
    compensated = {stamp: msg.concentration for stamp, msg in bag[COMPENSATED]}
    summaries = {stamp: live_summary(msg) for stamp, msg in bag[SUMMARY]}
    t_trial = next(stamp for stamp, event in bag[TRIAL]
                   if event.event_type == TaskEvent.TRIAL_START)
    (_, start), _ = find_start(raw, compensated, set(summaries), t_trial, params)

    replay = Replay(params, pf_seed)
    matched = {'compensated': 0, 'summary': 0}
    declared = []
    for stamp, msg in raw[start:]:
        z_ff, summary, position = replay.step(msg)
        matched['compensated'] += z_ff == compensated.get(stamp)
        if summary is not None:
            matched['summary'] += summary == summaries.get(stamp)
        if position is not None:
            declared.append(position)

    t_reset = raw[start][0]
    live_declared = [[event.position.x, event.position.y, event.position.z]
                     for stamp, event in bag[CONSUMED] if stamp >= t_reset]
    results = {
        'compensated': (matched['compensated'],
                        sum(stamp >= t_reset for stamp in compensated)),
        'summary': (matched['summary'], sum(stamp >= t_reset for stamp in summaries)),
        'declaration': (int(declared == live_declared), 1),
    }
    return results, start, replay.withheld


def main(argv=None):
    """Replay each bag; the exit code is 1 if any stream mismatches."""
    parser = argparse.ArgumentParser(description='Bit-exact estimator replay of trial bags.')
    parser.add_argument('bags', nargs='+', help='bag folders holding an mcap and trial.yaml')
    parser.add_argument('--pf-seed', type=int,
                        help="override trial.yaml's pf_seed (a wrong seed must fail)")
    args = parser.parse_args(argv)

    failed = False
    for path in args.bags:
        results, start, withheld = replay_bag(path, args.pf_seed)
        ok = all(matched == total for matched, total in results.values())
        failed = failed or not ok
        streams = '  '.join(f'{name} {matched}/{total}'
                            for name, (matched, total) in results.items())
        print(f'{"MATCH   " if ok else "MISMATCH"} {path}  reset at sample {start}  '
              f'withheld {withheld}  {streams}')
    return 1 if failed else 0
