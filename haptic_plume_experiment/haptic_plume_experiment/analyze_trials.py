"""
Turn a folder of per-trial bags into one metrics table.

Every metric is clipped to TRIAL_START -> TRIAL_END: the director starts the
bag before the drone reset, so anything earlier is the previous trial's end
pose and the jump back to the start.
"""

import argparse
import csv
import glob
import math
import os
import statistics

from haptic_plume_experiment.bag_io import read_bag, read_trial_yaml
from haptic_plume_experiment.eval_estimator import DECLARATION_OK_M, load_pipe_segments
from haptic_plume_interfaces.msg import TaskEvent
import numpy as np

# The airframe's collision radius (Holybro X500 V2), as drone_params.yaml has it.
DRONE_RADIUS = 0.377

# The only needed topics
TOPICS = ['/hp/odom', '/hp/events/trial', '/hp/events/source_consumed', '/hp/events/collision']

BAG_ROOT = os.path.expanduser('~/hp_bags/trials')


def trial_window(trial_events, last_stamp):
    """
    Return (t_start, t_end, ended) from the /hp/events/trial messages.

    :param trial_events: [(stamp_s, TaskEvent)] as read_bag returns them
    :param last_stamp: the bag's last stamp, used when TRIAL_END is missing
    :return: window bounds [s], and False when the trial never ended
    :rtype: tuple
    """
    starts = [stamp for stamp, event in trial_events
              if event.event_type == TaskEvent.TRIAL_START]
    ends = [stamp for stamp, event in trial_events
            if event.event_type == TaskEvent.TRIAL_END]
    if not starts:
        raise ValueError('no TRIAL_START in this bag')
    if ends:
        return starts[0], ends[-1], True
    return starts[0], last_stamp, False


def inside(items, t_start, t_end):
    """Keep the (stamp_s, msg) pairs stamped inside the trial window."""
    return [(stamp, msg) for stamp, msg in items if t_start <= stamp <= t_end]


def declaration(consumed_events, t_start, true_leak):
    """
    Score the trial's first declaration against the true leak.

    :param consumed_events: SOURCE_CONSUMED (stamp_s, TaskEvent) pairs,
        already clipped to the trial window
    :param t_start: TRIAL_START stamp [s]
    :param true_leak: [x, y, z] from trial.yaml [m]; z is ignored
    :return: (t_declare_s, error_m, outcome); None, None, 'none' if no declaration
    :rtype: tuple
    """
    if not consumed_events:
        return None, None, 'none'
    stamp, event = consumed_events[0]
    error = math.hypot(event.position.x - true_leak[0],
                       event.position.y - true_leak[1])
    outcome = 'right' if error <= DECLARATION_OK_M else 'wrong'
    return stamp - t_start, error, outcome


def path_length(points):
    """Sum of straight-line steps between consecutive (N, 3) positions [m]."""
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def min_clearance(points, segments, drone_radius=DRONE_RADIUS):
    """
    Return the closest the drone's body came to any pipe surface [m].

    0 is contact, room walls are not counted.

    :param points: (N, 3) drone positions [m]
    :param segments: [(p0, p1, radius)] capsules, as load_pipe_segments returns
    :param drone_radius: the drone is treated as a sphere of this radius [m]
    :return: the minimum clearance over all points and pipes [m]
    :rtype: float
    """
    nearest = np.inf
    for p0, p1, radius in segments:
        axis = p1 - p0
        t = np.clip((points - p0) @ axis / (axis @ axis), 0.0, 1.0)
        distance = np.linalg.norm(points - (p0 + t[:, None] * axis), axis=1)
        nearest = min(nearest, float(distance.min()) - radius - drone_radius)
    return nearest


def analyze_trial(path, segments):
    """
    Compute one metrics row from one trial's bag folder.

    The window includes both TRIAL_START and TRIAL_END.
    git_commit, git_dirty and ended_cleanly come along so every number
    can be traced to its code.

    :param path: bag folder holding the mcap and trial.yaml
    :param segments: [(p0, p1, radius)] capsules, as load_pipe_segments returns
    :return: one row, keyed by column name
    :rtype: dict
    """
    meta = read_trial_yaml(path)
    bag = read_bag(path, TOPICS)
    t_start, t_end, ended = trial_window(bag['/hp/events/trial'],
                                         bag['/hp/odom'][-1][0])
    points = np.array([[m.pose.pose.position.x, m.pose.pose.position.y,
                        m.pose.pose.position.z]
                       for _, m in inside(bag['/hp/odom'], t_start, t_end)])
    t_declare, error, outcome = declaration(
        inside(bag['/hp/events/source_consumed'], t_start, t_end),
        t_start, meta['true_leak'][0])
    return {
        'bag': os.path.basename(os.path.normpath(path)),
        'trial_id': meta['trial_id'], 'condition': meta['condition'],
        'scenario_seed': meta['scenario_seed'], 'pf_seed': meta['pf_seed'],
        'git_commit': meta['git_commit'][:7], 'git_dirty': meta['git_dirty'],
        'ended_cleanly': ended, 'duration_s': round(t_end - t_start, 2),
        't_declare_s': None if t_declare is None else round(t_declare, 2),
        'error_m': None if error is None else round(error, 3),
        'outcome': outcome,
        'path_length_m': round(path_length(points), 3),
        'min_pipe_clearance_m': round(min_clearance(points, segments), 3),
        'n_collisions': len(inside(bag['/hp/events/collision'], t_start, t_end)),
    }


def summarize(rows):
    """Return one summary line per condition: outcome shares and medians."""
    lines = []
    for condition in sorted({row['condition'] for row in rows}):
        group = [row for row in rows if row['condition'] == condition]
        declared = [row for row in group if row['outcome'] != 'none']
        share = {outcome: 100 * sum(row['outcome'] == outcome for row in group) / len(group)
                 for outcome in ('right', 'wrong', 'none')}

        def median(key, items):
            return statistics.median(row[key] for row in items) if items else math.nan

        lines.append(
            f'condition {condition}: n {len(group)}  right/wrong/none '
            f'{share["right"]:.0f}/{share["wrong"]:.0f}/{share["none"]:.0f} %  '
            f't_declare p50 {median("t_declare_s", declared):.1f} s  '
            f'error p50 {median("error_m", declared):.2f} m  '
            f'path p50 {median("path_length_m", group):.1f} m')
    return lines


def main(argv=None):
    """Analyze every trial bag, write metrics.csv, and print the table."""
    parser = argparse.ArgumentParser(description='Metrics table from per-trial bags.')
    parser.add_argument('bags', nargs='*',
                        help='bag folders; default: every folder under --bag-root '
                             'that holds a trial.yaml')
    parser.add_argument('--bag-root', default=BAG_ROOT)
    parser.add_argument('--out', help='CSV path; default <bag-root>/metrics.csv')
    args = parser.parse_args(argv)

    bags = args.bags or sorted(
        os.path.dirname(found)
        for found in glob.glob(os.path.join(args.bag_root, '*', 'trial.yaml')))
    if not bags:
        print(f'no trial bags under {args.bag_root}')
        return
    segments = load_pipe_segments()
    rows = [analyze_trial(bag, segments) for bag in bags]

    out = args.out or os.path.join(args.bag_root, 'metrics.csv')
    with open(out, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print('\n'.join(summarize(rows)))
    print(f'wrote {len(rows)} rows to {out}')
