"""
Tests for replay_bag's reset finder and replay step, on synthetic streams.

No test reads a bag: the trial bags live outside the repo. The streams are
built with the real compensator, reset at a known index, so the right answer
is known by construction. Each check comes with a control that must fail.
"""

from haptic_plume_experiment.replay_bag import find_start, make_compensator, Replay
from haptic_plume_interfaces.msg import GasConcentration
import numpy as np
import pytest

# The live estimator_params.yaml values, written out so the test does not need it.
PARAMS = {'tau_rise': 3.0, 'tau_rec': 3.0, 'sensor_rate': 20.0, 'gamma': 5.0,
          'pf_decimation': 4, 'n_particles': 200, 'q': 0.1, 'sigma_c': 0.5,
          'd_cluster': 1.8, 'n_twins': 3, 'agreement_m': 1.5, 'max_sources': 1}
RESET = 10      # index of the first sample the reset compensator sees
DT = 0.05


def sample(concentration, x=5.0, y=5.0):
    """Build one raw GasConcentration at a position."""
    msg = GasConcentration()
    msg.concentration = concentration
    msg.position.x, msg.position.y, msg.position.z = x, y, 1.0
    return msg


def stream(values, reset_at):
    """
    Build (raw, compensated, summary_stamps) the way the live node would.

    One compensator runs from sample 0 and is reset just before reset_at, as
    reset_estimator does; summaries land on every pf_decimation-th sample after.
    """
    raw = [(index * DT, sample(value)) for index, value in enumerate(values)]
    compensator = make_compensator(PARAMS)
    compensated = {}
    for index, (stamp, msg) in enumerate(raw):
        if index == reset_at:
            compensator.reset()
        compensated[stamp] = compensator.update(msg.concentration)
    decimation = PARAMS['pf_decimation']
    summaries = {raw[j][0] for j in range(reset_at + decimation - 1, len(raw), decimation)}
    return raw, compensated, summaries


def values_with_zero_at_reset():
    """Return a noisy, non-zero signal with an exact 0.0 at the reset sample."""
    values = 0.05 + 0.01 * np.random.default_rng(3).standard_normal(60)
    values[RESET] = 0.0
    return values.tolist()


def test_find_start_picks_the_true_reset_over_the_zero_sample_impostor():
    raw, compensated, summaries = stream(values_with_zero_at_reset(), RESET)
    (on_phase, index), candidates = find_start(raw, compensated, summaries, 1.0, PARAMS)
    assert index == RESET
    assert on_phase == len(summaries)
    # The trap: a raw 0.0 leaves a fresh compensator unchanged, so RESET + 1
    # also reproduces the stream, and only the decimation phase rejects it.
    assert (0, RESET + 1) in candidates


def test_find_start_refuses_a_stream_that_was_never_reset():
    # Control: drop the first 5 samples, so the compensator already carries
    # state when the bag begins and no index is a fresh start.
    raw, compensated, summaries = stream(values_with_zero_at_reset(), reset_at=-1)
    with pytest.raises(ValueError):
        find_start(raw[5:], compensated, summaries, 1.0, PARAMS)


def summaries_of(pf_seed, n_samples=16):
    """Run a Replay over a fixed stream and return what step() produced."""
    replay = Replay(PARAMS, pf_seed)
    return [replay.step(sample(0.02, x=4.0 + 0.1 * i))[1] for i in range(n_samples)]


def test_step_updates_the_filter_every_pf_decimation_samples():
    produced = summaries_of(0)
    assert [summary is not None for summary in produced] == [False, False, False, True] * 4


def test_replay_is_bit_exact_for_a_seed_and_differs_for_another():
    assert summaries_of(0) == summaries_of(0)
    assert summaries_of(0) != summaries_of(1)
