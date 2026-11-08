"""
Tests for analyze_trials' metric functions, on hand-built inputs.

No test reads a bag: the trial bags live outside the repo. Every expected
value here is worked out by hand from the geometry or the 1.5 m rule.
"""

from haptic_plume_experiment.analyze_trials import (
    declaration, inside, min_clearance, path_length, trial_window
)
from haptic_plume_interfaces.msg import TaskEvent
import numpy as np
import pytest

# main_run as obstacles.yaml has it, written out so the test does not depend on the file.
MAIN_RUN = [(np.array([10.0, 5.0, 1.0]), np.array([10.0, 15.0, 1.0]), 0.30)]
LEAK = [10.3, 6.5, 1.0]


def event(event_type='', x=0.0, y=0.0):
    """Build a TaskEvent of one type at one position."""
    msg = TaskEvent()
    msg.event_type = event_type
    msg.position.x = x
    msg.position.y = y
    return msg


def test_path_length_of_a_unit_square_is_four():
    square = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=float)
    assert path_length(square) == pytest.approx(4.0)


def test_clearance_at_the_clamp_boundary_is_the_margin():
    # 0.727 m from the axis: 0.727 - 0.30 pipe - 0.377 drone = 0.05, the clamp margin.
    assert min_clearance(np.array([[9.273, 10.0, 1.0]]), MAIN_RUN) == pytest.approx(0.05)


def test_clearance_beyond_the_end_cap_measures_from_the_end():
    # 1 m past p1 along the axis: the clip to [0, 1] measures from p1, not the infinite line.
    assert min_clearance(np.array([[10.0, 16.0, 1.0]]), MAIN_RUN) == pytest.approx(0.323)


def test_clearance_is_the_minimum_over_the_path():
    path = np.array([[5.0, 10.0, 1.0], [9.273, 10.0, 1.0], [5.0, 10.0, 1.0]])
    assert min_clearance(path, MAIN_RUN) == pytest.approx(0.05)


def test_no_declaration_is_none():
    assert declaration([], 0.0, LEAK) == (None, None, 'none')


@pytest.mark.parametrize('offset, outcome', [(1.0, 'right'), (1.5, 'right'), (2.0, 'wrong')])
def test_declaration_outcome_follows_the_one_and_a_half_metre_rule(offset, outcome):
    # Offset along y only, so the error is exactly the offset.
    consumed = [(105.0, event(TaskEvent.SOURCE_CONSUMED, LEAK[0], LEAK[1] + offset))]
    t_declare, error, result = declaration(consumed, 100.0, LEAK)
    assert t_declare == pytest.approx(5.0)
    assert error == pytest.approx(offset)
    assert result == outcome


def test_trial_window_from_start_and_end():
    events = [(10.0, event(TaskEvent.TRIAL_START)), (80.0, event(TaskEvent.TRIAL_END))]
    assert trial_window(events, 99.0) == (10.0, 80.0, True)


def test_trial_window_without_end_runs_to_the_last_stamp():
    assert trial_window([(10.0, event(TaskEvent.TRIAL_START))], 99.0) == (10.0, 99.0, False)


def test_trial_window_without_start_refuses():
    with pytest.raises(ValueError):
        trial_window([(80.0, event(TaskEvent.TRIAL_END))], 99.0)


def test_inside_keeps_both_window_ends():
    items = [(9.9, 'a'), (10.0, 'b'), (50.0, 'c'), (80.0, 'd'), (80.1, 'e')]
    assert [msg for _, msg in inside(items, 10.0, 80.0)] == ['b', 'c', 'd']
