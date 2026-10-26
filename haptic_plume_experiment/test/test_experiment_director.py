"""
Tests for experiment_director_node's condition table.

Every condition must switch on exactly its own cues and displays, and IDLE must
switch everything off, becuase IDLE is what makes the forces zero during a reset.

:author: premmm
:date: October 26, 2026
"""

from haptic_plume_experiment.experiment_director_node import (
    make_condition, make_idle
)

from haptic_plume_interfaces.msg import ConditionConfig

import pytest

def cues(config):
    """Return the three haptic cue flags."""
    return (config.prf_enabled, config.well_enabled, config.heaviness_enabled)

def displays(config):
    """Return the four visual display flags."""
    return (config.show_belief_markers, config.show_particles,
            config.show_gauge, config.show_wind_arrow)

@pytest.mark.parametrize('condition, haptic, visual', [
    (ConditionConfig.VISUAL_ONLY, False, True),
    (ConditionConfig.HAPTIC_ONLY, True, False),
    (ConditionConfig.VISUAL_HAPTIC, True, True)])
def test_condition_sets_its_flags(condition, haptic, visual):
    """Each condition turns on exactly its own channels and keeps the seeds."""
    config = make_condition(condition, 't1', 10, 0)
    assert cues(config) == (haptic,) * 3
    assert displays(config) == (visual,) * 4
    assert (config.trial_id, config.scenario_seed, config.pf_seed) == ('t1', 10, 0)

def test_idle_switches_everything_off():
    """IDLE has no cue and no display on."""
    idle = make_idle()
    assert cues(idle) == (False,) * 3
    assert displays(idle) == (False,) * 4

def test_unknown_condition_is_refused():
    """A condition number outside 0 to 2 is an error"""
    with pytest.raises(ValueError):
        make_condition(3, 't1', 10, 0)
