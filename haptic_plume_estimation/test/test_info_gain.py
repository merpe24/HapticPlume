"""
Unit tests for haptic_plume_estimation.info_gain (paper Eqs. 15-21).

Two of the expected values are exact and hand-derived:

  * a belief with no disagreement carries no information, I = 0;
  * a 50/50 belief whose two predictions are far apart relative to r carries
    exactly log 2 nats, because the mixture is then two disjoint Gaussians.

The rest pin the behaviour the haptic guidance depends on: information sits
where the models disagree, which is across the plume rather than up it.

:author: premmm
:date: July 29, 2026
"""

from haptic_plume_estimation.info_gain import (
    best_waypoint,
    candidate_waypoints,
    conditional_entropy,
    mixture_entropy,
    mutual_information,
    waypoint_utilities,
)

import numpy as np

import pytest

R = 0.01   # likelihood deviation used throughout [kg/m^3]


def two_mode_belief(y_a, y_b, n_each=250):
    """
    Build a belief split evenly between two candidate source positions.

    Both modes carry the same release rate and wind, so the only thing a
    measurement can resolve is where the source sits.
    """
    template = np.array([5.0, 10.0, 1.0, 0.0, 0.025, 0.5, 0.035, 5.0e-4])
    particles = np.tile(template, (2 * n_each, 1))
    particles[:n_each, 1] = y_a
    particles[n_each:, 1] = y_b
    weights = np.full(2 * n_each, 1.0 / (2 * n_each))
    return particles, weights


def test_conditional_entropy_matches_the_numerical_integral():
    """
    Eq. (19) in closed form equals Eq. (18) when the belief has one mode.

    With every particle predicting the same value the mixture IS a single
    Gaussian, so the two entropies must agree — the check that the analytic
    shortcut in conditional_entropy() is the same quantity Eq. (19) defines.
    """
    g = np.full(500, 0.05)
    weights = np.full(500, 1.0 / 500)
    assert mixture_entropy(g, weights, R) == pytest.approx(
        conditional_entropy(R), rel=1e-4)


def test_a_belief_that_agrees_carries_no_information():
    """I = H(z) - H(z|alpha) = 0 when no measurement can discriminate."""
    g = np.full(500, 0.05)
    weights = np.full(500, 1.0 / 500)
    assert mutual_information(g, weights, R) == pytest.approx(0.0, abs=1e-4)


def test_two_separated_modes_carry_log_two_nats():
    """
    A 50/50 belief with well-separated predictions gives exactly log 2.

    Hand-derived: if |g_a - g_b| >> r the mixture is two disjoint Gaussians,
    H(z) = 0.5 log(2 pi e r^2) + log 2, so I = log 2 = 0.693 nats — one bit,
    which is precisely the one yes/no question the measurement answers.
    """
    g = np.concatenate([np.full(250, 0.02), np.full(250, 0.20)])
    weights = np.full(500, 1.0 / 500)
    assert mutual_information(g, weights, R) == pytest.approx(
        np.log(2.0), abs=5e-3)


def test_uneven_modes_carry_the_binary_entropy():
    """
    A 25/75 split carries H(0.25) nats, not log 2.

    Confirms the weights reach the mixture rather than being ignored.
    """
    g = np.concatenate([np.full(250, 0.02), np.full(750, 0.20)])
    weights = np.concatenate([np.full(250, 0.25 / 250),
                              np.full(750, 0.75 / 750)])
    expected = -(0.25 * np.log(0.25) + 0.75 * np.log(0.75))
    assert mutual_information(g, weights, R) == pytest.approx(expected, abs=5e-3)


def test_overlapping_modes_carry_less_than_separated_ones():
    """Predictions within r of each other are nearly indistinguishable."""
    weights = np.full(500, 1.0 / 500)
    separated = np.concatenate([np.full(250, 0.02), np.full(250, 0.20)])
    overlapping = np.concatenate([np.full(250, 0.02), np.full(250, 0.021)])
    assert (mutual_information(overlapping, weights, R)
            < 0.1 * mutual_information(separated, weights, R))


def test_information_is_highest_where_the_models_disagree():
    """
    Paper Fig. 5: not where concentration is highest, but where beliefs split.

    The belief cannot decide between a source at y = 9 and one at y = 11.
    Sitting on the symmetry line at y = 10 measures both modes identically and
    learns nothing; offsetting to y = 9 makes the two hypotheses predict very
    different readings.
    """
    particles, weights = two_mode_belief(9.0, 11.0)
    on_symmetry_line = np.array([[10.0, 10.0, 1.0]])
    off_symmetry_line = np.array([[10.0, 9.0, 1.0]])
    _, mi_symmetric = waypoint_utilities(particles, weights, on_symmetry_line, R)
    _, mi_offset = waypoint_utilities(particles, weights, off_symmetry_line, R)
    assert mi_offset[0] > 5.0 * mi_symmetric[0]


def test_the_planner_prefers_crosswind_travel():
    """
    Crosswind candidates beat along-wind ones for a crosswind ambiguity.

    The wind blows along +x, and the belief is unsure of the source's
    crosswind position. Moving downwind samples the same streamline again;
    moving across the wind separates the hypotheses. This is the behaviour the
    haptic guidance cue exists to convey, and it is why the Phase E scenarios
    are laid out to force crosswind motion.
    """
    particles, weights = two_mode_belief(9.0, 11.0)
    start = np.array([10.0, 10.0, 1.0])
    downwind, crosswind = start + [2.0, 0.0, 0.0], start + [0.0, 2.0, 0.0]
    _, information = waypoint_utilities(
        particles, weights, np.vstack([downwind, crosswind]), R)
    assert information[1] > information[0]


def test_best_waypoint_returns_the_argmax_of_the_utility():
    """Eq. (21) maximizes V = I^2; the index, point and MI must agree."""
    particles, weights = two_mode_belief(9.0, 11.0)
    points = candidate_waypoints((10.0, 10.0, 1.0), step=2.0, n_directions=8)
    utilities, information = waypoint_utilities(particles, weights, points, R)
    index, point, mi = best_waypoint(particles, weights, points, R)
    assert index == int(np.argmax(utilities))
    np.testing.assert_allclose(point, points[index])
    assert mi == pytest.approx(information[index])
    np.testing.assert_allclose(utilities, information**2, rtol=1e-12)


def test_consumed_plumes_do_not_change_the_score():
    """
    Subtracting A-hat shifts the mixture rigidly, and entropy ignores that.

    Documented here because it is why info_gain.py never imports consume.py:
    the consumed set reaches the planner only through the particles.
    """
    _, weights = two_mode_belief(9.0, 11.0)
    g = np.concatenate([np.full(250, 0.02), np.full(250, 0.20)])
    shifted = g + 0.037
    assert mutual_information(shifted, weights, R) == pytest.approx(
        mutual_information(g, weights, R), rel=1e-6)


def test_candidate_ring_geometry():
    """Candidates sit at the requested radius, level, and inside the bounds."""
    points = candidate_waypoints((10.0, 10.0, 1.5), step=2.0, n_directions=4)
    assert points.shape == (4, 3)
    assert np.all(points[:, 2] == 1.5)
    np.testing.assert_allclose(
        np.linalg.norm(points[:, :2] - [10.0, 10.0], axis=1), 2.0)

    with_hold = candidate_waypoints((10.0, 10.0, 1.5), step=2.0,
                                    n_directions=4, include_hold=True)
    assert with_hold.shape == (5, 3)
    np.testing.assert_allclose(with_hold[-1], [10.0, 10.0, 1.5])

    clipped = candidate_waypoints((19.5, 10.0, 1.0), step=2.0, n_directions=4,
                                  bounds=np.array([[0.0, 20.0], [0.0, 20.0]]))
    assert np.all(clipped[:, 0] <= 20.0)


def test_invalid_arguments_are_rejected():
    """A non-positive deviation and mismatched lengths are programming errors."""
    weights = np.full(10, 0.1)
    with pytest.raises(ValueError):
        conditional_entropy(0.0)
    with pytest.raises(ValueError):
        mixture_entropy(np.zeros(10), weights, 0.0)
    with pytest.raises(ValueError):
        mixture_entropy(np.zeros(9), weights, R)
