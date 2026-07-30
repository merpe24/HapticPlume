"""
Unit tests for haptic_plume_core.scenario.

The promise being tested is the one the experiment depends on: a seed fully
determines a world, and that world survives a trip through plain Python types
without any column quietly changing meaning (risk R4).

:author: premmm
:date: July 29, 2026
"""

from haptic_plume_core.plume_model import PLUME_PARAM_NAMES

from haptic_plume_core.scenario import (
    MIN_SOURCE_SEPARATION,
    random_scenario,
    Scenario,
    SEARCH_AREA,
    SOURCE_HEIGHT,
    SOURCE_MARGIN,
)

import numpy as np

import pytest


def test_same_seed_gives_an_identical_world():
    """Scenario determinism is what makes a trial repeatable (risk R13)."""
    first = random_scenario(3, seed=7)
    second = random_scenario(3, seed=7)
    np.testing.assert_array_equal(first.alphas, second.alphas)
    np.testing.assert_array_equal(first.start_position, second.start_position)
    assert first.name == second.name == 's3_seed7'


def test_different_seeds_give_different_worlds():
    """The seed reaches the sampler rather than being decorative."""
    assert not np.allclose(random_scenario(3, seed=7).alphas,
                           random_scenario(3, seed=8).alphas)


def test_sources_respect_the_separation_and_margin():
    """
    Paper Section 4.1: leaks land in the centred 18 x 18 m box, 2 m apart.

    Overlapping leaks would make the source count ambiguous and the clustering
    threshold d = 1.8 m meaningless.
    """
    for seed in range(20):
        scenario = random_scenario(3, seed=seed)
        positions = scenario.source_positions
        assert np.all(positions[:, 0] >= SEARCH_AREA[0, 0] + SOURCE_MARGIN)
        assert np.all(positions[:, 0] <= SEARCH_AREA[0, 1] - SOURCE_MARGIN)
        assert np.all(positions[:, 1] >= SEARCH_AREA[1, 0] + SOURCE_MARGIN)
        assert np.all(positions[:, 1] <= SEARCH_AREA[1, 1] - SOURCE_MARGIN)
        assert np.all(positions[:, 2] == SOURCE_HEIGHT)
        for i in range(scenario.n_sources):
            for j in range(i + 1, scenario.n_sources):
                assert np.linalg.norm(positions[i] - positions[j]) >= \
                    MIN_SOURCE_SEPARATION


def test_shared_wind_gives_every_plume_the_same_direction_and_speed():
    """One physical wind field means one theta and one v (default)."""
    scenario = random_scenario(3, seed=1)
    assert len(set(scenario.alphas[:, 3])) == 1
    assert len(set(scenario.alphas[:, 5])) == 1


def test_per_plume_wind_can_be_requested():
    """The paper draws wind per plume; that option stays available."""
    scenario = random_scenario(3, seed=1, shared_wind=False)
    assert len(set(scenario.alphas[:, 3])) == 3


def test_parameters_stay_inside_the_paper_ranges():
    """Table 2 ranges bound every drawn parameter."""
    alphas = random_scenario(3, seed=5).alphas
    assert np.all(np.abs(alphas[:, 3]) <= np.pi / 6.0)
    assert np.all((alphas[:, 4] >= 2.0e-2) & (alphas[:, 4] <= 3.0e-2))
    assert np.all((alphas[:, 5] >= 0.2) & (alphas[:, 5] <= 0.8))
    assert np.all((alphas[:, 6] >= 2.0e-2) & (alphas[:, 6] <= 5.0e-2))
    assert np.all((alphas[:, 7] >= 3.0e-4) & (alphas[:, 7] <= 7.5e-4))


def test_start_position_is_inside_the_search_area():
    """The drone cannot begin a trial outside the world it may fly in."""
    for seed in range(10):
        scenario = random_scenario(2, seed=seed)
        assert SEARCH_AREA[0, 0] <= scenario.start_position[0] <= SEARCH_AREA[0, 1]
        assert SEARCH_AREA[1, 0] <= scenario.start_position[1] <= SEARCH_AREA[1, 1]


def test_diagonal_is_the_error_normalizer_of_eq_30():
    """A 20 x 20 m area at fixed altitude has a 28.28 m diagonal."""
    assert random_scenario(1, seed=0).diagonal == pytest.approx(
        np.sqrt(800.0), rel=1e-12)


def test_dict_round_trip_preserves_the_world():
    """A scenario written to metadata and read back is the same scenario."""
    original = random_scenario(3, seed=11)
    restored = Scenario.from_dict(original.to_dict())
    np.testing.assert_allclose(restored.alphas, original.alphas, rtol=0)
    np.testing.assert_allclose(restored.start_position,
                               original.start_position, rtol=0)
    np.testing.assert_allclose(restored.search_area, original.search_area)
    assert restored.name == original.name
    assert restored.seed == original.seed


def test_dict_records_the_parameter_order_and_refuses_a_mismatch():
    """
    A reordered alpha layout must fail loudly instead of silently swapping axes.

    This is the frame/unit trap of risk R4 in its most expensive form: a
    scenario file written by an older build being reinterpreted column by
    column against a newer one.
    """
    data = random_scenario(1, seed=0).to_dict()
    assert data['param_names'] == list(PLUME_PARAM_NAMES)
    data['param_names'] = ['y_s', 'x_s', 'z_s', 'theta', 'Q', 'v', 'd_y', 'd_z']
    with pytest.raises(ValueError):
        Scenario.from_dict(data)


def test_malformed_scenarios_are_rejected():
    """Wrong widths and shapes fail at construction."""
    with pytest.raises(ValueError):
        Scenario('bad', np.zeros((1, 7)), np.zeros(3), 0)
    with pytest.raises(ValueError):
        Scenario('bad', np.zeros((1, 8)), np.zeros(3), 0,
                 search_area=np.zeros((3, 2)))
    with pytest.raises(ValueError):
        random_scenario(0, seed=0)


def test_impossible_placement_raises_instead_of_hanging():
    """A search area too small for the requested leaks must fail, not loop."""
    with pytest.raises(RuntimeError):
        random_scenario(8, seed=0,
                        search_area=np.array([[0.0, 3.0], [0.0, 3.0]]))
