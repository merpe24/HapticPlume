"""
Unit tests for haptic_plume_estimation.consume (paper Eqs. 11-13).

:author: premmm
:date: July 29, 2026
"""

from haptic_plume_core.plume_model import concentration, total_concentration

from haptic_plume_estimation.consume import (
    ConsumedSet,
    residual_statistics,
    should_consume,
)

import numpy as np

import pytest

SIGMA_C = 0.7        # paper Table 2, Gaussian-plume environment
D_CLUSTER = 1.8

ALPHA_A = np.array([5.0, 10.0, 1.0, 0.0, 0.025, 0.5, 0.035, 5.0e-4])
ALPHA_B = np.array([14.0, 6.0, 1.0, 0.2, 0.030, 0.6, 0.040, 6.0e-4])


def test_consume_threshold_is_inclusive():
    """Eq. (11) consumes at ||sigma|| <= sigma_c, so the boundary counts."""
    assert should_consume(0.5, SIGMA_C)
    assert should_consume(SIGMA_C, SIGMA_C)
    assert not should_consume(0.71, SIGMA_C)


def test_empty_set_subtracts_exactly_nothing():
    """Before the first consume, Eq. (12) must be an identity."""
    consumed = ConsumedSet()
    assert len(consumed) == 0
    assert consumed.alphas.shape == (0, 8)
    assert consumed.concentration(5.0, 10.0, 1.0) == 0.0
    assert consumed.corrected_measurement(0.42, 5.0, 10.0, 1.0) == 0.42


def test_accumulated_models_subtract_their_own_field():
    """
    m_hat is the superposition of A-hat, so a perfect model cancels exactly.

    This is the whole premise of find-and-consume: after a source is
    localized, its contribution disappears from later measurements and the
    filter is free to chase the next one.
    """
    consumed = ConsumedSet()
    consumed.push(ALPHA_A)
    point = (9.0, 10.4, 1.0)
    z_truth = float(concentration(*point, ALPHA_A))
    assert consumed.concentration(*point) == pytest.approx(z_truth, rel=1e-12)
    assert consumed.corrected_measurement(z_truth, *point) == pytest.approx(
        0.0, abs=1e-15)


def test_two_consumed_plumes_superpose():
    """A-hat with two models subtracts the sum of both — Eq. (3)."""
    consumed = ConsumedSet([ALPHA_A, ALPHA_B])
    point = (16.0, 8.0, 1.0)
    assert len(consumed) == 2
    assert consumed.concentration(*point) == pytest.approx(
        total_concentration(*point, np.vstack([ALPHA_A, ALPHA_B])), rel=1e-12)


def test_positions_and_clear():
    """The positions property exposes the source columns; clear() empties A-hat."""
    consumed = ConsumedSet([ALPHA_A, ALPHA_B])
    np.testing.assert_allclose(consumed.positions,
                               np.vstack([ALPHA_A[:3], ALPHA_B[:3]]))
    consumed.clear()
    assert len(consumed) == 0


def test_push_rejects_wrong_length():
    """A short alpha means a layout bug upstream; fail loudly."""
    with pytest.raises(ValueError):
        ConsumedSet().push(np.zeros(7))


def test_merge_collapses_a_reappeared_source():
    """
    The same leak consumed twice becomes one model — paper Section 3.7.

    Reappearance is expected (the filter is stochastic), so the accumulator
    has to be idempotent about it or the reported source count inflates.
    """
    duplicate = ALPHA_A.copy()
    duplicate[0] += 0.8
    consumed = ConsumedSet([ALPHA_A, duplicate, ALPHA_B])
    removed = consumed.merge(D_CLUSTER)
    assert removed == 1
    assert len(consumed) == 2
    assert consumed.positions[0, 0] == pytest.approx(ALPHA_A[0] + 0.4)


def test_merge_is_a_no_op_below_two_models():
    """Nothing to cluster with zero or one model."""
    assert ConsumedSet().merge(D_CLUSTER) == 0
    assert ConsumedSet([ALPHA_A]).merge(D_CLUSTER) == 0


def test_residual_is_noise_like_when_a_hat_is_correct():
    """
    Risk R11 monitor: a correct A-hat leaves zero-mean noise behind.

    The alarm the estimator node raises is a residual whose mean has drifted
    away from zero relative to the known sensor noise.
    """
    rng = np.random.default_rng(4)
    points = rng.uniform([6.0, 8.0, 1.0], [18.0, 12.0, 1.0], size=(400, 3))
    sigma_noise = 2.0e-3
    truth = np.array([concentration(*p, ALPHA_A) for p in points])
    measurements = truth + rng.normal(0.0, sigma_noise, truth.shape)
    stats = residual_statistics(ConsumedSet([ALPHA_A]), points, measurements)
    assert stats['n'] == 400
    assert abs(stats['mean']) < 0.2 * sigma_noise
    assert stats['std'] == pytest.approx(sigma_noise, rel=0.15)


def test_residual_mean_goes_positive_when_a_source_is_missing():
    """An unconsumed source shows up as concentration A-hat cannot explain."""
    rng = np.random.default_rng(4)
    points = rng.uniform([6.0, 4.0, 1.0], [18.0, 12.0, 1.0], size=(400, 3))
    both = np.vstack([ALPHA_A, ALPHA_B])
    measurements = np.array([total_concentration(*p, both) for p in points])
    stats = residual_statistics(ConsumedSet([ALPHA_A]), points, measurements)
    assert stats['mean'] > 0.0
    assert stats['rms'] > 1.0e-3


def test_residual_mean_goes_negative_when_a_hat_is_poisoned():
    """Over-subtraction is the failure that silently blinds the filter."""
    rng = np.random.default_rng(4)
    points = rng.uniform([6.0, 8.0, 1.0], [18.0, 12.0, 1.0], size=(400, 3))
    measurements = np.array([concentration(*p, ALPHA_A) for p in points])
    inflated = ALPHA_A.copy()
    inflated[4] *= 3.0    # a wildly over-estimated release rate Q
    stats = residual_statistics(ConsumedSet([inflated]), points, measurements)
    assert stats['mean'] < 0.0


def test_residual_requires_matching_lengths():
    """Mismatched pose and measurement streams are a logging bug."""
    with pytest.raises(ValueError):
        residual_statistics(ConsumedSet([ALPHA_A]), np.zeros((3, 3)),
                            np.zeros(2))
