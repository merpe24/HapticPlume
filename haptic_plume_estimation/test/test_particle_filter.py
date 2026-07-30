"""
Unit tests for haptic_plume_estimation.particle_filter (paper Eqs. 4-11).

Two of these carry weight beyond "the code runs":

  * test_converges_and_localizes_a_clean_plume pins the Phase A definition of
    done — the error at the moment of consuming, expressed as a percentage of
    the search-area diagonal, has to land near the paper's reported ~1.2%.
  * the divergence tests are the evidence for risk R1: a trajectory that never
    crosses the plume leaves the belief at its prior, and the filter says so
    instead of quietly reporting a confident wrong answer.

:author: premmm
:date: July 29, 2026
"""

from haptic_plume_core.plume_model import concentration

from haptic_plume_estimation.particle_filter import (
    PAPER_BOUNDS,
    PlumeFilterConfig,
    PlumeParticleFilter,
    STATUS_CONVERGING,
    STATUS_DIVERGED,
    STATUS_INIT,
    STATUS_RESET,
    STATUS_TRACKING,
)

import numpy as np

import pytest

# One true leak inside the paper's 20 m x 20 m search area.
TRUE_ALPHA = np.array([5.0, 10.0, 1.0, 0.0, 0.025, 0.5, 0.035, 5.0e-4])

# Eq. (30) normalizer for a 20 x 20 x 0 m search area.
DOMAIN_DIAGONAL = np.sqrt(20.0**2 + 20.0**2)

SIGMA_C = 0.7


def make_filter(seed=0, n_particles=1500, **kwargs):
    """Build a seeded filter with the project's working configuration."""
    return PlumeParticleFilter(
        PlumeFilterConfig(n_particles=n_particles, **kwargs),
        np.random.default_rng(seed))


def crosswind_raster(y_step=0.3):
    """
    Scripted informative path: crosswind sweeps at increasing downwind range.

    Crosswind travel is what the plume model is identifiable from — flying
    along the wind samples one streamline and cannot separate a near weak
    source from a far strong one. The MI planner discovers this on its own
    (see test_info_gain); here it is hard-coded so the filter is tested
    against a known-good trajectory.
    """
    points, flip = [], False
    for x in np.arange(6.0, 18.01, 1.5):
        ys = np.arange(4.0, 16.01, y_step)
        for y in (ys[::-1] if flip else ys):
            points.append((x, y, 1.0))
        flip = not flip
    return np.array(points)


def true_measurements(path, alpha=TRUE_ALPHA):
    """Noise-free concentration along a path — the truth side of the test."""
    return np.array([float(concentration(p[0], p[1], p[2], alpha))
                     for p in path])


def test_prior_is_uniform_over_the_search_domain():
    """p(alpha_0) = U(S): particles fill the bounds, z_s stays pinned."""
    pf = make_filter()
    assert pf.particles.shape == (1500, 8)
    assert np.all(pf.particles >= PAPER_BOUNDS[:, 0] - 1e-12)
    assert np.all(pf.particles <= PAPER_BOUNDS[:, 1] + 1e-12)
    assert np.all(pf.particles[:, 2] == 1.0)
    assert pf.particles[:, 0].mean() == pytest.approx(10.0, abs=0.5)
    assert pf.status == STATUS_INIT
    assert pf.n_updates == 0


def test_first_update_uses_q_as_the_likelihood_deviation():
    """
    Eq. (10) at k = 0 gives r = q, because sigma is its own running maximum.

    That identity is what makes q meaningful as "the initial likelihood noise
    standard deviation" and is the anchor for tuning it per unit system.
    """
    pf = make_filter(q=0.1)
    pf.update((10.0, 10.0, 1.0), 0.0)
    assert pf.likelihood_sigma == pytest.approx(0.1, rel=1e-12)


def test_weights_are_uniform_after_resampling():
    """Systematic resampling leaves N equally weighted particles."""
    pf = make_filter()
    pf.update((10.0, 10.0, 1.0), 0.05)
    np.testing.assert_allclose(pf.weights, 1.0 / 1500)
    assert 1.0 <= pf.n_eff <= 1500.0


def test_seeded_runs_are_bit_identical():
    """
    Same seed, same data, same belief — decision D7 depends on this.

    Offline replay from a bag has to reproduce the estimate exactly, so
    nothing in the update may reach for global randomness.
    """
    path = crosswind_raster()[:30]
    z = true_measurements(path)
    estimates = []
    for _ in range(2):
        pf = make_filter(seed=42)
        for point, measurement in zip(path, z):
            pf.update(point, measurement)
        estimates.append(pf.mmse)
    np.testing.assert_array_equal(estimates[0], estimates[1])


def test_different_seeds_give_different_particle_sets():
    """The seed actually reaches the sampler (guards a hard-coded default)."""
    assert not np.allclose(make_filter(seed=1).particles,
                           make_filter(seed=2).particles)


def test_converges_and_localizes_a_clean_plume():
    """
    Phase A definition of done, on the trajectory the filter is entitled to.

    Across seeds the filter must reach ||sigma_(x,y,z)|| <= sigma_c and, at
    that moment, place the source within about 1% of the search-area diagonal
    — the paper reports 1.21% average localization error for infotaxis in the
    same 20 m x 20 m Gaussian-plume environment.
    """
    path = crosswind_raster()
    z = true_measurements(path)
    errors, steps = [], []
    for seed in range(5):
        pf = make_filter(seed=seed, q=0.1)
        for step, (point, measurement) in enumerate(zip(path, z)):
            pf.update(point, measurement)
            if pf.sigma_position_norm <= SIGMA_C:
                errors.append(np.linalg.norm(pf.mmse[:3] - TRUE_ALPHA[:3]))
                steps.append(step)
                break
        else:
            pytest.fail(f'seed {seed} never converged below sigma_c')
    assert max(steps) < 60
    assert np.mean(errors) / DOMAIN_DIAGONAL < 0.02
    assert max(errors) < 1.5


def test_converged_belief_is_reported_as_converging():
    """Status tracks the spread, so the composer can gate on it."""
    pf = make_filter(q=0.1)
    assert pf.status == STATUS_INIT
    path = crosswind_raster()
    for point, measurement in zip(path, true_measurements(path)):
        if pf.update(point, measurement) == STATUS_CONVERGING:
            break
    assert pf.status == STATUS_CONVERGING
    assert pf.sigma_position_norm <= pf.config.converging_factor * SIGMA_C


def test_hovering_outside_the_plume_is_reported_as_diverged():
    """
    Risk R1, the headline case: no information means no belief.

    A pilot who parks the drone off-plume gets DIVERGED rather than a
    confident wrong source, and the composer drops the belief-driven well.
    """
    pf = make_filter(q=0.1, diverged_patience=100)
    for _ in range(120):
        pf.update((2.0, 10.0, 1.0), 0.0)     # upwind of the source: reads zero
    assert pf.status == STATUS_DIVERGED
    assert pf.sigma_position_norm > 0.9 * np.linalg.norm(
        make_filter().sigma_position)


def test_an_informative_path_never_trips_the_divergence_detector():
    """The detector must not fire on the good case (it gates a haptic cue)."""
    pf = make_filter(q=0.1, diverged_patience=100)
    path = crosswind_raster()
    for point, measurement in zip(path[:120], true_measurements(path)[:120]):
        assert pf.update(point, measurement) != STATUS_DIVERGED


def test_divergence_needs_patience_to_elapse():
    """Early updates are TRACKING, not DIVERGED — the prior is allowed to be wide."""
    pf = make_filter(q=0.1, diverged_patience=100)
    for _ in range(20):
        assert pf.update((2.0, 10.0, 1.0), 0.0) == STATUS_TRACKING


def test_total_likelihood_collapse_is_detected():
    """
    An impossible measurement kills every particle; that is information.

    Rather than silently renormalizing forever, a run of collapses reports
    DIVERGED so the trial log shows the filter was fed something it could not
    explain (a mis-scaled unit, or a poisoned A-hat, risk R11).
    """
    pf = make_filter(q=0.1, diverged_collapse_limit=3)
    for _ in range(5):
        pf.update((10.0, 10.0, 1.0), np.inf)
    assert pf.status == STATUS_DIVERGED


def test_consumed_field_subtraction_matches_pre_subtracted_input():
    """
    Eq. (12) is applied to the measurement only, never to the prediction.

    Passing m_hat has to be identical to handing the filter an already
    corrected measurement; if the prediction were corrected too (paper
    Eq. 13) the two would still agree, but the consume step would do nothing
    at all — see the note in consume.py.
    """
    path = crosswind_raster()[:25]
    z = true_measurements(path)
    m_hat = 0.004

    pf_a = make_filter(seed=3)
    for point, measurement in zip(path, z):
        pf_a.update(point, measurement + m_hat, m_hat=m_hat)

    pf_b = make_filter(seed=3)
    for point, measurement in zip(path, z):
        pf_b.update(point, measurement)

    np.testing.assert_allclose(pf_a.mmse, pf_b.mmse, rtol=1e-12)


def test_reset_redraws_the_prior_and_reports_it():
    """After a consume the filter starts over, and the status says so."""
    pf = make_filter(q=0.1)
    path = crosswind_raster()
    for point, measurement in zip(path[:40], true_measurements(path)[:40]):
        pf.update(point, measurement)
    tightened = pf.sigma_position_norm
    pf.reset()
    assert pf.status == STATUS_RESET
    assert pf.n_resets == 1
    assert pf.sigma_position_norm > tightened
    assert pf.likelihood_sigma == pf.config.q


def test_mmse_theta_is_a_circular_mean():
    """
    A belief split across the +/-pi branch cut still yields a valid heading.

    Guards risk R4 on the wind-direction convention: an arithmetic mean of
    +3 rad and -3 rad would report a wind blowing the opposite way.
    """
    bounds = PAPER_BOUNDS.copy()
    bounds[3] = [-np.pi, np.pi]
    pf = make_filter(n_particles=4, bounds=bounds)
    pf.particles[:, 3] = np.array([3.0, -3.0, 3.0, -3.0])
    assert abs(pf.mmse[3]) == pytest.approx(np.pi, abs=1e-12)


def test_configuration_is_validated():
    """Bad shapes and impossible values fail at construction, not mid-flight."""
    with pytest.raises(ValueError):
        PlumeFilterConfig(bounds=np.zeros((7, 2)))
    with pytest.raises(ValueError):
        PlumeFilterConfig(process_sigma=np.zeros(7))
    with pytest.raises(ValueError):
        PlumeFilterConfig(bounds=np.column_stack([np.ones(8), np.zeros(8)]))
    with pytest.raises(ValueError):
        PlumeFilterConfig(process_sigma=-np.ones(8))
    with pytest.raises(ValueError):
        PlumeFilterConfig(n_particles=1)
    with pytest.raises(ValueError):
        PlumeFilterConfig(q=0.0)
