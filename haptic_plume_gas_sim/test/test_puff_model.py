"""
Unit tests for haptic_plume_gas_sim.puff_model.

The anchor: with a steady wind, the sum of puffs must equal the analytic
plume of haptic_plume_core.plume_model (Goodell Eq. 1), so the puff world
stays tied to the world every earlier number was validated on. Measured
2026-11-11: within 1.4% from 2 m downwind (2.75% at 1 m, where Eq. 1's
neglect of along-wind spreading matters most); 2% is the tolerance.

:author: premmm
:date: November 11, 2026
"""

from haptic_plume_core.plume_model import total_concentration

from haptic_plume_gas_sim.puff_model import AGE, PuffField, X, Y

import numpy as np

import pytest

ROOM = np.array([[0.0, 20.0], [0.0, 20.0]])
DT = 0.05


def make_alpha(theta):
    """Return the reference leak: (2, 10, 1), Q 0.025, v 0.5, mid-prior d_y, d_z."""
    return np.array([[2.0, 10.0, 1.0, theta, 0.025, 0.5, 0.035, 5.0e-4]])


@pytest.mark.parametrize('theta', [0.0, 0.4, -0.5])
@pytest.mark.parametrize('downwind, crosswind', [
    (2.0, 0.0), (5.0, 0.0), (10.0, 0.0), (5.0, 1.0), (5.0, -1.5), (10.0, 2.0)])
def test_steady_puffs_match_plume_formula(theta, downwind, crosswind):
    """With no meander the puff sum equals Eq. (1) within 2%, for any wind direction."""
    alpha = make_alpha(theta)
    field = PuffField(alpha, None, DT, ROOM)    # warm-up runs in the constructor
    x = 2.0 + downwind * np.cos(theta) - crosswind * np.sin(theta)
    y = 10.0 + downwind * np.sin(theta) + crosswind * np.cos(theta)

    expected = total_concentration(x, y, 1.0, alpha)
    assert field.concentration(x, y, 1.0) == pytest.approx(expected, rel=0.02)


def test_no_gas_far_upwind():
    """Upwind of the leak there is effectively nothing, as in Eq. (1)."""
    field = PuffField(make_alpha(0.0), None, DT, ROOM)
    assert field.concentration(0.5, 10.0, 1.0) < 1e-6


def test_puffs_kept_only_within_margin():
    """Every surviving puff sits inside the room plus the margin."""
    field = PuffField(make_alpha(0.4), None, DT, ROOM, margin=2.0)
    assert np.all(field.puffs[:, X] <= 22.0) and np.all(field.puffs[:, Y] <= 22.0)
    assert np.all(field.puffs[:, X] >= -2.0) and np.all(field.puffs[:, Y] >= -2.0)


def test_puffs_grow_with_age_and_stay_flat():
    """Older puffs are wider, and always much flatter than wide (the thin sheet)."""
    field = PuffField(make_alpha(0.0), None, DT, ROOM)
    sigma_y, sigma_z = field.sigmas()
    order = np.argsort(field.puffs[:, AGE])
    assert np.all(np.diff(sigma_y[order]) >= 0.0)
    # Newborns are round (both sizes start at sigma0); flatness builds with age.
    aged = field.puffs[:, AGE] > 1.0
    assert np.all(sigma_z[aged] < sigma_y[aged] / 5.0)
