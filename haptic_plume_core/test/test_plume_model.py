"""
Hand-computed unit tests for haptic_plume_core.plume_model (paper Eqs. 1-3).

Every expected value below is derived by hand from Eq. (1) so a frame or
unit bug (risk R4) cannot hide behind the implementation testing itself.

:author: premmm
:date: July 29, 2026
"""

from haptic_plume_core.plume_model import (
    concentration,
    concentration_plume_frame,
    N_PLUME_PARAMS,
    PLUME_PARAM_NAMES,
    total_concentration,
    world_to_plume,
)

import numpy as np

import pytest

# Reference plume: source at origin, wind toward +x_W, Q=1, v=1, d_y=d_z=0.1
ALPHA_REF = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.1, 0.1])

# By hand: c(5,0,0) = 1 / (4*pi*5*sqrt(0.1*0.1)) = 1/(2*pi)
C_CENTERLINE_5M = 1.0 / (2.0 * np.pi)


def test_param_layout():
    """The alpha layout is the paper's and nothing has silently reordered it."""
    assert PLUME_PARAM_NAMES == ('x_s', 'y_s', 'z_s', 'theta', 'Q', 'v', 'd_y', 'd_z')
    assert N_PLUME_PARAMS == 8


def test_centerline_hand_value():
    """c(x_p=5, y=z=0) = Q/(4*pi*x_p*sqrt(d_y*d_z)) = 1/(2*pi), exp term = 1."""
    c = concentration_plume_frame(5.0, 0.0, 0.0, q=1.0, v=1.0, d_y=0.1, d_z=0.1)
    assert c == pytest.approx(C_CENTERLINE_5M, rel=1e-12)


def test_crosswind_offset_hand_value():
    """y_p=1 multiplies the centerline by exp(-(v/4x_p)*y_p^2/d_y) = exp(-0.5)."""
    c = concentration_plume_frame(5.0, 1.0, 0.0, q=1.0, v=1.0, d_y=0.1, d_z=0.1)
    assert c == pytest.approx(C_CENTERLINE_5M * np.exp(-0.5), rel=1e-12)


def test_vertical_offset_matches_crosswind_when_symmetric():
    """With d_y == d_z the model is symmetric in y_p and z_p."""
    c_y = concentration_plume_frame(5.0, 1.0, 0.0, 1.0, 1.0, 0.1, 0.1)
    c_z = concentration_plume_frame(5.0, 0.0, 1.0, 1.0, 1.0, 0.1, 0.1)
    assert c_y == pytest.approx(c_z, rel=1e-12)


def test_upwind_and_source_are_zero():
    """Eq. (2) gates x_p <= 0 to exactly zero (no NaN/inf from 1/x_p)."""
    assert concentration_plume_frame(0.0, 0.0, 0.0, 1.0, 1.0, 0.1, 0.1) == 0.0
    assert concentration_plume_frame(-1.0, 0.0, 0.0, 1.0, 1.0, 0.1, 0.1) == 0.0
    assert concentration(-3.0, 0.0, 0.0, ALPHA_REF) == 0.0


def test_world_frame_along_wind_matches_plume_frame():
    """theta=0, source at origin: world coords ARE plume coords."""
    assert concentration(5.0, 0.0, 0.0, ALPHA_REF) == pytest.approx(
        C_CENTERLINE_5M, rel=1e-12)


def test_rotated_frame_hand_value():
    """
    Rotated frame reproduces the hand value.

    theta=pi/2 (wind toward +y_W), source (2,3,1): point (2,8,1) is 5 m
    downwind on the centerline -> same 1/(2*pi) hand value.
    """
    alpha = np.array([2.0, 3.0, 1.0, np.pi / 2.0, 1.0, 1.0, 0.1, 0.1])
    assert concentration(2.0, 8.0, 1.0, alpha) == pytest.approx(
        C_CENTERLINE_5M, rel=1e-12)
    # And the point 5 m along +x_W is now CROSSwind -> x_p = 0 -> zero
    assert concentration(7.0, 3.0, 1.0, alpha) == 0.0


def test_rotated_frame_45deg_hand_value():
    """
    A 45-degree wind reproduces the hand value.

    theta=pi/4, source at origin, point (sqrt(2), 0, 0):
    x_p = cos(pi/4)*sqrt(2) = 1, y_p = -sin(pi/4)*sqrt(2) = -1
    -> c = 1/(4*pi*0.1) * exp(-(1/4)*(1/0.1)) = (1/(0.4*pi))*exp(-2.5).
    """
    alpha = np.array([0.0, 0.0, 0.0, np.pi / 4.0, 1.0, 1.0, 0.1, 0.1])
    expected = (1.0 / (0.4 * np.pi)) * np.exp(-2.5)
    assert concentration(np.sqrt(2.0), 0.0, 0.0, alpha) == pytest.approx(
        expected, rel=1e-12)


def test_world_to_plume_round_values():
    """Transform alone, hand-checked: rotation is Rz(-theta) after translation."""
    alpha = np.array([1.0, 1.0, 0.0, np.pi / 2.0, 1.0, 1.0, 0.1, 0.1])
    x_p, y_p, z_p = world_to_plume(1.0, 4.0, 2.0, alpha)
    assert x_p == pytest.approx(3.0, abs=1e-12)   # 3 m downwind (+y_W)
    assert y_p == pytest.approx(0.0, abs=1e-12)
    assert z_p == pytest.approx(2.0, abs=1e-12)   # z untouched by rotation


def test_superposition_is_sum_of_singles():
    """Eq. (3): m = sum_j g_j, checked against individual evaluations."""
    alpha_b = np.array([10.0, -2.0, 0.0, np.pi, 2.0, 1.5, 0.2, 0.05])
    point = (4.0, 0.5, 0.0)
    expected = concentration(*point, ALPHA_REF) + concentration(*point, alpha_b)
    assert total_concentration(*point, np.vstack([ALPHA_REF, alpha_b])) == \
        pytest.approx(expected, rel=1e-12)
    # Two identical co-located sources double the concentration
    doubled = total_concentration(*point, np.vstack([ALPHA_REF, ALPHA_REF]))
    assert doubled == pytest.approx(2.0 * concentration(*point, ALPHA_REF), rel=1e-12)


def test_total_concentration_single_and_empty():
    """A (1,8) stack equals the single-plume call; L=0 gives exactly zero."""
    assert total_concentration(5.0, 0.0, 0.0, ALPHA_REF[np.newaxis, :]) == \
        pytest.approx(concentration(5.0, 0.0, 0.0, ALPHA_REF), rel=1e-12)
    assert total_concentration(5.0, 0.0, 0.0, np.empty((0, 8))) == 0.0


def test_particle_stack_vectorization():
    """(N,8) alpha stack at one point matches a Python loop over rows."""
    rng = np.random.default_rng(42)
    stack = np.column_stack([
        rng.uniform(-5, 5, 50),          # x_s
        rng.uniform(-5, 5, 50),          # y_s
        rng.uniform(0, 2, 50),           # z_s
        rng.uniform(-np.pi, np.pi, 50),  # theta
        rng.uniform(0.1, 2, 50),         # Q
        rng.uniform(0.5, 3, 50),         # v
        rng.uniform(0.05, 0.5, 50),      # d_y
        rng.uniform(0.05, 0.5, 50),      # d_z
    ])
    vec = concentration(1.0, 2.0, 0.5, stack)
    looped = np.array([concentration(1.0, 2.0, 0.5, a) for a in stack])
    assert vec.shape == (50,)
    np.testing.assert_allclose(vec, looped, rtol=1e-12)


def test_grid_evaluation_shape_and_gating():
    """Point arrays broadcast; upwind half of the grid is exactly zero."""
    xs = np.linspace(-5.0, 5.0, 11)
    ys = np.zeros_like(xs)
    zs = np.zeros_like(xs)
    c = concentration(xs, ys, zs, ALPHA_REF)
    assert c.shape == xs.shape
    assert np.all(c[xs <= 0.0] == 0.0)
    assert np.all(c[xs > 0.0] > 0.0)
