"""
Gaussian plume model — the single source of truth for plume math.

Implements the time-averaged Gaussian plume model of Goodell et al.,
"Mobile-robotic sensor for estimation and localization of multiple
chemical-gas leaks: A find-and-consume infotaxis approach" (RAS 2026),
Section 3.4, Eqs. (1)-(3). Both the ground-truth gas simulator
(haptic_plume_gas_sim) and the particle-filter estimator
(haptic_plume_estimation) import THIS module; neither may re-implement it.

Zero ROS dependencies — importable from plain pytest (Phase A).

Frames and conventions (authority: haptic_plume_description/frames.md):
    - World frame W: fixed, z up. Plume frame P: origin at the source,
      x_P points DOWNWIND (advection direction), z_P parallel to z_W.
    - theta is the angle from x_W to x_P, i.e. the direction the wind
      blows TOWARD (downwind bearing), in radians. It is NOT the
      meteorological "wind from" convention.
    - Concentration is zero everywhere upwind of the source (x_P <= 0).

Units:
    x, y, z [m] · theta [rad] · Q [kg/s] · v [m/s] · d_y, d_z [m^2/s]
    -> concentration [kg/m^3]

:author: premmm
:date: July 29, 2026
"""

import numpy as np

# Parameter vector alpha for one plume, in this exact order (paper Sec. 3.3).
# The particle filter's state and all (N, 8) parameter arrays use this layout.
PLUME_PARAM_NAMES = ('x_s', 'y_s', 'z_s', 'theta', 'Q', 'v', 'd_y', 'd_z')
N_PLUME_PARAMS = len(PLUME_PARAM_NAMES)


def concentration_plume_frame(x_p, y_p, z_p, q, v, d_y, d_z):
    """
    Time-averaged concentration in the plume frame — paper Eq. (1).

    c = Q / (4 pi x_p sqrt(d_y d_z)) * exp(-(v / 4 x_p) (y_p^2/d_y + z_p^2/d_z))

    for x_p > 0; zero at and upwind of the source (x_p <= 0). All inputs
    broadcast against each other (NumPy rules).

    :param x_p: downwind distance from the source [m]
    :param y_p: crosswind offset [m]
    :param z_p: vertical offset from source height [m]
    :param q: source release rate Q [kg/s]
    :param v: wind speed [m/s]
    :param d_y: crosswind diffusion constant [m^2/s]
    :param d_z: vertical diffusion constant [m^2/s]
    :return: concentration [kg/m^3], zero where x_p <= 0
    :rtype: numpy.ndarray or float
    """
    x_p, y_p, z_p, q, v, d_y, d_z = np.broadcast_arrays(
        x_p, y_p, z_p, q, v, d_y, d_z)
    downwind = np.asarray(x_p) > 0.0
    # Guard the 1/x_p singularity: evaluate with x_p=1 where gated to zero
    x_safe = np.where(downwind, x_p, 1.0)
    c = (q / (4.0 * np.pi * x_safe * np.sqrt(d_y * d_z))
         * np.exp(-(v / (4.0 * x_safe)) * (y_p**2 / d_y + z_p**2 / d_z)))
    result = np.where(downwind, c, 0.0)
    return result if result.ndim else float(result)


def world_to_plume(x_w, y_w, z_w, alpha):
    """
    Transform world-frame coordinates into a plume's frame — paper Eq. (2).

    Translates by the source location and rotates by -theta about z, so
    x_P points downwind. Supports a single alpha of shape (8,) or a stack
    of shape (N, 8); coordinates broadcast against the alpha stack.

    :param x_w: world x coordinate(s) [m]
    :param y_w: world y coordinate(s) [m]
    :param z_w: world z coordinate(s) [m]
    :param alpha: plume parameter vector(s), last axis ordered as PLUME_PARAM_NAMES
    :return: (x_p, y_p, z_p) plume-frame coordinates [m]
    :rtype: tuple
    """
    alpha = np.asarray(alpha, dtype=float)
    x_s, y_s, z_s, theta = (alpha[..., 0], alpha[..., 1],
                            alpha[..., 2], alpha[..., 3])
    dx = np.asarray(x_w, dtype=float) - x_s
    dy = np.asarray(y_w, dtype=float) - y_s
    dz = np.asarray(z_w, dtype=float) - z_s
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # p_P = Rz(-theta) @ (p_W - s)
    x_p = cos_t * dx + sin_t * dy
    y_p = -sin_t * dx + cos_t * dy
    return x_p, y_p, dz


def concentration(x_w, y_w, z_w, alpha):
    """
    Concentration of ONE plume at world-frame point(s) — paper Eq. (2).

    g = c(T_{W/P} p_W) for x_P > 0, else 0. alpha may be (8,) or (N, 8)
    (e.g. one row per particle); coordinates broadcast against it.

    :param x_w: world x coordinate(s) [m]
    :param y_w: world y coordinate(s) [m]
    :param z_w: world z coordinate(s) [m]
    :param alpha: plume parameter vector(s), last axis ordered as PLUME_PARAM_NAMES
    :return: concentration [kg/m^3]
    :rtype: numpy.ndarray or float
    """
    alpha = np.asarray(alpha, dtype=float)
    x_p, y_p, z_p = world_to_plume(x_w, y_w, z_w, alpha)
    return concentration_plume_frame(
        x_p, y_p, z_p,
        alpha[..., 4], alpha[..., 5], alpha[..., 6], alpha[..., 7])


def total_concentration(x_w, y_w, z_w, alphas):
    """
    Superposed concentration of L plumes — paper Eq. (3).

    m = sum_j g_j. Used both by the ground-truth field server and by the
    consume step (subtracting the accumulated set A-hat, paper Eq. (12)).

    :param x_w: world x coordinate(s) [m]
    :param y_w: world y coordinate(s) [m]
    :param z_w: world z coordinate(s) [m]
    :param alphas: (L, 8) array of plume parameter vectors; L may be 0
    :return: total concentration [kg/m^3]
    :rtype: numpy.ndarray or float
    """
    alphas = np.atleast_2d(np.asarray(alphas, dtype=float))
    if alphas.shape[0] == 0:
        zero = np.zeros(np.broadcast(np.asarray(x_w), np.asarray(y_w),
                                     np.asarray(z_w)).shape)
        return zero if zero.ndim else 0.0
    total = concentration(x_w, y_w, z_w, alphas[0])
    for a in alphas[1:]:
        total = total + concentration(x_w, y_w, z_w, a)
    return total
