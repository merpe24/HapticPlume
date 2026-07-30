"""
Mutual-information scoring of candidate waypoints (infotaxis).

Implements Goodell et al. (RAS 2026) Section 3.8, Eqs. (15)-(21). The belief is
a weighted particle set; each candidate sensing location turns it into a
Gaussian mixture of predicted measurements, and the informative place to go is
where that mixture is most spread out relative to any single component:

    I(z; alpha) = H(z) - H(z | alpha)                          (Eq. 15)
    H(z)        = -int sum_i w_i p(z|alpha_i) log sum_i w_i p(z|alpha_i) dz
                                                               (Eq. 18)
    H(z|alpha)  = -int sum_i w_i p(z|alpha_i) log p(z|alpha_i) dz
                                                               (Eq. 19)
    V           = I(z; alpha)^2                                (Eq. 21)

Eq. (19) is evaluated in closed form: every component shares the likelihood
deviation r, so H(z|alpha) = 0.5 log(2 pi e r^2) exactly, and only H(z) needs
quadrature. test_conditional_entropy_matches_the_numerical_integral checks the
two against each other.

Two consequences worth stating, because they shape the haptics:

  * Already-consumed plumes do not change the score. Subtracting m_hat shifts
    every component of the mixture by the same constant, and entropy is
    translation invariant, so A-hat never enters this module.
  * The information is where the models DISAGREE, not where concentration is
    highest (paper Fig. 5). The suggested waypoint therefore regularly points
    across the plume rather than up it, which is exactly the guidance a human
    pilot would not produce unaided -- the reason it is worth rendering as a
    haptic cue at all.

In this project the winning waypoint is a SUGGESTION rendered through the
haptic well, never an autopilot command: the human still flies (CLAUDE.md,
"haptic shared-control infotaxis").

:author: premmm
:date: July 29, 2026
"""

from haptic_plume_core.plume_model import concentration

import numpy as np

# Default quadrature settings for Eq. (18). The grid spans the bulk of the
# mixture, widened by a few deviations so the tails are integrated too.
DEFAULT_N_GRID = 256
DEFAULT_N_SIGMA = 6.0
DEFAULT_TAIL_QUANTILE = 1.0e-3


def conditional_entropy(r):
    """
    Differential entropy of one Gaussian likelihood component — Eq. (19).

    With a shared deviation r across particles the weighted integral collapses
    to the entropy of a single Gaussian, 0.5 log(2 pi e r^2).

    :param r: likelihood deviation from paper Eq. (10) [kg/m^3]
    :return: H(z | alpha) in nats
    :rtype: float
    """
    r = float(r)
    if r <= 0.0:
        raise ValueError('likelihood deviation r must be positive')
    return 0.5 * np.log(2.0 * np.pi * np.e * r**2)


def mixture_entropy(g, weights, r, n_grid=DEFAULT_N_GRID,
                    n_sigma=DEFAULT_N_SIGMA,
                    tail_quantile=DEFAULT_TAIL_QUANTILE):
    """
    Entropy of the predicted-measurement mixture by quadrature — Eq. (18).

    The integration range is set from the weighted quantiles of the predicted
    measurements rather than their extremes: a particle sitting almost on top
    of the sensor makes the Gaussian plume model blow up (1/x_p), and one such
    outlier would otherwise stretch the grid until the resolution is useless.
    The truncated mass is tail_quantile per side and affects every candidate
    waypoint alike, so the ranking is unchanged.

    :param g: (N,) predicted measurement per particle [kg/m^3]
    :param weights: (N,) particle weights, summing to one
    :param r: likelihood deviation from paper Eq. (10) [kg/m^3]
    :param n_grid: number of quadrature points
    :param n_sigma: how many deviations to widen the range by
    :param tail_quantile: fraction of weight trimmed from each side
    :return: H(z) in nats
    :rtype: float
    """
    g = np.asarray(g, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    r = float(r)
    if r <= 0.0:
        raise ValueError('likelihood deviation r must be positive')
    if g.size != weights.size:
        raise ValueError('g and weights must have equal length')

    low = _weighted_quantile(g, weights, tail_quantile) - n_sigma * r
    high = _weighted_quantile(g, weights, 1.0 - tail_quantile) + n_sigma * r
    grid = np.linspace(low, high, int(n_grid))

    # p(z) = sum_i w_i N(z; g_i, r), evaluated on the grid.
    z_scores = (grid[:, np.newaxis] - g[np.newaxis, :]) / r
    components = np.exp(-0.5 * z_scores**2) / (r * np.sqrt(2.0 * np.pi))
    density = components @ weights

    # -p log p, with the 0 log 0 = 0 limit taken by flooring the logarithm's
    # argument; where density is zero the leading factor kills the term anyway.
    integrand = -density * np.log(np.maximum(density, 1e-300))
    return float(np.trapezoid(integrand, grid))


def mutual_information(g, weights, r, **kwargs):
    """
    Mutual information between a measurement here and the plume model — Eq. (15).

    :param g: (N,) predicted measurement per particle [kg/m^3]
    :param weights: (N,) particle weights, summing to one
    :param r: likelihood deviation from paper Eq. (10) [kg/m^3]
    :param kwargs: quadrature settings forwarded to mixture_entropy
    :return: I(z; alpha) in nats, clamped at zero
    :rtype: float
    """
    entropy = mixture_entropy(g, weights, r, **kwargs)
    return max(entropy - conditional_entropy(r), 0.0)


def waypoint_utilities(particles, weights, waypoints, r, **kwargs):
    """
    Score candidate sensing locations — Eqs. (18)-(21).

    :param particles: (N, 8) particle set from the filter
    :param weights: (N,) particle weights, summing to one
    :param waypoints: (M, 3) candidate world-frame positions [m]
    :param r: likelihood deviation from paper Eq. (10) [kg/m^3]
    :param kwargs: quadrature settings forwarded to mixture_entropy
    :return: (utilities, information) each (M,), utility V = I^2 per Eq. (21)
    :rtype: tuple
    """
    particles = np.asarray(particles, dtype=float)
    points = np.atleast_2d(np.asarray(waypoints, dtype=float))
    information = np.empty(points.shape[0])
    for index, point in enumerate(points):
        g = np.asarray(concentration(point[0], point[1], point[2], particles),
                       dtype=float)
        information[index] = mutual_information(g, weights, r, **kwargs)
    return information**2, information


def best_waypoint(particles, weights, waypoints, r, **kwargs):
    """
    Pick the most informative candidate — the argmax of paper Eq. (21).

    :param particles: (N, 8) particle set from the filter
    :param weights: (N,) particle weights, summing to one
    :param waypoints: (M, 3) candidate world-frame positions [m]
    :param r: likelihood deviation from paper Eq. (10) [kg/m^3]
    :param kwargs: quadrature settings forwarded to mixture_entropy
    :return: (index, waypoint, information) of the winning candidate
    :rtype: tuple
    """
    points = np.atleast_2d(np.asarray(waypoints, dtype=float))
    utilities, information = waypoint_utilities(particles, weights, points, r,
                                                **kwargs)
    index = int(np.argmax(utilities))
    return index, points[index], float(information[index])


def candidate_waypoints(position, step, n_directions=8, include_hold=False,
                        bounds=None):
    """
    Build the candidate set Upsilon: a ring of reachable points around the robot.

    Fixed-altitude by construction (CLAUDE.md: all math is 3D internally, but
    Phase A flies level), so candidates share the robot's height.

    :param position: (3,) current world-frame position [m]
    :param step: distance to each candidate [m]
    :param n_directions: how many evenly spaced headings to offer
    :param include_hold: also offer staying put, which scores the value of
        another sample from here
    :param bounds: optional (2, 2) array [[x_min, x_max], [y_min, y_max]]
        clipping candidates into the search area
    :return: (M, 3) candidate world-frame positions [m]
    :rtype: numpy.ndarray
    """
    origin = np.asarray(position, dtype=float).reshape(3)
    headings = np.linspace(0.0, 2.0 * np.pi, int(n_directions), endpoint=False)
    points = np.column_stack([
        origin[0] + step * np.cos(headings),
        origin[1] + step * np.sin(headings),
        np.full(headings.size, origin[2]),
    ])
    if include_hold:
        points = np.vstack([points, origin])
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=float)
        points[:, 0] = np.clip(points[:, 0], bounds[0, 0], bounds[0, 1])
        points[:, 1] = np.clip(points[:, 1], bounds[1, 0], bounds[1, 1])
    return points


def _weighted_quantile(values, weights, quantile):
    """
    Weighted quantile of a sample, used to bound the quadrature range.

    :param values: (N,) sample values
    :param weights: (N,) non-negative weights summing to one
    :param quantile: quantile in [0, 1]
    :return: the value at that quantile
    :rtype: float
    """
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    index = int(np.searchsorted(cumulative, quantile, side='left'))
    return float(sorted_values[min(index, sorted_values.size - 1)])
