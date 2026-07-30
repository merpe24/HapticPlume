"""
Particle filter over the eight Gaussian-plume parameters.

Implements Goodell et al. (RAS 2026) Section 3.5, Eqs. (4)-(10): a bootstrap
particle filter whose state is one plume's alpha = (x_s, y_s, z_s, theta, Q,
v, d_y, d_z), with the state transition as proposal (so weights update by the
likelihood alone, Eq. 7), a Gaussian likelihood whose deviation shrinks as the
belief tightens (Eqs. 9-10), and systematic resampling every measurement.

One filter estimates ONE plume at a time. Multi-source operation is the
find-and-consume loop in consume.py: converge, push the model into A-hat,
reset this filter, keep flying.

Deliberate deviations and choices, all load-bearing for the paper:

  * The likelihood deviation r of Eq. (10) uses the positional spread carried
    into the step (the previous posterior), so on the very first update r == q
    exactly, matching the paper's description of q as "the initial likelihood
    noise standard deviation".
  * Weights are accumulated in log space. Near a source the Gaussian plume
    model is unbounded (1/x_p), so linear-space weights underflow to zero for
    the whole ensemble; that event is detected rather than hidden, because a
    total likelihood collapse is real information about divergence.
  * DIVERGED is a first-class status, not an afterthought. Risk R1 is that a
    human's non-informative flight path starves this filter; the composer
    suppresses the belief-driven haptic well whenever this status is set, and
    every trial logs it.

Zero ROS dependencies, and every random draw comes from an injected
numpy Generator, so a seeded run replays bit-exactly offline (decision D7).

:author: premmm
:date: July 29, 2026
"""

from dataclasses import dataclass, field

from haptic_plume_core.plume_model import concentration, N_PLUME_PARAMS

import numpy as np

# Belief status values, mirrored by haptic_plume_interfaces/BeliefSummary.
STATUS_INIT = 'INIT'
STATUS_TRACKING = 'TRACKING'
STATUS_CONVERGING = 'CONVERGING'
STATUS_RESET = 'RESET'
STATUS_DIVERGED = 'DIVERGED'

# Paper Table 2, Gaussian-plume environment: the search domain S and the
# random-walk standard deviations of the state transition model.
PAPER_BOUNDS = np.array([
    [0.0, 20.0],                    # x_s [m]
    [0.0, 20.0],                    # y_s [m]
    [1.0, 1.0],                     # z_s [m] (source height assumed known)
    [-np.pi / 6.0, np.pi / 6.0],    # theta [rad] (wind known to +/-30 deg)
    [2.0e-2, 3.0e-2],               # Q [kg/s]
    [0.2, 0.8],                     # v [m/s]
    [2.0e-2, 5.0e-2],               # d_y [m^2/s]
    [3.0e-4, 7.5e-4],               # d_z [m^2/s]
])
PAPER_PROCESS_SIGMA = np.array([
    0.1,                    # x_s [m]
    0.1,                    # y_s [m]
    0.0,                    # z_s (fixed: source height assumed known)
    np.deg2rad(0.33),       # theta -- see note below
    1.0e-6,                 # Q [kg/s]
    0.06,                   # v [m/s]
    2.0e-4,                 # d_y [m^2/s]
    5.0e-6,                 # d_z [m^2/s]
])
# Note on theta: Table 2 lists sigma_theta = 0.33 without units. It is read
# here as degrees, matching the domain on the line above it (theta in
# [-30 deg, 30 deg]). Taken as radians it would be 31% of the entire domain
# per step, which would randomize the wind direction faster than any
# measurement could inform it -- the filter would never hold a plume
# orientation. Flagged because it changes reproduced results.

# Likelihood scale q is quoted in Table 2 as 1.25, in whatever units that
# paper's simulated sensor reports. It is NOT unit-free: r = q sigma/max(sigma)
# is a concentration standard deviation and has to sit near the scale of the
# measurements actually being fed in. In this project's units (kg/m^3, with
# in-plume readings of order 1e-1) the equivalent value found in the Phase A
# sweep is 0.1: at 1.25 the likelihood is flat and the filter never converges,
# at 0.05 and below it commits early and consumes the wrong location.
PROJECT_Q = 0.1


@dataclass
class PlumeFilterConfig:
    """
    Tuning for one PlumeParticleFilter.

    Defaults are the Gaussian-plume column of paper Table 2 where the paper
    states a value. sigma_c and d belong to the consume/cluster stages but are
    carried here so a scenario configures one object.
    """

    bounds: np.ndarray = field(default_factory=lambda: PAPER_BOUNDS.copy())
    process_sigma: np.ndarray = field(
        default_factory=lambda: PAPER_PROCESS_SIGMA.copy())
    n_particles: int = 3500
    q: float = PROJECT_Q
    sigma_c: float = 0.7
    d_cluster: float = 1.8
    converging_factor: float = 2.0
    diverged_patience: int = 200
    diverged_sigma_fraction: float = 0.9
    diverged_collapse_limit: int = 5

    def __post_init__(self):
        """Validate shapes and ranges once, at construction."""
        self.bounds = np.asarray(self.bounds, dtype=float)
        self.process_sigma = np.asarray(self.process_sigma, dtype=float)
        if self.bounds.shape != (N_PLUME_PARAMS, 2):
            raise ValueError(f'bounds must be ({N_PLUME_PARAMS}, 2)')
        if self.process_sigma.shape != (N_PLUME_PARAMS,):
            raise ValueError(f'process_sigma must be ({N_PLUME_PARAMS},)')
        if np.any(self.bounds[:, 1] < self.bounds[:, 0]):
            raise ValueError('bounds must be ordered [low, high] per parameter')
        if np.any(self.process_sigma < 0.0):
            raise ValueError('process_sigma must be non-negative')
        if self.n_particles < 2:
            raise ValueError('n_particles must be at least 2')
        if self.q <= 0.0:
            raise ValueError('q must be positive')


class PlumeParticleFilter:
    """
    Bootstrap particle filter for a single Gaussian plume.

    Typical cycle, driven by the estimator node or by eval_estimator.py::

        m_hat = consumed.concentration(*position)
        pf.update(position, z_compensated, m_hat=m_hat)
        if should_consume(pf.sigma_position_norm, config.sigma_c):
            consumed.push(pf.mmse)
            pf.reset()
    """

    def __init__(self, config=None, rng=None):
        """
        Construct the filter and draw the prior p(alpha_0) = U(S).

        :param config: PlumeFilterConfig; None uses the paper's defaults
        :param rng: numpy.random.Generator; None creates an unseeded one
            (pass a seeded generator for reproducible runs)
        """
        self.config = config if config is not None else PlumeFilterConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.n_updates = 0
        self.n_resets = 0
        self.particles = np.empty((self.config.n_particles, N_PLUME_PARAMS))
        self.weights = np.empty(self.config.n_particles)
        self.reset()

    def reset(self):
        """
        Redraw the prior and clear all belief state — after a consume, Eq. (11).

        The generator is deliberately NOT reseeded, so consecutive sources in
        one trial explore differently while the trial as a whole stays
        reproducible from its seed.

        :return: None
        :rtype: NoneType
        """
        low, high = self.config.bounds[:, 0], self.config.bounds[:, 1]
        self.particles = self.rng.uniform(
            low, high, size=(self.config.n_particles, N_PLUME_PARAMS))
        self.weights = np.full(self.config.n_particles,
                               1.0 / self.config.n_particles)
        self._sigma_position = self._compute_sigma_position()
        self._sigma_norm_max = max(float(np.linalg.norm(self._sigma_position)),
                                   1e-12)
        self._n_eff = float(self.config.n_particles)
        self._likelihood_sigma = self.config.q
        self._collapse_streak = 0
        self._updates_since_reset = 0
        self.status = STATUS_RESET if self.n_updates else STATUS_INIT
        if self.n_updates:
            self.n_resets += 1

    def update(self, position, z, m_hat=0.0):
        """
        Run one measurement through the filter — Eqs. (7), (9), (10), (12).

        :param position: (3,) sensor position in the world frame [m]
        :param z: lag-compensated concentration measurement [kg/m^3]
        :param m_hat: concentration predicted at this position by the already
            consumed models, subtracted per Eq. (12) [kg/m^3]
        :return: the belief status after this update
        :rtype: str
        """
        x_w, y_w, z_w = (float(v) for v in np.asarray(position,
                                                      dtype=float).reshape(3))
        # 1. State transition: a bounded random walk, p(alpha[k] | alpha[k-1]).
        #    Used as the proposal, so Eq. (7) reduces to weight *= likelihood.
        self._propagate()

        # 2. Each particle's predicted measurement of its own plume, Eq. (2).
        g = np.asarray(concentration(x_w, y_w, z_w, self.particles), dtype=float)

        # 3. Remove the plumes already consumed, Eq. (12).
        z_u = float(z) - float(m_hat)

        # 4. Likelihood deviation, Eq. (10): less conservative as the particles
        #    cluster. Uses the spread carried into this step, so r == q on the
        #    first update after a reset.
        sigma_norm = float(np.linalg.norm(self._sigma_position))
        r = max(self.config.q * sigma_norm / self._sigma_norm_max, 1e-12)
        self._likelihood_sigma = r

        # 5. Weight update in log space, Eq. (9), then normalize.
        log_w = np.log(np.maximum(self.weights, 1e-300))
        log_w -= 0.5 * ((z_u - g) / r) ** 2
        log_w -= np.log(r * np.sqrt(2.0 * np.pi))
        self._normalize_from_log(log_w)

        # 6. Effective sample size before resampling, then systematic resample.
        self._n_eff = 1.0 / float(np.sum(self.weights ** 2))
        self._systematic_resample()

        # 7. Posterior spread drives both the consume test and the next r.
        self._sigma_position = self._compute_sigma_position()
        sigma_norm = float(np.linalg.norm(self._sigma_position))
        self._sigma_norm_max = max(self._sigma_norm_max, sigma_norm, 1e-12)

        self.n_updates += 1
        self._updates_since_reset += 1
        self.status = self._classify()
        return self.status

    @property
    def mmse(self):
        """
        Minimum mean-squared estimate of alpha — Eq. (8).

        theta is averaged circularly so a belief straddling the +/-pi branch
        cut cannot produce a wind direction pointing nowhere near the
        particles (risk R4). Every other parameter is a weighted mean.

        :return: (8,) estimated plume parameter vector
        :rtype: numpy.ndarray
        """
        estimate = self.weights @ self.particles
        theta = self.particles[:, 3]
        estimate[3] = np.arctan2(self.weights @ np.sin(theta),
                                 self.weights @ np.cos(theta))
        return estimate

    @property
    def sigma_position(self):
        """
        Per-axis standard deviation of the estimated source position.

        :return: (3,) standard deviations of (x_s, y_s, z_s) [m]
        :rtype: numpy.ndarray
        """
        return self._sigma_position.copy()

    @property
    def sigma_position_norm(self):
        """||sigma_(x,y,z)||_2, the quantity Eq. (11) thresholds [m]."""
        return float(np.linalg.norm(self._sigma_position))

    @property
    def n_eff(self):
        """Effective sample size before the most recent resampling step."""
        return self._n_eff

    @property
    def likelihood_sigma(self):
        """Return the likelihood deviation r of the last update, Eq. (10)."""
        return self._likelihood_sigma

    def _propagate(self):
        """Random-walk the particles and clip them back into the domain S."""
        sigma = self.config.process_sigma
        moving = sigma > 0.0
        if np.any(moving):
            noise = self.rng.normal(
                0.0, sigma[moving], size=(self.config.n_particles,
                                          int(np.count_nonzero(moving))))
            self.particles[:, moving] += noise
            np.clip(self.particles, self.config.bounds[:, 0],
                    self.config.bounds[:, 1], out=self.particles)

    def _normalize_from_log(self, log_w):
        """
        Exponentiate and normalize log-weights, detecting total collapse.

        :param log_w: (N,) unnormalized log weights
        :return: None
        :rtype: NoneType
        """
        finite = np.isfinite(log_w)
        if not np.any(finite):
            self._on_weight_collapse()
            return
        log_w = np.where(finite, log_w, -np.inf)
        weights = np.exp(log_w - log_w.max())
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            self._on_weight_collapse()
            return
        self.weights = weights / total
        self._collapse_streak = 0

    def _on_weight_collapse(self):
        """Fall back to a uniform belief and remember that the data ruled it out."""
        self.weights = np.full(self.config.n_particles,
                               1.0 / self.config.n_particles)
        self._collapse_streak += 1

    def _systematic_resample(self):
        """
        Systematic resampling: O(N) and low variance (paper Section 3.5).

        One uniform draw positions a comb of N equally spaced points on the
        cumulative weight distribution, so particles are duplicated in
        proportion to weight without the extra variance of N independent draws.
        """
        n = self.config.n_particles
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        comb = (self.rng.uniform() + np.arange(n)) / n
        indices = np.searchsorted(cumulative, comb, side='left')
        self.particles = self.particles[np.minimum(indices, n - 1)]
        self.weights = np.full(n, 1.0 / n)

    def _compute_sigma_position(self):
        """
        Weighted standard deviation of the particles' source position.

        :return: (3,) standard deviations of (x_s, y_s, z_s) [m]
        :rtype: numpy.ndarray
        """
        positions = self.particles[:, 0:3]
        mean = self.weights @ positions
        variance = self.weights @ (positions - mean) ** 2
        return np.sqrt(np.maximum(variance, 0.0))

    def _classify(self):
        """
        Map the current belief onto a BeliefSummary status.

        :return: one of the STATUS_* constants
        :rtype: str
        """
        if self._collapse_streak >= self.config.diverged_collapse_limit:
            return STATUS_DIVERGED
        sigma_norm = self.sigma_position_norm
        stagnant = (sigma_norm
                    > self.config.diverged_sigma_fraction * self._sigma_norm_max)
        if (self._updates_since_reset >= self.config.diverged_patience
                and stagnant):
            return STATUS_DIVERGED
        if sigma_norm <= self.config.converging_factor * self.config.sigma_c:
            return STATUS_CONVERGING
        return STATUS_TRACKING
