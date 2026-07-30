"""
The find-and-consume bookkeeping: the accumulated set A-hat.

Implements Goodell et al. (RAS 2026) Section 3.6, Eqs. (11)-(13). Once the
particle filter's positional spread drops below sigma_c the plume is declared
localized ("consumed"), its model is pushed into A-hat, and the filter resets
to hunt the next source. From then on the consumed models' predicted
concentration is subtracted from every later measurement, so the filter sees
only what A-hat fails to explain.

    ||sigma_(x,y,z)[k]||_2 <= sigma_c   ->  push to A-hat        (Eq. 11)
    z_u[k] = z[k] - m_hat(r_x, r_y, r_z)                          (Eq. 12)

On Eq. (13): the paper also subtracts m_hat from each particle's predicted
measurement, g_u^i = g^i - m_hat. Applying both Eq. (12) and Eq. (13) inside
the Gaussian likelihood of Eq. (9) would cancel exactly --
N(z - m_hat; g - m_hat, r) == N(z; g, r) -- leaving the consume step with no
effect at all. Only Eq. (12) is applied here, which is what the surrounding
prose describes ("the output of the accumulated models is subtracted from
subsequent measurements ... leaving the unmodeled behaviours in the
measurements for subsequent target estimation"). Worth flagging in the paper.

Subtraction is the mechanism behind risk R11 (consume-set poisoning): a bad
model in A-hat silently corrupts every future measurement. residual_statistics
below is the monitor for it — after consuming, what is left should look like
sensor noise and nothing else.

:author: premmm
:date: July 29, 2026
"""

from haptic_plume_core.plume_model import N_PLUME_PARAMS, total_concentration

from haptic_plume_estimation.clustering import agglomerative_merge

import numpy as np


def should_consume(sigma_position_norm, sigma_c):
    """
    Decide whether the current belief is tight enough to consume — Eq. (11).

    :param sigma_position_norm: ||sigma_(x,y,z)||_2 of the particle set [m]
    :param sigma_c: user-defined convergence threshold [m]; higher consumes
        sooner at the cost of localization accuracy
    :return: True if the plume should be pushed into A-hat
    :rtype: bool
    """
    return bool(float(sigma_position_norm) <= float(sigma_c))


class ConsumedSet:
    """
    A-hat: the ordered list of localized plume models, Eq. (11).

    Also the answer to "how many sources has the system found", which is the
    quantity the paper scores success on, and the source of the standoff
    targets the haptic well renders once the belief converges (decision D4).
    """

    def __init__(self, alphas=None):
        """
        Construct an empty accumulator, or seed it with known models.

        :param alphas: optional (L, 8) array of plume parameter vectors
        """
        self._alphas = []
        if alphas is not None:
            for alpha in np.atleast_2d(np.asarray(alphas, dtype=float)):
                self.push(alpha)

    def __len__(self):
        """Return the number of consumed plumes."""
        return len(self._alphas)

    @property
    def alphas(self):
        """
        Return the consumed models as an (L, 8) array — the matrix A-hat.

        :return: (L, 8) array of plume parameter vectors, L may be 0
        :rtype: numpy.ndarray
        """
        if not self._alphas:
            return np.empty((0, N_PLUME_PARAMS))
        return np.vstack(self._alphas)

    @property
    def positions(self):
        """
        Estimated source positions of the consumed models.

        :return: (L, 3) array of (x_s, y_s, z_s) [m]
        :rtype: numpy.ndarray
        """
        return self.alphas[:, 0:3]

    def push(self, alpha):
        """
        Accumulate one localized plume model into A-hat.

        :param alpha: (8,) plume parameter vector, ordered as
            haptic_plume_core.plume_model.PLUME_PARAM_NAMES
        :return: None
        :rtype: NoneType
        """
        vector = np.asarray(alpha, dtype=float).reshape(-1)
        if vector.size != N_PLUME_PARAMS:
            raise ValueError(f'alpha must have {N_PLUME_PARAMS} elements')
        self._alphas.append(vector.copy())

    def clear(self):
        """
        Forget every consumed model, for a per-trial reset.

        :return: None
        :rtype: NoneType
        """
        self._alphas = []

    def concentration(self, x_w, y_w, z_w):
        """
        Total concentration m_hat produced by the consumed models — Eq. (12).

        :param x_w: world x coordinate(s) [m]
        :param y_w: world y coordinate(s) [m]
        :param z_w: world z coordinate(s) [m]
        :return: concentration to subtract from the measurement [kg/m^3];
            exactly zero while A-hat is empty
        :rtype: numpy.ndarray or float
        """
        return total_concentration(x_w, y_w, z_w, self.alphas)

    def corrected_measurement(self, z, x_w, y_w, z_w):
        """
        Apply Eq. (12) to one measurement at one location.

        :param z: measurement (already lag-compensated) [kg/m^3]
        :param x_w: world x coordinate of the sensor [m]
        :param y_w: world y coordinate of the sensor [m]
        :param z_w: world z coordinate of the sensor [m]
        :return: z_u, the measurement with consumed plumes removed [kg/m^3]
        :rtype: float
        """
        return float(z) - float(self.concentration(x_w, y_w, z_w))

    def merge(self, d_threshold):
        """
        Collapse duplicate models with Algorithm 1 — paper Section 3.7.

        :param d_threshold: clustering distance threshold d [m]
        :return: number of models removed by the merge
        :rtype: int
        """
        before = len(self._alphas)
        if before < 2:
            return 0
        merged = agglomerative_merge(self.alphas, d_threshold)
        self._alphas = [row.copy() for row in merged]
        return before - len(self._alphas)


def residual_statistics(consumed, positions, measurements):
    """
    Summarize what is left after subtracting A-hat — the monitor for risk R11.

    If the consumed set is right, the residual over a stretch of flight is
    zero-mean noise. A residual with a persistent positive mean means A-hat
    under-explains the field (a source was missed, or its Q is too low); a
    persistent negative mean means a bad model is being over-subtracted, which
    is exactly the poisoning failure that would quietly blind the filter.

    :param consumed: ConsumedSet to evaluate
    :param positions: (K, 3) sensor positions the measurements were taken at [m]
    :param measurements: (K,) lag-compensated measurements [kg/m^3]
    :return: dict with 'n', 'mean', 'std' and 'rms' of the residual [kg/m^3]
    :rtype: dict
    """
    points = np.atleast_2d(np.asarray(positions, dtype=float))
    z = np.asarray(measurements, dtype=float).reshape(-1)
    if points.shape[0] != z.size:
        raise ValueError('positions and measurements must have equal length')
    m_hat = consumed.concentration(points[:, 0], points[:, 1], points[:, 2])
    residual = z - np.asarray(m_hat, dtype=float)
    return {
        'n': int(residual.size),
        'mean': float(residual.mean()),
        'std': float(residual.std()),
        'rms': float(np.sqrt(np.mean(residual**2))),
    }
