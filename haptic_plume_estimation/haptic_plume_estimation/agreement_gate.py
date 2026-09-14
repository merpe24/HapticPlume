"""
Agreement gate: independently seeded twin filters must agree before a declaration.

Risk R1's third gate, measured 2026-09-13 (CLAUDE.md). A particle filter's spread
drops below sigma_c for two different reasons: the measurements really have
pinned the source down, or systematic resampling has copied a handful of
particles until the cloud is tight with nothing behind it. From inside one
filter the two look identical -- sigma at declaration scored AUC 0.47-0.60 as a
right/wrong classifier. Twins fed the identical measurement stream but seeded
differently tell them apart: evidence reproduces across resampling luck,
depletion does not.

Rule: declare only when the MNE positions of the primary filter and every twin
lie within agreement_m of each other; otherwise reset every filter and keep
searching. The reset matters. Twins left running keep receiving the same
along-axis data, eventually deplete onto the same wrong spot, and then agree on
it. On 60 leaks-on-pipes worlds, in eval_estimator with the drone node's clamp,
the rule cut wrong declarations while flying down the plume from 40% to 10%,
mostly by turning them into no declaration at all. On the informative infotaxis
path it never made a right answer wrong: 2 wrong declarations became right
(88% -> 92%), and 48 of the 53 worlds right either way declared at the same
moment.

Pure NumPy and no ROS, so plume_estimator_node and eval_estimator run the same
code and a bag replay stays bit-exact (decision D7).

:author: premmm
:date: September 13, 2026
"""

from haptic_plume_estimation.particle_filter import PlumeParticleFilter

import numpy as np

# Twins are seeded from the primary filter's seed, so one logged pf_seed replays
# all of them. eval_estimator seeds the primary with trial_seed + 10000, which
# puts twin k at trial_seed + 20000 + k -- the streams the gate was measured on.
TWIN_SEED_OFFSET = 10_000

# Largest horizontal disagreement still trusted [m]. Chosen on selection seeds
# (1.494 m: most wrong declarations caught while keeping >= 80% of right ones)
# and verified once on held-out seeds. Rounding to 1.5 m changes no decision the
# reset rule made on either seed set.
AGREEMENT_M = 1.5

DEFAULT_N_TWINS = 3


def twin_seeds(pf_seed, n_twins=DEFAULT_N_TWINS):
    """
    Derive one Generator seed per twin from the primary filter's seed.

    :param pf_seed: the primary filter's seed
    :param n_twins: how many twins
    :return: one seed per twin
    :rtype: list
    """
    return [int(pf_seed) + TWIN_SEED_OFFSET + k for k in range(int(n_twins))]


class AgreementGate:
    """Twin particle filters that must agree with the primary before it may declare."""

    def __init__(self, filter_config, seeds, agreement_m=AGREEMENT_M):
        """
        Build one twin per seed; each draws its own prior.

        :param filter_config: PlumeFilterConfig shared with the primary filter
        :param seeds: one Generator seed per twin, normally from twin_seeds()
        :param agreement_m: largest horizontal disagreement still trusted [m]
        """
        if len(seeds) < 1:
            raise ValueError('the agreement gate needs at least one twin')
        if agreement_m <= 0.0:
            raise ValueError('agreement_m must be positive')
        self.twins = [PlumeParticleFilter(filter_config, np.random.default_rng(seed))
                      for seed in seeds]
        self.agreement_m = float(agreement_m)

    def update(self, position, z, m_hat=0.0):
        """
        Feed every twin the measurement the primary filter just received.

        :param position: (3,) sensor position in the world frame [m]
        :param z: lag-compensated concentration measurement [kg/m^3]
        :param m_hat: consumed-model prediction subtracted per Eq. (12) [kg/m^3]
        """
        for twin in self.twins:
            twin.update(position, z, m_hat=m_hat)

    def spread(self, primary):
        """
        Return the largest horizontal distance between any two MNE positions.

        The primary filter counts as one of the positions. MNE rather than mmse:
        it is the point the haptic well renders, and resampling outliers drag
        the mean around (particle_filter.mne_position).

        :param primary: the PlumeParticleFilter that wants to declare
        :return: spread [m]
        :rtype: float
        """
        points = np.array([primary.mne_position[0:2]]
                          + [twin.mne_position[0:2] for twin in self.twins])
        return float(np.max(np.linalg.norm(
            points[:, None, :] - points[None, :, :], axis=2)))

    def agrees(self, primary):
        """
        Report whether a declaration from the primary filter can be trusted.

        :param primary: the PlumeParticleFilter that wants to declare
        :return: True when every filter lies within agreement_m of every other
        :rtype: bool
        """
        return self.spread(primary) <= self.agreement_m

    def reset(self):
        """Redraw every twin's prior, alongside the primary filter's reset."""
        for twin in self.twins:
            twin.reset()
