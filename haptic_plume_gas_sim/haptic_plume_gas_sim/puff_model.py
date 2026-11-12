"""
Puff gas model: the plume built from many small blobs of gas.

The leak releases one puff per tick; puffs drift with the airflow.

Zero ROS dependencies

:author: premmm
:date: November 11, 2026
"""

import numpy as np

# Columns of PuffField.puffs, one row per puff.
X, Y, Z, MASS, AGE = range(5)


class PuffField:
    """Gas released as puffs that drift with the airflow."""

    def __init__(self, alphas, rng, dt, search_area, sigma0=0.01, margin=2.0,
                 meander_sigma_deg=0.0, meander_tau=10.0):
        """
        Start an empty room around the scenario's leaks.

        :param alphas: (L, 8) plume parameters. GasField takes them
        :param rng: numpy Generator, used only in step(), the gas never
        depends on where the drone flies
        :param dt: fixed tick [s] = 1 / publish_rate
        :param search_area: (2, 2) [[x_min, x_max], [y_min, y_max]] [m]
        :param sigma0: new puff size [m]
        :param margin: keep puffs this far past the walls
        """
        self.alphas = np.atleast_2d(np.asarray(alphas, dtype=float))
        self.rng = rng
        self.dt = float(dt)
        self.search_area = np.asarray(search_area, dtype=float)
        self.puffs = np.empty((0, 5))
        self.sigma0 = float(sigma0)     # new puff size [m]; >0.01 inflates the thin sheet
        self.margin = float(margin)     # keep puffs this far past the walls [m]
        self.meander_sigma = np.radians(meander_sigma_deg)      # typical swing [rad]; 0 = steady
        self.meander_tau = float(meander_tau)                   # how long one swing lasts [s]
        self.theta_offset = 0.0                                 # current swing [rad]

        # Warm-up: run the leak until gas reaches past the far corner, so a
        # trial starts in a fully developed plume, not an empty room.
        reach = np.hypot(*np.ptp(self.search_area, axis=1)) + 2.0 * self.margin
        for _ in range(int(np.ceil(reach / self.alphas[0, 5] / self.dt))):
            self.step()

    def step(self):
        """Release one puff per leak, move every puff, drop those outside the room."""
        new = np.zeros((len(self.alphas), 5))           # at the leak
        new[:, X:Z + 1] = self.alphas[:, 0:3]
        new[:, MASS] = self.alphas[:, 4] * self.dt      # Q * dt [kg]
        self.puffs = np.vstack([self.puffs, new])

        # Meander: the airflow direction wanders around its mean (a random
        # walk pulled back toward theta; Ornstein-Uhlenbeck).
        if self.meander_sigma > 0.0:
            pull = -self.theta_offset * self.dt / self.meander_tau
            kick = (self.meander_sigma * np.sqrt(2.0 * self.dt / self.meander_tau)
                    * self.rng.normal())
            self.theta_offset += pull + kick
        theta = self.alphas[0, 3] + self.theta_offset
        v = self.alphas[0, 5]
        self.puffs[:, X] += v * np.cos(theta) * self.dt
        self.puffs[:, Y] += v * np.sin(theta) * self.dt
        self.puffs[:, AGE] += self.dt

        (x_min, x_max), (y_min, y_max) = self.search_area
        m = self.margin
        inside = ((self.puffs[:, X] >= x_min - m) & (self.puffs[:, X] <= x_max + m)
                  & (self.puffs[:, Y] >= y_min - m) & (self.puffs[:, Y] <= y_max + m))
        self.puffs = self.puffs[inside]

    def sigmas(self):
        """Return every puff's sideways and vertical spread (sigma_y, sigma_z) [m]."""
        # Single-leak trials: row 0 speaks for every puff, like theta and v do
        d_y, d_z = self.alphas[0, 6], self.alphas[0, 7]
        age = self.puffs[:, AGE]
        sigma_y = np.sqrt(self.sigma0**2 + 2.0 * d_y * age)
        sigma_z = np.sqrt(self.sigma0**2 + 2.0 * d_z * age)
        return sigma_y, sigma_z

    def concentration(self, x_w, y_w, z_w):
        """Sum every puff's 3D Gaussian at one point [kg/m^3]. No randomness here."""
        sigma_y, sigma_z = self.sigmas()
        dx = x_w - self.puffs[:, X]
        dy = y_w - self.puffs[:, Y]
        dz = z_w - self.puffs[:, Z]
        # Round sideways (sigma_x = sigma_y), thin vertically.
        norm = self.puffs[:, MASS] / ((2.0 * np.pi) ** 1.5 * sigma_y**2 * sigma_z)
        bell = np.exp(-(dx**2 + dy**2) / (2.0 * sigma_y**2) - dz**2 / (2.0 * sigma_z**2))
        return float(np.sum(norm * bell))
