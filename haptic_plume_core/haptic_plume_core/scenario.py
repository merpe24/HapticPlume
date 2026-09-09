"""
Scenario schema — what a trial IS, as one serializable object.

A scenario pins everything about the world that a trial must reproduce: where
the leaks are, how hard they leak, which way the wind blows, and where the
drone starts. Both sides of the system read it — haptic_plume_gas_sim builds
its ground-truth field from the same object that haptic_plume_experiment logs
alongside the bag — so there is exactly one definition of "the same scenario".

Everything is derived from an integer seed. Determinism in this project is
promised at the level of the SCENARIO, not the trajectory (risk R13): two runs
of seed 7 place identical leaks in identical wind, but a human pilot will fly
them differently, and that is the point of the experiment.

Sampling ranges follow Goodell et al. (RAS 2026) Table 2, Gaussian-plume
environment: sources land inside an 18 m x 18 m box centred in the 20 m x 20 m
search area, no closer than 2 m to each other.

Units:
    positions [m] · theta [rad] · Q [kg/s] · v [m/s] · d_y, d_z [m^2/s]

:author: premmm
:date: July 29, 2026
"""

from dataclasses import dataclass, field

from haptic_plume_core.plume_model import N_PLUME_PARAMS, PLUME_PARAM_NAMES

import numpy as np

# Paper Table 2, Gaussian-plume environment.
SEARCH_AREA = np.array([[0.0, 20.0], [0.0, 20.0]])   # x, y extent [m]
SOURCE_MARGIN = 1.0        # keeps sources inside the centred 18 x 18 m box [m]
MIN_SOURCE_SEPARATION = 2.0
SOURCE_HEIGHT = 1.0
THETA_RANGE = (-np.pi / 6.0, np.pi / 6.0)
Q_RANGE = (2.0e-2, 3.0e-2)
WIND_SPEED_RANGE = (0.2, 0.8)
D_Y_RANGE = (2.0e-2, 5.0e-2)
D_Z_RANGE = (3.0e-4, 7.5e-4)

# Rejection sampling gives up rather than looping forever on an impossible ask.
MAX_PLACEMENT_ATTEMPTS = 1000


@dataclass
class Scenario:
    """
    One reproducible world: leaks, wind, drone start, and the seed behind them.

    :param name: human-readable identifier, also used in trial logs
    :param alphas: (L, 8) true plume parameters, columns per PLUME_PARAM_NAMES
    :param start_position: (3,) where the drone begins [m]
    :param seed: the integer this scenario was generated from
    :param search_area: (2, 2) [[x_min, x_max], [y_min, y_max]] the drone may
        fly in [m]
    """

    name: str
    alphas: np.ndarray
    start_position: np.ndarray
    seed: int
    search_area: np.ndarray = field(
        default_factory=lambda: SEARCH_AREA.copy())

    def __post_init__(self):
        """Coerce array fields and reject malformed scenarios at construction."""
        self.alphas = np.atleast_2d(np.asarray(self.alphas, dtype=float))
        if self.alphas.shape[1] != N_PLUME_PARAMS:
            raise ValueError(f'alphas must have {N_PLUME_PARAMS} columns')
        self.start_position = np.asarray(
            self.start_position, dtype=float).reshape(3)
        self.search_area = np.asarray(self.search_area, dtype=float)
        if self.search_area.shape != (2, 2):
            raise ValueError('search_area must be [[x_min, x_max], [y_min, y_max]]')

    @property
    def n_sources(self):
        """Return the leak count a successful run has to reproduce."""
        return int(self.alphas.shape[0])

    @property
    def source_positions(self):
        """
        Return the true source locations the localization error is scored on.

        :return: (L, 3) array of (x_s, y_s, z_s) [m]
        :rtype: numpy.ndarray
        """
        return self.alphas[:, 0:3]

    @property
    def diagonal(self):
        """
        Search-area diagonal, the normalizer of paper Eq. (30) [m].

        :return: sqrt(dx^2 + dy^2), with no vertical extent at fixed altitude
        :rtype: float
        """
        extent = self.search_area[:, 1] - self.search_area[:, 0]
        return float(np.hypot(extent[0], extent[1]))

    def to_dict(self):
        """
        Convert to plain Python types, ready for YAML or bag metadata.

        :return: nested dict with lists instead of arrays
        :rtype: dict
        """
        return {
            'name': self.name,
            'seed': int(self.seed),
            'search_area': self.search_area.tolist(),
            'start_position': self.start_position.tolist(),
            'param_names': list(PLUME_PARAM_NAMES),
            'alphas': self.alphas.tolist(),
        }

    @classmethod
    def from_dict(cls, data):
        """
        Rebuild a scenario from to_dict() output.

        :param data: dict as produced by to_dict()
        :return: the reconstructed scenario
        :rtype: Scenario
        """
        names = tuple(data.get('param_names', PLUME_PARAM_NAMES))
        if names != PLUME_PARAM_NAMES:
            raise ValueError(
                f'scenario parameter order {names} does not match this build '
                f'({PLUME_PARAM_NAMES}) — refusing to reinterpret the columns')
        return cls(
            name=data['name'],
            alphas=np.asarray(data['alphas'], dtype=float),
            start_position=np.asarray(data['start_position'], dtype=float),
            seed=int(data['seed']),
            search_area=np.asarray(data['search_area'], dtype=float),
        )


def random_scenario(n_sources, seed, name=None, shared_wind=True,
                    search_area=None, pipe_segments=None):
    """
    Draw a reproducible scenario from the paper's Table 2 ranges.

    :param n_sources: number of leaks to place
    :param seed: integer seed; the same seed always yields the same world
    :param name: optional label; defaults to "s<n>_seed<seed>"
    :param shared_wind: give every plume the same theta and v, which is what a
        single physical wind field does. The paper draws them per plume; the
        estimator does not care either way (it fits one plume at a time), but
        the simulator and the RViz wind arrow are only honest with one wind.
    :param search_area: optional (2, 2) override of the flyable extent [m]
    :param pipe_segments: [(p0, p1, radius)] capsules from obstacles.yaml. Given
        these, leaks sit on the pipe surfaces, which is what a pipeline
        inspection actually looks for. None keeps the uniform placement Phase A
        was validated with, so existing seeds reproduce bit-exactly.
    :return: the generated scenario
    :rtype: Scenario
    """
    if n_sources < 1:
        raise ValueError('a scenario needs at least one source')
    rng = np.random.default_rng(seed)
    area = SEARCH_AREA.copy() if search_area is None else np.asarray(
        search_area, dtype=float)

    if pipe_segments is None:
        positions = _place_sources(rng, n_sources, area)
    else:
        positions = _place_sources_on_pipes(
            rng, n_sources, pipe_segments, SOURCE_HEIGHT)
    theta = rng.uniform(*THETA_RANGE)
    wind_speed = rng.uniform(*WIND_SPEED_RANGE)

    alphas = np.empty((n_sources, N_PLUME_PARAMS))
    for index, (x_s, y_s, z_s) in enumerate(positions):
        alphas[index] = [
            x_s,
            y_s,
            z_s,
            theta if shared_wind else rng.uniform(*THETA_RANGE),
            rng.uniform(*Q_RANGE),
            wind_speed if shared_wind else rng.uniform(*WIND_SPEED_RANGE),
            rng.uniform(*D_Y_RANGE),
            rng.uniform(*D_Z_RANGE),
        ]

    start_position = np.array([
        rng.uniform(area[0, 0] + SOURCE_MARGIN, area[0, 1] - SOURCE_MARGIN),
        rng.uniform(area[1, 0] + SOURCE_MARGIN, area[1, 1] - SOURCE_MARGIN),
        SOURCE_HEIGHT,
    ])

    return Scenario(
        name=name if name is not None else f's{n_sources}_seed{seed}',
        alphas=alphas,
        start_position=start_position,
        seed=int(seed),
        search_area=area,
    )


def _place_sources(rng, n_sources, area):
    """
    Rejection-sample source positions at least MIN_SOURCE_SEPARATION apart.

    :param rng: numpy.random.Generator
    :param n_sources: how many positions to place
    :param area: (2, 2) search-area extent [m]
    :return: (n_sources, 3) array of (x_s, y_s, z_s) [m]
    :rtype: numpy.ndarray
    """
    low = area[:, 0] + SOURCE_MARGIN
    high = area[:, 1] - SOURCE_MARGIN
    placed = []
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        candidate = rng.uniform(low, high)
        if all(np.linalg.norm(candidate - other) >= MIN_SOURCE_SEPARATION
               for other in placed):
            placed.append(candidate)
            if len(placed) == n_sources:
                return np.column_stack(
                    [np.vstack(placed), np.full(n_sources, SOURCE_HEIGHT)])
    raise RuntimeError(
        f'could not place {n_sources} sources at least '
        f'{MIN_SOURCE_SEPARATION} m apart in {MAX_PLACEMENT_ATTEMPTS} attempts')


def _normal_frame(axis):
    """
    Build two unit vectors spanning the plane normal to a pipe's axis.

    :param axis: unit vector along the capsule axis
    :return: (first, second) orthonormal vectors normal to axis
    :rtype: tuple
    """
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ reference)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    first = np.cross(axis, reference)
    first /= np.linalg.norm(first)
    return first, np.cross(axis, first)


def _capsule_clearance(point, segments):
    """
    Signed distance from a point to the nearest capsule surface.

    :param point: (3,) position [m]
    :param segments: [(p0, p1, radius)] capsules
    :return: clearance [m], negative inside a capsule
    :rtype: float
    """
    best = float('inf')
    for p0, p1, radius in segments:
        axis = p1 - p0
        t = np.clip(float((point - p0) @ axis) / float(axis @ axis), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(point - (p0 + t * axis))) - radius)
    return best


def _place_sources_on_pipes(rng, n_sources, segments, altitude):
    """
    Sample leak positions where the pipe surface crosses the flight altitude.

    The height is not a free choice: particle_filter.py pins z_s to a single
    value with zero process noise ("source height assumed known"), and the
    plume is so thin vertically that a 0.10 m error costs 62 points of success.
    Solving for the angle at which each pipe's surface reaches the altitude
    keeps z_s exact. A vertical pipe -- the riser -- drops out on its own,
    because its surface height depends only on distance along the axis and
    never on angle, so there is no angle to solve for: every leak lands on
    main_run. Candidates buried inside a neighbouring capsule at a junction
    are dropped, since no gas would escape from there.

    :param rng: numpy.random.Generator
    :param n_sources: how many leaks to place
    :param segments: [(p0, p1, radius)] capsules to place them on
    :param altitude: the drone's fixed flight altitude [m]
    :return: (n_sources, 3) array of leak positions [m]
    :rtype: numpy.ndarray
    """
    lengths = np.array([np.linalg.norm(p1 - p0) for p0, p1, _ in segments])
    weights = lengths * np.array([radius for _, _, radius in segments])
    weights = weights / weights.sum()   # long pipes carry proportionally more

    placed = []
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        p0, p1, radius = segments[rng.choice(len(segments), p=weights)]
        axis = (p1 - p0) / np.linalg.norm(p1 - p0)
        first, second = _normal_frame(axis)
        base = p0 + rng.uniform() * (p1 - p0)
        amplitude = radius * float(np.hypot(first[2], second[2]))
        offset = altitude - float(base[2])
        if amplitude < 1e-9 or abs(offset) > amplitude:
            continue
        phase = float(np.arctan2(second[2], first[2]))
        angle = phase + rng.choice([-1.0, 1.0]) * float(
            np.arccos(np.clip(offset / amplitude, -1.0, 1.0)))
        candidate = base + radius * (np.cos(angle) * first
                                     + np.sin(angle) * second)
        if _capsule_clearance(candidate, segments) < -1e-9:
            continue
        if all(np.linalg.norm(candidate - other) >= MIN_SOURCE_SEPARATION
               for other in placed):
            placed.append(candidate)
            if len(placed) == n_sources:
                return np.vstack(placed)
    raise RuntimeError(
        f'could not place {n_sources} leaks on the pipe surfaces in '
        f'{MAX_PLACEMENT_ATTEMPTS} attempts')
