"""
Unit tests for haptic_plume_estimation.agreement_gate (risk R1's third gate).

The gate is only worth anything because differently seeded twins land in the
same place only when the evidence put them there. These tests pin the mechanics
that rests on: the spread is a maximum over MNE positions, the threshold is
inclusive, reset redraws every twin, and one pf_seed replays all of them (D7).
The statistical claim lives in CLAUDE.md, not here.

:author: premmm
:date: September 13, 2026
"""

from haptic_plume_estimation.agreement_gate import (
    AGREEMENT_M,
    AgreementGate,
    TWIN_SEED_OFFSET,
    twin_seeds,
)
from haptic_plume_estimation.particle_filter import (
    PlumeFilterConfig,
    PlumeParticleFilter,
)

import numpy as np

import pytest

N_PARTICLES = 300


def make_config():
    """Build a small filter configuration so the tests stay fast."""
    return PlumeFilterConfig(n_particles=N_PARTICLES, q=0.1, sigma_c=0.5)


def pin(pf, x, y, seed=0):
    """Collapse a filter's position particles into a tight clump at (x, y)."""
    rng = np.random.default_rng(seed)
    pf.particles[:, 0] = x + rng.normal(0.0, 0.05, N_PARTICLES)
    pf.particles[:, 1] = y + rng.normal(0.0, 0.05, N_PARTICLES)
    pf.weights = np.full(N_PARTICLES, 1.0 / N_PARTICLES)


def make_gate(pf_seed=7, n_twins=3, agreement_m=AGREEMENT_M):
    """Build a primary filter and its gate from one seed, as the node does."""
    config = make_config()
    primary = PlumeParticleFilter(config, np.random.default_rng(pf_seed))
    gate = AgreementGate(config, twin_seeds(pf_seed, n_twins), agreement_m)
    return primary, gate


def test_twin_seeds_follow_the_primary_seed():
    """One logged pf_seed replays every twin; eval_estimator relies on the offset."""
    assert twin_seeds(7, 3) == [7 + TWIN_SEED_OFFSET + k for k in range(3)]
    # eval_estimator seeds the primary with trial_seed + 10000, so trial 1000
    # puts its twins at 21000, 21001 -- the streams the 2026-09-13 study used.
    assert twin_seeds(1000 + 10_000, 2) == [21_000, 21_001]


def test_filters_on_one_spot_agree():
    """Twins clumped with the primary report a small spread and let it declare."""
    primary, gate = make_gate()
    pin(primary, 5.0, 5.0, seed=1)
    for k, twin in enumerate(gate.twins):
        pin(twin, 5.0 + 0.2 * k, 5.0, seed=2 + k)
    assert gate.spread(primary) == pytest.approx(0.4, abs=0.1)
    assert gate.agrees(primary)


def test_one_stray_twin_blocks_the_declaration():
    """A single twin elsewhere is enough: the spread is a maximum, not a mean."""
    primary, gate = make_gate()
    pin(primary, 5.0, 5.0, seed=1)
    for k, twin in enumerate(gate.twins):
        pin(twin, 5.0, 5.0, seed=2 + k)
    pin(gate.twins[-1], 9.0, 5.0, seed=9)
    assert gate.spread(primary) == pytest.approx(4.0, abs=0.1)
    assert not gate.agrees(primary)


def test_threshold_is_inclusive():
    """A spread exactly at agreement_m still declares, as the frozen study rule did."""
    primary, gate = make_gate()
    pin(primary, 5.0, 5.0, seed=1)
    for k, twin in enumerate(gate.twins):
        pin(twin, 6.0, 5.0, seed=2 + k)
    gate.agreement_m = gate.spread(primary)
    assert gate.agrees(primary)


def test_update_feeds_every_twin():
    """Every twin sees each measurement the primary sees."""
    _, gate = make_gate()
    gate.update(np.array([3.0, 4.0, 1.0]), 0.02)
    gate.update(np.array([3.5, 4.0, 1.0]), 0.03)
    assert all(twin.n_updates == 2 for twin in gate.twins)


def test_reset_redraws_every_twin():
    """After a reset no twin is still sitting on its old clump."""
    _, gate = make_gate()
    for k, twin in enumerate(gate.twins):
        pin(twin, 5.0, 5.0, seed=2 + k)
    gate.reset()
    for twin in gate.twins:
        assert np.std(twin.particles[:, 0]) > 3.0


def test_gate_replays_bit_exactly_from_its_seeds():
    """Decision D7: the same seeds and measurements give the same spread, exactly."""
    positions = [np.array([2.0 + 0.5 * i, 6.0, 1.0]) for i in range(15)]
    readings = [0.01 * (i % 4) for i in range(15)]
    spreads = []
    for _ in range(2):
        primary, gate = make_gate(pf_seed=11)
        for position, z in zip(positions, readings):
            primary.update(position, z)
            gate.update(position, z)
        spreads.append(gate.spread(primary))
    assert spreads[0] == spreads[1]


def test_rejects_a_gate_that_could_never_work():
    """No twins, or a non-positive distance, is a configuration error."""
    config = make_config()
    with pytest.raises(ValueError):
        AgreementGate(config, [])
    with pytest.raises(ValueError):
        AgreementGate(config, twin_seeds(0, 3), agreement_m=0.0)
