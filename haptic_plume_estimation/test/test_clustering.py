"""
Unit tests for haptic_plume_estimation.clustering (paper Algorithm 1).

:author: premmm
:date: July 29, 2026
"""

from haptic_plume_estimation.clustering import (
    agglomerative_merge,
    cluster_labels,
    merge_alphas,
)

import numpy as np

import pytest

D_CLUSTER = 1.8   # paper Table 2, Gaussian-plume environment


def alpha_at(x, y, z=1.0, theta=0.0, q=0.025, v=0.5, d_y=0.035, d_z=5.0e-4):
    """Build a plume parameter vector with a source at (x, y, z)."""
    return np.array([x, y, z, theta, q, v, d_y, d_z])


def test_nearby_duplicates_merge_to_their_average():
    """Two estimates 1 m apart are one source; the merge averages them."""
    alphas = np.vstack([alpha_at(5.0, 10.0), alpha_at(6.0, 10.0)])
    merged = agglomerative_merge(alphas, D_CLUSTER)
    assert merged.shape == (1, 8)
    assert merged[0, 0] == pytest.approx(5.5)
    assert merged[0, 1] == pytest.approx(10.0)


def test_distinct_sources_are_left_alone():
    """Sources 3 m apart exceed d = 1.8 m and stay separate."""
    alphas = np.vstack([alpha_at(5.0, 10.0), alpha_at(8.0, 10.0)])
    merged = agglomerative_merge(alphas, D_CLUSTER)
    assert merged.shape == (2, 8)
    np.testing.assert_allclose(merged, alphas)


def test_single_linkage_chains_through_an_intermediate():
    """
    A 1.5 m chain merges end to end even though the ends are 3.0 m apart.

    This is the defining behaviour of the minimum-distance metric D(.) in
    Algorithm 1, and the reason the threshold d has to stay well under the 2 m
    minimum source separation the paper's scenarios guarantee.
    """
    alphas = np.vstack([alpha_at(5.0, 10.0), alpha_at(6.5, 10.0),
                        alpha_at(8.0, 10.0)])
    assert list(cluster_labels(alphas, D_CLUSTER)) == [0, 0, 0]
    assert agglomerative_merge(alphas, D_CLUSTER).shape == (1, 8)
    assert agglomerative_merge(alphas, 1.0).shape == (3, 8)


def test_labels_number_clusters_in_order_of_appearance():
    """Cluster ids follow input order, so merged output is reproducible."""
    alphas = np.vstack([alpha_at(0.0, 0.0), alpha_at(10.0, 0.0),
                        alpha_at(0.5, 0.0), alpha_at(10.5, 0.0)])
    assert list(cluster_labels(alphas, D_CLUSTER)) == [0, 1, 0, 1]


def test_theta_is_averaged_circularly():
    """
    Wind directions straddling the +/-pi branch cut average to pi, not 0.

    cos(3) = cos(-3) and sin(3) = -sin(-3), so the circular mean is exactly
    pi; an arithmetic mean would report a wind blowing the opposite way (R4).
    """
    alphas = np.vstack([alpha_at(5.0, 10.0, theta=3.0),
                        alpha_at(5.5, 10.0, theta=-3.0)])
    merged = merge_alphas(alphas)
    assert abs(merged[3]) == pytest.approx(np.pi, abs=1e-12)


def test_remaining_parameters_are_arithmetic_means():
    """Q, v and the diffusion constants average the ordinary way."""
    alphas = np.vstack([alpha_at(5.0, 10.0, q=0.02, v=0.4),
                        alpha_at(5.5, 10.0, q=0.03, v=0.6)])
    merged = merge_alphas(alphas)
    assert merged[4] == pytest.approx(0.025)
    assert merged[5] == pytest.approx(0.5)


def test_empty_and_single_inputs():
    """No sources merges to nothing; one source is returned unchanged."""
    assert agglomerative_merge(np.empty((0, 8)), D_CLUSTER).shape == (0, 8)
    single = alpha_at(5.0, 10.0)
    np.testing.assert_allclose(agglomerative_merge(single, D_CLUSTER),
                               single[np.newaxis, :])


def test_wrong_width_is_rejected():
    """A stack that is not 8 wide is a frame/layout bug, not a cluster."""
    with pytest.raises(ValueError):
        agglomerative_merge(np.zeros((3, 5)), D_CLUSTER)
