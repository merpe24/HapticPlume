"""
Agglomerative clustering of duplicate source estimates.

Implements Goodell et al. (RAS 2026) Section 3.7, Algorithm 1. The particle
filter is stochastic and the same physical leak can be consumed twice, so the
consumed set is periodically collapsed: sources whose estimated positions are
within a distance threshold d are merged into one averaged model. The paper
also notes the useful side effect — a later, better estimate of an
already-consumed plume refines the stored one instead of duplicating it.

Algorithm 1 merges the closest pair of clusters repeatedly using the MINIMUM
inter-cluster distance, which is single-linkage clustering; for a fixed
distance threshold that is exactly the connected components of the graph whose
edges join sources closer than d. This module computes those components with a
union-find, which gives the same answer as the iterative merge loop without its
tie-breaking ambiguity (two equally close pairs would otherwise make the result
depend on array order, and Phase A owes the experiment reproducible runs).

Note on the paper's Algorithm 1 line 3: it reads `while min D >= d`, which
would merge only clusters that are FAR apart. That is taken to be a typo — the
surrounding text says clusters are merged "until ... a distance threshold has
been met", so merging is done while the closest pair is within d.

:author: premmm
:date: July 29, 2026
"""

from haptic_plume_core.plume_model import N_PLUME_PARAMS

import numpy as np

# Index of the wind-direction parameter theta inside an alpha vector. Averaging
# it needs a circular mean; every other parameter averages arithmetically.
THETA_INDEX = 3


def cluster_labels(alphas, d_threshold):
    """
    Assign each source estimate to a single-linkage cluster.

    :param alphas: (L, 8) array of plume parameter vectors
    :param d_threshold: distance threshold d [m]; sources closer than this in
        (x_s, y_s, z_s) are placed in the same cluster
    :return: (L,) integer labels, numbered in order of first appearance
    :rtype: numpy.ndarray
    """
    alphas = _as_alpha_stack(alphas)
    n = alphas.shape[0]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    positions = alphas[:, 0:3]
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) <= d_threshold:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_j] = root_i

    labels = np.empty(n, dtype=int)
    seen = {}
    for i in range(n):
        root = find(i)
        if root not in seen:
            seen[root] = len(seen)
        labels[i] = seen[root]
    return labels


def merge_alphas(alphas):
    """
    Average a group of plume models into one — the merge step of Algorithm 1.

    All eight parameters are averaged, with theta averaged circularly so that
    estimates straddling the +/-pi branch cut do not average to a wind
    direction pointing the wrong way (risk R4).

    :param alphas: (L, 8) array of plume parameter vectors to merge
    :return: (8,) merged plume parameter vector
    :rtype: numpy.ndarray
    """
    alphas = _as_alpha_stack(alphas)
    merged = alphas.mean(axis=0)
    theta = alphas[:, THETA_INDEX]
    merged[THETA_INDEX] = np.arctan2(np.sin(theta).mean(), np.cos(theta).mean())
    return merged


def agglomerative_merge(alphas, d_threshold):
    """
    Cluster nearby source estimates and merge each cluster — Algorithm 1.

    :param alphas: (L, 8) array of plume parameter vectors; L may be 0
    :param d_threshold: distance threshold d [m]
    :return: (M, 8) array of merged plume parameter vectors, M <= L, ordered
        by first appearance of each cluster in the input
    :rtype: numpy.ndarray
    """
    alphas = _as_alpha_stack(alphas)
    if alphas.shape[0] == 0:
        return alphas
    labels = cluster_labels(alphas, d_threshold)
    return np.vstack([merge_alphas(alphas[labels == label])
                      for label in range(labels.max() + 1)])


def _as_alpha_stack(alphas):
    """
    Coerce input to a float (L, 8) array, accepting a bare (8,) vector.

    :param alphas: (8,) or (L, 8) plume parameters
    :return: (L, 8) float array
    :rtype: numpy.ndarray
    """
    stack = np.atleast_2d(np.asarray(alphas, dtype=float))
    if stack.size == 0:
        return np.empty((0, N_PLUME_PARAMS))
    if stack.shape[1] != N_PLUME_PARAMS:
        raise ValueError(
            f'alphas must have {N_PLUME_PARAMS} columns, got {stack.shape[1]}')
    return stack
