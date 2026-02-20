"""Semantic entropy computation over clusters.

Implements the entropy calculation from:
    Kuhn et al., "Semantic Uncertainty" (Nature, 2024)

Semantic entropy is computed over clusters (semantic equivalence classes)
rather than individual token sequences, capturing meaning-level uncertainty.
"""

import math
from typing import Optional

import numpy as np


def compute_semantic_entropy(
    clusters: list[list[int]],
    probabilities: Optional[list[float]] = None,
    total: Optional[int] = None,
) -> float:
    """Compute semantic entropy over clusters of generations.

    SE(x) = -sum_c P(C=c|x) * log P(C=c|x)

    where P(C=c|x) = sum_{s in c} p(s|x) is the total probability mass
    assigned to cluster c.

    Args:
        clusters: List of clusters, each a list of generation indices.
        probabilities: Optional per-generation probabilities. If None,
            uniform distribution (1/N) is assumed.
        total: Total number of generations (used with uniform assumption).
            If None, inferred from clusters.

    Returns:
        Semantic entropy value (in nats). 0 = all same cluster, higher = more diverse.
    """
    if not clusters:
        return 0.0

    all_indices = [idx for cluster in clusters for idx in cluster]
    n = total or len(all_indices)

    if n == 0:
        return 0.0

    if probabilities is None:
        # Uniform probability: each generation has probability 1/N
        cluster_probs = [len(c) / n for c in clusters]
    else:
        # Sum probabilities within each cluster
        cluster_probs = []
        for cluster in clusters:
            p = sum(probabilities[i] for i in cluster)
            cluster_probs.append(p)

    # Normalize to ensure valid distribution
    total_prob = sum(cluster_probs)
    if total_prob > 0:
        cluster_probs = [p / total_prob for p in cluster_probs]

    return _entropy(cluster_probs)


def compute_normalized_entropy(
    clusters: list[list[int]],
    probabilities: Optional[list[float]] = None,
    total: Optional[int] = None,
) -> float:
    """Compute normalized semantic entropy in [0, 1].

    Normalized by log(K) where K is the number of clusters,
    so 0 = one dominant cluster, 1 = uniform across all clusters.
    """
    se = compute_semantic_entropy(clusters, probabilities, total)
    k = len(clusters)
    if k <= 1:
        return 0.0
    return se / math.log(k)


def _entropy(probs: list[float]) -> float:
    """Compute entropy H = -sum p * log(p) for a probability distribution."""
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log(p)
    return h


def log_probs_to_probs(log_probs: list[float]) -> list[float]:
    """Convert log-probabilities to normalized probabilities.

    Uses the log-sum-exp trick for numerical stability.
    """
    arr = np.array(log_probs)
    max_lp = np.max(arr)
    shifted = arr - max_lp
    probs = np.exp(shifted)
    probs = probs / np.sum(probs)
    return probs.tolist()
