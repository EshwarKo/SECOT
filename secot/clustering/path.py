"""Method 2: Path Clustering.

Given multiple chain-of-thought responses to the same prompt:
1. Parse each CoT into an ordered sequence of reasoning steps (a "path")
2. Compute pairwise distances between paths using either:
   a. Dynamic Time Warping (DTW) on step embeddings — captures
      structural similarity even when paths have different lengths
   b. Aggregate path embeddings — simpler but loses step ordering
3. Cluster paths based on the distance matrix
4. Compute semantic entropy over path clusters

High entropy over paths suggests the model arrives at answers via
fundamentally different reasoning routes, which may signal
unfaithful or inconsistent reasoning.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from secot.cot_parser import parse_cot_steps
from secot.embeddings import EmbeddingModel, cosine_similarity
from secot.entropy import compute_semantic_entropy, compute_normalized_entropy


@dataclass
class PathRepresentation:
    """A CoT response represented as an ordered path of step embeddings."""

    cot_text: str
    steps: list[str]
    step_embeddings: np.ndarray  # Shape: (num_steps, embedding_dim)
    path_embedding: np.ndarray  # Shape: (embedding_dim,) — aggregate


@dataclass
class PathClusteringResult:
    """Result of path clustering across multiple CoT responses."""

    paths: list[PathRepresentation]
    distance_matrix: np.ndarray
    clusters: list[list[int]]
    semantic_entropy: float
    normalized_entropy: float
    cluster_labels: list[int]


class PathClusterer:
    """Clusters CoT responses by comparing their full reasoning paths."""

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        distance_metric: str = "dtw",
        cluster_threshold: float = 0.5,
    ):
        self._embedding_model = embedding_model
        self.distance_metric = distance_metric
        self.cluster_threshold = cluster_threshold

    @property
    def embedding_model(self) -> EmbeddingModel:
        if self._embedding_model is None:
            self._embedding_model = EmbeddingModel()
        return self._embedding_model

    def build_path(self, cot_text: str) -> PathRepresentation:
        """Convert a CoT response into a path representation.

        Args:
            cot_text: Raw chain-of-thought text.

        Returns:
            PathRepresentation with step and aggregate embeddings.
        """
        steps = parse_cot_steps(cot_text)
        if not steps:
            steps = [cot_text]

        step_embeddings = self.embedding_model.embed(steps)

        # Aggregate path embedding: weighted average with position decay
        # Later steps get slightly more weight (they're closer to the conclusion)
        weights = np.linspace(0.5, 1.0, len(steps))
        weights = weights / weights.sum()
        path_embedding = np.average(step_embeddings, axis=0, weights=weights)

        return PathRepresentation(
            cot_text=cot_text,
            steps=steps,
            step_embeddings=step_embeddings,
            path_embedding=path_embedding,
        )

    def compute_distance_matrix(
        self,
        paths: list[PathRepresentation],
    ) -> np.ndarray:
        """Compute pairwise distance matrix between paths.

        Args:
            paths: List of path representations.

        Returns:
            Distance matrix of shape (n, n).
        """
        n = len(paths)
        dist = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                if self.distance_metric == "dtw":
                    d = dtw_distance(
                        paths[i].step_embeddings,
                        paths[j].step_embeddings,
                    )
                elif self.distance_metric == "embedding":
                    d = 1.0 - cosine_similarity(
                        paths[i].path_embedding,
                        paths[j].path_embedding,
                    )
                else:
                    raise ValueError(
                        f"Unknown distance metric: {self.distance_metric}"
                    )
                dist[i, j] = d
                dist[j, i] = d

        return dist

    def cluster_and_compute_entropy(
        self,
        cot_texts: list[str],
        probabilities: Optional[list[float]] = None,
    ) -> PathClusteringResult:
        """Full Method 2 pipeline: build paths, compute distances, cluster, entropy.

        Args:
            cot_texts: List of CoT response texts.
            probabilities: Optional per-generation probabilities.

        Returns:
            PathClusteringResult with clusters and entropy.
        """
        # Step 1: Build path representations
        paths = [self.build_path(text) for text in cot_texts]

        # Step 2: Compute pairwise distances
        dist_matrix = self.compute_distance_matrix(paths)

        # Step 3: Cluster paths using agglomerative clustering
        clusters = self._cluster_from_distance_matrix(dist_matrix)

        # Step 4: Compute entropy
        se = compute_semantic_entropy(clusters, probabilities)
        nse = compute_normalized_entropy(clusters, probabilities)

        # Build label array
        labels = [0] * len(cot_texts)
        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                labels[idx] = cluster_id

        return PathClusteringResult(
            paths=paths,
            distance_matrix=dist_matrix,
            clusters=clusters,
            semantic_entropy=se,
            normalized_entropy=nse,
            cluster_labels=labels,
        )

    def _cluster_from_distance_matrix(
        self,
        dist_matrix: np.ndarray,
    ) -> list[list[int]]:
        """Agglomerative clustering from a precomputed distance matrix."""
        n = dist_matrix.shape[0]
        if n <= 1:
            return [[i] for i in range(n)]

        # Convert to condensed form for scipy
        condensed = squareform(dist_matrix, checks=False)

        linkage_matrix = linkage(condensed, method="average")
        labels = fcluster(
            linkage_matrix,
            t=self.cluster_threshold,
            criterion="distance",
        )

        clusters: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(idx)

        return list(clusters.values())


def dtw_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
) -> float:
    """Compute Dynamic Time Warping distance between two embedding sequences.

    DTW finds the optimal alignment between two sequences of potentially
    different lengths, making it ideal for comparing reasoning paths
    that may have different numbers of steps.

    The cost function at each alignment point is cosine distance
    between step embeddings.

    Args:
        seq_a: Embeddings of shape (len_a, dim).
        seq_b: Embeddings of shape (len_b, dim).

    Returns:
        Normalized DTW distance (total cost / alignment length).
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return 1.0

    # Compute pairwise cosine distance cost matrix
    cost = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost[i, j] = 1.0 - cosine_similarity(seq_a[i], seq_b[j])

    # DTW dynamic programming
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw[i, j] = cost[i - 1, j - 1] + min(
                dtw[i - 1, j],      # insertion
                dtw[i, j - 1],      # deletion
                dtw[i - 1, j - 1],  # match
            )

    # Normalize by alignment path length
    path_length = n + m  # Upper bound on path length
    return float(dtw[n, m] / path_length)


def step_alignment_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
) -> float:
    """Simpler alternative to DTW: align by relative position.

    Linearly interpolates the shorter sequence to match the longer,
    then computes mean cosine distance between aligned steps.
    Faster than DTW but less flexible.

    Args:
        seq_a: Embeddings of shape (len_a, dim).
        seq_b: Embeddings of shape (len_b, dim).

    Returns:
        Mean cosine distance between position-aligned steps.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return 1.0

    # Interpolate shorter to match longer
    target_len = max(n, m)

    def interpolate(seq: np.ndarray, target: int) -> np.ndarray:
        if len(seq) == target:
            return seq
        indices = np.linspace(0, len(seq) - 1, target)
        result = np.zeros((target, seq.shape[1]))
        for i, idx in enumerate(indices):
            lo = int(np.floor(idx))
            hi = min(lo + 1, len(seq) - 1)
            frac = idx - lo
            result[i] = (1 - frac) * seq[lo] + frac * seq[hi]
        return result

    aligned_a = interpolate(seq_a, target_len)
    aligned_b = interpolate(seq_b, target_len)

    distances = [
        1.0 - cosine_similarity(aligned_a[i], aligned_b[i])
        for i in range(target_len)
    ]
    return float(np.mean(distances))
