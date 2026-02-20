"""Method 1: Key Step Clustering.

Given multiple chain-of-thought responses to the same prompt:
1. Parse each CoT into reasoning steps
2. Identify the KEY STEP — the pivotal reasoning step in each chain
3. Cluster key steps across all generations using semantic equivalence
4. Compute semantic entropy over the key step clusters

High entropy in key steps suggests the model is not consistently
using the same core reasoning, which may indicate unfaithful CoT.
"""

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from secot.cot_parser import parse_cot_steps
from secot.embeddings import EmbeddingModel, cosine_similarity
from secot.entropy import compute_semantic_entropy, compute_normalized_entropy
from secot.nli import NLIModel


# Linguistic markers indicating a pivotal reasoning step
CAUSAL_MARKERS = [
    "therefore", "thus", "hence", "so", "consequently",
    "this means", "this implies", "which means", "which implies",
    "it follows", "as a result", "because of this",
    "we can conclude", "this shows", "this tells us",
]

CONCLUSION_MARKERS = [
    "in conclusion", "to summarize", "overall", "finally",
    "the answer is", "the result is",
]


@dataclass
class KeyStepResult:
    """Result of key step analysis for a single CoT response."""

    cot_text: str
    steps: list[str]
    key_step_index: int
    key_step_text: str
    key_step_score: float


@dataclass
class KeyStepClusteringResult:
    """Result of clustering key steps across multiple CoT responses."""

    key_steps: list[KeyStepResult]
    clusters: list[list[int]]
    semantic_entropy: float
    normalized_entropy: float
    cluster_labels: list[int]


class KeyStepClusterer:
    """Identifies key steps in CoT responses and clusters them."""

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        nli_model: Optional[NLIModel] = None,
        entailment_threshold: float = 0.5,
    ):
        self._embedding_model = embedding_model
        self._nli_model = nli_model
        self.entailment_threshold = entailment_threshold

    @property
    def embedding_model(self) -> EmbeddingModel:
        if self._embedding_model is None:
            self._embedding_model = EmbeddingModel()
        return self._embedding_model

    @property
    def nli_model(self) -> NLIModel:
        if self._nli_model is None:
            self._nli_model = NLIModel()
        return self._nli_model

    def identify_key_step(
        self,
        cot_text: str,
        method: str = "heuristic",
    ) -> KeyStepResult:
        """Identify the key reasoning step in a chain of thought.

        Args:
            cot_text: The full CoT text.
            method: Identification method — "heuristic" or "embedding".

        Returns:
            KeyStepResult with the identified key step.
        """
        steps = parse_cot_steps(cot_text)
        if not steps:
            return KeyStepResult(
                cot_text=cot_text,
                steps=[],
                key_step_index=0,
                key_step_text=cot_text,
                key_step_score=0.0,
            )

        if method == "heuristic":
            scores = self._score_steps_heuristic(steps)
        elif method == "embedding":
            scores = self._score_steps_embedding(steps)
        else:
            raise ValueError(f"Unknown key step method: {method}")

        key_idx = int(np.argmax(scores))
        return KeyStepResult(
            cot_text=cot_text,
            steps=steps,
            key_step_index=key_idx,
            key_step_text=steps[key_idx],
            key_step_score=float(scores[key_idx]),
        )

    def cluster_and_compute_entropy(
        self,
        cot_texts: list[str],
        method: str = "heuristic",
        clustering: str = "nli",
        probabilities: Optional[list[float]] = None,
    ) -> KeyStepClusteringResult:
        """Full Method 1 pipeline: identify key steps, cluster, compute entropy.

        Args:
            cot_texts: List of CoT response texts (multiple generations
                for the same prompt).
            method: Key step identification method ("heuristic" or "embedding").
            clustering: Clustering approach — "nli" for bidirectional
                entailment, "embedding" for cosine similarity.
            probabilities: Optional per-generation probabilities.

        Returns:
            KeyStepClusteringResult with clusters and entropy.
        """
        # Step 1: Identify key step in each CoT
        key_step_results = [
            self.identify_key_step(text, method=method) for text in cot_texts
        ]
        key_step_texts = [r.key_step_text for r in key_step_results]

        # Step 2: Cluster key steps
        if clustering == "nli":
            clusters = self.nli_model.build_equivalence_classes(
                key_step_texts, threshold=self.entailment_threshold
            )
        elif clustering == "embedding":
            clusters = self._cluster_by_embedding(key_step_texts)
        else:
            raise ValueError(f"Unknown clustering method: {clustering}")

        # Step 3: Compute semantic entropy
        se = compute_semantic_entropy(clusters, probabilities)
        nse = compute_normalized_entropy(clusters, probabilities)

        # Build cluster label array
        labels = [0] * len(cot_texts)
        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                labels[idx] = cluster_id

        return KeyStepClusteringResult(
            key_steps=key_step_results,
            clusters=clusters,
            semantic_entropy=se,
            normalized_entropy=nse,
            cluster_labels=labels,
        )

    def _score_steps_heuristic(self, steps: list[str]) -> np.ndarray:
        """Score each step's likelihood of being the key step using heuristics.

        Scoring factors:
        1. Causal markers ("therefore", "thus", "hence", ...)
        2. Position bias (later steps are more likely to be key, but
           not the very last which is often just a restatement)
        3. Length (key steps tend to be substantive, not too short)
        4. Avoids pure conclusion/summary steps
        """
        n = len(steps)
        scores = np.zeros(n)

        for i, step in enumerate(steps):
            step_lower = step.lower()

            # Causal marker bonus
            for marker in CAUSAL_MARKERS:
                if marker in step_lower:
                    scores[i] += 2.0
                    break

            # Conclusion marker penalty (these restate, not reason)
            for marker in CONCLUSION_MARKERS:
                if marker in step_lower:
                    scores[i] -= 1.0
                    break

            # Position score: prefer steps in the 50-90% range of the chain
            # (after setup, before final restatement)
            relative_pos = (i + 1) / n
            if 0.4 <= relative_pos <= 0.85:
                scores[i] += 1.5
            elif relative_pos > 0.85:
                scores[i] += 0.5

            # Length score: substantive steps are typically 10-100 words
            word_count = len(step.split())
            if 10 <= word_count <= 80:
                scores[i] += 1.0
            elif word_count < 5:
                scores[i] -= 0.5

            # Presence of comparison or quantitative reasoning
            if re.search(r"\d+\.?\d*\s*(>|<|greater|less|more|fewer|larger|smaller)", step_lower):
                scores[i] += 1.0

        return scores

    def _score_steps_embedding(self, steps: list[str]) -> np.ndarray:
        """Score steps by how much they shift the embedding trajectory.

        The key step is the one that causes the largest directional change
        in the embedding space — i.e., the step where reasoning "turns".
        """
        if len(steps) < 3:
            # Not enough steps for trajectory analysis; fall back to position
            scores = np.zeros(len(steps))
            scores[-1] = 1.0
            return scores

        embeddings = self.embedding_model.embed(steps)

        # Compute directional change at each step:
        # delta[i] = 1 - cos(emb[i] - emb[i-1], emb[i+1] - emb[i])
        # High delta = the reasoning changed direction at step i
        scores = np.zeros(len(steps))
        for i in range(1, len(steps) - 1):
            before = embeddings[i] - embeddings[i - 1]
            after = embeddings[i + 1] - embeddings[i]
            sim = cosine_similarity(before, after)
            # Lower similarity = larger directional change = more "key"
            scores[i] = 1.0 - sim

        return scores

    def _cluster_by_embedding(
        self,
        texts: list[str],
        threshold: float = 0.3,
    ) -> list[list[int]]:
        """Cluster texts using agglomerative clustering on embeddings."""
        from scipy.cluster.hierarchy import fcluster, linkage

        if len(texts) <= 1:
            return [[i] for i in range(len(texts))]

        embeddings = self.embedding_model.embed(texts)

        # Agglomerative clustering with cosine distance
        linkage_matrix = linkage(embeddings, method="average", metric="cosine")
        labels = fcluster(linkage_matrix, t=threshold, criterion="distance")

        clusters: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(idx)

        return list(clusters.values())
