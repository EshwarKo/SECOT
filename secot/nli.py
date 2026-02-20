"""NLI-based semantic equivalence checking using DeBERTa.

Implements the bidirectional entailment clustering from the semantic
entropy paper (Kuhn et al., 2023). Two texts are semantically equivalent
if they mutually entail each other according to an NLI model.
"""

import numpy as np
import torch


class NLIModel:
    """Natural Language Inference model for entailment checking."""

    # DeBERTa MNLI label mapping: 0=contradiction, 1=neutral, 2=entailment
    CONTRADICTION = 0
    NEUTRAL = 1
    ENTAILMENT = 2

    def __init__(self, model_name: str = "microsoft/deberta-large-mnli"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def entailment_probability(self, premise: str, hypothesis: str) -> float:
        """Compute probability that premise entails hypothesis.

        Args:
            premise: The premise text.
            hypothesis: The hypothesis text.

        Returns:
            Probability of entailment (0 to 1).
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        return float(probs[self.ENTAILMENT])

    def check_bidirectional_entailment(
        self,
        text_a: str,
        text_b: str,
        threshold: float = 0.5,
    ) -> bool:
        """Check if two texts mutually entail each other.

        This is the core operation from the semantic entropy paper:
        two generations are semantically equivalent iff they
        bidirectionally entail each other.

        Args:
            text_a: First text.
            text_b: Second text.
            threshold: Minimum entailment probability for each direction.

        Returns:
            True if texts are semantically equivalent.
        """
        p_ab = self.entailment_probability(text_a, text_b)
        p_ba = self.entailment_probability(text_b, text_a)
        return p_ab >= threshold and p_ba >= threshold

    def build_equivalence_classes(
        self,
        texts: list[str],
        threshold: float = 0.5,
    ) -> list[list[int]]:
        """Cluster texts into semantic equivalence classes.

        Uses bidirectional entailment to build a graph, then finds
        connected components as equivalence classes.

        Args:
            texts: List of texts to cluster.
            threshold: Entailment probability threshold.

        Returns:
            List of clusters, each a list of text indices.
        """
        n = len(texts)
        if n == 0:
            return []
        if n == 1:
            return [[0]]

        # Union-Find structure
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Check all pairs for bidirectional entailment
        for i in range(n):
            for j in range(i + 1, n):
                if self.check_bidirectional_entailment(texts[i], texts[j], threshold):
                    union(i, j)

        # Collect equivalence classes
        clusters: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(i)

        return list(clusters.values())


def build_entailment_distance_matrix(
    nli_model: NLIModel,
    texts: list[str],
) -> np.ndarray:
    """Build a distance matrix based on entailment probabilities.

    Distance = 1 - mean(P(a|b), P(b|a)) for each pair.

    Args:
        nli_model: The NLI model to use.
        texts: List of texts.

    Returns:
        Distance matrix of shape (n, n).
    """
    n = len(texts)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p_ij = nli_model.entailment_probability(texts[i], texts[j])
            p_ji = nli_model.entailment_probability(texts[j], texts[i])
            d = 1.0 - (p_ij + p_ji) / 2.0
            dist[i, j] = d
            dist[j, i] = d
    return dist
