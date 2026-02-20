"""Sentence and step embedding computation using sentence-transformers."""

import numpy as np


class EmbeddingModel:
    """Wrapper around sentence-transformers for computing text embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        """Compute embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            Array of shape (len(texts), embedding_dim).
        """
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def embed_single(self, text: str) -> np.ndarray:
        """Compute embedding for a single text.

        Returns:
            1-D array of shape (embedding_dim,).
        """
        return self.embed([text])[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def pairwise_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.

    Args:
        embeddings: Array of shape (n, dim).

    Returns:
        Similarity matrix of shape (n, n).
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized = embeddings / norms
    return normalized @ normalized.T


def pairwise_cosine_distance(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance matrix (1 - cosine_similarity).

    Args:
        embeddings: Array of shape (n, dim).

    Returns:
        Distance matrix of shape (n, n) with values in [0, 2].
    """
    return 1.0 - pairwise_cosine_similarity(embeddings)
