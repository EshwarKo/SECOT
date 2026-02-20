"""Configuration for the SECOT pipeline."""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Central configuration for all SECOT components."""

    # --- API settings ---
    openrouter_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "")
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = "anthropic/claude-3.5-sonnet"
    temperature: float = 1.0  # High temperature for diverse CoT generations
    max_tokens: int = 2048
    num_generations: int = 10  # Number of CoT responses per prompt

    # --- Embedding model ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- NLI model (for bidirectional entailment clustering) ---
    nli_model: str = "microsoft/deberta-large-mnli"
    entailment_threshold: float = 0.5  # Min probability for entailment

    # --- Key step identification ---
    key_step_method: str = "heuristic"  # "heuristic" or "llm"

    # --- Path clustering ---
    path_distance_metric: str = "dtw"  # "dtw" or "embedding"
    path_cluster_method: str = "agglomerative"  # "agglomerative" or "dbscan"
    path_cluster_threshold: float = 0.5  # Distance threshold for clustering

    # --- Output directories ---
    output_dir: str = "data/results"
    responses_dir: str = "data/responses"
    prompts_dir: str = "data/prompts"
