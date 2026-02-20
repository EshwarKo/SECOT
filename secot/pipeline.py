"""Main pipeline orchestrating CoT generation, clustering, and analysis."""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import yaml

from secot.api_client import CoTResponse, generate_cot_responses
from secot.clustering.key_step import KeyStepClusterer, KeyStepClusteringResult
from secot.clustering.path import PathClusterer, PathClusteringResult
from secot.config import Config
from secot.embeddings import EmbeddingModel
from secot.entropy import compute_semantic_entropy
from secot.nli import NLIModel
from secot.unfaithfulness import (
    PromptPair,
    UnfaithfulnessResult,
    analyze_unfaithfulness,
    create_prompt_pair,
)


@dataclass
class PromptAnalysis:
    """Complete analysis results for a single prompt."""

    prompt: str
    responses: list[str]
    answers: list[Optional[str]]
    key_step_result: Optional[KeyStepClusteringResult]
    path_result: Optional[PathClusteringResult]
    unfaithfulness: Optional[UnfaithfulnessResult]


class Pipeline:
    """Orchestrates the full SECOT analysis pipeline.

    Usage:
        config = Config(openrouter_api_key="...")
        pipeline = Pipeline(config)

        # Analyze a single prompt
        result = pipeline.analyze_prompt("Is the population of France > Germany?")

        # Analyze with entity flipping (unfaithfulness detection)
        result = pipeline.analyze_prompt_pair(
            property_name="population",
            entity_a="France",
            entity_b="Germany",
            expected_gt="b",
        )

        # Analyze from stored responses (no API calls)
        result = pipeline.analyze_stored_responses(cot_texts)
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._embedding_model: Optional[EmbeddingModel] = None
        self._nli_model: Optional[NLIModel] = None
        self._key_step_clusterer: Optional[KeyStepClusterer] = None
        self._path_clusterer: Optional[PathClusterer] = None

    @property
    def embedding_model(self) -> EmbeddingModel:
        if self._embedding_model is None:
            self._embedding_model = EmbeddingModel(self.config.embedding_model)
        return self._embedding_model

    @property
    def nli_model(self) -> NLIModel:
        if self._nli_model is None:
            self._nli_model = NLIModel(self.config.nli_model)
        return self._nli_model

    @property
    def key_step_clusterer(self) -> KeyStepClusterer:
        if self._key_step_clusterer is None:
            self._key_step_clusterer = KeyStepClusterer(
                embedding_model=self.embedding_model,
                nli_model=self.nli_model,
                entailment_threshold=self.config.entailment_threshold,
            )
        return self._key_step_clusterer

    @property
    def path_clusterer(self) -> PathClusterer:
        if self._path_clusterer is None:
            self._path_clusterer = PathClusterer(
                embedding_model=self.embedding_model,
                distance_metric=self.config.path_distance_metric,
                cluster_threshold=self.config.path_cluster_threshold,
            )
        return self._path_clusterer

    def analyze_prompt(
        self,
        prompt: str,
        methods: Optional[list[str]] = None,
    ) -> PromptAnalysis:
        """Generate CoT responses and run both clustering methods.

        Args:
            prompt: The question to analyze.
            methods: Which methods to run — ["key_step", "path"].
                Defaults to both.

        Returns:
            PromptAnalysis with full results.
        """
        if methods is None:
            methods = ["key_step", "path"]

        # Generate responses
        print(f"Generating {self.config.num_generations} CoT responses...")
        responses = generate_cot_responses(prompt, self.config)
        cot_texts = [r.raw_text for r in responses]
        answers = [r.answer for r in responses]

        return self._analyze_texts(
            prompt=prompt,
            cot_texts=cot_texts,
            answers=answers,
            methods=methods,
        )

    def analyze_stored_responses(
        self,
        cot_texts: list[str],
        prompt: str = "",
        methods: Optional[list[str]] = None,
    ) -> PromptAnalysis:
        """Run analysis on pre-generated CoT responses (no API calls).

        Args:
            cot_texts: List of CoT response texts.
            prompt: The original prompt (for labeling).
            methods: Which methods to run.

        Returns:
            PromptAnalysis with results.
        """
        if methods is None:
            methods = ["key_step", "path"]

        return self._analyze_texts(
            prompt=prompt,
            cot_texts=cot_texts,
            answers=[None] * len(cot_texts),
            methods=methods,
        )

    def analyze_prompt_pair(
        self,
        property_name: str,
        entity_a: str,
        entity_b: str,
        expected_gt: str = "a",
        methods: Optional[list[str]] = None,
    ) -> tuple[PromptAnalysis, PromptAnalysis, UnfaithfulnessResult]:
        """Full analysis with entity flipping for unfaithfulness detection.

        Args:
            property_name: Property being compared.
            entity_a: First entity.
            entity_b: Second entity.
            expected_gt: Which entity is actually greater ("a" or "b").
            methods: Clustering methods to run.

        Returns:
            Tuple of (original_analysis, flipped_analysis, unfaithfulness_result).
        """
        pair = create_prompt_pair(property_name, entity_a, entity_b, expected_gt)

        # Generate responses for both orderings
        print(f"Generating responses for original prompt...")
        orig_responses = generate_cot_responses(pair.original, self.config)
        print(f"Generating responses for flipped prompt...")
        flip_responses = generate_cot_responses(pair.flipped, self.config)

        # Clustering analysis on both
        orig_analysis = self._analyze_texts(
            prompt=pair.original,
            cot_texts=[r.raw_text for r in orig_responses],
            answers=[r.answer for r in orig_responses],
            methods=methods or ["key_step", "path"],
        )
        flip_analysis = self._analyze_texts(
            prompt=pair.flipped,
            cot_texts=[r.raw_text for r in flip_responses],
            answers=[r.answer for r in flip_responses],
            methods=methods or ["key_step", "path"],
        )

        # Unfaithfulness analysis
        unfaithfulness = analyze_unfaithfulness(pair, orig_responses, flip_responses)

        return orig_analysis, flip_analysis, unfaithfulness

    def _analyze_texts(
        self,
        prompt: str,
        cot_texts: list[str],
        answers: list[Optional[str]],
        methods: list[str],
    ) -> PromptAnalysis:
        """Internal: run clustering methods on CoT texts."""
        key_step_result = None
        path_result = None

        if "key_step" in methods:
            print("Running Method 1: Key Step Clustering...")
            key_step_result = self.key_step_clusterer.cluster_and_compute_entropy(
                cot_texts,
                method=self.config.key_step_method,
                clustering="embedding",  # Default to embedding (NLI is expensive)
            )
            print(
                f"  Key Step SE: {key_step_result.semantic_entropy:.4f} "
                f"({len(key_step_result.clusters)} clusters)"
            )

        if "path" in methods:
            print("Running Method 2: Path Clustering...")
            path_result = self.path_clusterer.cluster_and_compute_entropy(cot_texts)
            print(
                f"  Path SE: {path_result.semantic_entropy:.4f} "
                f"({len(path_result.clusters)} clusters)"
            )

        return PromptAnalysis(
            prompt=prompt,
            responses=cot_texts,
            answers=answers,
            key_step_result=key_step_result,
            path_result=path_result,
            unfaithfulness=None,
        )

    def save_results(
        self,
        analysis: PromptAnalysis,
        output_dir: Optional[str] = None,
        label: str = "",
    ) -> str:
        """Save analysis results to disk.

        Returns:
            Path to the saved results file.
        """
        out_dir = output_dir or self.config.output_dir
        os.makedirs(out_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{label}_{timestamp}.json" if label else f"analysis_{timestamp}.json"
        filepath = os.path.join(out_dir, filename)

        result = {
            "prompt": analysis.prompt,
            "num_responses": len(analysis.responses),
            "answers": analysis.answers,
            "responses": analysis.responses,
        }

        if analysis.key_step_result:
            ks = analysis.key_step_result
            result["key_step"] = {
                "semantic_entropy": ks.semantic_entropy,
                "normalized_entropy": ks.normalized_entropy,
                "num_clusters": len(ks.clusters),
                "cluster_sizes": [len(c) for c in ks.clusters],
                "cluster_labels": ks.cluster_labels,
                "key_steps": [
                    {
                        "step_index": r.key_step_index,
                        "step_text": r.key_step_text,
                        "score": r.key_step_score,
                    }
                    for r in ks.key_steps
                ],
            }

        if analysis.path_result:
            pr = analysis.path_result
            result["path"] = {
                "semantic_entropy": pr.semantic_entropy,
                "normalized_entropy": pr.normalized_entropy,
                "num_clusters": len(pr.clusters),
                "cluster_sizes": [len(c) for c in pr.clusters],
                "cluster_labels": pr.cluster_labels,
                "distance_matrix": pr.distance_matrix.tolist(),
            }

        with open(filepath, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"Results saved to {filepath}")
        return filepath


def load_prompts(prompts_file: str) -> list[dict]:
    """Load prompts from a YAML file.

    Expected format:
        prompts:
          - prompt: "Is X > Y?"
            property: "population"
            entity_a: "France"
            entity_b: "Germany"
            expected_gt: "b"
    """
    with open(prompts_file) as f:
        data = yaml.safe_load(f)
    return data.get("prompts", [])
