#!/usr/bin/env python3
"""Main entry point for the SECOT pipeline.

Usage:
    # Run full pipeline with API generation:
    python scripts/run_pipeline.py --prompts data/prompts/sample_prompts.yaml

    # Run on stored responses (no API calls needed):
    python scripts/run_pipeline.py --responses data/responses/my_responses.json

    # Run with entity-flip unfaithfulness detection:
    python scripts/run_pipeline.py --prompts data/prompts/sample_prompts.yaml --detect-unfaithfulness

    # Choose specific methods:
    python scripts/run_pipeline.py --responses data/responses/my_responses.json --methods key_step path
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from secot.config import Config
from secot.pipeline import Pipeline, load_prompts


def main():
    parser = argparse.ArgumentParser(
        description="SECOT: Semantic Entropy for Chain-of-Thought analysis"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        help="Path to prompts YAML file (will generate responses via API)",
    )
    parser.add_argument(
        "--responses",
        type=str,
        help="Path to pre-generated responses JSON file (no API calls)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["key_step", "path"],
        choices=["key_step", "path"],
        help="Clustering methods to run",
    )
    parser.add_argument(
        "--detect-unfaithfulness",
        action="store_true",
        help="Run entity-flip unfaithfulness detection",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=10,
        help="Number of CoT generations per prompt",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-3.5-sonnet",
        help="Model to use via OpenRouter",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results",
        help="Directory for output files",
    )
    parser.add_argument(
        "--distance-metric",
        type=str,
        default="dtw",
        choices=["dtw", "embedding"],
        help="Distance metric for path clustering",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.5,
        help="Distance threshold for path clustering",
    )
    parser.add_argument(
        "--key-step-method",
        type=str,
        default="heuristic",
        choices=["heuristic", "embedding"],
        help="Method for identifying key steps",
    )

    args = parser.parse_args()

    if not args.prompts and not args.responses:
        parser.error("Either --prompts or --responses must be specified")

    # Build config
    config = Config(
        model_name=args.model,
        temperature=args.temperature,
        num_generations=args.num_generations,
        output_dir=args.output_dir,
        path_distance_metric=args.distance_metric,
        path_cluster_threshold=args.cluster_threshold,
        key_step_method=args.key_step_method,
    )

    pipeline = Pipeline(config)

    if args.responses:
        _run_on_stored_responses(pipeline, args)
    elif args.prompts:
        _run_on_prompts(pipeline, args)


def _run_on_stored_responses(pipeline: Pipeline, args):
    """Run analysis on pre-generated responses."""
    print(f"Loading responses from {args.responses}...")
    with open(args.responses) as f:
        data = json.load(f)

    if isinstance(data, list):
        # List of response texts
        cot_texts = data
        prompt = ""
    elif isinstance(data, dict):
        cot_texts = data.get("responses", data.get("cot_texts", []))
        prompt = data.get("prompt", "")
    else:
        print("Error: responses file must contain a list or dict")
        sys.exit(1)

    print(f"Loaded {len(cot_texts)} responses")
    analysis = pipeline.analyze_stored_responses(
        cot_texts, prompt=prompt, methods=args.methods
    )
    pipeline.save_results(analysis, label="stored")
    _print_summary(analysis)


def _run_on_prompts(pipeline: Pipeline, args):
    """Generate responses and run analysis for each prompt."""
    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts")

    for i, prompt_data in enumerate(prompts):
        prompt_text = prompt_data.get("prompt", "")
        label = prompt_data.get("label", f"prompt_{i}")
        print(f"\n{'='*60}")
        print(f"Prompt {i+1}/{len(prompts)}: {prompt_text[:80]}...")
        print(f"{'='*60}")

        if args.detect_unfaithfulness and "entity_a" in prompt_data:
            orig, flipped, unfaith = pipeline.analyze_prompt_pair(
                property_name=prompt_data["property"],
                entity_a=prompt_data["entity_a"],
                entity_b=prompt_data["entity_b"],
                expected_gt=prompt_data.get("expected_gt", "a"),
                methods=args.methods,
            )
            pipeline.save_results(orig, label=f"{label}_original")
            pipeline.save_results(flipped, label=f"{label}_flipped")
            _print_summary(orig)
            _print_unfaithfulness_summary(unfaith)
        else:
            analysis = pipeline.analyze_prompt(prompt_text, methods=args.methods)
            pipeline.save_results(analysis, label=label)
            _print_summary(analysis)


def _print_summary(analysis):
    """Print a summary of the analysis results."""
    print(f"\n--- Results for: {analysis.prompt[:60]}... ---")
    print(f"  Responses: {len(analysis.responses)}")

    answers = [a for a in analysis.answers if a is not None]
    if answers:
        yes_count = sum(1 for a in answers if a == "YES")
        no_count = sum(1 for a in answers if a == "NO")
        print(f"  Answers: YES={yes_count}, NO={no_count}")

    if analysis.key_step_result:
        ks = analysis.key_step_result
        print(f"  [Method 1: Key Step]")
        print(f"    Semantic Entropy: {ks.semantic_entropy:.4f}")
        print(f"    Normalized Entropy: {ks.normalized_entropy:.4f}")
        print(f"    Clusters: {len(ks.clusters)} (sizes: {[len(c) for c in ks.clusters]})")

    if analysis.path_result:
        pr = analysis.path_result
        print(f"  [Method 2: Path]")
        print(f"    Semantic Entropy: {pr.semantic_entropy:.4f}")
        print(f"    Normalized Entropy: {pr.normalized_entropy:.4f}")
        print(f"    Clusters: {len(pr.clusters)} (sizes: {[len(c) for c in pr.clusters]})")


def _print_unfaithfulness_summary(result):
    """Print unfaithfulness detection results."""
    print(f"\n  [Unfaithfulness Detection]")
    print(f"    Original accuracy: {result.original_accuracy:.2%}")
    print(f"    Flipped accuracy:  {result.flipped_accuracy:.2%}")
    print(f"    Flip rate:         {result.flip_rate:.2%}")
    print(f"    Agreement rate:    {result.agreement_rate:.2%}")
    print(f"    Unfaithfulness:    {result.unfaithfulness_score:.4f}")


if __name__ == "__main__":
    main()
