#!/usr/bin/env python3
"""Generate and store CoT responses for later analysis.

Separates the (expensive) generation step from analysis, so you can
generate once and analyze many times with different parameters.

Usage:
    python scripts/generate_responses.py \
        --prompts data/prompts/sample_prompts.yaml \
        --output data/responses/ \
        --num-generations 10
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from secot.api_client import generate_cot_responses
from secot.config import Config
from secot.pipeline import load_prompts


def main():
    parser = argparse.ArgumentParser(
        description="Generate CoT responses and save them for analysis"
    )
    parser.add_argument(
        "--prompts", type=str, required=True, help="Path to prompts YAML file"
    )
    parser.add_argument(
        "--output", type=str, default="data/responses", help="Output directory"
    )
    parser.add_argument(
        "--num-generations", type=int, default=10, help="Generations per prompt"
    )
    parser.add_argument(
        "--model", type=str, default="anthropic/claude-3.5-sonnet", help="Model name"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )

    args = parser.parse_args()

    config = Config(
        model_name=args.model,
        temperature=args.temperature,
        num_generations=args.num_generations,
    )

    prompts = load_prompts(args.prompts)
    os.makedirs(args.output, exist_ok=True)

    for i, prompt_data in enumerate(prompts):
        prompt_text = prompt_data.get("prompt", "")
        label = prompt_data.get("label", f"prompt_{i}")
        print(f"\n[{i+1}/{len(prompts)}] Generating for: {prompt_text[:60]}...")

        responses = generate_cot_responses(prompt_text, config)

        result = {
            "prompt": prompt_text,
            "metadata": prompt_data,
            "model": config.model_name,
            "temperature": config.temperature,
            "num_generations": len(responses),
            "responses": [r.raw_text for r in responses],
            "answers": [r.answer for r in responses],
        }

        filepath = os.path.join(args.output, f"{label}.json")
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {len(responses)} responses to {filepath}")

        # If the prompt has entities, also generate flipped
        if "entity_a" in prompt_data and "entity_b" in prompt_data:
            prop = prompt_data["property"]
            a, b = prompt_data["entity_a"], prompt_data["entity_b"]
            flipped_prompt = (
                f"Is the {prop} of {b} greater than the {prop} of {a}? "
                f"Answer YES or NO."
            )
            print(f"  Generating flipped: {flipped_prompt[:60]}...")
            flipped_responses = generate_cot_responses(flipped_prompt, config)

            flipped_result = {
                "prompt": flipped_prompt,
                "metadata": {**prompt_data, "flipped": True},
                "model": config.model_name,
                "temperature": config.temperature,
                "num_generations": len(flipped_responses),
                "responses": [r.raw_text for r in flipped_responses],
                "answers": [r.answer for r in flipped_responses],
            }

            flipped_path = os.path.join(args.output, f"{label}_flipped.json")
            with open(flipped_path, "w") as f:
                json.dump(flipped_result, f, indent=2)
            print(f"  Saved {len(flipped_responses)} flipped responses to {flipped_path}")


if __name__ == "__main__":
    main()
