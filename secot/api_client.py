"""OpenRouter API client for generating chain-of-thought responses."""

import json
import time
from dataclasses import dataclass
from typing import Optional

import requests

from secot.config import Config


@dataclass
class CoTResponse:
    """A single chain-of-thought response from the model."""

    prompt: str
    raw_text: str
    answer: Optional[str] = None  # Extracted YES/NO answer
    log_prob: Optional[float] = None  # Log-probability if available


def generate_cot_responses(
    prompt: str,
    config: Config,
    system_prompt: Optional[str] = None,
    n: Optional[int] = None,
) -> list[CoTResponse]:
    """Generate multiple CoT responses for a prompt via OpenRouter.

    Args:
        prompt: The question/prompt to send to the model.
        config: Pipeline configuration.
        system_prompt: Optional system prompt for CoT instruction.
        n: Number of generations (overrides config.num_generations).

    Returns:
        List of CoTResponse objects.
    """
    num = n or config.num_generations

    if system_prompt is None:
        system_prompt = (
            "You are a careful reasoner. Think step by step before giving "
            "your final answer. Show your complete chain of thought, then "
            "conclude with your answer on a new line starting with 'Answer: '."
        )

    headers = {
        "Authorization": f"Bearer {config.openrouter_api_key}",
        "Content-Type": "application/json",
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    responses = []
    for _ in range(num):
        response = _call_api_with_retry(
            url=f"{config.openrouter_base_url}/chat/completions",
            headers=headers,
            payload={
                "model": config.model_name,
                "messages": messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            },
        )

        if response is not None:
            text = response["choices"][0]["message"]["content"]
            answer = _extract_answer(text)
            log_prob = _extract_log_prob(response)
            responses.append(
                CoTResponse(
                    prompt=prompt,
                    raw_text=text,
                    answer=answer,
                    log_prob=log_prob,
                )
            )

    return responses


def _call_api_with_retry(
    url: str,
    headers: dict,
    payload: dict,
    max_retries: int = 4,
) -> Optional[dict]:
    """Call the API with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"API call failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"API call failed after {max_retries} attempts: {e}")
                return None


def _extract_answer(text: str) -> Optional[str]:
    """Extract a YES/NO answer from the end of a CoT response."""
    text_upper = text.upper()

    # Look for explicit "Answer: YES/NO" pattern
    for line in reversed(text.split("\n")):
        line_stripped = line.strip().upper()
        if line_stripped.startswith("ANSWER:"):
            remainder = line_stripped.replace("ANSWER:", "").strip()
            if "YES" in remainder:
                return "YES"
            if "NO" in remainder:
                return "NO"

    # Fallback: check last few lines for YES/NO
    last_lines = text_upper.split("\n")[-3:]
    for line in reversed(last_lines):
        line = line.strip()
        if line in ("YES", "YES.", "NO", "NO."):
            return line.rstrip(".")

    return None


def _extract_log_prob(response: dict) -> Optional[float]:
    """Extract log-probability from API response if available."""
    try:
        choice = response["choices"][0]
        if "logprobs" in choice and choice["logprobs"] is not None:
            token_logprobs = choice["logprobs"]["content"]
            return sum(t["logprob"] for t in token_logprobs)
    except (KeyError, TypeError):
        pass
    return None
