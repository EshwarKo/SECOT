"""Parse chain-of-thought text into individual reasoning steps."""

import re


def parse_cot_steps(text: str) -> list[str]:
    """Parse a chain-of-thought response into ordered reasoning steps.

    Handles multiple formats:
    - Numbered steps ("1.", "2.", "Step 1:")
    - Bullet points ("- ", "* ")
    - Sentence-based splitting (fallback)

    Args:
        text: Raw CoT text from the model.

    Returns:
        List of reasoning steps in order.
    """
    # Strip the final answer line if present
    text = _strip_answer_line(text)

    # Try numbered steps first
    steps = _parse_numbered_steps(text)
    if len(steps) >= 2:
        return [s.strip() for s in steps if s.strip()]

    # Try bullet points
    steps = _parse_bullet_steps(text)
    if len(steps) >= 2:
        return [s.strip() for s in steps if s.strip()]

    # Fallback: sentence splitting
    steps = segment_sentences(text)
    return [s.strip() for s in steps if s.strip()]


def segment_sentences(text: str) -> list[str]:
    """Split text into sentences using regex-based heuristics.

    Handles common abbreviations and decimal numbers to avoid
    false splits.

    Args:
        text: Raw text to segment.

    Returns:
        List of sentences.
    """
    # Protect common abbreviations from splitting
    protected = text
    abbreviations = [
        "Mr.", "Mrs.", "Dr.", "Prof.", "Sr.", "Jr.",
        "vs.", "etc.", "i.e.", "e.g.", "approx.",
    ]
    placeholders = {}
    for i, abbr in enumerate(abbreviations):
        placeholder = f"__ABBR{i}__"
        placeholders[placeholder] = abbr
        protected = protected.replace(abbr, placeholder)

    # Protect decimal numbers (e.g., "3.14")
    protected = re.sub(r"(\d)\.(\d)", r"\1__DEC__\2", protected)

    # Split on sentence-ending punctuation followed by space or newline
    raw_sentences = re.split(r"(?<=[.!?])\s+", protected)

    # Restore protected tokens
    sentences = []
    for sent in raw_sentences:
        for placeholder, original in placeholders.items():
            sent = sent.replace(placeholder, original)
        sent = sent.replace("__DEC__", ".")
        if sent.strip():
            sentences.append(sent.strip())

    return sentences


def _strip_answer_line(text: str) -> str:
    """Remove the final 'Answer: ...' line from CoT text."""
    lines = text.split("\n")
    filtered = []
    for line in lines:
        if re.match(r"^\s*(answer|final answer|conclusion)\s*:", line, re.IGNORECASE):
            continue
        filtered.append(line)
    return "\n".join(filtered)


def _parse_numbered_steps(text: str) -> list[str]:
    """Extract numbered steps from text."""
    # Match patterns like "1.", "1)", "Step 1:", "Step 1."
    pattern = r"(?:^|\n)\s*(?:step\s+)?\d+[.):\-]\s*"
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    # First part is usually preamble before step 1
    return [p.strip() for p in parts if p.strip()]


def _parse_bullet_steps(text: str) -> list[str]:
    """Extract bullet-pointed steps from text."""
    pattern = r"(?:^|\n)\s*[-*•]\s+"
    parts = re.split(pattern, text)
    return [p.strip() for p in parts if p.strip()]
