# SECOT — Semantic Entropy for Chain-of-Thought

Investigating whether **semantic entropy** can detect **unfaithful reasoning** in chain-of-thought (CoT) outputs from language models.

Based on:
- [Semantic Uncertainty (Kuhn et al., Nature 2024)](https://www.nature.com/articles/s41586-024-07421-0) — the semantic entropy method
- [ChainScope (Arcuschin et al., 2025)](https://arxiv.org/abs/2503.08679) — examples of unfaithful CoT reasoning

## Two Clustering Methods

### Method 1: Key Step Clustering

From each CoT response, identify the **key reasoning step** — the pivotal step that most directly determines the answer. Cluster these key steps across multiple generations and compute semantic entropy over the clusters.

**Intuition:** If the model is reasoning faithfully, the key step should be semantically consistent across generations. High entropy in key steps suggests the model is rationalizing rather than reasoning.

### Method 2: Path Clustering

Represent each CoT as a **reasoning path** (an ordered sequence of step embeddings). Compute pairwise distances between paths using Dynamic Time Warping (DTW) or aggregate embeddings, then cluster and compute semantic entropy.

**Intuition:** Unfaithful reasoning may take wildly different paths to reach the same answer, or similar paths to reach different answers. Path entropy captures this structural inconsistency.

## Project Structure

```
secot/
├── config.py              # Configuration
├── api_client.py          # OpenRouter API client
├── cot_parser.py          # Parse CoT into reasoning steps
├── embeddings.py          # Sentence embeddings (sentence-transformers)
├── nli.py                 # NLI-based semantic equivalence (DeBERTa)
├── entropy.py             # Semantic entropy computation
├── unfaithfulness.py      # Entity-flip unfaithfulness detection
├── pipeline.py            # Main orchestration pipeline
└── clustering/
    ├── key_step.py        # Method 1: Key Step clustering
    └── path.py            # Method 2: Path clustering
scripts/
├── run_pipeline.py        # CLI entry point for full analysis
└── generate_responses.py  # Generate and store CoT responses
data/
├── prompts/               # Comparative YES/NO prompts (YAML)
├── responses/             # Stored CoT responses (JSON)
└── results/               # Analysis output
```

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="your-key-here"
```

## Usage

### Generate CoT responses

```bash
python scripts/generate_responses.py \
    --prompts data/prompts/sample_prompts.yaml \
    --output data/responses/ \
    --num-generations 10 \
    --model anthropic/claude-3.5-sonnet
```

### Run analysis on stored responses

```bash
python scripts/run_pipeline.py \
    --responses data/responses/population_germany_france.json \
    --methods key_step path
```

### Full pipeline with unfaithfulness detection

```bash
python scripts/run_pipeline.py \
    --prompts data/prompts/sample_prompts.yaml \
    --detect-unfaithfulness \
    --methods key_step path
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--distance-metric` | `dtw` | Path distance: `dtw` or `embedding` |
| `--cluster-threshold` | `0.5` | Distance threshold for agglomerative clustering |
| `--key-step-method` | `heuristic` | Key step ID: `heuristic` or `embedding` |
| `--temperature` | `1.0` | Sampling temperature (higher = more diverse CoT) |
| `--num-generations` | `10` | CoT responses per prompt |

## How It Works

### Semantic Entropy

Standard entropy over token sequences treats "Paris is the capital of France" and "The capital of France is Paris" as different. **Semantic entropy** first clusters generations into meaning-equivalent groups, then computes entropy over the clusters:

```
SE(x) = −Σ_c P(C=c|x) · log P(C=c|x)
```

where `P(C=c|x)` is the total probability mass of cluster `c`.

### Key Step Identification (Method 1)

Steps are scored by:
- **Causal markers**: "therefore", "thus", "hence", etc. → high score
- **Position**: Steps in the 40-85% range of the chain (after setup, before restatement)
- **Substantiveness**: Steps with 10-80 words (not too short, not padding)
- **Quantitative reasoning**: Steps containing numerical comparisons
- Alternatively, **embedding trajectory**: the step where the embedding direction changes most

### Path Distance (Method 2)

**DTW (Dynamic Time Warping)** aligns two step-embedding sequences optimally, handling different chain lengths. Cost at each alignment point is cosine distance between step embeddings.

**Aggregate embedding** averages step embeddings (with position-based weighting) for a simpler but less structured comparison.

### Unfaithfulness Detection

For comparative YES/NO questions, flipping entities ("Is A > B?" → "Is B > A?") should flip the answer. When the model gives the **same answer regardless of order**, this signals **Implicit Post-Hoc Rationalization (IPHR)** — the CoT is a post-hoc justification rather than genuine reasoning.
