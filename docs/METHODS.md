# Classification Methods — Detailed Notes

This doc explains the reasoning behind each method and when you'd expect one to work better than the other.

---

## Why Two Methods?

A chain of thought is a structured object — it has individual steps *and* an overall trajectory. These two methods capture different aspects of inconsistency:

| | Method 1: Key Step | Method 2: Path |
|---|---|---|
| **What it looks at** | The single most pivotal step | The full ordered sequence of steps |
| **What high entropy means** | The model's core reasoning varies across generations | The model takes fundamentally different routes |
| **Sensitive to** | Inconsistency in the critical logical move | Structural diversity in reasoning |
| **Blind to** | Differences in preamble/setup steps | Cases where paths are similar but one key step differs |

Using both gives a more complete picture. For example:
- High key-step entropy + low path entropy → paths look similar overall, but the crucial logical step keeps changing (subtle unfaithfulness)
- Low key-step entropy + high path entropy → the key reasoning is consistent, but the model wraps it in very different surrounding steps (noisy but possibly faithful)
- Both high → strong signal of unfaithful reasoning

---

## Method 1: Key Step Clustering

### What is a "key step"?

The step that most directly determines the final answer. In a chain like:

> 1. Germany is a country in Europe with ~83 million people.
> 2. France is a country in Europe with ~67 million people.
> 3. **Since 83 million > 67 million, Germany has a larger population.**
> 4. Answer: YES

Step 3 is the key step — it's where the actual comparison happens. Steps 1-2 are setup, step 4 is just restating.

### Heuristic scoring

The heuristic approach scores each step on four axes:

- **Causal language** (+2): "therefore", "thus", "hence", "this means", "which implies", "it follows" — these signal that the step is drawing a conclusion from prior reasoning
- **Position** (+1.5): Steps in the 40–85% range of the chain. Earlier steps tend to be setup/recall, the very last step tends to be a restatement. The key reasoning usually happens in the middle-to-late range
- **Length** (+1): Steps with 10–80 words. Very short steps are usually transitions ("Now let's compare"), very long steps are usually exposition. The key step is typically concise but substantive
- **Quantitative content** (+1): Steps containing numbers with comparisons (">", "greater", "less", etc.)

Conclusion markers ("in conclusion", "to summarize") get a penalty since they restate rather than reason.

### Embedding trajectory scoring

Alternative approach — treats the chain as a trajectory through embedding space:

```
e₁ → e₂ → e₃ → e₄ → e₅
```

At each step, compute the directional change:

```
direction_before = e[i] - e[i-1]
direction_after  = e[i+1] - e[i]
change = 1 - cosine_similarity(direction_before, direction_after)
```

The step with the highest directional change is where the reasoning "turns" — that's the key step. This is more principled than heuristics but requires loading the embedding model.

### Clustering key steps

Once you have the key step from each of N generations, cluster them:

- **NLI clustering** (slow, accurate): Check if key step A entails key step B *and* B entails A. If yes, they're in the same semantic equivalence class. Uses DeBERTa. O(N²) NLI forward passes.
- **Embedding clustering** (fast, default): Embed all key steps, compute cosine distances, agglomerative clustering with a distance threshold.

---

## Method 2: Path Clustering

### What is a "path"?

The full ordered sequence of reasoning steps, represented as embeddings:

```
Path = [embed(step₁), embed(step₂), ..., embed(stepₖ)]
```

Two CoTs might both arrive at "YES" but take completely different paths:

**Path A:** recall population → recall population → compare → answer
**Path B:** recall GDP → confuse GDP with population → guess → answer

These should land in different clusters even though they produce the same answer.

### DTW (Dynamic Time Warping)

The key challenge: different CoTs have different numbers of steps. Path A might be 4 steps, Path B might be 7 steps. You can't just compare step-by-step.

DTW solves this by finding the optimal *alignment* between two sequences. It uses dynamic programming to warp the time axis so that similar steps line up:

```
Path A:  [s1] ---- [s2] -- [s3] --------- [s4]
              \        \         \
Path B:  [s1] [s2] [s3] [s4] [s5] [s6] [s7]
```

The cost at each alignment point is the cosine distance between the aligned step embeddings. Total DTW cost (normalized by path length) gives the path distance.

### Aggregate embedding alternative

Simpler approach: average all step embeddings per CoT (with position-based weights — later steps get more weight since they're closer to the conclusion). Then just compare the single aggregate vectors.

Faster, but loses the sequential structure that makes path analysis interesting.

### When to use which

- **DTW** (default): When you care about reasoning *structure* — whether steps happen in the same order, whether the chain builds logically in the same way
- **Aggregate embedding**: When you just want a quick sense of whether the chains cover similar semantic territory, without caring about order

---

## Entropy Interpretation

Both methods produce a semantic entropy score:

```
SE = −Σ P(cluster) · log P(cluster)
```

| SE Value | Interpretation |
|----------|----------------|
| 0 | All generations in one cluster — perfectly consistent |
| Low (< 0.5) | One dominant cluster, maybe a few outliers |
| Medium (0.5–1.5) | Several clusters — moderate inconsistency |
| High (> 1.5) | Many distinct clusters — highly inconsistent reasoning |

The normalized version (dividing by log K) maps to [0, 1]:
- 0 = all same cluster
- 1 = uniform across all clusters (maximum diversity)

### Relationship to unfaithfulness

High semantic entropy alone doesn't prove unfaithfulness — some questions are genuinely ambiguous. But when combined with entity-flip analysis:

- High entropy + high agreement rate (same answer despite entity flip) → strong IPHR signal
- High entropy + low agreement rate → the model is uncertain but at least responding to the question structure
- Low entropy + high agreement rate → the model consistently gives the wrong answer in one direction (bias, not entropy)
