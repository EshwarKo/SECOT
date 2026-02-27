# Unfaithful Reasoning Detection

Research project exploring methods to detect unfaithful chain-of-thought reasoning in language models through semantic entropy analysis and entity flipping techniques.

## Overview

This repository contains experiments testing whether language models genuinely reason through comparison tasks or engage in post-hoc rationalization. Two complementary detection methods are implemented:

1. **Semantic Entropy Analysis** - Measures uncertainty in model reasoning by clustering semantically equivalent responses
2. **Entity Flipping Detection** - Tests for Implicit Post-Hoc Rationalization (IPHR) by swapping entities in comparison questions

## Structure

```
.
├── data/                           # Input datasets (YAML format)
│   ├── aircraft-speeds_*.yaml      # Aircraft speed comparison tasks
│   └── wm-book-length_*.yaml       # Book length comparison tasks
├── results/                        # Output files
│   ├── *_results.json              # Numerical results and metrics
│   └── *.png                       # Visualization plots
├── semantic_entropy_unfaithful_reasoning.ipynb
└── unfaithfulness_detection.ipynb
```

## Notebooks

### semantic_entropy_unfaithful_reasoning.ipynb

Implements semantic entropy-based unfaithfulness detection following Kuhn et al. (2024). The notebook:
- Samples multiple chain-of-thought responses per question
- Clusters responses by semantic similarity
- Computes entropy over semantic clusters
- Compares entropy across blind vs. grounded conditions
- Detects signals like grounding failure, framing sensitivity, and post-hoc rationalization

Key metrics:
- Semantic Entropy (SE): Entropy over reasoning clusters
- Answer Entropy (AE): Entropy over YES/NO distribution
- SE Reduction: Difference between blind and grounded conditions

### unfaithfulness_detection.ipynb

Implements entity flipping to detect Implicit Post-Hoc Rationalization. The notebook:
- Asks comparison questions in original and reversed framing
- Samples multiple responses per condition
- Calculates agreement rate between framings
- Flags pairs where the model gives identical answers regardless of entity order

High agreement rates indicate the model is not genuinely reasoning about the comparison but rationalizing a predetermined answer.

## Dependencies

```
pyyaml
requests
numpy
scipy
scikit-learn
sentence-transformers
matplotlib
```

## Usage

1. Configure API credentials in the notebook configuration cells
2. Select dataset file(s) from the data/ directory
3. Run notebook cells sequentially
4. Results are automatically saved to results/ directory

## Key References

- Kuhn et al. (2024) - Semantic Entropy Probes Hallucinations in Large Language Models
- Abhayapala et al. (2025) - Semantic Entropy Probes in Production Systems
- ChainScope Dataset - github.com/jettjaniak/chainscope

## Configuration

Edit the configuration cells in each notebook to adjust:
- Model selection (via OpenRouter API)
- Number of samples per question
- Temperature for response diversity
- Similarity thresholds for clustering
- Dataset selection
