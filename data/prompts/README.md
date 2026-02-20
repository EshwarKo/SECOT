# Prompts

YAML files containing comparative YES/NO prompts for SECOT analysis.

## Format

```yaml
prompts:
  - label: "unique_identifier"
    prompt: "Is the X of A greater than the X of B? Answer YES or NO."
    property: "X"
    entity_a: "A"
    entity_b: "B"
    expected_gt: "a"  # or "b"
```

## Adding prompts

You can add prompts from the [ChainScope](https://github.com/jettjaniak/chainscope) dataset
or create your own comparative questions. The key requirement is that flipping
entity_a and entity_b should flip the correct answer.
