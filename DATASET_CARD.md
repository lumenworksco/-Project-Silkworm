---
license: cc-by-4.0
task_categories:
  - text-generation
  - question-answering
language:
  - en
tags:
  - dpo
  - preference
  - alignment
  - instruction-following
  - rlhf
size_categories:
  - n<1K
---

# Project Silkworm

A continuously growing DPO (Direct Preference Optimization) preference dataset spanning 10 knowledge domains. Generated automatically using the HuggingFace Serverless Inference API and uploaded incrementally — the dataset grows larger every day.

## Dataset Description

Each example contains a prompt paired with two responses of contrasting quality:
- **chosen**: generated with a thorough, expert-level system prompt — detailed, structured, with examples and reasoning
- **rejected**: generated with a brief, superficial system prompt — terse, unhelpful, no explanation

This natural quality contrast makes the dataset well-suited for DPO fine-tuning without requiring human annotation.

## Columns

| Column | Type | Description |
|---|---|---|
| `prompt` | string | The instruction or question posed to the model |
| `chosen` | string | The preferred, high-quality response |
| `rejected` | string | The dispreferred, low-quality response |
| `topic` | string | The subject area (e.g. "quantum entanglement", "Stoicism") |
| `timestamp` | string | UTC timestamp when the pair was generated |

## Domains

Prompts are sampled uniformly across 10 categories:

- Science, Technology, Mathematics
- History, Philosophy, Economics
- Psychology, Health, Culture, Practical

## Generation

- **Generator**: `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace Serverless Inference
- **Chosen system prompt**: expert assistant, thorough and structured
- **Rejected system prompt**: casual assistant, brief and superficial
- **Quality filter**: pairs where either response is below 80 characters or the responses are identical are discarded

## Intended Use

Designed for DPO fine-tuning of language models. Compatible with [TRL](https://github.com/huggingface/trl)'s `DPOTrainer` out of the box.

```python
from datasets import load_dataset

dataset = load_dataset("iPwnds/Project-Silkworm", split="train")
```

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
