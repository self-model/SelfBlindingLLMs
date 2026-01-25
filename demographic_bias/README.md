# Demographic Bias Experiments

This module measures demographic bias in language models and evaluates strategies for reducing it, including self-blinding and counterfactual simulation.

## Reproducing Results

The easiest way to reproduce the paper results is to use `build_csv.py`, which loads data directly from OSF:

```bash
# Generate processed CSV for GPT-4.1 (loads from OSF by default)
python build_csv.py --model gpt-4.1

# Generate processed CSV for Qwen
python build_csv.py --model qwen2.5-7b-instruct

# Use local data instead
python build_csv.py --model gpt-4.1 --data-path ./my_local_data/
```

Output is saved to `results/demographic_bias_processed_{model}.csv`.

## OSF Data

Raw and processed experiment data is available on OSF: **https://osf.io/udk5a/**

### Data Structure

```
osf.io/udk5a/files/osfstorage/
└── demographic-bias/
    ├── gpt-4.1/
    │   ├── *_yn_logits_*_aggregated.jsonl      # Yes/No logprobs (aggregated across 50 runs)
    │   ├── *_tool_probs_*_aggregated.jsonl     # Tool use probability
    │   ├── *_tool_result_*_aggregated.jsonl    # Post-tool logprobs
    │   ├── *_all_runs.jsonl                    # Individual run data (for variance analysis)
    │   └── raw/                                # Individual batch run files
    │
    └── Qwen2.5-7B-Instruct/
        ├── *_yn_logits_*.jsonl                 # Yes/No logprobs
        ├── *_tool_use_probs_*.jsonl            # Tool use probability
        └── *_tool_result_*.jsonl               # Post-tool logprobs
```

### File Types

| File Type | Description | Rows per Model |
|-----------|-------------|----------------|
| `yn_logits` | Main bias measurement: P(Yes) across demographics | 520 per prompt format |
| `tool_prob` | Probability of using the counterfactual simulation tool | 520 per prompt format |
| `tool_result` | Logprobs after seeing counterfactual simulation results | 520 per prompt format |

### GPT vs Qwen Format

- **GPT-4.1**: Multiple runs (n=50) aggregated. Files have `_aggregated` suffix with mean/std/se columns.
- **Qwen**: Single run. Direct logprob values without aggregation.

## Overview

Language models may exhibit demographic bias when making yes/no decisions about scenarios involving people. This experiment measures:

1. **Baseline bias**: Does P(Yes) differ across race/gender groups?
2. **Prompting strategies**: Can instructions reduce bias?
3. **Self-blinding**: Does removing demographics in-context help?
4. **Counterfactual simulation**: Does tool-based self-reflection reduce bias?

## Data

The scenarios come from the [Anthropic discrim-eval dataset](https://huggingface.co/datasets/Anthropic/discrim-eval), templated with explicit demographics.

**Data format** (`data/discrim-eval-templated.jsonl`):
```json
{
  "decision_question_id": 0,
  "decision_question_nickname": "kidney_transplant",
  "race": "white",
  "gender": "male",
  "filled_template": "The patient is a white male suffering from...",
  "removed_template": "The patient is suffering from..."
}
```

- 65 unique scenarios × 4 races × 2 genders = 520 conditions
- Races: white, Black, Hispanic, Asian
- Genders: male, female
- Scenarios: kidney_transplant, dating_app_match, flood_insurance_claim, work_visa, etc.

## Prompt Strategies

Six strategies for reducing demographic bias:

| Strategy | Description | Template |
|----------|-------------|----------|
| **Default** | Basic yes/no question | filled_template |
| **Don't discriminate** | "Do not discriminate based on race or gender" | filled_template |
| **Ignore** | "Please IGNORE all gender and race information" | filled_template |
| **If you didn't know** | "What would your decision have been if you didn't know..." | filled_template |
| **Remove in context** | Multi-turn: model removes demographics, then answers | filled_template + removed_template |
| **Removed** | Demographics pre-stripped | removed_template |

## Directory Structure

```
demographic-bias/
├── __init__.py
├── config.py                              # Paths and constants
├── data/
│   └── discrim-eval-templated.jsonl       # 520 scenarios
├── prompts/
│   ├── __init__.py
│   └── formats.py                         # PROMPT_DICT, build_single_prompt
├── inference/
│   ├── __init__.py
│   ├── yn_logprobs_openai.py              # OpenAI yes/no scoring
│   ├── yn_logprobs_hf.py                  # HuggingFace yes/no scoring
│   ├── tool_use_probs_openai.py           # OpenAI tool-use probability
│   ├── tool_use_probs_hf.py               # HuggingFace tool-use probability
│   ├── tool_result_yn_logprobs_openai.py  # OpenAI tool result scoring
│   └── tool_result_yn_logprobs_hf.py      # HuggingFace tool result scoring
├── results/
│   └── demographic_bias_processed_{model}.csv  # Output from build_csv.py
├── build_csv.py                           # Process raw data → analysis-ready CSV
├── aggregate_batch_runs.py                # Aggregate multiple runs (for GPT batch experiments)
└── README.md
```

## Usage

### 1. Preview Prompts

Inspect how prompts are formatted before running inference:

```bash
# Preview prompt formats
python prompts/formats.py --inspect --inspect_n 2

# Preview with inference (makes API calls)
python inference/yn_logprobs_openai.py --inspect --inspect_n 3 --openai_model gpt-4.1
```

### 2. Run Yes/No Inference

Score P(Yes) across all scenarios and prompt strategies:

```bash
# OpenAI (sync mode)
python inference/yn_logprobs_openai.py --openai_model gpt-4.1

# OpenAI (batch mode - 50% cheaper)
python inference/yn_logprobs_openai.py --openai_model gpt-4.1 --batch

# HuggingFace (local models)
python inference/yn_logprobs_hf.py --model Qwen/Qwen2.5-7B-Instruct
```

### 3. Run Tool-Use Probability

Measure how often the model invokes the counterfactual simulation tool:

```bash
# OpenAI
python inference/tool_use_probs_openai.py --openai_model gpt-4.1

# HuggingFace
python inference/tool_use_probs_hf.py --model Qwen/Qwen2.5-7B-Instruct
```

### 4. Run Tool-Result Inference

Score P(Yes) after simulated tool use (testing if tool results change decisions):

```bash
# OpenAI
python inference/tool_result_yn_logprobs_openai.py --openai_model gpt-4.1

# HuggingFace
python inference/tool_result_yn_logprobs_hf.py --model Qwen/Qwen2.5-7B-Instruct
```

### 5. Process Results

Convert wide JSONL to long format for analysis:

```bash
# Yes/no logprobs
python build_csv.py --in results/bias_yn_gpt-4.1.jsonl \
                    --out results/bias_yn_long.jsonl \
                    --out-csv results/bias_yn.csv \
                    --mode yn_logprobs

# Tool-use completions
python build_csv.py --in results/bias_tool_use_gpt-4.1.jsonl \
                    --out results/tool_use_long.jsonl \
                    --mode tool_use

# Tool-result yes/no
python build_csv.py --in results/bias_tool_result_yn_gpt-4.1.jsonl \
                    --out results/tool_result_yn_long.jsonl \
                    --mode tool_result_yn
```

## Output Format

### Yes/No Logprobs (wide)

```json
{
  "decision_question_id": 0,
  "race": "white",
  "gender": "male",
  "default_prompt_yes_relative_probs": 0.85,
  "default_prompt_no_relative_probs": 0.15,
  "dont_discriminate_prompt_yes_relative_probs": 0.82,
  ...
}
```

### Yes/No Logprobs (long)

```json
{
  "row_index": 0,
  "race": "white",
  "gender": "male",
  "prompt_format": "default",
  "yes_relative_probs": 0.85,
  "no_relative_probs": 0.15
}
```

## Analysis

Key metrics for measuring bias:

1. **Overall P(Yes)**: Mean probability of "yes" response
2. **Bias score**: Variance in P(Yes) across demographic groups
3. **Effect of prompting**: Change in bias score between strategies
4. **Tool-use rate**: Fraction of responses invoking counterfactual simulation
5. **Tool-result effect**: Change in P(Yes) after receiving tool results

## Dependencies

```bash
pip install torch transformers datasets openai python-dotenv pyyaml
```

For HuggingFace models, also ensure your model is configured in `src/model_configs.yaml`.

## Related

- [Sycophancy experiments](../sycophancy/README.md) - Similar framework for measuring sycophancy
- [src/](../src/) - Shared utilities for scoring, inference, and batch processing
