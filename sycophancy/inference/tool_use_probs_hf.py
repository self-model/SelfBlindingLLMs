#!/usr/bin/env python3
"""
Run tool-use probability inference via HuggingFace Transformers.

Measures whether models use counterfactual simulation tools when presented
with sycophancy scenarios. Key question: Does the model recognize it might
be biased toward the user and proactively use tools to check its reasoning?

Metric: P(tool_use_start_token) - probability the model begins a tool call

Usage:
    python model_inference_hf_tool_use_probs.py \
        --model "Qwen/Qwen2.5-7B-Instruct" \
        --data_path ../data/sycophancy-two-sides-eval.jsonl \
        --output_dir outputs/ \
        --inspect  # Preview prompts before running
"""

import argparse
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import yaml
from datasets import Dataset

# Add repo root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sycophancy.config import DEFAULT_SYCOPHANCY_DATA, DEFAULT_TOOL_PROMPTS_PATH
from src.inference import load_model_and_tokenizer, get_tool_use_start_token_id
from sycophancy.prompts.first_person import (
    load_scenarios,
    generate_full_experiment,
    ForcedChoiceCondition,
)

# =============================================================================
# Tool Definition
# =============================================================================

def load_tool_prompts(tool_prompts_path: Path) -> list:
    """Load tool prompts from YAML file."""
    with open(tool_prompts_path, 'r') as f:
        tool_prompts = yaml.safe_load(f)
    return tool_prompts


def create_tool_definition(tool_prompt: dict) -> list:
    """Create OpenAI-style tool definition from a tool prompt dict."""
    return [{
        "type": "function",
        "function": {
            "name": tool_prompt['name'],
            "description": tool_prompt['description'],
            "parameters": {
                "type": "object",
                "properties": {
                    tool_prompt['property_name']: {
                        "type": "string",
                        "description": tool_prompt['property_description']
                    }
                },
                "required": [tool_prompt['property_name']]
            }
        }
    }]


# =============================================================================
# Condition to Dict Conversion
# =============================================================================

def condition_to_dict(condition: ForcedChoiceCondition) -> dict:
    """Convert condition dataclass to dict, computing property values."""
    d = asdict(condition)
    # Add computed properties
    d['sycophantic_token'] = condition.sycophantic_token
    d['you_validates_version_a'] = condition.you_validates_version_a
    d['version_a_token'] = condition.version_a_token
    d['first_position_token'] = condition.first_position_token
    return d


# =============================================================================
# Inference Functions
# =============================================================================

def score_tool_use_condition(
    example: dict,
    tokenizer,
    model,
    tool_prompts: list,
    tool_use_start_token_id: int,
) -> dict:
    """
    Score tool use probability for a sycophancy condition.

    For each tool prompt, measures P(tool_use_start_token) when the model
    is presented with the sycophancy scenario and has access to the tool.

    Args:
        example: Dict with 'prompt' field
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        tool_prompts: List of tool prompt dicts
        tool_use_start_token_id: Token ID for tool use start

    Returns:
        Dict with tool_prob__{tool_name} for each tool
    """
    results = {}

    for tool_prompt in tool_prompts:
        tool_block = create_tool_definition(tool_prompt)

        # Build conversation (no system message, matching GPT-4.1 approach)
        conversation = [
            {'role': 'user', 'content': example['prompt']}
        ]

        # Apply chat template with tools
        try:
            prompt_str = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tools=tool_block,
                tokenize=False
            )
        except Exception as e:
            # Model may not support tools in chat template
            print(f"Warning: Could not apply chat template with tools: {e}")
            results[f"tool_prob__{tool_prompt['name']}"] = None
            continue

        # Tokenize and get logits
        inputs = tokenizer(prompt_str, return_tensors='pt').to(model.device)

        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits

        # Get probability of tool use start token
        probs = torch.softmax(logits, dim=-1)
        tool_prob = probs[tool_use_start_token_id].item()

        results[f"tool_prob__{tool_prompt['name']}"] = tool_prob

    return results


# =============================================================================
# Inspection Mode
# =============================================================================

def run_inspect_mode(
    data: Dataset,
    tokenizer,
    model,
    tool_prompts: list,
    tool_use_start_token_id: int,
    n_samples: int,
    seed: int,
):
    """Run inspection mode on a few samples."""
    random.seed(seed)

    n_samples = min(n_samples, len(data))
    random_indices = random.sample(range(len(data)), n_samples)

    print("\n" + "#" * 70)
    print(f"# INSPECT MODE: Examining {n_samples} random examples")
    print(f"# Tool use start token ID: {tool_use_start_token_id}")
    print(f"# Tool use start token: {repr(tokenizer.decode([tool_use_start_token_id]))}")
    print(f"# Indices: {random_indices}")
    print("#" * 70)

    for idx in random_indices:
        example = data[idx]

        print("\n" + "=" * 70)
        print(f"EXAMPLE INDEX: {idx}")
        print("=" * 70)

        # Print example metadata
        print("\nEXAMPLE METADATA:")
        print(f"  scenario_id: {example['scenario_id']}")
        print(f"  scenario_nickname: {example['scenario_nickname']}")
        print(f"  my_version: {example['my_version']}")
        print(f"  my_first: {example['my_first']}")
        print(f"  instruction: {example['instruction'][:50]}..." if example['instruction'] else "  instruction: (none)")

        # Show prompt with first tool
        tool_block = create_tool_definition(tool_prompts[0])
        conversation = [{'role': 'user', 'content': example['prompt']}]

        try:
            prompt_str = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tools=tool_block,
                tokenize=False
            )

            print("\n" + "-" * 70)
            print(f"PROMPT WITH TOOL '{tool_prompts[0]['name']}' (truncated):")
            print("-" * 70)
            print(prompt_str[:2000] + "..." if len(prompt_str) > 2000 else prompt_str)

        except Exception as e:
            print(f"\nWarning: Could not format prompt with tools: {e}")

        # Score and print results
        results = score_tool_use_condition(
            example, tokenizer, model, tool_prompts, tool_use_start_token_id
        )

        print("\n" + "-" * 70)
        print("TOOL USE PROBABILITIES:")
        print("-" * 70)
        for tool_prompt in tool_prompts:
            prob = results.get(f"tool_prob__{tool_prompt['name']}")
            if prob is not None:
                print(f"  P(use {tool_prompt['name']}) = {prob:.6f} ({prob*100:.4f}%)")
            else:
                print(f"  P(use {tool_prompt['name']}) = N/A")

    print("\n" + "#" * 70)
    print("# INSPECT MODE COMPLETE")
    print("#" * 70 + "\n")


# =============================================================================
# Full Inference Mode
# =============================================================================

def run_full_inference(
    data: Dataset,
    tokenizer,
    model,
    tool_prompts: list,
    tool_use_start_token_id: int,
    output_path: Path,
    model_name: str,
):
    """Run inference on all conditions."""
    print("=" * 60)
    print(f"RUNNING FULL INFERENCE")
    print(f"Model:       {model_name}")
    print(f"Conditions:  {len(data)}")
    print(f"Tools:       {[tp['name'] for tp in tool_prompts]}")
    print(f"Output:      {output_path}")
    print("=" * 60)

    def score_fn(example):
        return score_tool_use_condition(
            example, tokenizer, model, tool_prompts, tool_use_start_token_id
        )

    # Create unique fingerprint for this run
    model_nickname = model_name.replace('/', '_')

    data = data.map(
        score_fn,
        batched=False,
        load_from_cache_file=False,
        new_fingerprint=f"{model_nickname}_sycophancy_tool_use",
        desc="Scoring tool use probability"
    )

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_json(str(output_path), orient='records', lines=True, force_ascii=False)
    print(f"\nSaved to {output_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for tool_prompt in tool_prompts:
        col = f"tool_prob__{tool_prompt['name']}"
        probs = [ex[col] for ex in data if ex[col] is not None]
        if probs:
            mean_prob = sum(probs) / len(probs)
            print(f"{tool_prompt['name']}: Mean P(tool) = {mean_prob:.6f}")
        else:
            print(f"{tool_prompt['name']}: No valid probabilities")

    return output_path


# =============================================================================
# Main
# =============================================================================

def run(model, tokenizer, args):
    """Run inference with a pre-loaded model and tokenizer."""
    # Setup
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Resolve paths
    data_path = Path(args.data_path) if args.data_path else DEFAULT_SYCOPHANCY_DATA
    tool_prompts_path = Path(args.tool_prompts_path) if args.tool_prompts_path else DEFAULT_TOOL_PROMPTS_PATH

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    if not tool_prompts_path.exists():
        print(f"Error: Tool prompts file not found: {tool_prompts_path}")
        sys.exit(1)

    # Load tool prompts
    print(f"Loading tool prompts from {tool_prompts_path}...")
    tool_prompts = load_tool_prompts(tool_prompts_path)
    print(f"  Loaded {len(tool_prompts)} tools: {[tp['name'] for tp in tool_prompts]}")

    # Load scenarios and generate conditions
    print(f"\nLoading scenarios from {data_path}...")
    scenarios = load_scenarios(str(data_path))
    print(f"  Loaded {len(scenarios)} scenarios")

    print("Generating conditions...")
    experiment = generate_full_experiment(scenarios)
    conditions = experiment['conditions']
    print(f"  Generated {len(conditions)} conditions")

    # Convert to Dataset with computed properties
    data = Dataset.from_list([condition_to_dict(c) for c in conditions])
    print(f"  Created dataset with columns: {data.column_names}")

    # Get tool use start token ID
    tool_use_start_token_id = get_tool_use_start_token_id(args.model)
    print(f"Tool use start token ID: {tool_use_start_token_id}")
    print(f"Tool use start token: {repr(tokenizer.decode([tool_use_start_token_id]))}")

    # Run appropriate mode
    if args.inspect:
        run_inspect_mode(
            data, tokenizer, model, tool_prompts, tool_use_start_token_id,
            args.inspect_n, args.seed
        )
        print("Exiting after inspect mode. Run without --inspect for full inference.")
    else:
        # Setup output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_nickname = args.model.split('/')[-1]
        output_dir = Path(args.output_dir)
        output_path = output_dir / f"{timestamp}_sycophancy_tool_use_{model_nickname}.jsonl"

        run_full_inference(
            data, tokenizer, model, tool_prompts, tool_use_start_token_id,
            output_path, args.model
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run tool-use probability inference via HuggingFace Transformers"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name (e.g., Qwen/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Path to scenarios JSONL file (default: sycophancy/data/sycophancy-two-sides-eval.jsonl)"
    )
    parser.add_argument(
        "--tool_prompts_path", type=str, default=None,
        help="Path to tool prompts YAML file"
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(SCRIPT_DIR.parent / "results"),
        help="Directory for output files"
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Run inspection mode: show prompts and scores for a few samples"
    )
    parser.add_argument(
        "--inspect_n", type=int, default=3,
        help="Number of samples to inspect (default: 3)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)

    run(model, tokenizer, args)


if __name__ == "__main__":
    main()
