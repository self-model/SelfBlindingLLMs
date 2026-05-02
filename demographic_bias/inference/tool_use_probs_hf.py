#!/usr/bin/env python3
"""
Run demographic bias inference measuring tool-use probability via HuggingFace Transformers.

Measures the probability that the model invokes a counterfactual simulation tool
when presented with demographic scenarios.

Usage:
    python tool_use_probs_hf.py \
        --model "Qwen/Qwen2.5-1.5B-Instruct" \
        --data_path ../data/discrim-eval-templated.jsonl \
        --output_dir results/ \
        --inspect
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

# Add repo root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from datasets import Dataset

from demographic_bias.config import DEFAULT_BIAS_DATA, DEFAULT_TOOL_PROMPTS_PATH
from demographic_bias.prompts.formats import PROMPT_DICT, build_single_prompt, create_tool_definition
from src.inference import load_model_and_tokenizer, get_tool_use_start_token_id
from src.thinking import ThinkingConfig, render_with_thinking


# =============================================================================
# Data Loading
# =============================================================================

def load_scenarios(path: Path) -> list[dict]:
    """Load scenarios from JSONL file."""
    scenarios = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                scenarios.append(json.loads(line))
    return scenarios


def load_tool_prompts(path: Path) -> list[dict]:
    """Load tool prompts from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# =============================================================================
# Inference Functions
# =============================================================================

def get_tool_use_prob(example: dict, prompt_format: dict, tool_prompt: dict,
                      tokenizer, model, tool_start_token_id: int,
                      *, model_name: str, thinking_config: ThinkingConfig) -> tuple[float, dict]:
    """
    Get probability of tool-use start token for a given example and prompt format.

    Returns:
        (tool_prob, thinking_meta) — the probability of the tool-use start token,
        and the thinking metadata (trace, truncated, etc.) from render_with_thinking.
    """
    # Build conversation
    row_values = [example[col] for col in prompt_format['prompt_column']]
    conversation = build_single_prompt(prompt_format['conversation'], row_values)

    # Remove assistant prefill if present
    if conversation[-1]['role'] == 'assistant':
        conversation = conversation[:-1]

    # Create tool definition
    tools = create_tool_definition(tool_prompt)

    # Render prompt (handles enable_thinking=False on the off path; full thinking
    # generate-then-measure on the on path).
    prompt_str, thinking_meta = render_with_thinking(
        model_name, model, tokenizer, conversation,
        tools=tools,
        add_generation_prompt=True,
        thinking_config=thinking_config,
    )

    # Tokenize and run model
    inputs = tokenizer(prompt_str, return_tensors='pt').to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token logits

    # Get probability for tool-use start token
    probs = torch.softmax(logits, dim=-1)
    tool_prob = probs[tool_start_token_id].item()

    return tool_prob, thinking_meta


def score_example(example: dict, prompt_formats: dict, tool_prompts: list,
                  tokenizer, model, tool_start_token_id: int,
                  *, model_name: str, thinking_config: ThinkingConfig) -> dict:
    """
    Score a single example across all prompt formats and tool prompts.

    Returns:
        Dict with tool-use probabilities and thinking traces per (prompt_format, tool)
    """
    results = {}

    for prompt_name, prompt_format in prompt_formats.items():
        snake_case = prompt_format['snake_case']

        for tool_prompt in tool_prompts:
            tool_name = tool_prompt['name']
            col_name = f"{snake_case}__{tool_name}__tool_prob"
            trace_col = f"{snake_case}__{tool_name}__thinking_trace"
            truncated_col = f"{snake_case}__{tool_name}__thinking_truncated"

            try:
                prob, thinking_meta = get_tool_use_prob(
                    example, prompt_format, tool_prompt,
                    tokenizer, model, tool_start_token_id,
                    model_name=model_name, thinking_config=thinking_config,
                )
                results[col_name] = prob
                results[trace_col] = thinking_meta['thinking_trace']
                results[truncated_col] = thinking_meta['truncated']
            except Exception as e:
                print(f"Error scoring {snake_case}/{tool_name}: {e}")
                results[col_name] = None
                results[trace_col] = None
                results[truncated_col] = False

    return results


# =============================================================================
# Inspection Mode
# =============================================================================

def run_inspect_mode(data: Dataset, tool_prompts: list, tokenizer, model,
                     tool_start_token_id: int, n_samples: int, seed: int,
                     *, model_name: str, thinking_config: ThinkingConfig):
    """Run inspection mode on a few samples."""
    random.seed(seed)

    idx = random.randint(0, len(data) - 1)
    example = data[idx]

    # Use first prompt format and tool prompt
    prompt_name = list(PROMPT_DICT.keys())[0]
    prompt_format = PROMPT_DICT[prompt_name]
    tool_prompt = tool_prompts[0]

    print("\n" + "#" * 70)
    print(f"# INSPECT MODE: Tool-Use Probability")
    print(f"# Example index: {idx}")
    print("#" * 70)

    # Print metadata
    print("\n--- EXAMPLE METADATA ---")
    for field in ['race', 'gender', 'decision_question_id', 'decision_question_nickname']:
        if field in example:
            print(f"  {field}: {example[field]}")

    # Build conversation
    row_values = [example[col] for col in prompt_format['prompt_column']]
    conversation = build_single_prompt(prompt_format['conversation'], row_values)

    if conversation[-1]['role'] == 'assistant':
        conversation = conversation[:-1]

    tools = create_tool_definition(tool_prompt)

    # Render through render_with_thinking so inspect mode reflects what the
    # full inference path will see (thinking-on or off, depending on config).
    prompt_str, _ = render_with_thinking(
        model_name, model, tokenizer, conversation,
        tools=tools,
        add_generation_prompt=True,
        thinking_config=thinking_config,
    )

    print(f"\n--- PROMPT (truncated) ---")
    print(prompt_str[:1500] + "..." if len(prompt_str) > 1500 else prompt_str)

    # Get tool probability
    prob, _ = get_tool_use_prob(
        example, prompt_format, tool_prompt, tokenizer, model, tool_start_token_id,
        model_name=model_name, thinking_config=thinking_config,
    )

    print(f"\n--- RESULTS ---")
    print(f"  Tool start token ID: {tool_start_token_id}")
    print(f"  Tool start token: {repr(tokenizer.decode([tool_start_token_id]))}")
    print(f"  P(tool use) = {prob:.6f}")

    # Show top tokens for context
    inputs = tokenizer(prompt_str, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]

    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=10)

    print(f"\n  Top 10 tokens:")
    for p, idx in zip(top_probs, top_indices):
        tok = tokenizer.decode([idx.item()])
        marker = " <-- TOOL" if idx.item() == tool_start_token_id else ""
        print(f"    {repr(tok):20s} {p.item():.6f}{marker}")

    print("\n" + "#" * 70)
    print("# INSPECT MODE COMPLETE")
    print("#" * 70)


# =============================================================================
# Full Inference Mode
# =============================================================================

def run_full_inference(data: Dataset, tool_prompts: list, tokenizer, model,
                       tool_start_token_id: int, output_path: Path, model_name: str,
                       thinking_config: ThinkingConfig):
    """Run inference on all conditions."""
    # Determine which prompt formats to use
    prompt_formats_to_use = {}
    for name, fmt in PROMPT_DICT.items():
        # Skip multi-turn formats for tool-use
        if fmt['snake_case'] in ('remove_in_context',):
            print(f"Skipping format: {fmt['snake_case']}")
            continue
        prompt_formats_to_use[name] = fmt

    print("=" * 60)
    print(f"RUNNING FULL INFERENCE")
    print(f"Model:          {model_name}")
    print(f"Scenarios:      {len(data)}")
    print(f"Prompt formats: {len(prompt_formats_to_use)}")
    print(f"Tool prompts:   {len(tool_prompts)}")
    print(f"Thinking:       {thinking_config.mode} (budget={thinking_config.budget}, T={thinking_config.temperature}, N={thinking_config.n_samples})")
    print(f"Output:         {output_path}")
    print("=" * 60)

    # Use bare basename to keep fingerprint under HF datasets' 64-char cap.
    model_nickname = model_name.split('/')[-1]

    def score_fn(example):
        return score_example(example, prompt_formats_to_use, tool_prompts,
                            tokenizer, model, tool_start_token_id,
                            model_name=model_name, thinking_config=thinking_config)

    run_data = data.map(
        score_fn,
        batched=False,
        load_from_cache_file=False,
        new_fingerprint=f"{model_nickname}_tool_use_think{thinking_config.mode}",
        desc="Scoring tool-use probability"
    )

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_data.to_json(str(output_path), orient='records', lines=True, force_ascii=False)
    print(f"\nSaved to {output_path}")

    return output_path


# =============================================================================
# Main
# =============================================================================

def run(model, tokenizer, args):
    """Run inference with a pre-loaded model and tokenizer."""
    # Setup
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load tool prompts
    tool_prompts_path = Path(args.tool_prompts_path) if args.tool_prompts_path else DEFAULT_TOOL_PROMPTS_PATH
    tool_prompts = load_tool_prompts(tool_prompts_path)
    print(f"Loaded {len(tool_prompts)} tool prompts: {[tp['name'] for tp in tool_prompts]}")

    # Load data
    data_path = Path(args.data_path) if args.data_path else DEFAULT_BIAS_DATA
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    print(f"Loading scenarios from {data_path}...")
    scenarios = load_scenarios(data_path)
    print(f"  Loaded {len(scenarios)} scenarios")

    # Limit scenarios if requested
    if args.n_scenarios:
        scenarios = scenarios[:args.n_scenarios]
        print(f"  Limited to {len(scenarios)} scenarios")

    # Convert to Dataset
    data = Dataset.from_list(scenarios)
    print(f"  Dataset columns: {data.column_names}")

    # Get tool-use start token ID
    tool_start_id = get_tool_use_start_token_id(args.model)
    print(f"Tool-use start token ID: {tool_start_id}")
    print(f"Tool-use start token: {repr(tokenizer.decode([tool_start_id]))}")

    # Build thinking config from args (defaults yield mode='off')
    thinking_config = ThinkingConfig(
        mode=getattr(args, 'thinking', 'off'),
        budget=getattr(args, 'thinking_budget', -1),
        temperature=getattr(args, 'thinking_temperature', 0.0),
        n_samples=getattr(args, 'thinking_n_samples', 1),
    )

    # Run appropriate mode
    if args.inspect:
        run_inspect_mode(data, tool_prompts, tokenizer, model, tool_start_id,
                         args.inspect_n, args.seed,
                         model_name=args.model, thinking_config=thinking_config)
        print("Exiting after inspect mode. Run without --inspect for full inference.")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_nickname = args.model.split('/')[-1]
        output_dir = Path(args.output_dir)
        output_path = output_dir / f"{timestamp}_bias_tool_use_{model_nickname}.jsonl"

        run_full_inference(data, tool_prompts, tokenizer, model, tool_start_id,
                           output_path, args.model, thinking_config)


def main():
    parser = argparse.ArgumentParser(description="Run demographic bias tool-use inference via HuggingFace")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name (e.g., Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to scenarios JSONL file")
    parser.add_argument("--output_dir", type=str, default=str(SCRIPT_DIR.parent / "results"),
                        help="Directory for output files")
    parser.add_argument("--tool_prompts_path", type=str, default=str(DEFAULT_TOOL_PROMPTS_PATH),
                        help="Path to tool descriptions YAML file")
    parser.add_argument("--inspect", action="store_true",
                        help="Run inspection mode and exit")
    parser.add_argument("--inspect_n", type=int, default=2,
                        help="Number of samples to inspect")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_scenarios", type=int, default=None,
                        help="Limit number of scenarios (for testing)")
    parser.add_argument("--thinking", choices=['off', 'on'], default='off',
                        help="Thinking mode (only meaningful for models with thinking_family set; default off)")
    parser.add_argument("--thinking-budget", dest='thinking_budget', type=int, default=-1,
                        help="Max thinking tokens; -1 = unlimited (capped at safety_max)")
    parser.add_argument("--thinking-temperature", dest='thinking_temperature', type=float, default=0.0,
                        help="Generation temperature for the thinking trace; 0 = greedy")
    parser.add_argument("--thinking-n-samples", dest='thinking_n_samples', type=int, default=1,
                        help="Number of trace samples per condition (only meaningful with --thinking-temperature > 0)")
    args = parser.parse_args()

    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)

    run(model, tokenizer, args)


if __name__ == "__main__":
    main()
