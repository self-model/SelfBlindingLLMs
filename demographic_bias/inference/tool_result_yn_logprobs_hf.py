#!/usr/bin/env python3
"""
Run demographic bias inference with simulated tool use via HuggingFace Transformers.

Simulates tool calls and responses, then measures yes/no logprobs.
Tests whether the model's decision changes after receiving counterfactual simulation results.

Usage:
    python tool_result_yn_logprobs_hf.py \
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
from demographic_bias.prompts.formats import (
    PROMPT_DICT, build_single_prompt, create_tool_definition,
    create_tool_call, create_tool_response
)
from src.inference import load_model_and_tokenizer, get_yes_no_token_ids
from src.scoring import score_yes_no_from_logits


# =============================================================================
# Prefill for after tool response
# =============================================================================

TOOL_RESULT_PREFILL = 'Based on the counterfactual simulation result, my final answer is "'


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
# Conversation Building
# =============================================================================

def build_conversation_with_tool(example: dict, prompt_format: dict, tool_prompt: dict,
                                  tool_response_text: str) -> list[dict]:
    """Build conversation including tool call and response."""
    # Build base conversation
    row_values = [example[col] for col in prompt_format['prompt_column']]
    conversation = build_single_prompt(prompt_format['conversation'], row_values)

    # Remove assistant prefill if present
    if conversation[-1]['role'] == 'assistant':
        conversation = conversation[:-1]

    # Add tool call (using removed_template as the counterfactual prompt)
    tool_call = create_tool_call(
        tool_prompt['name'],
        {tool_prompt['property_name']: example['removed_template']}
    )
    conversation.append(tool_call)

    # Add tool response
    tool_response = create_tool_response(
        tool_prompt['name'],
        tool_response_text
    )
    conversation.append(tool_response)

    # Add final assistant prefill
    conversation.append({
        'role': 'assistant',
        'content': TOOL_RESULT_PREFILL
    })

    return conversation


# =============================================================================
# Inference Functions
# =============================================================================

def score_with_tool_result(example: dict, prompt_format: dict, tool_prompt: dict,
                           tool_response_text: str, tokenizer, model,
                           yes_token_ids: dict, no_token_ids: dict) -> dict:
    """
    Score yes/no logprobs after simulated tool use.

    Args:
        example: Scenario dict
        prompt_format: Prompt format dict
        tool_prompt: Tool prompt dict
        tool_response_text: Simulated tool response ("Yes." or "No.")
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        yes_token_ids: Dict mapping variants to token IDs
        no_token_ids: Dict mapping variants to token IDs

    Returns:
        dict with yes/no logits and probabilities
    """
    # Build conversation with tool
    conversation = build_conversation_with_tool(
        example, prompt_format, tool_prompt, tool_response_text
    )

    # Create tools for chat template
    tools = create_tool_definition(tool_prompt)

    # Apply chat template
    prompt_str = tokenizer.apply_chat_template(
        conversation,
        tools=tools,
        add_generation_prompt=False,
        continue_final_message=True,
        tokenize=False
    )

    # Tokenize and run model
    inputs = tokenizer(prompt_str, return_tensors='pt').to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token logits

    # Score Yes/No
    result = score_yes_no_from_logits(logits, yes_token_ids, no_token_ids)

    return {
        'yes_logit': result['yes_logit'],
        'no_logit': result['no_logit'],
        'yes_prob': result['yes_prob'],
        'no_prob': result['no_prob'],
    }


def score_example(example: dict, prompt_formats: dict, tool_prompts: list,
                  tokenizer, model, yes_token_ids: dict, no_token_ids: dict) -> dict:
    """
    Score a single example across all prompt formats, tool prompts, and tool responses.

    Returns:
        Dict with nested structure: {prompt_format: {tool_name: {"Yes.": {...}, "No.": {...}}}}
    """
    results = {}

    for prompt_name, prompt_format in prompt_formats.items():
        snake_case = prompt_format['snake_case']
        results[snake_case] = {}

        for tool_prompt in tool_prompts:
            tool_name = tool_prompt['name']
            results[snake_case][tool_name] = {}

            for tool_response_text in ['Yes.', 'No.']:
                try:
                    result = score_with_tool_result(
                        example, prompt_format, tool_prompt, tool_response_text,
                        tokenizer, model, yes_token_ids, no_token_ids
                    )
                    results[snake_case][tool_name][tool_response_text] = result
                except Exception as e:
                    print(f"Error scoring {snake_case}/{tool_name}/{tool_response_text}: {e}")
                    results[snake_case][tool_name][tool_response_text] = {
                        'error': str(e),
                        'yes_prob': 0.0,
                        'no_prob': 0.0,
                    }

    return results


# =============================================================================
# Inspection Mode
# =============================================================================

def run_inspect_mode(data: Dataset, tool_prompts: list, tokenizer, model,
                     yes_token_ids: dict, no_token_ids: dict, n_samples: int, seed: int):
    """Run inspection mode on a few samples."""
    random.seed(seed)

    idx = random.randint(0, len(data) - 1)
    example = data[idx]

    # Use first prompt format and tool prompt
    prompt_name = list(PROMPT_DICT.keys())[0]
    prompt_format = PROMPT_DICT[prompt_name]
    tool_prompt = tool_prompts[0]
    tool_response_text = 'Yes.'

    print("\n" + "#" * 70)
    print(f"# INSPECT MODE: Tool Result Logprobs")
    print(f"# Example index: {idx}")
    print("#" * 70)

    # Print metadata
    print("\n--- EXAMPLE METADATA ---")
    for field in ['race', 'gender', 'decision_question_id', 'decision_question_nickname']:
        if field in example:
            print(f"  {field}: {example[field]}")

    # Build conversation
    conversation = build_conversation_with_tool(
        example, prompt_format, tool_prompt, tool_response_text
    )

    tools = create_tool_definition(tool_prompt)

    prompt_str = tokenizer.apply_chat_template(
        conversation,
        tools=tools,
        add_generation_prompt=False,
        continue_final_message=True,
        tokenize=False
    )

    print(f"\n--- CONVERSATION ({len(conversation)} messages) ---")
    for i, msg in enumerate(conversation):
        role = msg.get('role', 'unknown')
        print(f"\n[{i}] ROLE: {role}")
        content = msg.get('content', '')
        if role == 'assistant' and msg.get('tool_calls'):
            print("    CONTENT: (tool call)")
            for tc in msg['tool_calls']:
                print(f"    TOOL_CALL: {tc['function']['name']}")
        elif content:
            if len(content) > 150:
                content = content[:150] + "..."
            print(f"    CONTENT: {content}")

    print(f"\n--- TOKENIZED PROMPT (truncated) ---")
    print(prompt_str[:1500] + "..." if len(prompt_str) > 1500 else prompt_str)

    # Score and show results
    result = score_with_tool_result(
        example, prompt_format, tool_prompt, tool_response_text,
        tokenizer, model, yes_token_ids, no_token_ids
    )

    print(f"\n--- RESULTS (tool response = {tool_response_text}) ---")
    print(f"  P(Yes) = {result['yes_prob']:.4f}")
    print(f"  P(No)  = {result['no_prob']:.4f}")

    # Also test with No. response
    result_no = score_with_tool_result(
        example, prompt_format, tool_prompt, 'No.',
        tokenizer, model, yes_token_ids, no_token_ids
    )

    print(f"\n--- RESULTS (tool response = No.) ---")
    print(f"  P(Yes) = {result_no['yes_prob']:.4f}")
    print(f"  P(No)  = {result_no['no_prob']:.4f}")

    print("\n" + "#" * 70)
    print("# INSPECT MODE COMPLETE")
    print("#" * 70)


# =============================================================================
# Full Inference Mode
# =============================================================================

def run_full_inference(data: Dataset, tool_prompts: list, tokenizer, model,
                       yes_token_ids: dict, no_token_ids: dict,
                       output_path: Path, model_name: str):
    """Run inference on all conditions."""
    print("=" * 60)
    print(f"RUNNING FULL INFERENCE")
    print(f"Model:          {model_name}")
    print(f"Scenarios:      {len(data)}")
    print(f"Prompt formats: {len(PROMPT_DICT)}")
    print(f"Tool prompts:   {len(tool_prompts)}")
    print(f"Output:         {output_path}")
    print("=" * 60)

    model_nickname = model_name.replace('/', '_')

    def score_fn(example):
        return score_example(example, PROMPT_DICT, tool_prompts,
                            tokenizer, model, yes_token_ids, no_token_ids)

    run_data = data.map(
        score_fn,
        batched=False,
        load_from_cache_file=False,
        new_fingerprint=f"{model_nickname}_tool_result_yn",
        desc="Scoring tool-result yes/no"
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

    # Get token IDs
    yes_token_ids, no_token_ids = get_yes_no_token_ids(tokenizer)
    print(f"Yes tokens: {yes_token_ids}")
    print(f"No tokens: {no_token_ids}")

    # Run appropriate mode
    if args.inspect:
        run_inspect_mode(data, tool_prompts, tokenizer, model,
                         yes_token_ids, no_token_ids, args.inspect_n, args.seed)
        print("Exiting after inspect mode. Run without --inspect for full inference.")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_nickname = args.model.split('/')[-1]
        output_dir = Path(args.output_dir)
        output_path = output_dir / f"{timestamp}_bias_tool_result_yn_{model_nickname}.jsonl"

        run_full_inference(data, tool_prompts, tokenizer, model,
                           yes_token_ids, no_token_ids, output_path, args.model)


def main():
    parser = argparse.ArgumentParser(description="Run demographic bias tool-result inference via HuggingFace")
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
    args = parser.parse_args()

    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)

    run(model, tokenizer, args)


if __name__ == "__main__":
    main()
