#!/usr/bin/env python3
"""
Run demographic bias inference measuring tool-use probability via OpenAI API.

Measures the probability that the model invokes a counterfactual simulation tool
when presented with demographic scenarios.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Add repo root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_client = OpenAI()

from demographic_bias.config import DEFAULT_BIAS_DATA, DEFAULT_TOOL_PROMPTS_PATH
from demographic_bias.prompts.formats import PROMPT_DICT, build_single_prompt, create_tool_definition


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
# Batch Request Building
# =============================================================================

def prepare_batch_openai(example: dict, prompt_format: dict, tool_prompts: list, model_name: str) -> list[dict]:
    """
    Prepare batch API requests for a single example.

    Args:
        example: Scenario dict
        prompt_format: Prompt format dict from PROMPT_DICT
        tool_prompts: List of tool prompt dicts
        model_name: OpenAI model name

    Returns:
        List of batch request dicts
    """
    prompt_format_name = prompt_format['snake_case']

    # Build conversation
    row_values = [example[col] for col in prompt_format['prompt_column']]
    conversation = build_single_prompt(prompt_format['conversation'], row_values)

    # Remove assistant prefill if present
    if conversation[-1]['role'] == "assistant":
        conversation = conversation[:-1]

    batch = []

    for tool_prompt in tool_prompts:
        tool_block = create_tool_definition(tool_prompt)

        request_body = {
            'model': model_name,
            'messages': conversation,
            'tools': tool_block,
            'seed': 42,
            'max_completion_tokens': 64,
        }

        batch.append({
            'custom_id': f"{example['scenario_id']}_scenario__{prompt_format_name}_prompt__{tool_prompt['name']}_tool",
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': request_body
        })

    return batch


# =============================================================================
# Result Processing
# =============================================================================

def extract_tool_call_info(response_body: dict) -> dict:
    """
    Extract tool call information from API response.

    Returns:
        Dict with has_tool_call, finish_reason, tool_name, tool_arguments
    """
    choice = response_body.get('choices', [{}])[0]
    message = choice.get('message', {})
    finish_reason = choice.get('finish_reason', '')

    tool_calls = message.get('tool_calls', [])
    has_tool_call = len(tool_calls) > 0

    tool_name = None
    tool_arguments = None

    if has_tool_call:
        tc = tool_calls[0]
        fn = tc.get('function', {})
        tool_name = fn.get('name', '')
        tool_arguments = fn.get('arguments', '')

    return {
        'has_tool_call': has_tool_call,
        'finish_reason': finish_reason,
        'tool_name': tool_name,
        'tool_arguments': tool_arguments,
    }


# =============================================================================
# Batch Inference Mode
# =============================================================================

def run_batch_inference(data, tool_prompts: list, args):
    """Run inference using OpenAI Batch API."""
    model_nickname = args.openai_model.replace('/', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"{timestamp}_bias_tool_use_{model_nickname}.jsonl"

    # Determine which prompt formats to use
    prompt_formats_to_use = {}
    for name, fmt in PROMPT_DICT.items():
        # Skip multi-turn formats for tool-use
        if fmt['snake_case'] in ('remove_in_context',):
            print(f"Skipping format: {fmt['snake_case']}")
            continue
        if args.prompt and fmt['snake_case'] not in args.prompt:
            print(f"Skipping format: {fmt['snake_case']}")
            continue
        prompt_formats_to_use[name] = fmt

    print("=" * 60)
    print(f"BATCH MODE")
    print(f"Model:            {args.openai_model}")
    print(f"Output:           {output_filename}")
    print(f"Scenarios:        {len(data)}")
    print(f"Prompt formats:   {len(prompt_formats_to_use)}")
    print(f"Tool prompts:     {len(tool_prompts)}")
    print("=" * 60)

    # Build batch requests
    batch = []
    for prompt_format in prompt_formats_to_use.values():
        print(f"Preparing format: {prompt_format['snake_case']}")
        for example in data:
            minibatch = prepare_batch_openai(example, prompt_format, tool_prompts, args.openai_model)
            batch.extend(minibatch)

    # Save batch to file
    batch_path = output_filename.with_name(f"{output_filename.stem}_batch.jsonl")
    with open(batch_path, 'w') as f:
        for item in batch:
            f.write(json.dumps(item) + "\n")
    print(f"\nGenerated {len(batch)} batch requests -> {batch_path}")

    # Upload and submit
    with open(batch_path, 'rb') as f:
        batch_file = openai_client.files.create(file=f, purpose="batch")
    print(f"Uploaded file: {batch_file.id}")

    batch_job = openai_client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"Submitted batch: {batch_job.id}")

    # Poll until complete
    print(f"\nPolling for completion (every {args.batch_poll_interval}s)...")
    start_time = time.time()

    while True:
        batch_status = openai_client.batches.retrieve(batch_job.id)
        counts = batch_status.request_counts
        elapsed = time.time() - start_time
        elapsed_str = f"{int(elapsed // 60):02d}m {int(elapsed % 60):02d}s"

        print(f"[{elapsed_str}] {batch_status.status} | {counts.completed}/{counts.total} done, {counts.failed} failed")

        if batch_status.status in ["completed", "failed", "expired", "cancelled"]:
            break

        time.sleep(args.batch_poll_interval)

    print(f"\nBatch finished with status: {batch_status.status}")

    if batch_status.status != "completed":
        if batch_status.error_file_id:
            errors = openai_client.files.content(batch_status.error_file_id)
            print(f"Errors:\n{errors.text[:3000]}")
        raise SystemExit(f"Batch failed: {batch_status.status}")

    # Download results
    print("Downloading results...")
    results_content = openai_client.files.content(batch_status.output_file_id)

    # Save raw response
    raw_results_path = batch_path.with_suffix('.results.jsonl')
    with open(raw_results_path, 'w') as f:
        f.write(results_content.text)
    print(f"Raw results saved to {raw_results_path}")

    # Parse results
    results_by_id = {}
    for line in results_content.text.strip().split("\n"):
        item = json.loads(line)
        results_by_id[item["custom_id"]] = item["response"]["body"]

    print(f"Parsed {len(results_by_id)} results")

    # Build new columns
    new_columns = {}
    for custom_id, response_body in results_by_id.items():
        # Parse: "{scenario_id}_scenario__{prompt_format}_prompt__{tool_name}_tool"
        parts = custom_id.split("__")
        scenario_id = int(parts[0].replace("_scenario", ""))
        prompt_format = parts[1].replace("_prompt", "")
        tool_name = parts[2].replace("_tool", "")

        col_name = f"{prompt_format}__{tool_name}__completion_json"

        if col_name not in new_columns:
            new_columns[col_name] = [None] * len(data)

        new_columns[col_name][scenario_id] = json.dumps(response_body)

    # Add columns to dataset
    run_data = data
    for col_name, values in new_columns.items():
        run_data = run_data.add_column(col_name, values)

    print(f"Added {len(new_columns)} columns to dataset")

    # Save final dataset
    run_data.to_json(str(output_filename), orient='records', lines=True, force_ascii=False)
    print(f"Final dataset saved to {output_filename}")

    return output_filename


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run demographic bias tool-use inference via OpenAI API")
    parser.add_argument("--openai_model", type=str, default="gpt-4.1",
                        help="OpenAI chat model name")
    parser.add_argument("--output_dir", type=str, default=str(SCRIPT_DIR.parent / "results"),
                        help="Directory for output files")
    parser.add_argument("--tool_prompts_path", type=str, default=str(DEFAULT_TOOL_PROMPTS_PATH),
                        help="Path to tool descriptions YAML file")
    parser.add_argument("--prompt", type=str, action="append", default=None,
                        help="Which user prompt(s) to use (can be repeated)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to scenarios JSONL file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_scenarios", type=int, default=None,
                        help="Limit number of scenarios (for testing)")
    parser.add_argument("--batch_poll_interval", type=int, default=30,
                        help="Seconds between batch status checks")
    args = parser.parse_args()

    # Setup
    random.seed(args.seed)

    # Load tool prompts
    tool_prompts_path = Path(args.tool_prompts_path)
    tool_prompts = load_tool_prompts(tool_prompts_path)
    print(f"Loaded {len(tool_prompts)} tool prompts from {tool_prompts_path}")

    # Load data
    data_path = Path(args.data_path) if args.data_path else DEFAULT_BIAS_DATA
    print(f"Loading scenarios from {data_path}...")
    scenarios = load_scenarios(data_path)
    print(f"  Loaded {len(scenarios)} scenarios")

    # Limit scenarios if requested
    if args.n_scenarios:
        scenarios = scenarios[:args.n_scenarios]
        print(f"  Limited to {len(scenarios)} scenarios")

    # Add scenario_id for batch matching
    for i, s in enumerate(scenarios):
        s['scenario_id'] = i

    # Convert to Dataset
    data = Dataset.from_list(scenarios)
    print(f"  Dataset columns: {data.column_names}")
    print(f"  Decision questions: {set(data['decision_question_id'])}")

    # Run batch inference
    output_file = run_batch_inference(data, tool_prompts, args)
    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()
