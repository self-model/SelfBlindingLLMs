#!/usr/bin/env python3
"""
Run sycophancy tool-use inference using OpenAI batch API.

Measures whether models use counterfactual simulation tools when presented
with sycophancy scenarios. Key question: Does the model recognize it might
be biased toward the user and proactively use tools to check its reasoning?
"""

import argparse
import json
import sys
from pathlib import Path
import time
from dataclasses import asdict

# Add repo root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sycophancy.config import DEFAULT_SYCOPHANCY_DATA, DEFAULT_TOOL_PROMPTS_PATH

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
openai_client = OpenAI()


# =============================================================================
# Tool Definition
# =============================================================================

def load_tool_prompts(tool_prompts_path: Path) -> list:
    """Load tool prompts from YAML file."""
    import yaml
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
# Batch Preparation
# =============================================================================

def prepare_batch_request(example: dict, tool_prompt: dict, model_name: str) -> dict:
    """
    Prepare a single batch request for a sycophancy condition with a tool.
    """
    tool_block = create_tool_definition(tool_prompt)

    messages = [
        {'role': 'user', 'content': example['prompt']}
    ]

    request_body = {
        'model': model_name,
        'messages': messages,
        'tools': tool_block,
        'seed': 42,
        'max_completion_tokens': 256,
    }

    # Create unique ID: scenario_id + condition details + tool name
    condition_id = f"{example['scenario_id']}__{example['my_version']}_{example['my_first']}"
    instruction_short = "ctrl" if not example['instruction'] else example['instruction'][:10].replace(" ", "_").rstrip("_")
    custom_id = f"{condition_id}__{instruction_short}__{tool_prompt['name']}"

    return {
        'custom_id': custom_id,
        'method': 'POST',
        'url': '/v1/chat/completions',
        'body': request_body
    }


# =============================================================================
# Single batch run
# =============================================================================

def run_single_batch(data, tool_prompts, args, run_index=None):
    """
    Run a single batch inference and save to a file.

    If run_index is provided, it will be appended to the filename.
    Returns the output filename.
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_nickname = args.openai_model.replace("/", "_")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_suffix = f"_run{run_index:03d}" if run_index is not None else ""
    output_filename = Path(args.output_dir) / f"{timestamp}_sycophancy_tool_calls_{model_nickname}{run_suffix}.jsonl"

    print("=" * 60)
    print(f"Model:        {args.openai_model}")
    print(f"Output:       {output_filename}")
    if run_index is not None:
        print(f"Run:          {run_index + 1} of {args.n_times}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Build batch
    # -------------------------------------------------------------------------
    batch = []
    for example in data:
        for tool_prompt in tool_prompts:
            request = prepare_batch_request(example, tool_prompt, args.openai_model)
            batch.append(request)

    print(f"Generated {len(batch)} batch requests")

    # Save batch to JSONL
    batch_path = output_filename.with_name(f"{output_filename.stem}_batch.jsonl")
    with open(batch_path, 'w') as f:
        for item in batch:
            f.write(json.dumps(item) + "\n")
    print(f"Batch saved to {batch_path}")

    # -------------------------------------------------------------------------
    # Upload and submit batch
    # -------------------------------------------------------------------------
    with open(batch_path, 'rb') as f:
        batch_file = openai_client.files.create(file=f, purpose="batch")
    print(f"Uploaded file: {batch_file.id}")

    batch_job = openai_client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"Submitted batch: {batch_job.id}")
    print(f"\nPoll status with:")
    print(f"  python -c \"from openai import OpenAI; b = OpenAI().batches.retrieve('{batch_job.id}'); print(f'{{b.status}} {{b.request_counts}}')\"\n")

    # Save batch metadata
    batch_meta = {
        "batch_id": batch_job.id,
        "input_file_id": batch_file.id,
        "num_requests": len(batch),
        "model": args.openai_model,
        "timestamp": timestamp,
    }
    if run_index is not None:
        batch_meta['run_index'] = run_index
        batch_meta['n_times'] = args.n_times
    batch_meta_path = batch_path.with_suffix('.batch_meta.json')
    with open(batch_meta_path, 'w') as f:
        json.dump(batch_meta, f, indent=2)
    print(f"Batch metadata saved to {batch_meta_path}")

    # -------------------------------------------------------------------------
    # Poll until complete
    # -------------------------------------------------------------------------
    print(f"\nPolling for completion (every 30s)...")
    start_time = time.time()

    while True:
        batch_status = openai_client.batches.retrieve(batch_job.id)
        counts = batch_status.request_counts
        elapsed = time.time() - start_time
        elapsed_str = f"{int(elapsed // 60):02d}m {int(elapsed % 60):02d}s"

        print(f"[{elapsed_str}] {batch_status.status} | {counts.completed}/{counts.total} done, {counts.failed} failed")

        if batch_status.status in ["completed", "failed", "expired", "cancelled"]:
            break

        time.sleep(30)

    print(f"\nBatch finished with status: {batch_status.status}")

    if batch_status.status != "completed":
        if batch_status.error_file_id:
            errors = openai_client.files.content(batch_status.error_file_id)
            print(f"Errors:\n{errors.text[:3000]}")
        raise SystemExit(f"Batch failed: {batch_status.status}")

    # -------------------------------------------------------------------------
    # Download raw results
    # -------------------------------------------------------------------------
    print("Downloading results...")
    results_content = openai_client.files.content(batch_status.output_file_id)

    raw_results_path = batch_path.with_suffix('.results.jsonl')
    with open(raw_results_path, 'w') as f:
        f.write(results_content.text)
    print(f"Raw results saved to {raw_results_path}")

    # -------------------------------------------------------------------------
    # Parse results
    # -------------------------------------------------------------------------
    print("Parsing results...")

    results_by_id = {}
    for line in results_content.text.strip().split("\n"):
        item = json.loads(line)
        results_by_id[item["custom_id"]] = item["response"]["body"]

    print(f"Parsed {len(results_by_id)} results")

    # Extract tool call info
    parsed_results = []
    for custom_id, response_body in results_by_id.items():
        # Parse custom_id: "{scenario_id}__{my_version}_{my_first}__{instruction}__{tool_name}"
        parts = custom_id.split("__")
        scenario_id = int(parts[0])
        version_first = parts[1]
        instruction_short = parts[2]
        tool_name = parts[3]

        # Check if model made a tool call
        message = response_body['choices'][0]['message']
        made_tool_call = message.get('tool_calls') is not None
        tool_call_content = None
        if made_tool_call:
            tool_call_content = message['tool_calls'][0]['function']['arguments']

        parsed_results.append({
            'custom_id': custom_id,
            'scenario_id': scenario_id,
            'tool_name': tool_name,
            'made_tool_call': made_tool_call,
            'tool_call_arguments': tool_call_content,
            'response_content': message.get('content'),
            'finish_reason': response_body['choices'][0]['finish_reason'],
        })

    # Save parsed results
    parsed_path = output_filename.with_name(f"{output_filename.stem}_parsed.jsonl")
    with open(parsed_path, 'w') as f:
        for item in parsed_results:
            f.write(json.dumps(item) + "\n")
    print(f"Parsed results saved to {parsed_path}")

    # -------------------------------------------------------------------------
    # Merge with original data
    # -------------------------------------------------------------------------
    print("Merging with original data...")

    # Make a copy of data to avoid mutating the original
    data_copy = [dict(d) for d in data]

    # Build lookup by (scenario_id, tool_name)
    results_lookup = {}
    for r in parsed_results:
        key = (r['scenario_id'], r['tool_name'])
        results_lookup[key] = r

    # Add columns to data
    for example in data_copy:
        for tool_prompt in tool_prompts:
            tool_name = tool_prompt['name']
            key = (example['scenario_id'], tool_name)
            result = results_lookup.get(key, {})
            example[f"made_tool_call__{tool_name}"] = result.get('made_tool_call', None)
            example[f"tool_call_args__{tool_name}"] = result.get('tool_call_arguments', None)
            example[f"response_content__{tool_name}"] = result.get('response_content', None)

    # Save final merged dataset
    with open(output_filename, 'w') as f:
        for item in data_copy:
            f.write(json.dumps(item) + "\n")
    print(f"Final dataset saved to {output_filename}")

    # -------------------------------------------------------------------------
    # Quick summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for tool_prompt in tool_prompts:
        tool_name = tool_prompt['name']
        col = f"made_tool_call__{tool_name}"
        tool_calls = sum(1 for d in data_copy if d.get(col) is True)
        total = sum(1 for d in data_copy if d.get(col) is not None)
        rate = tool_calls / total if total > 0 else 0
        print(f"{tool_name}: {tool_calls}/{total} = {rate:.1%} made tool calls")

    print(f"\nDone! Output: {output_filename}")

    return output_filename


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run sycophancy tool-use inference")
    parser.add_argument("--openai_model", type=str, default="gpt-4.1",
                        help="OpenAI model name")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to scenarios JSONL file (default: sycophancy/data/sycophancy-two-sides-eval.jsonl)")
    parser.add_argument("--output_dir", type=str, default=str(SCRIPT_DIR.parent / "results"),
                        help="Directory for output files")
    parser.add_argument("--tool_prompts_path", type=str, default=str(DEFAULT_TOOL_PROMPTS_PATH),
                        help="Path to tool descriptions YAML file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_times", type=int, default=1,
                        help="Number of times to run the batch, creating separate output files (default: 1)")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load sycophancy scenarios
    # -------------------------------------------------------------------------
    from sycophancy.prompts.first_person import (
        load_scenarios,
        generate_full_experiment,
        experiment_summary,
    )

    data_path = Path(args.data_path) if args.data_path else DEFAULT_SYCOPHANCY_DATA
    scenarios = load_scenarios(str(data_path))
    print(f"Loaded {len(scenarios)} scenarios")

    experiment = generate_full_experiment(scenarios)
    conditions = experiment['conditions']

    summary = experiment_summary(experiment)
    print(f"Total conditions: {summary['total_conditions']}")

    # Convert to dicts
    def condition_to_dict(condition, idx):
        d = asdict(condition)
        d['scenario_id'] = idx
        d['sycophantic_token'] = condition.sycophantic_token
        d['you_validates_version_a'] = condition.you_validates_version_a
        d['version_a_token'] = condition.version_a_token
        d['first_position_token'] = condition.first_position_token
        return d

    data = [condition_to_dict(c, i) for i, c in enumerate(conditions)]
    print(f"Prepared {len(data)} condition dicts")

    # -------------------------------------------------------------------------
    # Load tool prompts
    # -------------------------------------------------------------------------
    tool_prompts_path = Path(args.tool_prompts_path)
    tool_prompts = load_tool_prompts(tool_prompts_path)
    print(f"Loaded {len(tool_prompts)} tool prompts from {tool_prompts_path}")
    for tp in tool_prompts:
        print(f"  - {tp['name']}")

    # -------------------------------------------------------------------------
    # Run batch(es)
    # -------------------------------------------------------------------------
    output_files = []

    if args.n_times == 1:
        output_file = run_single_batch(data, tool_prompts, args, run_index=None)
        output_files.append(output_file)
    else:
        for i in range(args.n_times):
            print(f"\n{'#' * 60}")
            print(f"# STARTING RUN {i + 1} OF {args.n_times}")
            print(f"{'#' * 60}\n")

            output_file = run_single_batch(data, tool_prompts, args, run_index=i)
            output_files.append(output_file)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    if args.n_times > 1:
        print(f"\n{'=' * 60}")
        print(f"ALL {args.n_times} RUNS COMPLETE")
        print(f"{'=' * 60}")
        print("Output files:")
        for f in output_files:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
