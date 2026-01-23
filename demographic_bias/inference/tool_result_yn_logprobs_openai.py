#!/usr/bin/env python3
"""
Run demographic bias inference with simulated tool use via OpenAI API.

Simulates tool calls and responses, then measures yes/no logprobs.
Tests whether the model's decision changes after receiving counterfactual simulation results.
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

import torch
import yaml
from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_client = OpenAI()

from demographic_bias.config import DEFAULT_BIAS_DATA, DEFAULT_TOOL_PROMPTS_PATH, YES_NO_SYSTEM_PROMPT
from demographic_bias.prompts.formats import PROMPT_DICT, build_single_prompt, create_tool_definition
from src.scoring import score_yes_no_from_top_logprobs


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
# Tool Call Helpers
# =============================================================================

def create_tool_call_openai(tool_name: str, arguments: dict, call_id: str = "call_001") -> dict:
    """Create an assistant message containing a tool call in OpenAI API format."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments)
            }
        }]
    }


def create_tool_response_openai(tool_name: str, response: str, call_id: str = "call_001") -> dict:
    """Create a tool response message in OpenAI API format."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": response
    }


def build_conversation_with_tool(example: dict, prompt_format: dict, tool_prompt: dict, tool_response_text: str) -> list:
    """Build conversation including tool call and response."""
    # Build base conversation
    row_values = [example[col] for col in prompt_format['prompt_column']]
    conversation = build_single_prompt(prompt_format['conversation'], row_values)

    # Remove assistant prefill if present
    if conversation[-1]['role'] == "assistant":
        conversation = conversation[:-1]

    # Add system prompt
    messages = [{"role": "system", "content": YES_NO_SYSTEM_PROMPT}]
    messages.extend(conversation)

    # Add tool call (using removed_template as the counterfactual prompt)
    tool_call = create_tool_call_openai(
        tool_prompt['name'],
        {tool_prompt['property_name']: example['removed_template']},
        call_id="call_001"
    )
    messages.append(tool_call)

    # Add tool response
    tool_response = create_tool_response_openai(
        tool_prompt['name'],
        tool_response_text,
        call_id="call_001"
    )
    messages.append(tool_response)

    return messages


# =============================================================================
# Batch Request Building
# =============================================================================

def prepare_batch_request(example: dict, prompt_format: dict, tool_prompts: list, model_name: str) -> list[dict]:
    """Prepare batch API requests for a single example."""
    requests = []
    prompt_format_name = prompt_format['snake_case']

    for tool_prompt in tool_prompts:
        tool_name = tool_prompt['name']
        tools = create_tool_definition(tool_prompt)

        for tool_response_text in ['Yes.', 'No.']:
            messages = build_conversation_with_tool(
                example, prompt_format, tool_prompt, tool_response_text
            )

            # Sanitize for custom_id
            response_key = tool_response_text.rstrip('.')

            requests.append({
                'custom_id': f"{example['scenario_id']}__{prompt_format_name}__{tool_name}__{response_key}",
                'method': 'POST',
                'url': '/v1/chat/completions',
                'body': {
                    'model': model_name,
                    'messages': messages,
                    'tools': tools,
                    'max_completion_tokens': 2,
                    'temperature': 0,
                    'logprobs': True,
                    'top_logprobs': 20,
                    'seed': 42,
                }
            })

    return requests


# =============================================================================
# Result Parsing
# =============================================================================

def parse_logprobs_from_response(response_body: dict) -> dict:
    """Parse yes/no logprobs from a batch API response body."""
    choice = response_body['choices'][0]

    # Check if we got content with logprobs
    if not choice.get('logprobs') or not choice['logprobs'].get('content'):
        return {
            'yes_logit': float('-inf'),
            'no_logit': float('-inf'),
            'yes_relative_prob': 0.0,
            'no_relative_prob': 0.0,
            'error': 'no_content_logprobs'
        }

    top_logprobs = choice['logprobs']['content'][0]['top_logprobs']

    try:
        result = score_yes_no_from_top_logprobs(top_logprobs)
        return {
            'yes_logit': result['yes_logit'],
            'no_logit': result['no_logit'],
            'yes_relative_prob': result['yes_prob'],
            'no_relative_prob': result['no_prob'],
        }
    except ValueError:
        return {
            'yes_logit': float('-inf'),
            'no_logit': float('-inf'),
            'yes_relative_prob': 0.0,
            'no_relative_prob': 0.0,
            'error': 'both_missing'
        }


def merge_batch_results(data, results_by_id: dict) -> dict:
    """Merge batch results back into dataset with nested structure."""
    # Initialize columns
    columns = {}
    for prompt_format in PROMPT_DICT.values():
        columns[prompt_format['snake_case']] = [None] * len(data)

    # Group results by (scenario_id, prompt_format)
    grouped = {}

    for custom_id, response_body in results_by_id.items():
        # Parse: "{scenario_id}__{prompt_format}__{tool_name}__{response_key}"
        parts = custom_id.split("__")
        scenario_id = int(parts[0])
        prompt_format = parts[1]
        tool_name = parts[2]
        response_key = parts[3]  # "Yes" or "No"

        key = (scenario_id, prompt_format)
        if key not in grouped:
            grouped[key] = {}
        if tool_name not in grouped[key]:
            grouped[key][tool_name] = {}

        # Parse logprobs
        result = parse_logprobs_from_response(response_body)
        grouped[key][tool_name][f"{response_key}."] = result

    # Build final column values
    for (scenario_id, prompt_format), tool_results in grouped.items():
        columns[prompt_format][scenario_id] = tool_results

    return columns


# =============================================================================
# Inspection Mode
# =============================================================================

def run_inspect_mode(data, tool_prompts: list, model_name: str, n_samples: int, seed: int):
    """Run inspection mode on a few samples."""
    random.seed(seed)

    # Pick one random example
    idx = random.randint(0, len(data) - 1)
    example = data[idx]

    # Use first prompt format and tool prompt
    prompt_name = list(PROMPT_DICT.keys())[0]
    prompt_format = PROMPT_DICT[prompt_name]
    tool_prompt = tool_prompts[0]
    tool_response_text = 'Yes.'

    print("\n" + "#" * 70)
    print(f"# INSPECT MODE: Tool Use Logprobs")
    print(f"# Model: {model_name}")
    print(f"# Example index: {idx}")
    print("#" * 70)

    # Print metadata
    print("\n--- EXAMPLE METADATA ---")
    for field in ['race', 'gender', 'decision_question_id', 'decision_question_nickname']:
        if field in example:
            print(f"  {field}: {example[field]}")

    # Build and print conversation
    messages = build_conversation_with_tool(example, prompt_format, tool_prompt, tool_response_text)

    print(f"\n--- CONVERSATION ({len(messages)} messages) ---")
    for i, msg in enumerate(messages):
        role = msg.get('role', 'unknown')
        print(f"\n[{i}] ROLE: {role}")

        if role == 'assistant' and msg.get('tool_calls'):
            print("    CONTENT: null (tool call)")
            for tc in msg['tool_calls']:
                print(f"    TOOL_CALL ID: {tc['id']}")
                print(f"    TOOL_CALL FUNCTION: {tc['function']['name']}")
        elif role == 'tool':
            print(f"    TOOL_CALL_ID: {msg.get('tool_call_id', '')}")
            print(f"    CONTENT: {msg.get('content', '')}")
        else:
            content = msg.get('content', '')
            if content and len(content) > 200:
                content = content[:200] + "..."
            print(f"    CONTENT: {content}")

    # Query model
    print("\n--- QUERYING MODEL ---")
    tools = create_tool_definition(tool_prompt)

    resp = openai_client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        max_completion_tokens=2,
        temperature=0,
        logprobs=True,
        top_logprobs=20,
        seed=42,
    )

    choice = resp.choices[0]
    print(f"  Finish reason: {choice.finish_reason}")
    print(f"  Generated text: {repr(choice.message.content)}")

    if choice.logprobs and choice.logprobs.content:
        print(f"\n  Top logprobs:")
        for cand in choice.logprobs.content[0].top_logprobs[:10]:
            print(f"    {repr(cand.token):12s} -> {cand.logprob:.4f}")

        # Parse result
        top_logprobs = [{'token': c.token, 'logprob': c.logprob} for c in choice.logprobs.content[0].top_logprobs]
        result = score_yes_no_from_top_logprobs(top_logprobs)
        print(f"\n  P(Yes) = {result['yes_prob']:.4f}")
        print(f"  P(No)  = {result['no_prob']:.4f}")
    else:
        print("  No logprobs in response")

    print("\n" + "#" * 70)
    print("# INSPECT MODE COMPLETE")
    print("#" * 70)


# =============================================================================
# Batch Inference Mode
# =============================================================================

def run_batch_inference(data, tool_prompts: list, args):
    """Run batch inference using OpenAI Batch API."""
    model_nickname = args.openai_model.replace('/', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"{timestamp}_bias_tool_result_yn_{model_nickname}.jsonl"

    print("=" * 60)
    print(f"BATCH MODE")
    print(f"Model:          {args.openai_model}")
    print(f"Output:         {output_filename}")
    print(f"Scenarios:      {len(data)}")
    print(f"Tool prompts:   {[tp['name'] for tp in tool_prompts]}")
    print("=" * 60)

    # Build batch requests
    batch = []
    for prompt_format in PROMPT_DICT.values():
        print(f"Preparing format: {prompt_format['snake_case']}")
        for example in data:
            requests = prepare_batch_request(example, prompt_format, tool_prompts, args.openai_model)
            batch.extend(requests)

    # Save batch
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

    # Merge results
    new_columns = merge_batch_results(data, results_by_id)

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
    parser = argparse.ArgumentParser(description="Run demographic bias tool-result inference via OpenAI API")
    parser.add_argument("--openai_model", type=str, default="gpt-4.1",
                        help="OpenAI chat model name")
    parser.add_argument("--output_dir", type=str, default=str(SCRIPT_DIR.parent / "results"),
                        help="Directory for output files")
    parser.add_argument("--tool_prompts_path", type=str, default=str(DEFAULT_TOOL_PROMPTS_PATH),
                        help="Path to tool descriptions YAML file")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to scenarios JSONL file")
    parser.add_argument("--inspect", action="store_true",
                        help="Run inspection mode and exit")
    parser.add_argument("--inspect_n", type=int, default=2,
                        help="Number of samples to inspect")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_scenarios", type=int, default=None,
                        help="Limit number of scenarios (for testing)")
    parser.add_argument("--batch_poll_interval", type=int, default=30,
                        help="Seconds between batch status checks")
    args = parser.parse_args()

    # Setup
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load tool prompts
    tool_prompts_path = Path(args.tool_prompts_path)
    tool_prompts = load_tool_prompts(tool_prompts_path)
    print(f"Loaded {len(tool_prompts)} tool prompts: {[tp['name'] for tp in tool_prompts]}")

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

    # Run appropriate mode
    if args.inspect:
        run_inspect_mode(data, tool_prompts, args.openai_model, args.inspect_n, args.seed)
        return

    output_file = run_batch_inference(data, tool_prompts, args)
    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()
