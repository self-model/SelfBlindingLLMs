#!/usr/bin/env python3
"""
Run demographic bias inference (yes/no scoring) via OpenAI API.

Measures bias by comparing P(Yes) across demographic groups for each prompt strategy.

Design: 520 scenarios × 6 prompt strategies = 3,120 conditions
"""

import argparse
import json
import os
import random
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Add repo root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_client = OpenAI()

from demographic_bias.config import DEFAULT_BIAS_DATA, YES_NO_SYSTEM_PROMPT
from demographic_bias.prompts.formats import PROMPT_DICT, build_single_prompt
from src.scoring import score_yes_no_from_top_logprobs


# =============================================================================
# Token Scoring
# =============================================================================

def compute_yes_no_scores(top_logprobs: list, prompt_format_name: str) -> dict:
    """
    Compute Yes/No scores from top logprobs.

    Args:
        top_logprobs: List of dicts with 'token' and 'logprob' keys
        prompt_format_name: Snake case name for output keys

    Returns:
        Dict with logits, probs, and token counts for Yes/No
    """
    result = score_yes_no_from_top_logprobs(top_logprobs)

    return {
        f"{prompt_format_name}_prompt_yes_logits": result['yes_logit'],
        f"{prompt_format_name}_prompt_no_logits": result['no_logit'],
        f"{prompt_format_name}_prompt_yes_relative_probs": result['yes_prob'],
        f"{prompt_format_name}_prompt_no_relative_probs": result['no_prob'],
        f"{prompt_format_name}_prompt_yes_tokens_in_top_20": result['yes_tokens_in_top_20'],
        f"{prompt_format_name}_prompt_no_tokens_in_top_20": result['no_tokens_in_top_20'],
    }


# =============================================================================
# Batch API Support
# =============================================================================

def build_batch_request(custom_id: str, messages: list, model_name: str) -> dict:
    """Build a single batch request object for the OpenAI Batch API."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "messages": messages,
            "max_completion_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20,
            "seed": 42,
        }
    }


def create_batch_file_for_prompt_format(data, prompt_format: dict, model_name: str) -> str:
    """
    Create a JSONL batch file for all examples with a given prompt format.
    Returns the path to the temporary file.
    """
    prompt_format_name = prompt_format['snake_case']

    fd, batch_file_path = tempfile.mkstemp(suffix='.jsonl', prefix=f'batch_{prompt_format_name}_')

    with os.fdopen(fd, 'w') as f:
        for idx in range(len(data)):
            example = data[idx]

            # Build conversation
            row_values = [example[col] for col in prompt_format['prompt_column']]
            conversation = build_single_prompt(prompt_format['conversation'], row_values)

            # Remove assistant prefill if present (OpenAI Chat API doesn't accept prefills)
            if conversation[-1]['role'] == "assistant":
                conversation = conversation[:-1]

            # Add system prompt
            messages = [{"role": "system", "content": YES_NO_SYSTEM_PROMPT}]
            messages.extend(conversation)

            # Create batch request
            custom_id = f"{prompt_format_name}__idx_{idx}"
            request = build_batch_request(custom_id, messages, model_name)

            f.write(json.dumps(request) + '\n')

    return batch_file_path


def parse_batch_response(response: dict, prompt_format_name: str) -> dict:
    """Parse a single batch response and compute Yes/No scores."""
    if response.get('status_code') != 200:
        raise ValueError(f"Request failed with status {response.get('status_code')}: {response.get('error')}")

    body = response['body']
    choice = body['choices'][0]
    top_logprobs = choice['logprobs']['content'][0]['top_logprobs']

    return compute_yes_no_scores(top_logprobs, prompt_format_name)


# =============================================================================
# Synchronous Scoring
# =============================================================================

def score_example_openai(example, prompt_format, model_name: str) -> dict:
    """Score a single row using ChatGPT."""
    prompt_format_name = prompt_format['snake_case']

    # Build conversation
    row_values = [example[col] for col in prompt_format['prompt_column']]
    conversation = build_single_prompt(prompt_format['conversation'], row_values)

    # Remove assistant prefill if present
    if conversation[-1]['role'] == "assistant":
        conversation = conversation[:-1]

    # Add system prompt
    messages = [{"role": "system", "content": YES_NO_SYSTEM_PROMPT}]
    messages.extend(conversation)

    # Query API
    def query():
        return openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=42,
        )

    try:
        resp = query()
    except Exception as e:
        print(f"Error during OpenAI API call: {e}. Retrying...")
        time.sleep(1)
        resp = query()

    choice = resp.choices[0]
    top = choice.logprobs.content[0].top_logprobs

    # Convert to dicts
    top_logprobs = [{'token': c.token, 'logprob': c.logprob} for c in top]

    return compute_yes_no_scores(top_logprobs, prompt_format_name)


# =============================================================================
# Inspection Mode
# =============================================================================

def run_inspect_mode(data, model_name: str, n_samples: int, seed: int):
    """Run inspection mode on a few samples."""
    import math
    random.seed(seed)

    n_samples = min(n_samples, len(data))
    random_indices = random.sample(range(len(data)), n_samples)

    print("\n" + "#" * 70)
    print(f"# INSPECT MODE: Examining {n_samples} random examples")
    print(f"# Model: {model_name}")
    print(f"# Indices: {random_indices}")
    print("#" * 70)

    for idx in random_indices:
        example = data[idx]

        print("\n" + "=" * 70)
        print(f"EXAMPLE INDEX: {idx}")
        print("=" * 70)

        # Print example metadata
        print("\nEXAMPLE METADATA:")
        for field in ['race', 'gender', 'decision_question_id', 'decision_question_nickname']:
            if field in example:
                print(f"  {field}: {example[field]}")

        # Inspect first prompt format only
        prompt_name = list(PROMPT_DICT.keys())[0]
        prompt_format = PROMPT_DICT[prompt_name]

        print(f"\n{'─' * 70}")
        print(f"PROMPT FORMAT: {prompt_name}")
        print(f"{'─' * 70}")

        # Build and print conversation
        row_values = [example[col] for col in prompt_format['prompt_column']]
        conversation = build_single_prompt(prompt_format['conversation'], row_values)

        print("\nCONVERSATION:")
        for msg in conversation:
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')
            if len(content) > 300:
                content = content[:300] + "..."
            print(f"\n[{role}]")
            print(content)

        # Score and show results
        try:
            result = score_example_openai(example, prompt_format, model_name)
            snake_case = prompt_format['snake_case']
            print(f"\nRESULTS:")
            print(f"  P(Yes) = {result[f'{snake_case}_prompt_yes_relative_probs']:.4f}")
            print(f"  P(No)  = {result[f'{snake_case}_prompt_no_relative_probs']:.4f}")
        except Exception as e:
            print(f"\n  ERROR: {e}")

    print("\n" + "#" * 70)
    print("# INSPECT MODE COMPLETE")
    print("#" * 70 + "\n")


# =============================================================================
# Batch Inference Mode
# =============================================================================

def run_batch_inference(data, args):
    """Run inference using OpenAI Batch API."""
    model_nickname = args.openai_model.replace('/', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"{timestamp}_bias_yn_{model_nickname}.jsonl"

    print("=" * 60)
    print(f"BATCH MODE")
    print(f"Model:           {args.openai_model}")
    print(f"Output:          {output_filename}")
    print(f"Examples:        {len(data)}")
    print(f"Prompt formats:  {len(PROMPT_DICT)}")
    print(f"Total requests:  {len(data) * len(PROMPT_DICT)}")
    print("=" * 60)

    # Convert to list for manipulation
    data_list = [data[i] for i in range(len(data))]

    # Process each prompt format as a separate batch
    for prompt_name, prompt_format in PROMPT_DICT.items():
        snake_case = prompt_format['snake_case']
        print(f"\n{'─' * 60}")
        print(f"Processing prompt format: {prompt_name}")
        print(f"{'─' * 60}")

        # Create batch file
        batch_file_path = create_batch_file_for_prompt_format(data, prompt_format, args.openai_model)
        print(f"  Batch file: {batch_file_path}")

        # Submit batch
        with open(batch_file_path, 'rb') as f:
            file_obj = openai_client.files.create(file=f, purpose='batch')
        print(f"  Uploaded: {file_obj.id}")

        batch = openai_client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"bias_{model_nickname}_{snake_case}"}
        )
        print(f"  Batch: {batch.id}")

        # Wait for completion
        while True:
            batch = openai_client.batches.retrieve(batch.id)
            status = batch.status
            completed = batch.request_counts.completed
            total = batch.request_counts.total
            failed = batch.request_counts.failed

            print(f"  {status} ({completed}/{total} complete, {failed} failed)")

            if status in ('completed', 'failed', 'expired', 'cancelled'):
                break

            time.sleep(args.batch_poll_interval)

        if batch.status != 'completed':
            print(f"  Batch failed: {batch.status}")
            continue

        # Download results
        content = openai_client.files.content(batch.output_file_id)
        results = []
        for line in content.text.strip().split('\n'):
            if line:
                obj = json.loads(line)
                results.append((obj['custom_id'], obj['response']))

        # Map results back to data
        results_by_idx = {}
        for custom_id, response in results:
            parts = custom_id.split('__idx_')
            if len(parts) == 2:
                idx = int(parts[1])
                results_by_idx[idx] = response

        # Add results to data_list
        success_count = 0
        for idx in range(len(data_list)):
            if idx in results_by_idx:
                try:
                    parsed = parse_batch_response(results_by_idx[idx], snake_case)
                    data_list[idx].update(parsed)
                    success_count += 1
                except Exception as e:
                    print(f"  Warning: Failed to parse response for idx {idx}: {e}")

        print(f"  Processed {success_count} results")

        # Checkpoint
        checkpoint_data = Dataset.from_list(data_list)
        checkpoint_data.to_json(str(output_filename), orient='records', lines=True, force_ascii=False)

        # Cleanup
        try:
            os.unlink(batch_file_path)
        except:
            pass

    print(f"\n{'=' * 60}")
    print(f"BATCH MODE COMPLETE")
    print(f"Output: {output_filename}")
    print(f"{'=' * 60}")

    return output_filename


# =============================================================================
# Sync Inference Mode
# =============================================================================

def run_sync_inference(data, args):
    """Run inference using synchronous API calls."""
    model_nickname = args.openai_model.replace('/', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"{timestamp}_bias_yn_{model_nickname}.jsonl"

    print("=" * 60)
    print(f"SYNC MODE")
    print(f"Model:      {args.openai_model}")
    print(f"Output:     {output_filename}")
    print(f"Examples:   {len(data)}")
    print("=" * 60)

    run_data = data

    for prompt_name, prompt_format in PROMPT_DICT.items():
        snake_case = prompt_format['snake_case']
        print(f"Scoring: {prompt_name}")

        def score_fn(example):
            return score_example_openai(example, prompt_format, args.openai_model)

        run_data = run_data.map(
            score_fn,
            batched=False,
            load_from_cache_file=False,
            new_fingerprint=f"{model_nickname}_{snake_case}",
            desc=f"Scoring '{prompt_name}'"
        )

        # Checkpoint
        run_data.to_json(str(output_filename), orient='records', lines=True, force_ascii=False)

    print(f"\nSaved to {output_filename}")
    return output_filename


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


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run demographic bias inference via OpenAI API")
    parser.add_argument("--openai_model", type=str, default="gpt-4.1",
                        help="OpenAI chat model name (e.g., gpt-4o, gpt-4.1-mini)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to scenarios JSONL file")
    parser.add_argument("--output_dir", type=str, default=str(SCRIPT_DIR.parent / "results"),
                        help="Directory for output files")
    parser.add_argument("--inspect", action="store_true",
                        help="Run inspection mode and exit")
    parser.add_argument("--inspect_n", type=int, default=3,
                        help="Number of samples to inspect")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_scenarios", type=int, default=None,
                        help="Limit number of scenarios (for testing)")
    parser.add_argument("--batch", action="store_true",
                        help="Use OpenAI Batch API (50%% cheaper)")
    parser.add_argument("--batch_poll_interval", type=int, default=30,
                        help="Seconds between batch status checks")
    args = parser.parse_args()

    # Setup
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data
    data_path = Path(args.data_path) if args.data_path else DEFAULT_BIAS_DATA
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

    # Run appropriate mode
    if args.inspect:
        run_inspect_mode(data, args.openai_model, args.inspect_n, args.seed)
        print("Exiting after inspect mode.")
        return

    if args.batch:
        output_file = run_batch_inference(data, args)
    else:
        output_file = run_sync_inference(data, args)

    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()
