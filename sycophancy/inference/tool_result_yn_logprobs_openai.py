#!/usr/bin/env python3
"""
Run sycophancy inference with simulated tool use, using OpenAI Chat Completions API.

This script measures You/Them logprobs AFTER a simulated tool call and response.
The tool simulates asking: "Would you side with Person D or Person E?" (third-person version)
and returns one of the letters as its response.

Flow:
1. Present sycophancy scenario (first-person: "My side" vs "Their side")
2. Simulate tool call with third-person version ("Person D" vs "Person E")
3. Simulate tool response: one of the letters (e.g., "D" or "E")
4. Measure model's You/Them logprobs
"""

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Add repo root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sycophancy.config import DEFAULT_SYCOPHANCY_DATA

import torch
import torch.nn.functional as F
import yaml

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_client = OpenAI()

# Import system prompt constant
from sycophancy.prompts.first_person import SYSTEM_PROMPT


# =============================================================================
# Tool call helpers (adapted for OpenAI API format)
# =============================================================================

def create_tool_call_openai(tool_name: str, arguments: dict, call_id: str = "call_001") -> dict:
    """
    Create an assistant message containing a tool call in OpenAI API format.

    OpenAI requires an 'id' field in tool_calls that matches the tool response.
    """
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
    """
    Create a tool response message in OpenAI API format.

    OpenAI requires 'tool_call_id' to match the assistant's tool call id.
    """
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": response
    }


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
# Removed prompt generation
# =============================================================================

def generate_removed_prompt_and_letters(condition_dict: dict, scenario) -> tuple[str, tuple[str, str]]:
    """
    Generate third-person 'removed' version of a forced-choice condition.

    Preserves the same presentation order but uses neutral letter labels
    instead of first-person framing.

    Returns:
        (removed_prompt, letter_pair) where letter_pair is (version_a_label, version_b_label)
    """
    from sycophancy.prompts.third_person import build_third_person_prompt, deterministic_letter_pair

    # Determine version_a_first to match presentation order
    my_version = condition_dict['my_version']
    my_first = condition_dict['my_first']
    version_a_first = my_first if my_version == "A" else not my_first

    # Get deterministic letter pair for this scenario
    letter_pair = deterministic_letter_pair(scenario.id)
    version_a_label = letter_pair[0]
    version_b_label = letter_pair[1]

    prompt = build_third_person_prompt(
        stories=scenario.stories,
        version_a_label=version_a_label,
        version_b_label=version_b_label,
        version_a_first=version_a_first,
    )

    return prompt, letter_pair


# =============================================================================
# Scoring logic
# =============================================================================

def score_example_openai(example: dict, tool_prompts: list, model_name: str) -> dict:
    """
    Score a single row using ChatGPT with simulated tool calls.

    For each tool prompt and each tool response (letter), we:
    1. Build the conversation with the tool call and response
    2. Query the model for logprobs on the next token
    3. Extract You/Them probabilities
    """
    results = {}
    letter_pair = example['letter_pair']  # e.g., ("D", "E")

    for tool_prompt in tool_prompts:
        tool_name = tool_prompt['name']
        results[tool_name] = {}

        # Tool responses are the letters from the third-person version
        for tool_response_letter in letter_pair:
            # Build the conversation with tool call and response
            messages = build_conversation_with_tool(
                example, tool_prompt, tool_response_letter
            )

            # Query for logprobs
            logprob_result = query_logprobs_you_them(messages, tool_prompt, model_name)

            results[tool_name][tool_response_letter] = logprob_result

    return results


def build_conversation_with_tool(example: dict, tool_prompt: dict, tool_response_letter: str) -> list:
    """
    Build the conversation including tool call and response.
    """
    # System prompt constraining to You/Them response (parallel to Yes/No in base experiments)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Add the sycophancy prompt
    messages.append({"role": "user", "content": example['prompt']})

    # Add tool call (assistant message with tool_calls)
    # The tool receives the removed_prompt (third-person version)
    tool_call = create_tool_call_openai(
        tool_prompt['name'],
        {tool_prompt['property_name']: example['removed_prompt']},
        call_id="call_001"
    )
    messages.append(tool_call)

    # Add tool response - the letter indicating which party the model would side with
    tool_response = create_tool_response_openai(
        tool_prompt['name'],
        tool_response_letter,
        call_id="call_001"
    )
    messages.append(tool_response)

    return messages


def query_logprobs_you_them(messages: list, tool_prompt: dict, model_name: str, return_raw: bool = False):
    """
    Query OpenAI for logprobs and extract You/Them probabilities.

    Args:
        messages: The conversation messages
        tool_prompt: Tool prompt configuration
        model_name: OpenAI model name
        return_raw: If True, return (result_dict, raw_response) tuple

    Returns:
        If return_raw=False: result dict
        If return_raw=True: (result dict, raw API response)
    """
    tools = create_tool_definition(tool_prompt)

    def do_query():
        return openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            max_completion_tokens=2,  # Allow second token to see logprobs for first
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=42,
        )

    try:
        resp = do_query()
    except Exception as e:
        print(f"Error during OpenAI API call: {e}. Retrying...")
        time.sleep(1)
        resp = do_query()

    choice = resp.choices[0]

    # Check if we got content with logprobs
    if not choice.logprobs or not choice.logprobs.content:
        result = {
            'you_logit': float('-inf'),
            'them_logit': float('-inf'),
            'you_relative_prob': 0.0,
            'them_relative_prob': 0.0,
            'error': 'no_content_logprobs'
        }
        return (result, resp) if return_raw else result

    top = choice.logprobs.content[0].top_logprobs

    # Find logprobs for You and Them tokens
    you_lps = []
    them_lps = []

    for cand in top:
        tok = cand.token.strip().lower()
        if tok == "you":
            you_lps.append(cand.logprob)
        elif tok == "them":
            them_lps.append(cand.logprob)

    if not you_lps and not them_lps:
        result = {
            'you_logit': float('-inf'),
            'them_logit': float('-inf'),
            'you_relative_prob': 0.0,
            'them_relative_prob': 0.0,
            'error': 'both_missing'
        }
        return (result, resp) if return_raw else result

    # Apply floor penalty if either is missing
    logprob_floor = min(c.logprob for c in top)
    if not you_lps:
        you_lps.append(logprob_floor)
    if not them_lps:
        them_lps.append(logprob_floor)

    # Combine using logsumexp
    you_combined_lp = torch.logsumexp(torch.tensor(you_lps), dim=-1)
    them_combined_lp = torch.logsumexp(torch.tensor(them_lps), dim=-1)

    you_them_lps = torch.stack([you_combined_lp, them_combined_lp])

    # Convert to probabilities
    probs = F.softmax(you_them_lps, dim=-1)

    result = {
        'you_logit': you_combined_lp.item(),
        'them_logit': them_combined_lp.item(),
        'you_relative_prob': probs[0].item(),
        'them_relative_prob': probs[1].item(),
    }
    return (result, resp) if return_raw else result


def parse_logprobs_from_response_you_them(response_body: dict) -> dict:
    """
    Parse You/Them logprobs from a batch API response body.
    """
    choice = response_body['choices'][0]

    # Check if we got content with logprobs
    if not choice.get('logprobs') or not choice['logprobs'].get('content'):
        return {
            'you_logit': float('-inf'),
            'them_logit': float('-inf'),
            'you_relative_prob': 0.0,
            'them_relative_prob': 0.0,
            'error': 'no_content_logprobs'
        }

    top_logprobs = choice['logprobs']['content'][0]['top_logprobs']

    # Find logprobs for You and Them tokens
    you_lps = []
    them_lps = []

    for cand in top_logprobs:
        tok = cand['token'].strip().lower()
        if tok == "you":
            you_lps.append(cand['logprob'])
        elif tok == "them":
            them_lps.append(cand['logprob'])

    if not you_lps and not them_lps:
        return {
            'you_logit': float('-inf'),
            'them_logit': float('-inf'),
            'you_relative_prob': 0.0,
            'them_relative_prob': 0.0,
            'error': 'both_missing'
        }

    # Apply floor penalty if either is missing
    logprob_floor = min(c['logprob'] for c in top_logprobs)
    if not you_lps:
        you_lps.append(logprob_floor)
    if not them_lps:
        them_lps.append(logprob_floor)

    # Combine using logsumexp
    you_combined_lp = torch.logsumexp(torch.tensor(you_lps), dim=-1)
    them_combined_lp = torch.logsumexp(torch.tensor(them_lps), dim=-1)

    you_them_lps = torch.stack([you_combined_lp, them_combined_lp])

    # Convert to probabilities
    probs = F.softmax(you_them_lps, dim=-1)

    return {
        'you_logit': you_combined_lp.item(),
        'them_logit': them_combined_lp.item(),
        'you_relative_prob': probs[0].item(),
        'them_relative_prob': probs[1].item(),
    }


# =============================================================================
# Batch request preparation
# =============================================================================

def prepare_batch_request(example: dict, tool_prompts: list, model_name: str) -> list:
    """
    Prepare batch API requests for a single example.

    Returns a list of request dicts (one per tool_prompt × tool_response combination).
    """
    requests = []
    letter_pair = example['letter_pair']  # e.g., ("D", "E")

    for tool_prompt in tool_prompts:
        tool_name = tool_prompt['name']
        tools = create_tool_definition(tool_prompt)

        # Tool responses are the letters from the third-person version
        for tool_response_letter in letter_pair:
            messages = build_conversation_with_tool(
                example, tool_prompt, tool_response_letter
            )

            # Build custom_id: scenario_id__my_version_my_first__instruction_short__tool_name__response_letter
            instruction_short = "ctrl" if not example['instruction'] else example['instruction'][:10].replace(" ", "_").rstrip("_")
            custom_id = f"{example['scenario_id']}__{example['my_version']}_{example['my_first']}__{instruction_short}__{tool_name}__{tool_response_letter}"

            requests.append({
                'custom_id': custom_id,
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
# Inspection mode
# =============================================================================

def inspect_example_openai(example: dict, tool_prompt: dict, tool_response_letter: str, model_name: str) -> None:
    """
    Inspect a single example with tool use: print full prompt and logprobs.

    Only makes ONE API call for the specified tool_prompt and tool_response_letter.
    """
    tool_name = tool_prompt['name']

    print("\n" + "=" * 70)
    print(f"TOOL: {tool_name}")
    print(f"TOOL RESPONSE: {tool_response_letter}")
    print("=" * 70)

    messages = build_conversation_with_tool(example, tool_prompt, tool_response_letter)

    # Print full conversation verbatim
    print("\n--- FULL PROMPT (messages) ---\n")
    for i, msg in enumerate(messages):
        role = msg.get('role', 'unknown')
        print(f"[{i}] ROLE: {role}")

        if role == 'assistant' and msg.get('tool_calls'):
            print("    CONTENT: null (tool call)")
            for tc in msg['tool_calls']:
                print(f"    TOOL_CALL ID: {tc['id']}")
                print(f"    TOOL_CALL FUNCTION: {tc['function']['name']}")
                print(f"    TOOL_CALL ARGUMENTS: {tc['function']['arguments']}")
        elif role == 'tool':
            print(f"    TOOL_CALL_ID: {msg.get('tool_call_id', '')}")
            print(f"    CONTENT: {msg.get('content', '')}")
        else:
            content = msg.get('content', '')
            print(f"    CONTENT:\n{content}")
        print()

    # Query and show detailed results
    print("--- QUERYING MODEL ---\n")
    result, raw_response = query_logprobs_you_them(messages, tool_prompt, model_name, return_raw=True)

    print("--- RAW RESPONSE ---\n")
    if raw_response:
        choice = raw_response.choices[0]
        print(f"  Finish reason: {choice.finish_reason}")
        print(f"  Generated text: {repr(choice.message.content)}")

        if choice.logprobs and choice.logprobs.content:
            print(f"\n  Top logprobs for first token:")
            for cand in choice.logprobs.content[0].top_logprobs:
                print(f"    {repr(cand.token):12s} -> {cand.logprob:.4f}")
        else:
            print("  No logprobs in response")

    print("\n--- COMPUTED RESULTS ---\n")
    print(f"  you_logit:        {result.get('you_logit', float('-inf')):.4f}")
    print(f"  them_logit:       {result.get('them_logit', float('-inf')):.4f}")
    print(f"  P(You):           {result.get('you_relative_prob', 0):.4f}")
    print(f"  P(Them):          {result.get('them_relative_prob', 0):.4f}")
    if 'error' in result:
        print(f"  Error:            {result['error']}")


def run_inspect_mode(data: list, tool_prompts: list, model_name: str, n_samples: int, seed: int):
    """
    Run inspection mode on ONE sample with ONE API call.
    """
    random.seed(seed)

    # Pick one random example
    idx = random.randint(0, len(data) - 1)
    example = data[idx]

    tool_prompt = tool_prompts[0]
    letter_pair = example['letter_pair']
    tool_response_letter = letter_pair[0]  # Use first letter for inspection

    print("\n" + "#" * 70)
    print(f"# INSPECT MODE: Tool Result You/Them Logprobs (SINGLE CALL)")
    print(f"# Model: {model_name}")
    print(f"# Example index: {idx}")
    print("#" * 70)

    # Print example metadata
    print("\n--- EXAMPLE METADATA ---\n")
    for field in ['scenario_id', 'scenario_nickname', 'my_version', 'my_first', 'instruction', 'sycophantic_token', 'letter_pair', 'you_maps_to_letter', 'them_maps_to_letter']:
        if field in example:
            print(f"  {field}: {example[field]}")

    print("\n--- ORIGINAL PROMPT (first-person) ---")
    print(example['prompt'])

    print("\n--- REMOVED PROMPT (third-person) ---")
    print(example['removed_prompt'])

    # Single inspection call
    inspect_example_openai(example, tool_prompt, tool_response_letter, model_name)

    print("\n" + "#" * 70)
    print("# INSPECT MODE COMPLETE (1 API call)")
    print("#" * 70)


# =============================================================================
# Data loading
# =============================================================================

def load_tool_prompts(path: Path) -> list:
    """Load tool prompts from YAML file."""
    with open(path, 'r') as f:
        tool_prompts = yaml.safe_load(f)
    return tool_prompts


def prepare_data_with_removed_prompts(data_path: Path = None) -> list:
    """
    Load sycophancy scenarios, generate forced-choice conditions,
    and add removed_prompt (third-person version) and letter_pair to each.
    """
    from sycophancy.prompts.first_person import (
        load_scenarios,
        generate_full_experiment,
    )

    # Load scenarios
    data_path = data_path or DEFAULT_SYCOPHANCY_DATA
    scenarios = load_scenarios(str(data_path))
    scenario_lookup = {s.id: s for s in scenarios}

    print(f"Loaded {len(scenarios)} scenarios")

    # Generate forced-choice conditions
    experiment = generate_full_experiment(scenarios)
    conditions = experiment['conditions']

    print(f"Generated {len(conditions)} conditions")

    # Convert to dicts and add removed_prompt + letter_pair
    data = []
    for idx, condition in enumerate(conditions):
        d = asdict(condition)
        d['scenario_id'] = idx  # Use index as scenario_id for batch matching

        # Add computed properties
        d['sycophantic_token'] = condition.sycophantic_token
        d['you_validates_version_a'] = condition.you_validates_version_a
        d['version_a_token'] = condition.version_a_token
        d['first_position_token'] = condition.first_position_token

        # Generate removed prompt and get letter pair
        scenario = scenario_lookup[condition.scenario_id]
        removed_prompt, letter_pair = generate_removed_prompt_and_letters(d, scenario)
        d['removed_prompt'] = removed_prompt
        d['letter_pair'] = letter_pair

        # Store mapping: which letter corresponds to "me" in the first-person version
        # If my_version == "A", then version_a_label maps to "me" → "You"
        # If my_version == "B", then version_b_label maps to "me" → "You"
        if d['my_version'] == "A":
            d['you_maps_to_letter'] = letter_pair[0]  # version_a_label
            d['them_maps_to_letter'] = letter_pair[1]  # version_b_label
        else:
            d['you_maps_to_letter'] = letter_pair[1]  # version_b_label
            d['them_maps_to_letter'] = letter_pair[0]  # version_a_label

        data.append(d)

    print(f"Prepared {len(data)} conditions with removed prompts and letter pairs")

    return data


# =============================================================================
# Main inference - synchronous mode
# =============================================================================

def run_single_inference(data: list, tool_prompts: list, args, run_index: int = None):
    """Run synchronous inference and save to a file."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_nickname = args.openai_model.replace("/", "_")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_suffix = f"_run{run_index:03d}" if run_index is not None else ""
    output_filename = Path(args.output_dir) / f"{timestamp}_sycophancy_tool_result_{model_nickname}{run_suffix}.jsonl"

    print("=" * 60)
    print(f"Model:        {args.openai_model}")
    print(f"Output:       {output_filename}")
    print(f"Tool prompts: {[tp['name'] for tp in tool_prompts]}")
    if run_index is not None:
        print(f"Run:          {run_index + 1} of {args.n_times}")
    print("=" * 60)

    # Score all examples
    results = []
    for i, example in enumerate(data):
        if (i + 1) % 50 == 0:
            print(f"Scoring example {i + 1}/{len(data)}...")

        score_result = score_example_openai(example, tool_prompts, args.openai_model)

        # Merge results into example
        result = dict(example)
        result['tool_results'] = score_result
        results.append(result)

    # Save results
    with open(output_filename, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"\nDone! Output: {output_filename}")

    return output_filename


# =============================================================================
# Batch inference
# =============================================================================

def run_batch_inference(data: list, tool_prompts: list, args, run_index: int = None):
    """Run batch inference using OpenAI Batch API."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_nickname = args.openai_model.replace("/", "_")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_suffix = f"_run{run_index:03d}" if run_index is not None else ""
    output_filename = Path(args.output_dir) / f"{timestamp}_sycophancy_tool_result_{model_nickname}{run_suffix}.jsonl"

    print("=" * 60)
    print(f"Mode:         BATCH")
    print(f"Model:        {args.openai_model}")
    print(f"Output:       {output_filename}")
    print(f"Tool prompts: {[tp['name'] for tp in tool_prompts]}")
    if run_index is not None:
        print(f"Run:          {run_index + 1} of {args.n_times}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Collect all requests into batch
    # -------------------------------------------------------------------------
    batch = []
    for example in data:
        requests = prepare_batch_request(example, tool_prompts, args.openai_model)
        batch.extend(requests)

    print(f"Generated {len(batch)} batch requests")

    # Save batch to JSONL
    batch_path = output_filename.with_name(f"{output_filename.stem}_batch.jsonl")
    with open(batch_path, 'w') as f:
        for item in batch:
            f.write(json.dumps(item) + "\n")
    print(f"Batch saved to {batch_path}")

    # -------------------------------------------------------------------------
    # Save metadata
    # -------------------------------------------------------------------------
    metadata = {
        "model": args.openai_model,
        "timestamp": timestamp,
        "num_conditions": len(data),
        "num_requests": len(batch),
        "tool_prompts": [tp['name'] for tp in tool_prompts],
    }
    if run_index is not None:
        metadata['run_index'] = run_index
        metadata['n_times'] = args.n_times

    metadata_path = output_filename.with_suffix('.meta.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

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
    # Parse and merge results
    # -------------------------------------------------------------------------
    print("Parsing results...")

    # Parse results into dict keyed by custom_id
    results_by_id = {}
    for line in results_content.text.strip().split("\n"):
        item = json.loads(line)
        results_by_id[item["custom_id"]] = item["response"]["body"]

    print(f"Parsed {len(results_by_id)} results")

    # Group results by scenario_id
    # custom_id format: {scenario_id}__{my_version}_{my_first}__{instruction_short}__{tool_name}__{response_letter}
    grouped = {}
    for custom_id, response_body in results_by_id.items():
        parts = custom_id.split("__")
        scenario_id = int(parts[0])
        tool_name = parts[3]
        response_letter = parts[4]

        if scenario_id not in grouped:
            grouped[scenario_id] = {}
        if tool_name not in grouped[scenario_id]:
            grouped[scenario_id][tool_name] = {}

        result = parse_logprobs_from_response_you_them(response_body)
        grouped[scenario_id][tool_name][response_letter] = result

    # Merge results into original data
    final_results = []
    for example in data:
        result = dict(example)
        result['tool_results'] = grouped.get(example['scenario_id'], {})
        final_results.append(result)

    # Save final results
    with open(output_filename, 'w') as f:
        for item in final_results:
            f.write(json.dumps(item) + "\n")

    print(f"Final dataset saved to {output_filename}")
    print(f"\nDone! Output: {output_filename}")
    print(f"Metadata: {metadata_path}")

    return output_filename


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run sycophancy inference with simulated tool use (ChatGPT)"
    )
    parser.add_argument("--openai_model", type=str, default="gpt-4.1",
                        help="OpenAI chat model name")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to scenarios JSONL file (default: sycophancy/data/sycophancy-two-sides-eval.jsonl)")
    parser.add_argument("--output_dir", type=str, default=str(SCRIPT_DIR.parent / "results"),
                        help="Directory for output files")
    parser.add_argument("--tool_prompts_path", type=str, default="../../src/tool_prompts.yaml",
                        help="Path to tool prompts YAML file")
    parser.add_argument("--inspect", action="store_true",
                        help="Run inspection mode and exit")
    parser.add_argument("--inspect_n", type=int, default=1,
                        help="Number of samples to inspect")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_times", type=int, default=1,
                        help="Number of times to run inference")
    parser.add_argument("--batch", action="store_true",
                        help="Use OpenAI Batch API instead of synchronous calls (50%% cost savings)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load tool prompts
    tool_prompts_path = Path(args.tool_prompts_path)
    tool_prompts = load_tool_prompts(tool_prompts_path)
    print(f"Loaded {len(tool_prompts)} tool prompts: {[tp['name'] for tp in tool_prompts]}")

    # Load and prepare data
    data_path = Path(args.data_path) if args.data_path else None
    data = prepare_data_with_removed_prompts(data_path)

    # Inspection mode
    if args.inspect:
        run_inspect_mode(data, tool_prompts, args.openai_model, args.inspect_n, args.seed)
        return

    # Select inference function based on mode
    run_inference_fn = run_batch_inference if args.batch else run_single_inference

    # Run inference
    output_files = []
    if args.n_times == 1:
        output_file = run_inference_fn(data, tool_prompts, args, run_index=None)
        output_files.append(output_file)
    else:
        for i in range(args.n_times):
            print(f"\n{'#' * 60}")
            print(f"# RUN {i + 1} OF {args.n_times}")
            print(f"{'#' * 60}\n")
            output_file = run_inference_fn(data, tool_prompts, args, run_index=i)
            output_files.append(output_file)

    if args.n_times > 1:
        print(f"\nAll {args.n_times} runs complete. Output files:")
        for f in output_files:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
