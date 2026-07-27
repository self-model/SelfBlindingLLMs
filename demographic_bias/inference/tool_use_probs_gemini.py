#!/usr/bin/env python3
"""
Run demographic bias tool-use probability inference via Vertex AI Gemini.

For each (scenario × prompt_format × tool), ask the model whether to call the
counterfactual-simulation tool. Records the response in OpenAI-shaped form
under the column `{snake_case}__{tool_name}__completion_json` so the existing
build_csv.py logic can detect tool calls without modification.

Skips the `remove_in_context` prompt format (multi-turn; not used for tool-use
in the OpenAI version either).
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Public-repo root for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Private repo's sycophancy/ for gemini_client
PRIVATE_SYCOPHANCY = REPO_ROOT.parent / "sycophancy"
if str(PRIVATE_SYCOPHANCY) not in sys.path:
    sys.path.insert(0, str(PRIVATE_SYCOPHANCY))

import yaml
from datasets import Dataset

from demographic_bias.config import DEFAULT_BIAS_DATA, DEFAULT_TOOL_PROMPTS_PATH
from demographic_bias.prompts.formats import PROMPT_DICT, build_single_prompt, create_tool_definition

from gemini_client import (
    call_gemini,
    extract_function_calls,
    extract_text,
    finish_reason as get_finish_reason,
    model_nickname,
)


def load_scenarios(path: Path) -> list[dict]:
    scenarios = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                scenarios.append(json.loads(line))
    return scenarios


def load_tool_prompts(path: Path) -> list[dict]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# =============================================================================
# Build OpenAI-shaped completion_json from Gemini response so build_csv parses it
# =============================================================================

def gemini_to_openai_response(response: dict, tool_name_hint: str = "") -> dict:
    """
    Convert a Vertex Gemini response into the same OpenAI Chat Completion shape
    that build_csv.py expects to find under `{prompt}__{tool}__completion_json`.
    Specifically: choices[0].message.{content, tool_calls}, choices[0].finish_reason.

    `tool_name_hint` is used only when Gemini reports MALFORMED_FUNCTION_CALL —
    the model tried to invoke the tool but emitted bad arguments. We still want
    that to register as has_tool_call=True (it's a tool-use attempt), so we
    synthesize a tool_calls entry with empty arguments and the hinted name.
    """
    text = extract_text(response) or ""
    fr = get_finish_reason(response) or ""
    fn_calls = extract_function_calls(response)

    tool_calls = []
    for i, call in enumerate(fn_calls):
        tool_calls.append({
            "id": f"call_{i+1:03d}",
            "type": "function",
            "function": {
                "name": call.get("name", ""),
                "arguments": json.dumps(call.get("args", {})),
            },
        })

    # Map Vertex finish_reason to OpenAI-ish strings used by build_csv
    # OpenAI uses 'tool_calls', 'stop', 'length'. Vertex uses STOP, MAX_TOKENS, TOOL_USE, etc.
    finish_map = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "TOOL_USE": "tool_calls",
        "MALFORMED_FUNCTION_CALL": "tool_calls",
    }
    openai_finish = finish_map.get(fr, fr.lower() if fr else "stop")

    # MALFORMED_FUNCTION_CALL: Gemini intended to call the tool but couldn't
    # produce well-formed args. Preserve the tool-use signal by synthesizing a
    # placeholder tool_calls entry so build_csv counts has_tool_call=True.
    if fr == "MALFORMED_FUNCTION_CALL" and not tool_calls:
        tool_calls.append({
            "id": "call_001",
            "type": "function",
            "function": {
                "name": tool_name_hint or "unknown",
                "arguments": json.dumps({"_malformed": True}),
            },
        })

    if tool_calls and openai_finish == "stop":
        openai_finish = "tool_calls"

    message = {"role": "assistant", "content": text if text else None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": openai_finish,
        }],
        "model": "gemini",
    }


# =============================================================================
# Per-call worker
# =============================================================================

def call_one(example: dict, prompt_format: dict, tool_prompt: dict, args) -> dict:
    """Returns dict with one column: {prompt}__{tool}__completion_json -> JSON string."""
    snake_case = prompt_format['snake_case']
    tool_name = tool_prompt['name']

    # Build conversation
    row_values = [example[col] for col in prompt_format['prompt_column']]
    conversation = build_single_prompt(prompt_format['conversation'], row_values)
    if conversation[-1]['role'] == 'assistant':
        conversation = conversation[:-1]

    tools = create_tool_definition(tool_prompt)

    try:
        resp = call_gemini(
            model=args.gemini_model,
            messages=conversation,
            project=args.project,
            region=args.region,
            tools=tools,
            thinking_budget=args.thinking_budget,
            response_logprobs=False,  # tool_use_probs doesn't need logprobs
            max_output_tokens=args.max_output_tokens,
            temperature=0.0,
            seed=args.seed,
        )
    except Exception as e:
        # Stash a synthesized error response so build_csv handles missing-tool-call gracefully
        err_resp = {
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": None},
                "finish_reason": "error",
            }],
            "error": repr(e)[:300],
        }
        return {f"{snake_case}__{tool_name}__completion_json": json.dumps(err_resp)}

    openai_shaped = gemini_to_openai_response(resp, tool_name_hint=tool_name)
    return {f"{snake_case}__{tool_name}__completion_json": json.dumps(openai_shaped)}


# =============================================================================
# Main run
# =============================================================================

def run_inference(data: Dataset, tool_prompts: list, args) -> Path:
    nick = model_nickname(args.gemini_model, args.thinking_budget)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{timestamp}_bias_tool_use_{nick}.jsonl"

    # Filter prompt formats: skip multi-turn 'remove_in_context' (matches OpenAI script)
    formats_to_use = {
        name: fmt for name, fmt in PROMPT_DICT.items()
        if fmt['snake_case'] not in ('remove_in_context',)
        and (not args.prompt or fmt['snake_case'] in args.prompt)
    }

    print("=" * 60)
    print(f"Model:          {args.gemini_model}  (region {args.region})")
    print(f"Thinking:       budget={args.thinking_budget}")
    print(f"Scenarios:      {len(data)}")
    print(f"Prompt formats: {len(formats_to_use)}")
    print(f"Tool prompts:   {len(tool_prompts)} ({[tp['name'] for tp in tool_prompts]})")
    print(f"Total calls:    {len(data) * len(formats_to_use) * len(tool_prompts)}")
    print(f"Workers:        {args.max_workers}")
    print(f"Output:         {out_path}")
    print("=" * 60)

    rows = [dict(data[i]) for i in range(len(data))]

    # Iterate per (prompt_format, tool_prompt) and parallelize across scenarios
    for prompt_name, prompt_format in formats_to_use.items():
        for tool_prompt in tool_prompts:
            tool_name = tool_prompt['name']
            print(f"\n--- {prompt_format['snake_case']} × {tool_name} ---")
            start = time.time()
            errors = 0

            with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                futures = {ex.submit(call_one, rows[i], prompt_format, tool_prompt, args): i
                           for i in range(len(rows))}
                done = 0
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        r = fut.result()
                        rows[i].update(r)
                    except Exception as e:
                        errors += 1
                        rows[i][f"{prompt_format['snake_case']}__{tool_name}__completion_json"] = json.dumps({
                            "choices": [{"index": 0, "message": {"role": "assistant", "content": None},
                                          "finish_reason": "error"}],
                            "error": repr(e)[:300],
                        })
                    done += 1
                    if done % 200 == 0 or done == len(rows):
                        elapsed = time.time() - start
                        print(f"  [{done}/{len(rows)}] elapsed {elapsed:.1f}s  errors {errors}")

            # Checkpoint
            with open(out_path, 'w') as f:
                for r in rows:
                    f.write(json.dumps(r, default=str) + "\n")
            print(f"  Checkpoint saved.")

    print(f"\nSaved {len(rows)} rows to {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Demographic-bias tool-use probability via Vertex Gemini")
    p.add_argument("--gemini_model", default="gemini-2.5-flash-lite")
    p.add_argument("--region", default="us-central1")
    p.add_argument("--project", default=os.environ.get("GCP_PROJECT", "matan-self-model"))
    p.add_argument("--thinking_budget", type=int, default=0)
    p.add_argument("--no_thinking_config", action="store_true")
    p.add_argument("--output_dir", default=str(SCRIPT_DIR.parent / "results"))
    p.add_argument("--tool_prompts_path", default=str(DEFAULT_TOOL_PROMPTS_PATH))
    p.add_argument("--prompt", action="append", default=None,
                   help="Restrict to specific prompt format(s); repeatable")
    p.add_argument("--data_path", default=None)
    p.add_argument("--n_scenarios", type=int, default=None)
    p.add_argument("--max_workers", type=int, default=16)
    p.add_argument("--max_output_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.no_thinking_config:
        args.thinking_budget = None

    random.seed(args.seed)

    tool_prompts = load_tool_prompts(Path(args.tool_prompts_path))
    print(f"Loaded {len(tool_prompts)} tool prompts: {[tp['name'] for tp in tool_prompts]}")

    data_path = Path(args.data_path) if args.data_path else DEFAULT_BIAS_DATA
    print(f"Loading scenarios from {data_path}")
    scenarios = load_scenarios(data_path)
    print(f"  Loaded {len(scenarios)}")
    if args.n_scenarios:
        scenarios = scenarios[:args.n_scenarios]
        print(f"  Limited to {len(scenarios)}")

    for i, s in enumerate(scenarios):
        s['scenario_id'] = i

    data = Dataset.from_list(scenarios)

    out = run_inference(data, tool_prompts, args)
    print(f"\nOutput: {out}")


if __name__ == "__main__":
    main()
