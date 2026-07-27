#!/usr/bin/env python3
"""
Run demographic bias yes/no logprobs after simulated tool result via Vertex AI Gemini.

For each (scenario × prompt_format × tool × {Yes., No.} simulated tool response),
build a conversation [system, user, assistant(tool_call), tool(response)] and ask
for next-token yes/no logprobs.

Output schema mirrors tool_result_yn_logprobs_openai.py: per-prompt-format dict
column whose value is {tool_name: {response_text: {yes_logit, no_logit, ...}}}.
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

# Public-repo root
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

from demographic_bias.config import DEFAULT_BIAS_DATA, DEFAULT_TOOL_PROMPTS_PATH, YES_NO_SYSTEM_PROMPT
from demographic_bias.prompts.formats import PROMPT_DICT, build_single_prompt, create_tool_definition
from src.scoring import score_yes_no_from_top_logprobs

from gemini_client import (
    call_gemini,
    extract_top_logprobs,
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
# Conversation builders (OpenAI-format messages; gemini_client converts internally)
# =============================================================================

def create_tool_call_openai(tool_name: str, arguments: dict, call_id: str = "call_001") -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": tool_name, "arguments": json.dumps(arguments)},
        }],
    }


def create_tool_response_openai(tool_name: str, response: str, call_id: str = "call_001") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": response}


def build_conversation_with_tool(example: dict, prompt_format: dict, tool_prompt: dict,
                                 tool_response_text: str) -> list[dict]:
    row_values = [example[col] for col in prompt_format['prompt_column']]
    conversation = build_single_prompt(prompt_format['conversation'], row_values)
    if conversation[-1]['role'] == 'assistant':
        conversation = conversation[:-1]

    messages = [{"role": "system", "content": YES_NO_SYSTEM_PROMPT}]
    messages.extend(conversation)
    messages.append(create_tool_call_openai(
        tool_prompt['name'],
        {tool_prompt['property_name']: example['removed_template']},
        call_id="call_001",
    ))
    messages.append(create_tool_response_openai(
        tool_prompt['name'], tool_response_text, call_id="call_001",
    ))
    return messages


# =============================================================================
# Per-call worker
# =============================================================================

def call_one(example: dict, prompt_format: dict, tool_prompt: dict,
             tool_response_text: str, args) -> dict:
    messages = build_conversation_with_tool(example, prompt_format, tool_prompt, tool_response_text)
    tools = create_tool_definition(tool_prompt)

    try:
        resp = call_gemini(
            model=args.gemini_model,
            messages=messages,
            project=args.project,
            region=args.region,
            tools=tools,
            thinking_budget=args.thinking_budget,
            response_logprobs=True,
            logprobs_top_k=20,
            max_output_tokens=args.max_output_tokens,
            temperature=0.0,
            seed=args.seed,
        )
    except Exception as e:
        return {
            "yes_logit": float("-inf"), "no_logit": float("-inf"),
            "yes_relative_prob": 0.0, "no_relative_prob": 0.0,
            "error": repr(e)[:200],
        }

    top_logprobs = extract_top_logprobs(resp, position=0)
    if not top_logprobs:
        return {
            "yes_logit": float("-inf"), "no_logit": float("-inf"),
            "yes_relative_prob": 0.0, "no_relative_prob": 0.0,
            "error": "no_logprobs",
        }
    try:
        result = score_yes_no_from_top_logprobs(top_logprobs)
        return {
            "yes_logit": result['yes_logit'],
            "no_logit": result['no_logit'],
            "yes_relative_prob": result['yes_prob'],
            "no_relative_prob": result['no_prob'],
        }
    except ValueError:
        return {
            "yes_logit": float("-inf"), "no_logit": float("-inf"),
            "yes_relative_prob": 0.0, "no_relative_prob": 0.0,
            "error": "both_missing",
        }


# =============================================================================
# Main run
# =============================================================================

def run_inference(data: Dataset, tool_prompts: list, args) -> Path:
    nick = model_nickname(args.gemini_model, args.thinking_budget)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{timestamp}_bias_tool_result_yn_{nick}.jsonl"

    response_texts = ['Yes.', 'No.']
    formats_to_use = list(PROMPT_DICT.values())

    # Build the full job list: (scenario_idx, prompt_fmt, tool_prompt, response_text)
    jobs = []
    for prompt_fmt in formats_to_use:
        for tool_prompt in tool_prompts:
            for resp_text in response_texts:
                for i in range(len(data)):
                    jobs.append((i, prompt_fmt, tool_prompt, resp_text))

    print("=" * 60)
    print(f"Model:          {args.gemini_model}  (region {args.region})")
    print(f"Thinking:       budget={args.thinking_budget}")
    print(f"Scenarios:      {len(data)}")
    print(f"Prompt formats: {len(formats_to_use)}")
    print(f"Tool prompts:   {len(tool_prompts)}")
    print(f"Tool responses: {len(response_texts)} ({response_texts})")
    print(f"Total calls:    {len(jobs)}")
    print(f"Workers:        {args.max_workers}")
    print(f"Output:         {out_path}")
    print("=" * 60)

    rows = [dict(data[i]) for i in range(len(data))]
    # Initialize per-prompt-format columns as nested dicts
    for prompt_fmt in formats_to_use:
        snake = prompt_fmt['snake_case']
        for r in rows:
            r[snake] = {}

    start = time.time()
    errors = 0
    done = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(call_one, rows[i], prompt_fmt, tp, resp_text, args):
                (i, prompt_fmt['snake_case'], tp['name'], resp_text)
            for (i, prompt_fmt, tp, resp_text) in jobs
        }
        last_checkpoint = time.time()
        for fut in as_completed(futures):
            i, snake, tool_name, resp_text = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                errors += 1
                result = {"yes_logit": float("-inf"), "no_logit": float("-inf"),
                          "yes_relative_prob": 0.0, "no_relative_prob": 0.0,
                          "error": repr(e)[:200]}
            if "error" in result:
                errors += 1
            rows[i][snake].setdefault(tool_name, {})[resp_text] = result
            done += 1
            if done % 500 == 0 or done == len(jobs):
                elapsed = time.time() - start
                print(f"  [{done}/{len(jobs)}] elapsed {elapsed:.1f}s  errors {errors}")
            # Checkpoint every 5 minutes
            if time.time() - last_checkpoint > 300:
                with open(out_path, 'w') as f:
                    for r in rows:
                        f.write(json.dumps(r, default=str) + "\n")
                last_checkpoint = time.time()
                print(f"    [checkpoint @ {done}/{len(jobs)}]")

    with open(out_path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nSaved {len(rows)} rows to {out_path}  ({errors} errors)")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Demographic-bias tool-result yes/no logprobs via Vertex Gemini")
    p.add_argument("--gemini_model", default="gemini-2.5-flash-lite")
    p.add_argument("--region", default="us-central1")
    p.add_argument("--project", default=os.environ.get("GCP_PROJECT", "matan-self-model"))
    p.add_argument("--thinking_budget", type=int, default=0)
    p.add_argument("--no_thinking_config", action="store_true")
    p.add_argument("--output_dir", default=str(SCRIPT_DIR.parent / "results"))
    p.add_argument("--tool_prompts_path", default=str(DEFAULT_TOOL_PROMPTS_PATH))
    p.add_argument("--data_path", default=None)
    p.add_argument("--n_scenarios", type=int, default=None)
    p.add_argument("--max_workers", type=int, default=16)
    p.add_argument("--max_output_tokens", type=int, default=4)
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
