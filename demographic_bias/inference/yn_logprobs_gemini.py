#!/usr/bin/env python3
"""
Run demographic bias yes/no logprobs inference via Vertex AI Gemini.

Mirrors yn_logprobs_openai.py output schema. Uses gemini_client.call_gemini
(sync calls with ThreadPoolExecutor) since Vertex doesn't have an OpenAI-style
batch API exposed here.

Output: {timestamp}_bias_yn_{model_nickname}.jsonl with one row per scenario;
each row has per-prompt-format columns:
  {snake_case}_prompt_yes_logits / _no_logits / _yes_relative_probs / _no_relative_probs
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

# Public-repo root for src.scoring + demographic_bias.*
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Private repo's sycophancy/ for gemini_client
PRIVATE_SYCOPHANCY = REPO_ROOT.parent / "sycophancy"
if str(PRIVATE_SYCOPHANCY) not in sys.path:
    sys.path.insert(0, str(PRIVATE_SYCOPHANCY))

from datasets import Dataset

from demographic_bias.config import DEFAULT_BIAS_DATA, YES_NO_SYSTEM_PROMPT
from demographic_bias.prompts.formats import PROMPT_DICT, build_single_prompt
from src.scoring import score_yes_no_from_top_logprobs

from gemini_client import (
    call_gemini,
    extract_top_logprobs,
    extract_text,
    model_nickname,
)


# =============================================================================
# Scoring
# =============================================================================

def compute_yes_no_scores(top_logprobs: list, prompt_format_name: str) -> dict:
    """Match yn_logprobs_openai.py output schema."""
    if not top_logprobs:
        return {
            f"{prompt_format_name}_prompt_yes_logits": float("-inf"),
            f"{prompt_format_name}_prompt_no_logits": float("-inf"),
            f"{prompt_format_name}_prompt_yes_relative_probs": 0.0,
            f"{prompt_format_name}_prompt_no_relative_probs": 0.0,
            f"{prompt_format_name}_prompt_yes_tokens_in_top_20": 0,
            f"{prompt_format_name}_prompt_no_tokens_in_top_20": 0,
            f"{prompt_format_name}_prompt_error": "no_logprobs",
        }
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
# Per-call worker
# =============================================================================

def score_one(example: dict, prompt_format: dict, args) -> dict:
    snake_case = prompt_format['snake_case']

    # Build conversation
    row_values = [example[col] for col in prompt_format['prompt_column']]
    conversation = build_single_prompt(prompt_format['conversation'], row_values)

    # Strip assistant prefill if present (Vertex doesn't accept trailing assistant turn)
    if conversation[-1]['role'] == 'assistant':
        conversation = conversation[:-1]

    # Prepend system prompt
    messages = [{"role": "system", "content": YES_NO_SYSTEM_PROMPT}]
    messages.extend(conversation)

    try:
        resp = call_gemini(
            model=args.gemini_model,
            messages=messages,
            project=args.project,
            region=args.region,
            thinking_budget=args.thinking_budget,
            response_logprobs=True,
            logprobs_top_k=20,
            max_output_tokens=args.max_output_tokens,
            temperature=0.0,
            seed=args.seed,
        )
    except Exception as e:
        return {f"{snake_case}_prompt_error": repr(e)[:200]}

    top_logprobs = extract_top_logprobs(resp, position=0)
    out = compute_yes_no_scores(top_logprobs, snake_case)
    out[f"{snake_case}_prompt_text"] = extract_text(resp) or ""
    return out


# =============================================================================
# Main run
# =============================================================================

def run_inference(data: Dataset, args) -> Path:
    nick = model_nickname(args.gemini_model, args.thinking_budget)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{timestamp}_bias_yn_{nick}.jsonl"

    print("=" * 60)
    print(f"Model:          {args.gemini_model}  (region {args.region})")
    print(f"Thinking:       budget={args.thinking_budget}")
    print(f"Scenarios:      {len(data)}")
    print(f"Prompt formats: {len(PROMPT_DICT)}")
    print(f"Total calls:    {len(data) * len(PROMPT_DICT)}")
    print(f"Workers:        {args.max_workers}")
    print(f"Output:         {out_path}")
    print("=" * 60)

    rows = [dict(data[i]) for i in range(len(data))]

    for prompt_name, prompt_format in PROMPT_DICT.items():
        snake_case = prompt_format['snake_case']
        print(f"\n--- Scoring '{prompt_name}' ({snake_case}) ---")
        start = time.time()
        errors = 0

        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {ex.submit(score_one, rows[i], prompt_format, args): i for i in range(len(rows))}
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    r = fut.result()
                    rows[i].update(r)
                    if f"{snake_case}_prompt_error" in r:
                        errors += 1
                except Exception as e:
                    errors += 1
                    rows[i][f"{snake_case}_prompt_error"] = repr(e)[:200]
                done += 1
                if done % 200 == 0 or done == len(rows):
                    elapsed = time.time() - start
                    print(f"  [{done}/{len(rows)}] elapsed {elapsed:.1f}s  errors {errors}")

        # Checkpoint after each prompt format
        with open(out_path, 'w') as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"  Checkpoint saved.")

    print(f"\nSaved {len(rows)} rows to {out_path}")
    return out_path


def load_scenarios(path: Path) -> list[dict]:
    scenarios = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                scenarios.append(json.loads(line))
    return scenarios


def main():
    p = argparse.ArgumentParser(description="Demographic-bias yes/no logprobs via Vertex Gemini")
    p.add_argument("--gemini_model", default="gemini-2.5-flash-lite")
    p.add_argument("--region", default="us-central1")
    p.add_argument("--project", default=os.environ.get("GCP_PROJECT", "matan-self-model"))
    p.add_argument("--thinking_budget", type=int, default=0)
    p.add_argument("--no_thinking_config", action="store_true",
                   help="Pass thinking_budget=None (omit thinkingConfig from request)")
    p.add_argument("--data_path", default=None)
    p.add_argument("--output_dir", default=str(SCRIPT_DIR.parent / "results"))
    p.add_argument("--n_scenarios", type=int, default=None)
    p.add_argument("--max_workers", type=int, default=16)
    p.add_argument("--max_output_tokens", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.no_thinking_config:
        args.thinking_budget = None

    random.seed(args.seed)

    data_path = Path(args.data_path) if args.data_path else DEFAULT_BIAS_DATA
    print(f"Loading scenarios from {data_path}")
    scenarios = load_scenarios(data_path)
    print(f"  Loaded {len(scenarios)} scenarios")
    if args.n_scenarios:
        scenarios = scenarios[:args.n_scenarios]
        print(f"  Limited to {len(scenarios)}")

    for i, s in enumerate(scenarios):
        s['scenario_id'] = i

    data = Dataset.from_list(scenarios)
    print(f"  Columns: {data.column_names}")

    out = run_inference(data, args)
    print(f"\nOutput: {out}")


if __name__ == "__main__":
    main()
