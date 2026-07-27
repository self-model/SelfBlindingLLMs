#!/usr/bin/env python3
"""
Run sycophancy first-person (forced-choice) inference via Vertex AI Gemini.

Mirrors first_person_openai.py: uses You/Them framing (SYSTEM_PROMPT from
sycophancy.prompts.first_person) so build_csv.py's process_first_person works
without modification.

Design: 60 scenarios × 2 my_version × 2 my_first × N instructions, sync calls
with a thread pool.
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Public-repo root for sycophancy.* + src.scoring
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Private repo's sycophancy/ for gemini_client
PRIVATE_SYCOPHANCY = REPO_ROOT.parent / "sycophancy"
if str(PRIVATE_SYCOPHANCY) not in sys.path:
    sys.path.insert(0, str(PRIVATE_SYCOPHANCY))

from datasets import Dataset

from sycophancy.config import DEFAULT_SYCOPHANCY_DATA
from sycophancy.prompts.first_person import (
    load_scenarios,
    generate_full_experiment,
    ForcedChoiceCondition,
    SYSTEM_PROMPT,
)
from src.scoring import score_you_them_from_top_logprobs

from gemini_client import (
    call_gemini,
    extract_top_logprobs,
    extract_text,
    model_nickname,
)


def condition_to_dict(condition: ForcedChoiceCondition) -> dict:
    d = asdict(condition)
    d["you_validates_version_a"] = condition.you_validates_version_a
    d["sycophantic_token"] = condition.sycophantic_token
    d["version_a_token"] = condition.version_a_token
    d["first_position_token"] = condition.first_position_token
    # ForcedChoiceCondition has 'sycophantic_token' but not 'you_validates_speaker';
    # derive it (True iff "You" is the speaker-validating token).
    d["you_validates_speaker"] = (condition.sycophantic_token == "You")
    return d


def score_one(example: dict, args) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["prompt"]},
    ]
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
        return {
            "you_logit": float("-inf"), "them_logit": float("-inf"),
            "you_relative_prob": 0.0, "them_relative_prob": 0.0,
            "you_tokens_in_top_20": 0, "them_tokens_in_top_20": 0,
            "_text": None, "_error": repr(e)[:200],
        }
    top = extract_top_logprobs(resp, position=0)
    if not top:
        return {
            "you_logit": float("-inf"), "them_logit": float("-inf"),
            "you_relative_prob": 0.0, "them_relative_prob": 0.0,
            "you_tokens_in_top_20": 0, "them_tokens_in_top_20": 0,
            "_text": extract_text(resp), "_error": "no_logprobs",
        }
    res = score_you_them_from_top_logprobs(top)
    return {
        "you_logit": res["you_logit"],
        "them_logit": res["them_logit"],
        "you_relative_prob": res["you_prob"],
        "them_relative_prob": res["them_prob"],
        "you_tokens_in_top_20": res["you_tokens_in_top_20"],
        "them_tokens_in_top_20": res["them_tokens_in_top_20"],
        "_text": extract_text(resp),
    }


def run_inference(data: Dataset, args, run_idx: int = 0) -> Path:
    nick = model_nickname(args.gemini_model, args.thinking_budget)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_run{run_idx:02d}" if args.n_times > 1 else ""
    out_path = output_dir / f"{timestamp}_sycophancy_first_person_{nick}{suffix}.jsonl"

    print("=" * 60)
    print(f"Model:        {args.gemini_model}  (region {args.region})")
    print(f"Thinking:     budget={args.thinking_budget}")
    print(f"Conditions:   {len(data)}")
    print(f"Workers:      {args.max_workers}")
    if args.n_times > 1:
        print(f"Run:          {run_idx + 1}/{args.n_times}")
    print(f"Output:       {out_path}")
    print("=" * 60)

    items = [data[i] for i in range(len(data))]
    results = [None] * len(items)
    errors = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(score_one, items[i], args): i for i in range(len(items))}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                r = fut.result()
                items[i].update(r)
                if "_error" in r:
                    errors += 1
                results[i] = items[i]
            except Exception as e:
                errors += 1
                items[i]["_error"] = repr(e)[:200]
                results[i] = items[i]
            done += 1
            if done % 100 == 0 or done == len(items):
                elapsed = time.time() - start
                print(f"  [{done}/{len(items)}] elapsed {elapsed:.1f}s  errors {errors}")

    with open(out_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nSaved {len(results)} rows to {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Sycophancy first-person (You/Them) via Vertex Gemini")
    p.add_argument("--gemini_model", default="gemini-2.5-flash-lite")
    p.add_argument("--region", default="us-central1")
    p.add_argument("--project", default=os.environ.get("GCP_PROJECT", "matan-self-model"))
    p.add_argument("--thinking_budget", type=int, default=0)
    p.add_argument("--no_thinking_config", action="store_true")
    p.add_argument("--output_dir", default=str(SCRIPT_DIR.parent / "results"))
    p.add_argument("--data_path", default=None)
    p.add_argument("--n_times", type=int, default=1)
    p.add_argument("--max_workers", type=int, default=16)
    p.add_argument("--max_output_tokens", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.no_thinking_config:
        args.thinking_budget = None

    random.seed(args.seed)

    print("Loading scenarios...")
    data_path = Path(args.data_path) if args.data_path else DEFAULT_SYCOPHANCY_DATA
    scenarios = load_scenarios(data_path)
    print(f"  {len(scenarios)} scenarios")
    experiment = generate_full_experiment(scenarios)
    conditions = experiment["conditions"]
    print(f"  {len(conditions)} conditions")

    data = Dataset.from_list([condition_to_dict(c) for c in conditions])

    out_files = []
    for run_idx in range(args.n_times):
        if args.n_times > 1:
            print(f"\n{'#' * 60}\n# RUN {run_idx + 1}/{args.n_times}\n{'#' * 60}")
        out_files.append(run_inference(data, args, run_idx=run_idx))

    if args.n_times > 1:
        print("\nAll runs complete:")
        for f in out_files:
            print(f"  {f}")


if __name__ == "__main__":
    main()
