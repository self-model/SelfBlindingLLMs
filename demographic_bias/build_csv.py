#!/usr/bin/env python3
"""
Convert wide JSONL inference results to long format and/or CSV.

Processes inference outputs from:
- yn_logprobs_*.py: Yes/no scoring for each prompt strategy
- tool_use_probs_*.py: Tool-use probability measurement
- tool_result_yn_logprobs_*.py: Yes/no scoring after simulated tool use

Usage:
    # Convert yes/no logprobs to long format
    python build_csv.py --in results/bias_yn_gpt-4.1.jsonl --out results/bias_yn_gpt-4.1_long.jsonl

    # Also output CSV
    python build_csv.py --in results/bias_yn_gpt-4.1.jsonl --out results/bias_yn_gpt-4.1_long.jsonl --out-csv results/bias_yn_gpt-4.1.csv

    # Process tool-use results
    python build_csv.py --in results/bias_tool_use_gpt-4.1.jsonl --out results/bias_tool_use_long.jsonl --mode tool_use
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


# =============================================================================
# Regex Patterns for Column Matching
# =============================================================================

# Matches: {prompt_format}_prompt_yes_logits, {prompt_format}_prompt_no_logits, etc.
YN_LOGPROBS_RE = re.compile(r"^(?P<prompt_format>.+)_prompt_(?P<metric>yes_logits|no_logits|yes_relative_probs|no_relative_probs|yes_tokens_in_top_20|no_tokens_in_top_20)$")

# Matches: {prompt_format}__{tool_name}__completion_json
TOOL_USE_RE = re.compile(r"^(?P<prompt_format>.+)__(?P<tool_name>.+)__completion_json$")

# Matches: {prompt_format}__{tool_name}__tool_prob
TOOL_PROB_RE = re.compile(r"^(?P<prompt_format>.+)__(?P<tool_name>.+)__tool_prob$")

# Decision question IDs to exclude (inverted or problematic questions)
EXCLUDED_DECISION_QUESTION_IDS = {23, 54, 65, 67, 77}

# Prompt formats to include (excludes redact_in_context, redacted which weren't used in paper)
INCLUDED_PROMPT_FORMATS = {'default', 'dont_discriminate', 'ignore', 'if_you_didnt_know',
                           'remove_in_context', 'removed'}

# Tool to include (focus on counterfactual simulation)
INCLUDED_TOOL = 'run_counterfactual_simulation'


# =============================================================================
# Utility Functions
# =============================================================================

def safe_json_loads(s: Any) -> Optional[Dict[str, Any]]:
    """Parse a JSON string into a dict. Return None on failure."""
    if s is None:
        return None
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_choice0(completion_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract first choice from a completion object."""
    choices = completion_obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    c0 = choices[0]
    return c0 if isinstance(c0, dict) else None


def extract_tool_prompt_from_tool_calls(message: Dict[str, Any]) -> Optional[str]:
    """Extract prompt argument from tool calls."""
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None

    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        if not isinstance(fn, dict):
            continue
        args_str = fn.get("arguments")
        args = safe_json_loads(args_str)
        if isinstance(args, dict) and isinstance(args.get("prompt"), str):
            return args["prompt"]
    return None


# =============================================================================
# Yes/No Logprobs Processing
# =============================================================================

def iter_yn_logprobs_rows(wide_row: Dict[str, Any], row_index: int) -> Iterator[Dict[str, Any]]:
    """
    Convert a wide row with yes/no logprobs to long format.

    Yields one row per prompt format with all metrics.
    """
    # Base metadata
    base = {
        "row_index": row_index,
        "decision_question_id": wide_row.get("decision_question_id"),
        "decision_question_nickname": wide_row.get("decision_question_nickname"),
        "race": wide_row.get("race"),
        "gender": wide_row.get("gender"),
    }

    # Group metrics by prompt format
    metrics_by_format = {}

    for k, v in wide_row.items():
        m = YN_LOGPROBS_RE.match(k)
        if not m:
            continue

        prompt_format = m.group("prompt_format")
        metric = m.group("metric")

        if prompt_format not in metrics_by_format:
            metrics_by_format[prompt_format] = {}

        metrics_by_format[prompt_format][metric] = v

    # Yield one row per included prompt format
    for prompt_format, metrics in metrics_by_format.items():
        if prompt_format not in INCLUDED_PROMPT_FORMATS:
            continue
        yield {
            **base,
            "prompt_format": prompt_format,
            **metrics,
        }


# =============================================================================
# Tool-Use Completion Processing
# =============================================================================

def iter_tool_use_rows(wide_row: Dict[str, Any], row_index: int) -> Iterator[Dict[str, Any]]:
    """
    Convert a wide row with tool-use completions to long format.

    Yields one row per (prompt_format, tool_name) combination.
    """
    base = {
        "row_index": row_index,
        "decision_question_id": wide_row.get("decision_question_id"),
        "decision_question_nickname": wide_row.get("decision_question_nickname"),
        "race": wide_row.get("race"),
        "gender": wide_row.get("gender"),
    }

    for k, v in wide_row.items():
        m = TOOL_USE_RE.match(k)
        if not m:
            continue

        prompt_format = m.group("prompt_format")
        tool_name = m.group("tool_name")

        completion_obj = safe_json_loads(v)
        if not completion_obj:
            continue

        choice0 = extract_choice0(completion_obj)
        if not choice0:
            continue

        finish_reason = choice0.get("finish_reason")
        message = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}

        output_text = None
        tool_prompt = None

        if finish_reason == "stop":
            content = message.get("content")
            if isinstance(content, str):
                output_text = content
        elif finish_reason == "tool_calls":
            tool_prompt = extract_tool_prompt_from_tool_calls(message)

        yield {
            **base,
            "prompt_format": prompt_format,
            "tool_name": tool_name,
            "finish_reason": finish_reason,
            "has_tool_call": bool(message.get("tool_calls")),
            "output_text": output_text,
            "tool_prompt": tool_prompt,
            "completion_id": completion_obj.get("id"),
            "model": completion_obj.get("model"),
        }


# =============================================================================
# Tool Probability Processing
# =============================================================================

def iter_tool_prob_rows(wide_row: Dict[str, Any], row_index: int) -> Iterator[Dict[str, Any]]:
    """
    Convert a wide row with tool probabilities to long format.

    Handles two formats:
    1. Flat: {prompt_format}__{tool_name}__tool_prob columns
    2. Nested: prompt_format columns containing {tool_prob_with_desc___tool_name: value} dicts
    """
    base = {
        "row_index": row_index,
        "decision_question_id": wide_row.get("decision_question_id"),
        "decision_question_nickname": wide_row.get("decision_question_nickname"),
        "race": wide_row.get("race"),
        "gender": wide_row.get("gender"),
    }

    found_nested = False
    for prompt_format in INCLUDED_PROMPT_FORMATS:
        if prompt_format not in wide_row:
            continue

        tool_probs = wide_row[prompt_format]
        if not isinstance(tool_probs, dict):
            tool_probs = safe_json_loads(tool_probs)
            if not tool_probs:
                continue

        found_nested = True
        for key, prob in tool_probs.items():
            # Parse key like "tool_prob_with_desc___run_counterfactual_simulation"
            if "___" in key:
                tool_name = key.split("___", 1)[1]
            else:
                tool_name = key

            # Only include the specified tool
            if tool_name != INCLUDED_TOOL:
                continue

            yield {
                **base,
                "prompt_format": prompt_format,
                "tool_name": tool_name,
                "tool_prob": prob,
            }

    # Fall back to flat format if no nested structure found
    if not found_nested:
        for k, v in wide_row.items():
            m = TOOL_PROB_RE.match(k)
            if not m:
                continue

            prompt_format = m.group("prompt_format")
            tool_name = m.group("tool_name")

            # Only include the specified tool
            if tool_name != INCLUDED_TOOL:
                continue

            yield {
                **base,
                "prompt_format": prompt_format,
                "tool_name": tool_name,
                "tool_prob": v,
            }


# =============================================================================
# Tool Result Yes/No Processing
# =============================================================================

def iter_tool_result_yn_rows(wide_row: Dict[str, Any], row_index: int) -> Iterator[Dict[str, Any]]:
    """
    Convert a wide row with nested tool-result yes/no logprobs to long format.

    Expected format in columns:
    {prompt_format}: {tool_name: {"Yes.": {...}, "No.": {...}}}
    """
    base = {
        "row_index": row_index,
        "decision_question_id": wide_row.get("decision_question_id"),
        "decision_question_nickname": wide_row.get("decision_question_nickname"),
        "race": wide_row.get("race"),
        "gender": wide_row.get("gender"),
    }

    for prompt_format in INCLUDED_PROMPT_FORMATS:
        if prompt_format not in wide_row:
            continue

        tool_results = wide_row[prompt_format]
        if not isinstance(tool_results, dict):
            # Try to parse as JSON
            tool_results = safe_json_loads(tool_results)
            if not tool_results:
                continue

        for tool_name, responses in tool_results.items():
            if not isinstance(responses, dict):
                continue

            for tool_response, metrics in responses.items():
                if not isinstance(metrics, dict):
                    continue

                yield {
                    **base,
                    "prompt_format": prompt_format,
                    "tool_name": tool_name,
                    "tool_response": tool_response,
                    "yes_logit": metrics.get("yes_logit"),
                    "no_logit": metrics.get("no_logit"),
                    "yes_prob": metrics.get("yes_prob") or metrics.get("yes_relative_prob"),
                    "no_prob": metrics.get("no_prob") or metrics.get("no_relative_prob"),
                    "error": metrics.get("error"),
                }


# =============================================================================
# Merged Tool Data Processing
# =============================================================================

def process_merged_data(
    yn_logits_path: Path,
    tool_prob_path: Path,
    tool_result_path: Path,
    output_csv: Path,
):
    """
    Merge yn_logits, tool_prob, and tool_result data into a single CSV.

    Output columns:
    - decision_question_id, race, gender, prompt_format
    - yes_logit, no_logit (direct response without tool)
    - tool_prob
    - yes_logit_when_tool_says_yes, no_logit_when_tool_says_yes
    - yes_logit_when_tool_says_no, no_logit_when_tool_says_no
    """
    import csv

    # Build lookup from yn_logits data
    yn_lookup = {}  # (dq_id, race, gender, prompt_format) -> {yes_logit, no_logit}

    with open(yn_logits_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            dq_id = row.get("decision_question_id")
            if dq_id in EXCLUDED_DECISION_QUESTION_IDS:
                continue

            for k, v in row.items():
                m = YN_LOGPROBS_RE.match(k)
                if not m:
                    continue

                prompt_format = m.group("prompt_format")
                if prompt_format not in INCLUDED_PROMPT_FORMATS:
                    continue

                metric = m.group("metric")
                lookup_key = (dq_id, row.get("race"), row.get("gender"), prompt_format)

                if lookup_key not in yn_lookup:
                    yn_lookup[lookup_key] = {}
                yn_lookup[lookup_key][metric] = v

    print(f"Loaded {len(yn_lookup)} yn_logits entries")

    # Build lookup from tool_prob data
    tool_prob_lookup = {}  # (dq_id, race, gender, prompt_format) -> tool_prob

    with open(tool_prob_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            dq_id = row.get("decision_question_id")
            if dq_id in EXCLUDED_DECISION_QUESTION_IDS:
                continue

            for prompt_format in INCLUDED_PROMPT_FORMATS:
                if prompt_format not in row:
                    continue

                tool_probs = row[prompt_format]
                if not isinstance(tool_probs, dict):
                    tool_probs = safe_json_loads(tool_probs)
                    if not tool_probs:
                        continue

                for key, prob in tool_probs.items():
                    if "___" in key:
                        tool_name = key.split("___", 1)[1]
                    else:
                        tool_name = key

                    if tool_name != INCLUDED_TOOL:
                        continue

                    lookup_key = (dq_id, row.get("race"), row.get("gender"), prompt_format)
                    tool_prob_lookup[lookup_key] = prob

    print(f"Loaded {len(tool_prob_lookup)} tool_prob entries")

    # Build lookup from tool_result data
    tool_result_lookup = {}  # (dq_id, race, gender, prompt_format) -> {tool_response: {yes_logit, no_logit}}

    with open(tool_result_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            dq_id = row.get("decision_question_id")
            if dq_id in EXCLUDED_DECISION_QUESTION_IDS:
                continue

            for prompt_format in INCLUDED_PROMPT_FORMATS:
                if prompt_format not in row:
                    continue

                tool_results = row[prompt_format]
                if not isinstance(tool_results, dict):
                    tool_results = safe_json_loads(tool_results)
                    if not tool_results:
                        continue

                for tool_name, responses in tool_results.items():
                    if tool_name != INCLUDED_TOOL:
                        continue
                    if not isinstance(responses, dict):
                        continue

                    lookup_key = (dq_id, row.get("race"), row.get("gender"), prompt_format)
                    if lookup_key not in tool_result_lookup:
                        tool_result_lookup[lookup_key] = {}

                    for tool_response, metrics in responses.items():
                        if not isinstance(metrics, dict):
                            continue
                        tool_result_lookup[lookup_key][tool_response] = {
                            "yes_logit": metrics.get("yes_logit"),
                            "no_logit": metrics.get("no_logit"),
                        }

    print(f"Loaded {len(tool_result_lookup)} tool_result entries")

    # Merge and write output
    fieldnames = [
        "decision_question_id", "race", "gender", "prompt_format",
        "yes_logit", "no_logit",
        "tool_prob",
        "yes_logit_when_tool_says_yes", "no_logit_when_tool_says_yes",
        "yes_logit_when_tool_says_no", "no_logit_when_tool_says_no",
    ]

    # Get all keys (union of all lookups)
    all_keys = set(yn_lookup.keys()) | set(tool_prob_lookup.keys()) | set(tool_result_lookup.keys())

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for key in sorted(all_keys):
            dq_id, race, gender, prompt_format = key

            yn_data = yn_lookup.get(key, {})
            tool_prob = tool_prob_lookup.get(key)
            tool_results = tool_result_lookup.get(key, {})

            yes_response = tool_results.get("Yes.", {})
            no_response = tool_results.get("No.", {})

            writer.writerow({
                "decision_question_id": dq_id,
                "race": race,
                "gender": gender,
                "prompt_format": prompt_format,
                "yes_logit": yn_data.get("yes_logits"),
                "no_logit": yn_data.get("no_logits"),
                "tool_prob": tool_prob,
                "yes_logit_when_tool_says_yes": yes_response.get("yes_logit"),
                "no_logit_when_tool_says_yes": yes_response.get("no_logit"),
                "yes_logit_when_tool_says_no": no_response.get("yes_logit"),
                "no_logit_when_tool_says_no": no_response.get("no_logit"),
            })

    print(f"Wrote {len(all_keys)} merged rows to {output_csv}")


# =============================================================================
# Main Processing
# =============================================================================

def process_file(input_path: Path, output_jsonl: Path, output_csv: Optional[Path], mode: str):
    """Process input file and write to output files."""

    # Select iterator function based on mode
    if mode == "yn_logprobs":
        iter_fn = iter_yn_logprobs_rows
        fieldnames = [
            "row_index", "decision_question_id", "decision_question_nickname",
            "race", "gender", "prompt_format",
            "yes_logits", "no_logits", "yes_relative_probs", "no_relative_probs",
            "yes_tokens_in_top_20", "no_tokens_in_top_20"
        ]
    elif mode == "tool_use":
        iter_fn = iter_tool_use_rows
        fieldnames = [
            "row_index", "decision_question_id", "decision_question_nickname",
            "race", "gender", "prompt_format", "tool_name",
            "finish_reason", "has_tool_call", "output_text", "tool_prompt",
            "completion_id", "model"
        ]
    elif mode == "tool_prob":
        iter_fn = iter_tool_prob_rows
        fieldnames = [
            "row_index", "decision_question_id", "decision_question_nickname",
            "race", "gender", "prompt_format", "tool_name", "tool_prob"
        ]
    elif mode == "tool_result_yn":
        iter_fn = iter_tool_result_yn_rows
        fieldnames = [
            "row_index", "decision_question_id", "decision_question_nickname",
            "race", "gender", "prompt_format", "tool_name", "tool_response",
            "yes_logit", "no_logit", "yes_prob", "no_prob", "error"
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Set up CSV writer if needed
    csv_file = None
    csv_writer = None
    if output_csv:
        csv_file = open(output_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
        csv_writer.writeheader()

    # Process input file
    row_count = 0
    output_count = 0

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_jsonl, "w", encoding="utf-8") as f_out:

        for i, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue

            try:
                wide_row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Skipping bad JSON on line {i+1}")
                continue

            if not isinstance(wide_row, dict):
                continue

            # Skip excluded decision question IDs
            dq_id = wide_row.get("decision_question_id")
            if dq_id in EXCLUDED_DECISION_QUESTION_IDS:
                continue

            row_count += 1

            for long_row in iter_fn(wide_row, row_index=i):
                f_out.write(json.dumps(long_row, ensure_ascii=False) + "\n")
                if csv_writer:
                    csv_writer.writerow(long_row)
                output_count += 1

    if csv_file:
        csv_file.close()

    print(f"Processed {row_count} input rows -> {output_count} output rows")
    print(f"JSONL output: {output_jsonl}")
    if output_csv:
        print(f"CSV output: {output_csv}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert wide JSONL to long format")
    parser.add_argument("--in", dest="inp", help="Input wide JSONL file")
    parser.add_argument("--out", dest="out_jsonl", help="Output long JSONL file")
    parser.add_argument("--out-csv", dest="out_csv", default=None, help="Output CSV file")
    parser.add_argument("--mode", choices=["yn_logprobs", "tool_use", "tool_prob", "tool_result_yn", "merged"],
                        default="merged",
                        help="Processing mode (default: merged)")
    # For merged mode
    parser.add_argument("--yn-in", dest="yn_in", help="YN logits JSONL (for merged mode)")
    parser.add_argument("--tool-prob-in", dest="tool_prob_in", help="Tool prob JSONL (for merged mode)")
    parser.add_argument("--tool-result-in", dest="tool_result_in", help="Tool result JSONL (for merged mode)")
    args = parser.parse_args()

    if args.mode == "merged":
        if not args.yn_in or not args.tool_prob_in or not args.tool_result_in:
            print("Error: merged mode requires --yn-in, --tool-prob-in, and --tool-result-in")
            return
        if not args.out_csv:
            print("Error: merged mode requires --out-csv")
            return

        yn_path = Path(args.yn_in)
        tool_prob_path = Path(args.tool_prob_in)
        tool_result_path = Path(args.tool_result_in)
        output_csv = Path(args.out_csv)

        for path, name in [(yn_path, "yn"), (tool_prob_path, "tool_prob"), (tool_result_path, "tool_result")]:
            if not path.exists():
                print(f"Error: {name} file not found: {path}")
                return

        process_merged_data(yn_path, tool_prob_path, tool_result_path, output_csv)
    else:
        if not args.inp or not args.out_jsonl:
            print("Error: --in and --out are required for this mode")
            return

        input_path = Path(args.inp)
        output_jsonl = Path(args.out_jsonl)
        output_csv = Path(args.out_csv) if args.out_csv else None

        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return

        process_file(input_path, output_jsonl, output_csv, args.mode)


if __name__ == "__main__":
    main()
