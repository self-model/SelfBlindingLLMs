#!/usr/bin/env python3
"""
Convert wide JSONL inference results to long format and/or CSV.

Processes inference outputs from:
- yn_logprobs_*.py: Yes/no scoring for each prompt strategy
- tool_use_probs_*.py: Tool-use probability measurement
- tool_result_yn_logprobs_*.py: Yes/no scoring after simulated tool use

By default, loads data from OSF for reproducibility. Use --data-path to
load from a local folder instead.

Usage:
    # Load from OSF (default) - merged mode
    python build_csv.py --model GPT-4.1
    python build_csv.py --model Qwen2.5-7B-Instruct

    # Load from local folder
    python build_csv.py --model GPT-4.1 --data-path ./results/

    # Legacy: explicit file paths
    python build_csv.py --yn-in FILE --tool-prob-in FILE --tool-result-in FILE --out-csv OUTPUT

    # Convert yes/no logprobs to long format (non-merged mode)
    python build_csv.py --in results/bias_yn_gpt-4.1.jsonl --out results/bias_yn_gpt-4.1_long.jsonl --mode yn_logprobs

    # Process tool-use results (non-merged mode)
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

# Matches aggregated tool call rate: {prompt_format}__{tool_name}__tool_call_rate
TOOL_CALL_RATE_RE = re.compile(r"^(?P<prompt_format>.+)__(?P<tool_name>.+)__tool_call_rate$")

# Matches aggregated tool result: {prompt_format}__{tool_name}__{response}__{metric}
# e.g., default__run_counterfactual_simulation__Yes__yes_logit
TOOL_RESULT_FLAT_RE = re.compile(
    r"^(?P<prompt_format>.+)__(?P<tool_name>.+)__(?P<response>Yes|No)__(?P<metric>yes_logit|no_logit|yes_relative_prob|no_relative_prob)$"
)

# Decision question IDs to exclude (inverted or problematic questions)
EXCLUDED_DECISION_QUESTION_IDS = {23, 54, 65, 67, 77}

# Prompt formats to include (excludes redact_in_context, redacted which weren't used in paper)
INCLUDED_PROMPT_FORMATS = {'default', 'dont_discriminate', 'ignore', 'if_you_didnt_know',
                           'remove_in_context', 'removed'}

# Tool to include (focus on counterfactual simulation)
INCLUDED_TOOL = 'run_counterfactual_simulation'


# =============================================================================
# OSF Configuration
# =============================================================================

# OSF file IDs for direct downloads.
# Project: https://osf.io/udk5a/
# Browse: https://osf.io/udk5a/files/osfstorage -> demographic_bias/{model}/
OSF_FILE_IDS = {
    "GPT-4.1": {
        "yn_logits": "6976507b7f5caaf4d7834678",
        "tool_prob": "6976507bfd6c714eae8345ed",
        "tool_result": "697650794bf50c960a44fe48",
    },
    "Qwen2.5-7B-Instruct": {
        "yn_logits": "697698ad9b49fe625065c6ab",
        "tool_prob": "697698af53876d3356f22662",
        "tool_result": "697698cb9b49fe625065c6b1",
    },
}

OUTPUT_FOLDER = Path(__file__).parent / "results"

# File type patterns for local file discovery
FILE_TYPES = ["yn_logits", "tool_prob", "tool_result"]


def get_osf_url(file_id: str) -> str:
    """Convert OSF file ID to download URL."""
    return f"https://osf.io/download/{file_id}/"


def resolve_file_paths(model: str, data_path: str | None) -> dict[str, str]:
    """Resolve file paths for a model, either from OSF or local folder.

    Args:
        model: Model name (e.g., "gpt-4.1" or "qwen2.5-7b-instruct")
        data_path: Optional local folder path. If None, uses OSF.

    Returns:
        Dict mapping file type to path/URL
    """
    if data_path is None:
        # Use OSF URLs
        file_ids = OSF_FILE_IDS[model]
        return {ft: get_osf_url(file_ids[ft]) for ft in FILE_TYPES}

    # Local folder - look for files matching expected patterns
    folder = Path(data_path)
    paths = {}

    # Pattern mappings for local files (order matters - more specific patterns first)
    # Note: tool_use_logprobs is tool_result, not tool_prob
    pattern_map = {
        "yn_logits": ["yn_logits", "yn_logprobs", "bias_yn"],
        "tool_prob": ["tool_prob", "tool_probs", "tool_use_probs", "tool_calls"],
        "tool_result": ["tool_result", "tool_results", "tool_result_yn", "tool_use_logprobs", "response_logprobs"],
    }

    for file_type in FILE_TYPES:
        patterns = pattern_map[file_type]
        found = False

        # Search for files matching any pattern
        for pattern in patterns:
            candidates = list(folder.glob(f"*{pattern}*.jsonl"))
            if candidates:
                # Prefer aggregated files if multiple matches
                aggregated = [c for c in candidates if "aggregated" in c.name]
                paths[file_type] = str(aggregated[0] if aggregated else candidates[0])
                found = True
                break

        if not found:
            raise FileNotFoundError(
                f"Could not find {file_type} file in {folder}. "
                f"Tried patterns: {patterns}"
            )

    return paths


# =============================================================================
# URL/File Loading Support
# =============================================================================

import tempfile
import urllib.request
from contextlib import contextmanager


@contextmanager
def open_file_or_url(path: str, encoding: str = "utf-8"):
    """Open a local file or download from URL.

    Args:
        path: Local file path or URL
        encoding: Text encoding (default: utf-8)

    Yields:
        File-like object for reading
    """
    if path.startswith("http://") or path.startswith("https://"):
        # Download URL to temp file
        with urllib.request.urlopen(path) as response:
            with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as tmp:
                tmp.write(response.read())
                tmp_path = tmp.name
        try:
            with open(tmp_path, 'r', encoding=encoding) as f:
                yield f
        finally:
            Path(tmp_path).unlink()
    else:
        with open(path, 'r', encoding=encoding) as f:
            yield f


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

def detect_aggregated_format(file_path: str) -> bool:
    """
    Detect if the file uses aggregated format (flat columns with tool_call_rate or flattened tool result).

    Args:
        file_path: Local path or URL to the file

    Returns True if aggregated format, False if raw format.
    """
    with open_file_or_url(file_path) as f:
        first_line = f.readline().strip()
        if not first_line:
            return False
        row = json.loads(first_line)

    # Check for aggregated tool prob format
    if any(TOOL_CALL_RATE_RE.match(k) for k in row.keys()):
        return True

    # Check for aggregated tool result format
    if any(TOOL_RESULT_FLAT_RE.match(k) for k in row.keys()):
        return True

    return False


def load_yn_logits(yn_logits_path: str) -> dict:
    """
    Load YN logits data into lookup dict.

    Works with both raw and aggregated formats (same column pattern).

    Args:
        yn_logits_path: Local path or URL to the file

    Returns: {(dq_id, race, gender, prompt_format): {metric: value}}
    """
    yn_lookup = {}

    with open_file_or_url(yn_logits_path) as f:
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

    return yn_lookup


def load_tool_prob_nested(tool_prob_path: str) -> dict:
    """
    Load tool prob data from raw format (nested dicts).

    Args:
        tool_prob_path: Local path or URL to the file

    Returns: {(dq_id, race, gender, prompt_format): tool_prob}
    """
    tool_prob_lookup = {}

    with open_file_or_url(tool_prob_path) as f:
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

    return tool_prob_lookup


def load_tool_prob_aggregated(tool_prob_path: str) -> dict:
    """
    Load tool prob data from aggregated format (flat columns with tool_call_rate).

    Args:
        tool_prob_path: Local path or URL to the file

    Returns: {(dq_id, race, gender, prompt_format): tool_prob}
    """
    tool_prob_lookup = {}

    with open_file_or_url(tool_prob_path) as f:
        for line in f:
            row = json.loads(line.strip())
            dq_id = row.get("decision_question_id")
            if dq_id in EXCLUDED_DECISION_QUESTION_IDS:
                continue

            for k, v in row.items():
                m = TOOL_CALL_RATE_RE.match(k)
                if not m:
                    continue

                prompt_format = m.group("prompt_format")
                tool_name = m.group("tool_name")

                if prompt_format not in INCLUDED_PROMPT_FORMATS:
                    continue
                if tool_name != INCLUDED_TOOL:
                    continue

                lookup_key = (dq_id, row.get("race"), row.get("gender"), prompt_format)
                tool_prob_lookup[lookup_key] = v

    return tool_prob_lookup


def load_tool_result_nested(tool_result_path: str) -> dict:
    """
    Load tool result data from raw format (nested dicts).

    Args:
        tool_result_path: Local path or URL to the file

    Returns: {(dq_id, race, gender, prompt_format): {tool_response: {yes_logit, no_logit}}}
    """
    tool_result_lookup = {}

    with open_file_or_url(tool_result_path) as f:
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

    return tool_result_lookup


def load_tool_result_aggregated(tool_result_path: str) -> dict:
    """
    Load tool result data from aggregated format (flat columns).

    Columns like: default__run_counterfactual_simulation__Yes__yes_logit

    Args:
        tool_result_path: Local path or URL to the file

    Returns: {(dq_id, race, gender, prompt_format): {tool_response: {yes_logit, no_logit}}}
    """
    tool_result_lookup = {}

    with open_file_or_url(tool_result_path) as f:
        for line in f:
            row = json.loads(line.strip())
            dq_id = row.get("decision_question_id")
            if dq_id in EXCLUDED_DECISION_QUESTION_IDS:
                continue

            for k, v in row.items():
                m = TOOL_RESULT_FLAT_RE.match(k)
                if not m:
                    continue

                prompt_format = m.group("prompt_format")
                tool_name = m.group("tool_name")
                response = m.group("response")  # "Yes" or "No"
                metric = m.group("metric")  # "yes_logit", "no_logit", etc.

                if prompt_format not in INCLUDED_PROMPT_FORMATS:
                    continue
                if tool_name != INCLUDED_TOOL:
                    continue

                # Normalize response to match expected format (e.g., "Yes." or "No.")
                tool_response = response + "."

                lookup_key = (dq_id, row.get("race"), row.get("gender"), prompt_format)
                if lookup_key not in tool_result_lookup:
                    tool_result_lookup[lookup_key] = {}
                if tool_response not in tool_result_lookup[lookup_key]:
                    tool_result_lookup[lookup_key][tool_response] = {}

                # Map metric name (e.g., "yes_logit" -> "yes_logit")
                tool_result_lookup[lookup_key][tool_response][metric] = v

    return tool_result_lookup


def process_merged_data(
    yn_logits_path: str,
    tool_prob_path: str,
    tool_result_path: str,
    output_csv: Path,
):
    """
    Merge yn_logits, tool_prob, and tool_result data into a single CSV.

    Handles both raw and aggregated formats by auto-detecting the format.

    Output columns:
    - decision_question_id, race, gender, prompt_format
    - yes_logit, no_logit (direct response without tool)
    - tool_prob
    - yes_logit_when_tool_says_yes, no_logit_when_tool_says_yes
    - yes_logit_when_tool_says_no, no_logit_when_tool_says_no
    """
    import csv

    # Load YN logits (same format for raw and aggregated)
    yn_lookup = load_yn_logits(yn_logits_path)
    print(f"Loaded {len(yn_lookup)} yn_logits entries")

    # Detect and load tool_prob data
    tool_prob_aggregated = detect_aggregated_format(tool_prob_path)
    if tool_prob_aggregated:
        print("Detected aggregated tool_prob format")
        tool_prob_lookup = load_tool_prob_aggregated(tool_prob_path)
    else:
        print("Detected raw tool_prob format")
        tool_prob_lookup = load_tool_prob_nested(tool_prob_path)
    print(f"Loaded {len(tool_prob_lookup)} tool_prob entries")

    # Detect and load tool_result data
    tool_result_aggregated = detect_aggregated_format(tool_result_path)
    if tool_result_aggregated:
        print("Detected aggregated tool_result format")
        tool_result_lookup = load_tool_result_aggregated(tool_result_path)
    else:
        print("Detected raw tool_result format")
        tool_result_lookup = load_tool_result_nested(tool_result_path)
    print(f"Loaded {len(tool_result_lookup)} tool_result entries")

    # Merge and write output
    fieldnames = [
        "decision_question_id", "race", "gender", "prompt_format",
        "yes_logit", "no_logit",
        "tool_prob",
        "yes_logit_when_tool_says_yes", "no_logit_when_tool_says_yes",
        "yes_logit_when_tool_says_no", "no_logit_when_tool_says_no",
    ]

    # Use yn_logits as the primary source (left join)
    # This ensures we keep all (dq_id, race, gender, prompt_format) combinations from yn_logits
    # even if tool_prob or tool_result data is missing (will be NaN)
    all_keys = set(yn_lookup.keys())
    print(f"Keys from yn_logits: {len(all_keys)} (prob: {len(tool_prob_lookup)}, result: {len(tool_result_lookup)})")

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
    parser = argparse.ArgumentParser(
        description="Convert wide JSONL to long format. "
        "By default, loads data from OSF for reproducibility."
    )
    parser.add_argument("--in", dest="inp", help="Input wide JSONL file")
    parser.add_argument("--out", dest="out_jsonl", help="Output long JSONL file")
    parser.add_argument("--out-csv", dest="out_csv", default=None, help="Output CSV file")
    parser.add_argument("--mode", choices=["yn_logprobs", "tool_use", "tool_prob", "tool_result_yn", "merged"],
                        default="merged",
                        help="Processing mode (default: merged)")
    # For merged mode - new OSF-based arguments
    parser.add_argument(
        "--model",
        type=str,
        choices=["GPT-4.1", "Qwen2.5-7B-Instruct"],
        help="Model to process (for merged mode with OSF or local folder)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Local folder containing input JSONL files. If not specified, loads from OSF.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output folder for processed CSV. Default: ./results/",
    )
    # Legacy arguments for backward compatibility
    parser.add_argument("--yn-in", dest="yn_in", help="YN logits JSONL (legacy, for merged mode)")
    parser.add_argument("--tool-prob-in", dest="tool_prob_in", help="Tool prob JSONL (legacy, for merged mode)")
    parser.add_argument("--tool-result-in", dest="tool_result_in", help="Tool result JSONL (legacy, for merged mode)")
    args = parser.parse_args()

    if args.mode == "merged":
        if args.model:
            # New OSF-based loading (or local folder with --data-path)
            source = "OSF" if args.data_path is None else args.data_path
            print(f"Loading data for model: {args.model} (from {source})")

            paths = resolve_file_paths(args.model, args.data_path)
            for file_type, path in paths.items():
                print(f"  {file_type}: {path}")

            output_folder = Path(args.output_path) if args.output_path else OUTPUT_FOLDER
            output_folder.mkdir(parents=True, exist_ok=True)
            output_csv = output_folder / f"demographic_bias_processed_{args.model}.csv"

            process_merged_data(
                paths["yn_logits"],
                paths["tool_prob"],
                paths["tool_result"],
                output_csv,
            )
        elif args.yn_in and args.tool_prob_in and args.tool_result_in:
            # Legacy: explicit file paths
            if not args.out_csv:
                print("Error: legacy merged mode requires --out-csv")
                return

            yn_path = Path(args.yn_in)
            tool_prob_path = Path(args.tool_prob_in)
            tool_result_path = Path(args.tool_result_in)
            output_csv = Path(args.out_csv)

            for path, name in [(yn_path, "yn"), (tool_prob_path, "tool_prob"), (tool_result_path, "tool_result")]:
                if not path.exists():
                    print(f"Error: {name} file not found: {path}")
                    return

            process_merged_data(str(yn_path), str(tool_prob_path), str(tool_result_path), output_csv)
        else:
            print("Error: merged mode requires --model OR (--yn-in, --tool-prob-in, --tool-result-in, --out-csv)")
            return
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
