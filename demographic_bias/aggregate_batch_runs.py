#!/usr/bin/env python3
"""
Aggregate multiple batch run files into single output files.

Given a directory of run files (e.g., *_run00.jsonl, *_run01.jsonl, ...),
produces two output files:
  1. *_aggregated.jsonl - means across runs (same format as single run)
  2. *_all_runs.jsonl - all rows concatenated with run_idx column

Usage:
    python aggregate_batch_runs.py <input_dir> [--output_dir <dir>] [--prefix <prefix>]

Example:
    python aggregate_batch_runs.py "results/2026-01-01 Gpt 4.1 YN logits Tamkin Revised"
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np


def find_run_files(input_dir: Path) -> list[Path]:
    """
    Find all run files in a directory.

    Looks for files matching pattern *_run##.jsonl or *_run###.jsonl
    (but not *_batch.jsonl or *_batch.results.jsonl).
    Returns list of paths sorted by run index.
    """
    # Match _run followed by 2 or 3 digits at end of filename
    pattern = re.compile(r'_run(\d{2,3})\.jsonl$')

    run_files = []
    for f in input_dir.glob('*.jsonl'):
        # Skip batch files
        if '_batch' in f.name:
            continue
        match = pattern.search(f.name)
        if match:
            run_idx = int(match.group(1))
            run_files.append((run_idx, f))

    # Sort by run index
    run_files.sort(key=lambda x: x[0])

    return [f for _, f in run_files]


def extract_base_name(run_files: list[Path]) -> str:
    """
    Extract the base name from run files (without _run## suffix).
    """
    if not run_files:
        raise ValueError("No run files provided")

    name = run_files[0].name
    # Remove _run##.jsonl or _run###.jsonl suffix
    base = re.sub(r'_run\d{2,3}\.jsonl$', '', name)
    return base


def load_and_concat_runs(run_files: list[Path]) -> pd.DataFrame:
    """
    Load all run files and concatenate with run_idx column.
    """
    dfs = []
    for run_idx, f in enumerate(run_files):
        df = pd.read_json(f, lines=True)
        df['run_idx'] = run_idx
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Convert list columns to tuples (for hashability in groupby)
    for col in df_all.columns:
        if df_all[col].dtype == object:
            # Check if first non-null value is a list
            first_val = df_all[col].dropna().iloc[0] if len(df_all[col].dropna()) > 0 else None
            if isinstance(first_val, list):
                df_all[col] = df_all[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    return df_all


def identify_numeric_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify columns that should be aggregated (numeric, varying across runs).

    Excludes ID columns and columns that are constant across conditions.
    """
    # Common numeric columns to aggregate
    numeric_patterns = [
        'logit', 'prob', 'logprob', 'score'
    ]

    # Columns to never aggregate (even if numeric)
    exclude_columns = {
        'run_idx', 'scenario_id', 'idx', 'decision_question_id'
    }

    numeric_cols = []
    for col in df.columns:
        if col in exclude_columns:
            continue
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            # Check if column name matches patterns or is actually varying
            if any(p in col.lower() for p in numeric_patterns):
                numeric_cols.append(col)
            elif df[col].nunique() > 1:
                # Include other numeric columns that vary
                numeric_cols.append(col)

    return numeric_cols


def identify_groupby_columns(df: pd.DataFrame, numeric_cols: list[str]) -> list[str]:
    """
    Identify columns to group by (non-numeric, non-run_idx).

    These are the condition identifiers that should be identical across runs.
    """
    # Columns that are too long or variable to group by
    exclude_patterns = [
        'template', 'prompt', 'completion_json'
    ]
    exclude = set(numeric_cols) | {'run_idx'}

    groupby_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        # Skip columns matching exclude patterns
        if any(p in col.lower() for p in exclude_patterns):
            continue
        groupby_cols.append(col)

    return groupby_cols


def aggregate_runs(df_all: pd.DataFrame, numeric_cols: list[str], groupby_cols: list[str]) -> pd.DataFrame:
    """
    Aggregate across runs, computing mean (and optionally std/se) for numeric columns.
    """
    # Build aggregation dict
    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = ['mean', 'std']

    # Group and aggregate
    df_agg = df_all.groupby(groupby_cols, as_index=False).agg(agg_dict)

    # Flatten column names
    new_columns = []
    for col in df_agg.columns:
        if isinstance(col, tuple):
            if col[1] == '' or col[1] == 'mean' or col[1] == 'first':
                new_columns.append(col[0])  # Keep original name
            elif col[1] == 'std':
                new_columns.append(f"{col[0]}_std")
            else:
                new_columns.append(f"{col[0]}_{col[1]}")
        else:
            new_columns.append(col)

    df_agg.columns = new_columns

    # Add count and SE columns
    n_runs = df_all['run_idx'].nunique()
    for col in numeric_cols:
        std_col = f"{col}_std"
        if std_col in df_agg.columns:
            df_agg[f"{col}_se"] = df_agg[std_col] / np.sqrt(n_runs)

    # Add metadata
    df_agg['n_runs'] = n_runs

    return df_agg


# =============================================================================
# Format Detection and Special Handlers
# =============================================================================

def detect_format(df: pd.DataFrame) -> str:
    """
    Detect data format from columns.

    Returns:
        'gpt_yn_logits': Has {prompt}_prompt_yes_logits columns
        'gpt_tool_probs': Has {prompt}__{tool}__completion_json columns
        'gpt_tool_result': Has prompt columns containing nested tool result dicts
        'standard': Default format
    """
    cols = df.columns.tolist()

    # Check for GPT YN logits format
    if any('_prompt_yes_logits' in c for c in cols):
        return 'gpt_yn_logits'

    # Check for GPT tool probs format (completion_json columns)
    if any('__completion_json' in c for c in cols):
        return 'gpt_tool_probs'

    # Check for GPT tool result format (prompt columns with nested dicts)
    # These have prompt_format columns (like 'default', 'dont_discriminate') containing dicts
    prompt_formats = ['default', 'dont_discriminate', 'ignore', 'if_you_didnt_know',
                      'remove_in_context', 'removed', 'redacted']
    if any(pf in cols for pf in prompt_formats):
        # Check if one of these columns contains a dict
        for pf in prompt_formats:
            if pf in cols:
                first_val = df[pf].dropna().iloc[0] if len(df[pf].dropna()) > 0 else None
                if isinstance(first_val, dict):
                    return 'gpt_tool_result'

    return 'standard'


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


def parse_completion_json(completion_str: Any) -> Dict[str, Any]:
    """
    Parse a completion JSON and extract relevant info.

    Returns dict with:
    - made_tool_call: bool
    - finish_reason: str
    - response_content: str or None
    - tool_name: str or None (if tool call was made)
    """
    result = {
        'made_tool_call': False,
        'finish_reason': None,
        'response_content': None,
        'tool_name': None,
    }

    completion = safe_json_loads(completion_str)
    if not completion:
        return result

    choices = completion.get('choices', [])
    if not choices:
        return result

    choice = choices[0]
    result['finish_reason'] = choice.get('finish_reason')

    message = choice.get('message', {})
    if message.get('tool_calls'):
        result['made_tool_call'] = True
        # Extract tool name from first tool call
        tool_calls = message['tool_calls']
        if tool_calls and isinstance(tool_calls, list):
            first_call = tool_calls[0]
            if isinstance(first_call, dict) and 'function' in first_call:
                result['tool_name'] = first_call['function'].get('name')
    else:
        result['response_content'] = message.get('content')

    return result


def flatten_gpt_tool_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten GPT tool probs format: parse completion_json columns and extract
    whether a tool call was made for each prompt/tool combination.

    Creates columns:
    - {prompt}__{tool}__made_tool_call: bool
    - {prompt}__{tool}__response_content: str (when no tool call)
    """
    df = df.copy()

    # Find all completion_json columns
    completion_cols = [c for c in df.columns if '__completion_json' in c]

    for col in completion_cols:
        # Parse the column name: {prompt}__{tool}__completion_json
        parts = col.replace('__completion_json', '').split('__')
        if len(parts) != 2:
            continue
        prompt_format, tool_name = parts

        # Parse each completion and extract info
        parsed = df[col].apply(parse_completion_json)

        # Create new columns
        df[f'{prompt_format}__{tool_name}__made_tool_call'] = parsed.apply(lambda x: x['made_tool_call'])
        df[f'{prompt_format}__{tool_name}__response_content'] = parsed.apply(lambda x: x['response_content'])

    return df


def aggregate_gpt_tool_probs(df_all: pd.DataFrame, groupby_cols: list[str]) -> pd.DataFrame:
    """
    Aggregate GPT tool-use data: compute tool call rate and response distribution.

    Returns dataframe with:
    - {prompt}__{tool}__tool_call_rate: proportion of runs that made a tool call
    - {prompt}__{tool}__tool_call_count: number of runs that made a tool call
    """
    df = df_all.copy()
    n_runs = df['run_idx'].nunique()

    # Find all made_tool_call columns
    tool_call_cols = [c for c in df.columns if '__made_tool_call' in c]

    # Build aggregation dict
    agg_dict = {}
    for col in tool_call_cols:
        # Convert bool to int for summing
        df[f'{col}_int'] = df[col].astype(int)
        agg_dict[f'{col}_int'] = 'sum'

    # Group and aggregate
    df_agg = df.groupby(groupby_cols, as_index=False).agg(agg_dict)

    # Flatten column names and compute rates
    new_columns = []
    for col in df_agg.columns:
        if isinstance(col, tuple):
            new_columns.append(col[0])
        else:
            new_columns.append(col)
    df_agg.columns = new_columns

    # Rename count columns and add rate columns
    for col in tool_call_cols:
        count_col = f'{col}_int'
        if count_col in df_agg.columns:
            base = col.replace('__made_tool_call', '')
            df_agg = df_agg.rename(columns={count_col: f'{base}__tool_call_count'})
            df_agg[f'{base}__tool_call_rate'] = df_agg[f'{base}__tool_call_count'] / n_runs

    # Add metadata
    df_agg['n_runs'] = n_runs

    return df_agg


def flatten_gpt_tool_result(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten GPT tool result format: extract yes/no logits from nested dicts.

    Input columns like 'default' contain:
    {"run_counterfactual_simulation": {"No.": {...}, "Yes.": {...}}}

    Output columns:
    - {prompt}__{tool}__Yes__yes_logit
    - {prompt}__{tool}__Yes__no_logit
    - {prompt}__{tool}__No__yes_logit
    - {prompt}__{tool}__No__no_logit
    """
    df = df.copy()

    prompt_formats = ['default', 'dont_discriminate', 'ignore', 'if_you_didnt_know',
                      'remove_in_context', 'removed', 'redacted']

    for pf in prompt_formats:
        if pf not in df.columns:
            continue

        # Extract values from nested dicts
        def extract_nested(row):
            results = {}
            val = row.get(pf)
            if not isinstance(val, dict):
                return results

            for tool_name, tool_results in val.items():
                if not isinstance(tool_results, dict):
                    continue

                for response_type, metrics in tool_results.items():
                    if not isinstance(metrics, dict):
                        continue

                    # Clean up response type (e.g., "Yes." -> "Yes")
                    resp = response_type.rstrip('.')

                    for metric_name, metric_val in metrics.items():
                        col_name = f'{pf}__{tool_name}__{resp}__{metric_name}'
                        results[col_name] = metric_val

            return results

        extracted = df.apply(extract_nested, axis=1, result_type='expand')

        # Drop original column and add extracted columns
        df = df.drop(columns=[pf])
        df = pd.concat([df, extracted], axis=1)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multiple batch run files into single output files."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing run files (*_run##.jsonl)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as input_dir)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Output file prefix (default: auto-detect from input files)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find run files
    print(f"Scanning {input_dir} for run files...")
    run_files = find_run_files(input_dir)

    if not run_files:
        print(f"ERROR: No run files found in {input_dir}")
        print("Expected files matching pattern *_run##.jsonl (excluding *_batch.jsonl)")
        return 1

    print(f"  Found {len(run_files)} run files")

    # Determine output prefix
    prefix = args.prefix or extract_base_name(run_files)
    print(f"  Base name: {prefix}")

    # Load and concatenate
    print("\nLoading run files...")
    df_all = load_and_concat_runs(run_files)
    print(f"  Total rows: {len(df_all)}")
    print(f"  Columns: {list(df_all.columns)[:10]}...")  # Show first 10

    # Detect format
    data_format = detect_format(df_all)
    print(f"\nDetected format: {data_format}")

    # Handle format-specific preprocessing and aggregation
    if data_format == 'gpt_yn_logits':
        # Standard aggregation for YN logits
        numeric_cols = identify_numeric_columns(df_all)
        groupby_cols = identify_groupby_columns(df_all, numeric_cols)
        print(f"\nNumeric columns to aggregate: {numeric_cols[:5]}...")
        print(f"Groupby columns: {groupby_cols}")

        df_agg = aggregate_runs(df_all, numeric_cols, groupby_cols)

    elif data_format == 'gpt_tool_probs':
        # Flatten completion_json columns to extract tool call info
        print("Flattening GPT tool probs format...")
        df_all = flatten_gpt_tool_probs(df_all)

        # Find groupby columns (exclude the new extracted columns)
        numeric_cols = []
        groupby_cols = identify_groupby_columns(df_all, numeric_cols)
        # Also exclude the new tool call columns from groupby
        groupby_cols = [c for c in groupby_cols
                        if '__made_tool_call' not in c
                        and '__response_content' not in c
                        and '__completion_json' not in c]
        print(f"Groupby columns: {groupby_cols}")

        df_agg = aggregate_gpt_tool_probs(df_all, groupby_cols)

    elif data_format == 'gpt_tool_result':
        # Flatten nested tool result dicts
        print("Flattening GPT tool result format...")
        df_all = flatten_gpt_tool_result(df_all)
        print(f"  Flattened columns sample: {[c for c in df_all.columns if '__Yes__' in c or '__No__' in c][:5]}...")

        # Now aggregate as standard
        numeric_cols = identify_numeric_columns(df_all)
        groupby_cols = identify_groupby_columns(df_all, numeric_cols)
        print(f"\nNumeric columns to aggregate: {numeric_cols[:5]}...")
        print(f"Groupby columns: {groupby_cols}")

        df_agg = aggregate_runs(df_all, numeric_cols, groupby_cols)

    else:  # standard
        # Identify columns
        numeric_cols = identify_numeric_columns(df_all)
        groupby_cols = identify_groupby_columns(df_all, numeric_cols)

        print(f"\nNumeric columns to aggregate: {numeric_cols}")
        print(f"Groupby columns: {groupby_cols}")

        df_agg = aggregate_runs(df_all, numeric_cols, groupby_cols)

    print(f"\nAggregated to {len(df_agg)} rows")

    # Save all_runs file
    all_runs_path = output_dir / f"{prefix}_all_runs.jsonl"
    print(f"\nSaving all runs to {all_runs_path}...")
    df_all.to_json(all_runs_path, orient='records', lines=True, force_ascii=False)
    print(f"  Saved {len(df_all)} rows")

    # Save aggregated file
    agg_path = output_dir / f"{prefix}_aggregated.jsonl"
    print(f"\nSaving aggregated data to {agg_path}...")
    df_agg.to_json(agg_path, orient='records', lines=True, force_ascii=False)
    print(f"  Saved {len(df_agg)} rows")

    # Summary stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Input: {len(run_files)} run files from {input_dir}")
    print(f"Format: {data_format}")
    print(f"Output:")
    print(f"  {all_runs_path.name}: {len(df_all)} rows (all runs, with run_idx)")
    print(f"  {agg_path.name}: {len(df_agg)} rows (aggregated across {df_all['run_idx'].nunique()} runs)")

    # Show sample stats based on format
    if data_format == 'gpt_tool_probs':
        rate_cols = [c for c in df_agg.columns if '__tool_call_rate' in c]
        if rate_cols:
            print(f"\nTool call rate stats:")
            for col in rate_cols[:3]:
                print(f"  {col}: mean={df_agg[col].mean():.4f}")
    elif data_format == 'gpt_yn_logits':
        logit_cols = [c for c in df_agg.columns if '_yes_logits' in c and '_std' not in c and '_se' not in c]
        if logit_cols:
            print(f"\nSample yes_logits stats:")
            for col in logit_cols[:3]:
                print(f"  {col}: mean={df_agg[col].mean():.4f}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
