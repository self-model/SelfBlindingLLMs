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
    python aggregate_batch_runs.py "sycophancy/results/2026-01-12 GPT Third Person"
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np


def find_run_files(input_dir: Path) -> list[Path]:
    """
    Find all run files in a directory.

    Looks for files matching pattern *_run##.jsonl
    Returns list of paths sorted by run index.
    """
    pattern = re.compile(r'_run(\d+)\.jsonl$')

    run_files = []
    for f in input_dir.glob('*.jsonl'):
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
    # Remove _run##.jsonl suffix
    base = re.sub(r'_run\d+\.jsonl$', '', name)
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
        'run_idx', 'scenario_id', 'idx'
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
    exclude = set(numeric_cols) | {'run_idx', 'prompt'}  # prompt is too long to group by efficiently

    groupby_cols = []
    for col in df.columns:
        if col not in exclude:
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

    # Also keep first value of prompt if it exists
    if 'prompt' in df_all.columns:
        agg_dict['prompt'] = 'first'

    # Group and aggregate
    df_agg = df_all.groupby(groupby_cols, as_index=False).agg(agg_dict)

    # Flatten column names
    new_columns = []
    for col in df_agg.columns:
        if isinstance(col, tuple):
            if col[1] == '' or col[1] == 'mean' or col[1] == 'first':
                new_columns.append(col[0])  # Keep original name (groupby cols have '', mean/first keep original)
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
        'tool_use_probs': Has made_tool_call__* boolean column
        'tool_result_logits': Has nested tool_results dict column
        'third_person': Has label_a_logit column (third person experiment)
        'standard': Default numeric logits/probs format
    """
    if 'made_tool_call__run_counterfactual_simulation' in df.columns:
        return 'tool_use_probs'
    elif 'tool_results' in df.columns:
        return 'tool_result_logits'
    elif 'label_a_logit' in df.columns:
        return 'third_person'
    else:
        return 'standard'


def rename_third_person_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename confusing third_person column names.

    The original names 'label_a_*' and 'label_b_*' are misleading because they
    sound like they refer to version A/B, but they actually refer to the first
    and second letters in the letter_pair (positional, not semantic).

    Renames:
        label_a_* → first_label_*
        label_b_* → second_label_*
    """
    rename_map = {}
    for col in df.columns:
        if col.startswith('label_a_'):
            new_name = col.replace('label_a_', 'first_label_')
            rename_map[col] = new_name
        elif col.startswith('label_b_'):
            new_name = col.replace('label_b_', 'second_label_')
            rename_map[col] = new_name

    if rename_map:
        print(f"  Renaming columns: {rename_map}")
        df = df.rename(columns=rename_map)

    return df


def flatten_tool_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten nested tool_results column into separate columns.

    Extracts:
    - tool_results.run_counterfactual_simulation.{you_letter}.you_logit → you_letter_you_logit
    - tool_results.run_counterfactual_simulation.{them_letter}.you_logit → them_letter_you_logit
    - etc.
    """
    df = df.copy()

    # Extract nested values
    def extract_tool_result_values(row):
        results = {}
        tool_results = row.get('tool_results')
        if not tool_results or not isinstance(tool_results, dict):
            return results

        sim_results = tool_results.get('run_counterfactual_simulation', {})
        if not sim_results:
            return results

        you_letter = row.get('you_maps_to_letter')
        them_letter = row.get('them_maps_to_letter')

        # Extract values for each letter, using semantic names
        for letter, letter_name in [(you_letter, 'you_letter'), (them_letter, 'them_letter')]:
            if letter and letter in sim_results:
                letter_data = sim_results[letter]
                for key, value in letter_data.items():
                    results[f'{letter_name}_{key}'] = value

        return results

    # Apply extraction to each row
    extracted = df.apply(extract_tool_result_values, axis=1, result_type='expand')

    # Concatenate with original dataframe (excluding tool_results column)
    cols_to_keep = [c for c in df.columns if c != 'tool_results']
    df_flat = pd.concat([df[cols_to_keep], extracted], axis=1)

    return df_flat


def aggregate_tool_use_probs(df_all: pd.DataFrame, groupby_cols: list[str]) -> pd.DataFrame:
    """
    Aggregate tool-use data: compute tool call rate and response distribution.

    Returns dataframe with:
    - tool_call_rate: proportion of runs that made a tool call (0.0 to 1.0)
    - tool_call_count: number of runs that made a tool call
    - response_you_count: count of "You" responses (when no tool call)
    - response_them_count: count of "Them" responses (when no tool call)
    - response_you_rate: rate of "You" among text responses
    - response_them_rate: rate of "Them" among text responses
    """
    df = df_all.copy()
    n_runs = df['run_idx'].nunique()

    # Convert boolean to int for aggregation
    tool_call_col = 'made_tool_call__run_counterfactual_simulation'
    df['_tool_call_int'] = df[tool_call_col].astype(int)

    # Count response types (only when no tool call)
    response_col = 'response_content__run_counterfactual_simulation'
    df['_response_you'] = ((~df[tool_call_col]) & (df[response_col] == 'You')).astype(int)
    df['_response_them'] = ((~df[tool_call_col]) & (df[response_col] == 'Them')).astype(int)
    df['_no_tool_call'] = (~df[tool_call_col]).astype(int)

    # Build aggregation dict
    agg_dict = {
        '_tool_call_int': 'sum',
        '_response_you': 'sum',
        '_response_them': 'sum',
        '_no_tool_call': 'sum',
    }

    # Keep first value of prompt if exists
    if 'prompt' in df.columns:
        agg_dict['prompt'] = 'first'

    # Group and aggregate
    df_agg = df.groupby(groupby_cols, as_index=False).agg(agg_dict)

    # Flatten column names (groupby columns become tuples with empty second element)
    new_columns = []
    for col in df_agg.columns:
        if isinstance(col, tuple):
            new_columns.append(col[0])  # Just use the first element
        else:
            new_columns.append(col)
    df_agg.columns = new_columns

    # Rename and compute derived columns
    df_agg = df_agg.rename(columns={
        '_tool_call_int': 'tool_call_count',
        '_response_you': 'response_you_count',
        '_response_them': 'response_them_count',
        '_no_tool_call': 'no_tool_call_count',
    })

    # Compute rates
    df_agg['tool_call_rate'] = df_agg['tool_call_count'] / n_runs
    df_agg['response_you_rate'] = df_agg['response_you_count'] / df_agg['no_tool_call_count'].replace(0, np.nan)
    df_agg['response_them_rate'] = df_agg['response_them_count'] / df_agg['no_tool_call_count'].replace(0, np.nan)

    # Add metadata
    df_agg['n_runs'] = n_runs

    return df_agg


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
        print("Expected files matching pattern *_run##.jsonl")
        return 1

    print(f"  Found {len(run_files)} run files")

    # Determine output prefix
    prefix = args.prefix or extract_base_name(run_files)
    print(f"  Base name: {prefix}")

    # Load and concatenate
    print("\nLoading run files...")
    df_all = load_and_concat_runs(run_files)
    print(f"  Total rows: {len(df_all)}")
    print(f"  Columns: {list(df_all.columns)}")

    # Detect format
    data_format = detect_format(df_all)
    print(f"\nDetected format: {data_format}")

    # Handle format-specific preprocessing and aggregation
    if data_format == 'tool_result_logits':
        # Flatten nested tool_results before aggregation
        print("Flattening nested tool_results...")
        df_all = flatten_tool_results(df_all)
        print(f"  Flattened columns: {[c for c in df_all.columns if c.startswith('you_letter_') or c.startswith('them_letter_')]}")

        # Now aggregate as standard
        numeric_cols = identify_numeric_columns(df_all)
        groupby_cols = identify_groupby_columns(df_all, numeric_cols)
        print(f"\nNumeric columns to aggregate: {numeric_cols}")
        print(f"Groupby columns: {groupby_cols}")

        df_agg = aggregate_runs(df_all, numeric_cols, groupby_cols)

    elif data_format == 'third_person':
        # Rename confusing label_a/label_b columns to first_label/second_label
        print("Renaming third_person columns...")
        df_all = rename_third_person_columns(df_all)

        # Now aggregate as standard
        numeric_cols = identify_numeric_columns(df_all)
        groupby_cols = identify_groupby_columns(df_all, numeric_cols)
        print(f"\nNumeric columns to aggregate: {numeric_cols}")
        print(f"Groupby columns: {groupby_cols}")

        df_agg = aggregate_runs(df_all, numeric_cols, groupby_cols)

    elif data_format == 'tool_use_probs':
        # Aggregate tool-use boolean data
        numeric_cols = []  # No standard numeric columns
        groupby_cols = identify_groupby_columns(df_all, numeric_cols)
        # Remove columns that shouldn't be grouped by
        groupby_cols = [c for c in groupby_cols if not c.startswith('made_tool_call__')
                        and not c.startswith('tool_call_args__')
                        and not c.startswith('response_content__')]
        print(f"Groupby columns: {groupby_cols}")

        df_agg = aggregate_tool_use_probs(df_all, groupby_cols)

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

    # Show sample of aggregation stats based on format
    if data_format == 'tool_use_probs':
        print(f"\nTool-use aggregation stats:")
        print(f"  Mean tool_call_rate: {df_agg['tool_call_rate'].mean():.4f}")
        if 'response_you_rate' in df_agg.columns:
            you_rate = df_agg['response_you_rate'].dropna().mean()
            them_rate = df_agg['response_them_rate'].dropna().mean()
            print(f"  Mean response_you_rate (when no tool): {you_rate:.4f}")
            print(f"  Mean response_them_rate (when no tool): {them_rate:.4f}")
    else:
        print(f"\nSample aggregation stats:")
        for col in numeric_cols[:3]:
            if col not in df_agg.columns:
                continue
            mean_val = df_agg[col].mean()
            std_col = f"{col}_std"
            if std_col in df_agg.columns:
                mean_std = df_agg[std_col].mean()
                print(f"  {col}: mean={mean_val:.4f}, avg_std_across_conditions={mean_std:.4f}")
            else:
                print(f"  {col}: mean={mean_val:.4f}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
