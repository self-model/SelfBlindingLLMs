"""Shared configuration for sycophancy experiments."""
from pathlib import Path

# Repository root directory (parent of sycophancy/)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Default path to sycophancy scenarios data file
DEFAULT_SYCOPHANCY_DATA = REPO_ROOT / "sycophancy" / "data" / "sycophancy-two-sides-eval.jsonl"

# Default path to tool prompts YAML file
DEFAULT_TOOL_PROMPTS_PATH = REPO_ROOT / "src" / "tool_prompts.yaml"
