import os
from pathlib import Path
import yaml
import torch

def load_model_configs(config_path: str | Path | None = None) -> dict:
    """
    Load model configurations from YAML file or use built-in defaults.

    Args:
        config_path: Optional path to YAML config file. If None, tries
                    'model_config.yaml' in current directory, then falls
                    back to built-in defaults.

    Returns:
        Dict mapping model names to their configurations with torch dtypes.
    """
    # Try to load from YAML file
    yaml_path = Path(config_path) if config_path else Path("model_config.yaml")

    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            MODEL_CONFIGS = yaml.safe_load(f)
        # Convert dtype strings to torch dtypes
        for model, settings in MODEL_CONFIGS.items():
            if isinstance(settings.get('dtype'), str):
                settings['dtype'] = getattr(torch, settings['dtype'])
        return MODEL_CONFIGS
    else:
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

def clear_gpu_memory(model=None, tokenizer=None):
    import gc
    """
    Aggressively clears GPU memory and Python garbage.

    Args:
        model: The model object to delete (optional)
        tokenizer: The tokenizer object to delete (optional)
    """
    # Delete the model and tokenizer if provided
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer

    # Run garbage collection multiple times to ensure cleanup
    for _ in range(3):
        gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Reset peak memory stats for cleaner tracking
        torch.cuda.reset_peak_memory_stats()

        # Report current memory usage
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    print("  ✓ Memory cleared")

def set_random_seed(seed: int = 42):
    import random
    import numpy as np
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"  ✓ Random seed set to {seed}")

def set_determinism():
    """Set PyTorch to deterministic mode for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print("  ✓ PyTorch set to deterministic mode")

def save_to_gdrive(local_filename, gdrive_dir='/content/drive/MyDrive/Oxford/self_blinding/outputs'):
    """Copy a local file to Google Drive."""
    import os
    import shutil
    os.makedirs(gdrive_dir, exist_ok=True)
    gdrive_path = os.path.join(gdrive_dir, os.path.basename(local_filename))
    shutil.copy2(local_filename, gdrive_path)
    print(f"Saved to Google Drive: {gdrive_path}")

def mount_gdrive():
    """Mount Google Drive in Colab environment."""
    from google.colab import drive
    drive.mount('/content/drive')

def normalize_dataset_columns(data):
    """
    Normalize dataset column names for compatibility with Tamkin et al. (2023) dataset from HF hub.

    Renames:
    - 'text' -> 'filled_template' (if filled_template doesn't exist)
    - 'scenario_type' -> 'decision_question_id' (if decision_question_id doesn't exist)

    Returns the modified dataset.
    """
    if 'text' in data.column_names and 'filled_template' not in data.column_names:
        data = data.rename_column('text', 'filled_template')
    if 'scenario_type' in data.column_names and 'decision_question_id' not in data.column_names:
        data = data.rename_column('scenario_type', 'decision_question_id')
    return data


def load_data(path: str, name: str = None, split: str = "train", normalize: bool = True):
    """
    Load data from HuggingFace Hub, local parquet, or local JSONL.

    Args:
        path: HF Hub dataset name (e.g., "Anthropic/discrim-eval")
              or local file path (.parquet, .jsonl, .json)
        name: Dataset configuration/subset name for HF Hub datasets
              (e.g., "explicit" for Anthropic/discrim-eval)
        split: Dataset split to return (default: "train")
        normalize: Whether to apply normalize_dataset_columns (default: True)

    Returns:
        HuggingFace Dataset with optionally normalized column names

    Examples:
        data = load_data("Anthropic/discrim-eval", "explicit")  # HF Hub with config
        data = load_data("Anthropic/discrim-eval")              # HF Hub default config
        data = load_data("scenarios.parquet")                   # Local parquet
        data = load_data("outputs/results.jsonl")               # Local JSONL
    """
    from datasets import load_dataset

    path_obj = Path(path)

    # Check if it's a HF Hub dataset (contains / but no file extension)
    is_hub = '/' in path and not path_obj.suffix

    if is_hub:
        data = load_dataset(path, name=name, split=split)
    elif path_obj.suffix == '.parquet':
        data = load_dataset("parquet", data_files=path, split=split)
    elif path_obj.suffix in ('.jsonl', '.json'):
        data = load_dataset("json", data_files=path, split=split)
    else:
        raise ValueError(f"Unsupported file format: {path_obj.suffix}. "
                         "Supported: .parquet, .jsonl, .json, or HF Hub path")

    if normalize:
        data = normalize_dataset_columns(data)

    return data
