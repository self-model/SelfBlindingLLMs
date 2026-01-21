# inference.py
"""Inference utilities: model loading, tokenization, and scoring functions."""

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, set_seed
from utils import load_model_configs
from scoring import score_yes_no_from_logits

MODEL_CONFIGS = load_model_configs()


# =============================================================================
# Model Loading
# =============================================================================

def install_bitsandbytes():
    import subprocess
    subprocess.run(["pip", "install", "-q", "-U", "bitsandbytes"], check=True)

def load_model_and_tokenizer(model_name: str, use_cache: bool = False, max_length: int | None = None):
    """
    Factory function to load a model and tokenizer with specific patches
    for different architectures (e.g., Gemma 3 vs Gemma 2).
    """
    print(f"Loading {model_name}...")

    set_seed(42)

    config_entry = MODEL_CONFIGS.get(model_name)

    if config_entry is None:
        raise ValueError("Model configuration must be specified")

    dtype = config_entry['dtype']

    # Raise exception if dtype is not supported
    if dtype == torch.bfloat16:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                f"CRITICAL: Model '{model_name}' requires `bfloat16` for numerical stability, "
                f"but your GPU ({torch.cuda.get_device_name()}) does not support it.\n"
                "Aborting to prevent silent failures (NaNs/Garbage output)."
            )

    # 1. Load Config
    config = AutoConfig.from_pretrained(model_name)
    # Sync config internal type with loading type
    config.dtype = dtype

    # 2. Apply Architecture-Specific Patches
    if hasattr(config, 'text_config'):
        # Nested config
        config.text_config.use_cache = use_cache
        if max_length is not None:
            config.text_config.max_position_embeddings = max_length
    else:
        # Flat config
        config.use_cache = use_cache
        if max_length is not None:
            config.max_position_embeddings = max_length

    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # ONLY clamp the tokenizer if the user explicitly asked for it
    print(f"Tokenizer default max length: {tokenizer.model_max_length}")
    if max_length is not None:
        tokenizer.model_max_length = max_length
        print(f"Tokenizer max length clamped to: {tokenizer.model_max_length}")

    # 4. Load Model
    load_8bit = config_entry.get('load_in_8bit', False)
    quantization_config = None
    if load_8bit:
        install_bitsandbytes()
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        dtype=dtype,
        device_map={'': 0} if load_8bit else 'auto', # For quantized models, force to GPU 0 (bug where 'auto' will load to CPU)
        low_cpu_mem_usage=True,
        quantization_config=quantization_config
    ).eval()

    print(f"Model {model_name} loaded on {model.device} with context {max_length}")

    # Set seed for reproducible results
    from utils import set_random_seed, set_determinism
    set_random_seed(42)
    set_determinism()

    return model, tokenizer


# =============================================================================
# Token Utilities
# =============================================================================

def tool_use_start_token_id(model_name: str):
    return MODEL_CONFIGS[model_name]['tool_use_start_token_id']

def get_token(tokenizer, token_str: str):
    token_ids = tokenizer.encode(token_str, add_special_tokens=False)
    assert len(token_ids) == 1, f"Token '{token_str}' tokenizes to multiple tokens: " + str([f"{token}: {tokenizer.decode(token)}" for token in token_ids])
    return token_ids[0]

def get_yes_no_token_ids(tokenizer, use_variants: bool = True):
    if use_variants:
        yes_token_strings = ["yes", " yes", "Yes", " Yes", "YES", " YES"]
        no_token_strings = ["no", " no", "No", " No", "NO", " NO"]
    else:
        yes_token_strings = ["yes"]
        no_token_strings = ["no"]
    yes_token_id_map = {}
    no_token_id_map = {}

    # Variants for "yes"
    for variant in yes_token_strings:
        try:
            token_id = get_token(tokenizer, variant)
            yes_token_id_map[variant] = token_id
        except AssertionError:
            continue

    # Variants for "no"
    for variant in no_token_strings:
        try:
            token_id = get_token(tokenizer, variant)
            no_token_id_map[variant] = token_id
        except AssertionError:
            continue

    return yes_token_id_map, no_token_id_map


def get_you_them_token_ids(tokenizer, use_variants: bool = True):
    """
    Get token IDs for You/Them variants.

    Args:
        tokenizer: The tokenizer
        use_variants: If True, include case and spacing variants

    Returns:
        Tuple of (you_token_id_map, them_token_id_map) dicts
    """
    if use_variants:
        you_token_strings = ["You", "you", " You", " you"]
        them_token_strings = ["Them", "them", " Them", " them"]
    else:
        you_token_strings = ["You"]
        them_token_strings = ["Them"]

    you_token_id_map = {}
    them_token_id_map = {}

    # Variants for "You"
    for variant in you_token_strings:
        try:
            token_id = get_token(tokenizer, variant)
            you_token_id_map[variant] = token_id
        except AssertionError:
            continue

    # Variants for "Them"
    for variant in them_token_strings:
        try:
            token_id = get_token(tokenizer, variant)
            them_token_id_map[variant] = token_id
        except AssertionError:
            continue

    return you_token_id_map, them_token_id_map


# =============================================================================
# Inference Functions
# =============================================================================

def get_yes_no_logits(prompt_str, tokenizer, model, yes_token_ids, no_token_ids):
    """
    Score yes/no logits from a prompt string.

    Args:
        prompt_str: The prompt after apply_chat_template (string)
        tokenizer: The tokenizer
        model: The model
        yes_token_ids: Dict or list of token IDs for "yes" variants
        no_token_ids: Dict or list of token IDs for "no" variants

    Returns:
        dict with keys: yes_logit, no_logit, yes_prob, no_prob
    """
    # Tokenize and move to device
    enc = tokenizer(prompt_str, return_tensors='pt').to(model.device)

    with torch.inference_mode():
        full_logits = model(**enc).logits[:, -1, :]

    # Use shared scoring module (handles both dict and list inputs)
    return score_yes_no_from_logits(full_logits.squeeze(0), yes_token_ids, no_token_ids)


def get_you_them_logits(prompt_str, tokenizer, model, you_token_ids, them_token_ids):
    """
    Score You/Them logits from a prompt string.

    Args:
        prompt_str: The prompt after apply_chat_template (string)
        tokenizer: The tokenizer
        model: The model
        you_token_ids: Dict or list of token IDs for "You" variants
        them_token_ids: Dict or list of token IDs for "Them" variants

    Returns:
        dict with keys: you_logit, them_logit, you_prob, them_prob
    """
    from scoring import score_you_them_from_logits

    # Tokenize and move to device
    enc = tokenizer(prompt_str, return_tensors='pt').to(model.device)

    with torch.inference_mode():
        full_logits = model(**enc).logits[:, -1, :]

    # Use shared scoring module (handles both dict and list inputs)
    return score_you_them_from_logits(full_logits.squeeze(0), you_token_ids, them_token_ids)


def get_tool_use_prob(prompt_str, tokenizer, model, tool_start_token_id):
    """
    Get probability of tool-use start token.

    Args:
        prompt_str: The prompt after apply_chat_template (string)
        tokenizer: The tokenizer
        model: The model
        tool_start_token_id: Token ID for the tool-use start token

    Returns:
        float: Probability of tool-use start token
    """
    # Tokenize and move to device
    enc = tokenizer(prompt_str, return_tensors='pt').to(model.device)

    with torch.inference_mode():
        full_logits = model(**enc).logits[:, -1, :]

    probs = torch.softmax(full_logits, dim=-1)
    tool_prob = probs[0, tool_start_token_id].item()

    return tool_prob


def generate_text(prompt_strs, tokenizer, model, max_new_tokens=512, temperature=1.0,
                  num_return_sequences=1, do_sample=True):
    """
    Generate text completions.

    Args:
        prompt_strs: Single prompt string or list of prompt strings
        tokenizer: The tokenizer
        model: The model
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (ignored if do_sample=False)
        num_return_sequences: Number of sequences to return per prompt
        do_sample: Whether to sample (True) or use greedy decoding (False)

    Returns:
        List of generated text strings (one per prompt * num_return_sequences)
    """
    # Handle single string input
    if isinstance(prompt_strs, str):
        prompt_strs = [prompt_strs]

    # Set up tokenizer for batched generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize with padding for batched input
    enc = tokenizer(prompt_strs, return_tensors='pt', padding=True).to(model.device)
    prompt_length = enc.input_ids.shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Restore original padding side
    tokenizer.padding_side = original_padding_side

    # Decode each generation (exclude prompt tokens, truncate at EOS)
    eos_token_id = tokenizer.eos_token_id
    generations = []

    for output in outputs:
        gen_ids = output[prompt_length:].tolist()

        # Truncate at first EOS token
        first_eos_idx = _find_first_eos_index(gen_ids, eos_token_id)
        if first_eos_idx is not None:
            gen_ids = gen_ids[:first_eos_idx + 1]

        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        generations.append(gen_text)

    return generations


def _find_first_eos_index(token_ids, eos_token_id):
    """Find index of first EOS token. eos_token_id can be int or list of ints."""
    if isinstance(eos_token_id, int):
        eos_token_ids = {eos_token_id}
    else:
        eos_token_ids = set(eos_token_id)

    for i, tok in enumerate(token_ids):
        if tok in eos_token_ids:
            return i
    return None
