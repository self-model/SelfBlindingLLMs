"""
Shared scoring utilities for binary logprob extraction.

Two paradigms:
1. Token IDs (Qwen/local models) - extract from logits tensor
2. Token Strings (ChatGPT API) - extract from top_logprobs list

Both use logsumexp to aggregate multiple token variants (e.g., "Yes", "yes", " Yes")
before computing binary probabilities via softmax.
"""

import torch
import torch.nn.functional as F
from typing import Sequence


# =============================================================================
# Token Variants
# =============================================================================

def get_token_variants(word: str) -> list[str]:
    """
    Generate token variants with case and leading space variations.

    Args:
        word: Any token (e.g., "yes", "You", "D")

    Returns:
        List of 4 variants: [lowercase, capitalized, " lowercase", " capitalized"]
        Example: "yes" -> ["yes", "Yes", " yes", " Yes"]
    """
    lower = word.lower()
    cap = word.capitalize()
    return [lower, cap, f" {lower}", f" {cap}"]


# =============================================================================
# Core: logsumexp binary probability calculation
# =============================================================================

def logsumexp_binary_probs(
    logprobs_a: Sequence[float],
    logprobs_b: Sequence[float],
) -> dict:
    """
    Compute binary probabilities from two sets of logprobs using logsumexp.

    Args:
        logprobs_a: Logprobs for first category (e.g., Yes/You/label_a)
        logprobs_b: Logprobs for second category (e.g., No/Them/label_b)

    Returns:
        Dict with:
            - a_logit: Combined logprob for category A
            - b_logit: Combined logprob for category B
            - a_prob: Normalized probability for category A
            - b_prob: Normalized probability for category B
            - a_count: Number of tokens found for category A
            - b_count: Number of tokens found for category B

    Raises:
        ValueError: If either category has no logprobs
    """
    a_lps = list(logprobs_a)
    b_lps = list(logprobs_b)

    if not a_lps or not b_lps:
        raise ValueError("Both categories must have at least one logprob")

    a_count = len(a_lps)
    b_count = len(b_lps)

    # Logsumexp to aggregate variants
    a_combined = torch.logsumexp(torch.tensor(a_lps), dim=-1)
    b_combined = torch.logsumexp(torch.tensor(b_lps), dim=-1)

    # Stack for softmax
    combined = torch.stack([a_combined, b_combined])
    probs = F.softmax(combined, dim=-1)

    return {
        'a_logit': a_combined.item(),
        'b_logit': b_combined.item(),
        'a_prob': probs[0].item(),
        'b_prob': probs[1].item(),
        'a_count': a_count,
        'b_count': b_count,
    }


# =============================================================================
# Token ID Paradigm (Qwen/local models)
# =============================================================================

def score_binary_from_logits(
    logits: torch.Tensor,
    token_ids_a: Sequence[int],
    token_ids_b: Sequence[int],
) -> dict:
    """
    Score binary choice from model logits tensor using token IDs.

    Used for local models (Qwen, etc.) where we have direct access to the
    full logits tensor and can index by token ID.

    Args:
        logits: 1D tensor of logits for all tokens (shape: [vocab_size])
        token_ids_a: Token IDs for first category (e.g., Yes variants)
        token_ids_b: Token IDs for second category (e.g., No variants)

    Returns:
        Dict with a_logit, b_logit, a_prob, b_prob, a_count, b_count
    """
    # Extract logits for each category
    a_logits = [logits[tid].item() for tid in token_ids_a]
    b_logits = [logits[tid].item() for tid in token_ids_b]

    return logsumexp_binary_probs(a_logits, b_logits)


# =============================================================================
# Token String Paradigm (ChatGPT API)
# =============================================================================

def _extract_logprobs_by_tokens(
    top_logprobs: list[dict],
    target_tokens: Sequence[str],
    normalize: bool = True,
) -> list[float]:
    """
    Extract logprobs for target tokens from ChatGPT top_logprobs.

    Args:
        top_logprobs: List of dicts with 'token' and 'logprob' keys
        target_tokens: Tokens to match (e.g., ["yes", " yes", "Yes"])
        normalize: If True, normalize tokens by stripping whitespace for matching

    Returns:
        List of logprobs for matched tokens
    """
    matched = []
    target_set = set(target_tokens)
    target_normalized = {t.strip().lower() for t in target_tokens} if normalize else set()

    for cand in top_logprobs:
        tok = cand['token']
        tok_normalized = tok.strip().lower() if normalize else None

        # Direct match or normalized match
        if tok in target_set or (normalize and tok_normalized in target_normalized):
            matched.append(cand['logprob'])

    return matched


def score_binary_from_top_logprobs(
    top_logprobs: list[dict],
    tokens_a: Sequence[str],
    tokens_b: Sequence[str],
    normalize: bool = True,
) -> dict:
    """
    Score binary choice from ChatGPT top_logprobs list.

    Used for ChatGPT API responses where we only have access to the
    top-k (typically 20) tokens with their logprobs.

    Args:
        top_logprobs: List of dicts with 'token' and 'logprob' keys
        tokens_a: Token strings for first category
        tokens_b: Token strings for second category
        normalize: If True, normalize tokens for matching (strip + lowercase)

    Returns:
        Dict with a_logit, b_logit, a_prob, b_prob, a_count, b_count
    """
    a_lps = _extract_logprobs_by_tokens(top_logprobs, tokens_a, normalize)
    b_lps = _extract_logprobs_by_tokens(top_logprobs, tokens_b, normalize)

    if not a_lps and not b_lps:
        raise ValueError("Neither category found in top_logprobs")

    # Use floor for missing categories
    if not a_lps or not b_lps:
        logprob_floor = min(c['logprob'] for c in top_logprobs)
        if not a_lps:
            a_lps = [logprob_floor]
        if not b_lps:
            b_lps = [logprob_floor]

    return logsumexp_binary_probs(a_lps, b_lps)


# =============================================================================
# Convenience Functions for Common Token Pairs
# =============================================================================

def score_yes_no_from_top_logprobs(
    top_logprobs: list[dict],
) -> dict:
    """
    Score Yes/No from ChatGPT top_logprobs.

    Returns:
        Dict with yes_logit, no_logit, yes_prob, no_prob,
        yes_tokens_in_top_20, no_tokens_in_top_20
    """
    result = score_binary_from_top_logprobs(
        top_logprobs,
        tokens_a=get_token_variants("yes"),
        tokens_b=get_token_variants("no"),
    )

    return {
        'yes_logit': result['a_logit'],
        'no_logit': result['b_logit'],
        'yes_prob': result['a_prob'],
        'no_prob': result['b_prob'],
        'yes_tokens_in_top_20': result['a_count'],
        'no_tokens_in_top_20': result['b_count'],
    }


def score_you_them_from_top_logprobs(
    top_logprobs: list[dict],
) -> dict:
    """
    Score You/Them from ChatGPT top_logprobs.

    Returns:
        Dict with you_logit, them_logit, you_prob, them_prob,
        you_tokens_in_top_20, them_tokens_in_top_20
    """
    result = score_binary_from_top_logprobs(
        top_logprobs,
        tokens_a=get_token_variants("you"),
        tokens_b=get_token_variants("them"),
    )

    return {
        'you_logit': result['a_logit'],
        'them_logit': result['b_logit'],
        'you_prob': result['a_prob'],
        'them_prob': result['b_prob'],
        'you_tokens_in_top_20': result['a_count'],
        'them_tokens_in_top_20': result['b_count'],
    }


def score_letters_from_top_logprobs(
    top_logprobs: list[dict],
    label_a: str,
    label_b: str,
) -> dict:
    """
    Score any letter pair from ChatGPT top_logprobs.

    Third-person experiments use randomly assigned letters (D, E, F, etc.)
    rather than fixed A/B. This function generates variants dynamically.

    Args:
        top_logprobs: List of dicts with 'token' and 'logprob' keys
        label_a: First letter label (e.g., "D")
        label_b: Second letter label (e.g., "E")

    Returns:
        Dict with label_a_logit, label_b_logit, label_a_prob, label_b_prob,
        label_a_tokens_in_top_20, label_b_tokens_in_top_20
    """
    a_variants = get_token_variants(label_a)
    b_variants = get_token_variants(label_b)

    result = score_binary_from_top_logprobs(
        top_logprobs,
        tokens_a=a_variants,
        tokens_b=b_variants,
        normalize=False,  # Letters need exact matching (case-sensitive variants)
    )

    return {
        'label_a_logit': result['a_logit'],
        'label_b_logit': result['b_logit'],
        'label_a_prob': result['a_prob'],
        'label_b_prob': result['b_prob'],
        'label_a_tokens_in_top_20': result['a_count'],
        'label_b_tokens_in_top_20': result['b_count'],
    }


# =============================================================================
# Token ID Convenience Functions
# =============================================================================

def score_yes_no_from_logits(
    logits: torch.Tensor,
    yes_token_ids: dict | Sequence[int],
    no_token_ids: dict | Sequence[int],
) -> dict:
    """
    Score Yes/No from model logits using token IDs.

    Args:
        logits: 1D tensor of logits
        yes_token_ids: Token IDs for Yes (dict values or sequence)
        no_token_ids: Token IDs for No (dict values or sequence)

    Returns:
        Dict with yes_logit, no_logit, yes_prob, no_prob
    """
    # Handle dict input (from get_yes_no_token_ids)
    yes_ids = list(yes_token_ids.values()) if isinstance(yes_token_ids, dict) else list(yes_token_ids)
    no_ids = list(no_token_ids.values()) if isinstance(no_token_ids, dict) else list(no_token_ids)

    result = score_binary_from_logits(logits, yes_ids, no_ids)

    return {
        'yes_logit': result['a_logit'],
        'no_logit': result['b_logit'],
        'yes_prob': result['a_prob'],
        'no_prob': result['b_prob'],
    }


def score_you_them_from_logits(
    logits: torch.Tensor,
    you_token_ids: dict | Sequence[int],
    them_token_ids: dict | Sequence[int],
) -> dict:
    """
    Score You/Them from model logits using token IDs.

    Args:
        logits: 1D tensor of logits
        you_token_ids: Token IDs for You (dict values or sequence)
        them_token_ids: Token IDs for Them (dict values or sequence)

    Returns:
        Dict with you_logit, them_logit, you_prob, them_prob
    """
    you_ids = list(you_token_ids.values()) if isinstance(you_token_ids, dict) else list(you_token_ids)
    them_ids = list(them_token_ids.values()) if isinstance(them_token_ids, dict) else list(them_token_ids)

    result = score_binary_from_logits(logits, you_ids, them_ids)

    return {
        'you_logit': result['a_logit'],
        'them_logit': result['b_logit'],
        'you_prob': result['a_prob'],
        'them_prob': result['b_prob'],
    }


def score_letters_from_logits(
    logits: torch.Tensor,
    label_a_token_ids: dict | Sequence[int],
    label_b_token_ids: dict | Sequence[int],
) -> dict:
    """
    Score letter pair from model logits using token IDs.

    Args:
        logits: 1D tensor of logits
        label_a_token_ids: Token IDs for label A (dict values or sequence)
        label_b_token_ids: Token IDs for label B (dict values or sequence)

    Returns:
        Dict with label_a_logit, label_b_logit, label_a_prob, label_b_prob
    """
    a_ids = list(label_a_token_ids.values()) if isinstance(label_a_token_ids, dict) else list(label_a_token_ids)
    b_ids = list(label_b_token_ids.values()) if isinstance(label_b_token_ids, dict) else list(label_b_token_ids)

    result = score_binary_from_logits(logits, a_ids, b_ids)

    return {
        'label_a_logit': result['a_logit'],
        'label_b_logit': result['b_logit'],
        'label_a_prob': result['a_prob'],
        'label_b_prob': result['b_prob'],
    }
