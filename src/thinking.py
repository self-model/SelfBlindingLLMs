"""
Thinking-mode integration for inference scripts.

Provides ThinkingConfig and render_with_thinking() — a unified prompt-rendering
function that handles both thinking-off (passthrough to apply_chat_template)
and thinking-on (generate-then-measure) for supported model families.

Currently supported families:
  - qwen3:  off-path + on-path implemented
  - gemma4: off-path implemented (suppresses thinking via enable_thinking=False);
            on-path is a stub (raises NotImplementedError)

To add a new family (e.g. gemini): write a `_<family>_render_with_thinking`
function with the same signature/return shape as `_qwen3_render_with_thinking`
and add a dispatch case in `render_with_thinking`. Then set
`thinking_family: <family>` in `model_config.yaml` for each model that uses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class ThinkingConfig:
    """Per-run thinking-mode configuration.

    Attributes:
        mode: 'off' = standard apply_chat_template behavior. 'on' = generate
            a thinking trace and then assemble the final prompt for measurement.
        budget: Max tokens to generate during the thinking phase. -1 = unlimited
            (capped at safety_max_tokens for runaway protection).
        temperature: Generation temperature for the thinking trace. 0 = greedy.
        n_samples: Number of trace samples per condition. Only meaningful when
            temperature > 0; the inference script's outer loop is responsible
            for invoking render_with_thinking n_samples times and writing a
            separate _run{NNN}.jsonl per sample.
        safety_max_tokens: Hard cap when budget == -1.
    """
    mode: Literal['off', 'on'] = 'off'
    budget: int = -1
    temperature: float = 0.0
    n_samples: int = 1
    safety_max_tokens: int = 8192


def get_thinking_family(model_name: str) -> str | None:
    """Look up the thinking_family field for a model. Returns None if not set
    in model_config.yaml (which means the model doesn't support thinking)."""
    # Lazy import to avoid circular dependency at module-load time.
    from src.inference import MODEL_CONFIGS
    return MODEL_CONFIGS.get(model_name, {}).get('thinking_family')


def render_with_thinking(
    model_name: str,
    model,
    tokenizer,
    conversation: list[dict],
    *,
    tools: list | None = None,
    add_generation_prompt: bool = False,
    continue_final_message: bool = False,
    thinking_config: ThinkingConfig,
) -> tuple[str, dict]:
    """
    Render a prompt for logit measurement with thinking-mode controls.

    Args:
        model_name: HF model name; looks up thinking_family from MODEL_CONFIGS.
        model: Loaded HF model. Only used when thinking_config.mode == 'on'.
        tokenizer: HF tokenizer.
        conversation: Chat-format message list.
        tools: Optional tools list passed to apply_chat_template.
        add_generation_prompt: Standard apply_chat_template kwarg.
        continue_final_message: Standard apply_chat_template kwarg. When True,
            the trailing assistant message is treated as a prefill that the
            model continues. For thinking-on, the prefill is preserved and
            re-attached after the generated thinking trace.
        thinking_config: ThinkingConfig instance.

    Returns:
        (final_prompt_str, metadata)

        metadata keys:
            thinking_trace:    str | None  — generated trace, None if mode='off'
            trace_token_count: int         — token count of the trace (0 if off)
            truncated:         bool        — True if generation hit budget cap
                                              before </think>
            thinking_family:   str | None  — which adapter handled this call
    """
    family = get_thinking_family(model_name)

    if thinking_config.mode == 'off':
        prompt = _render_thinking_off(
            family, tokenizer, conversation,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )
        return prompt, {
            'thinking_trace': None,
            'trace_token_count': 0,
            'truncated': False,
            'thinking_family': family,
        }

    # mode == 'on' from here on.
    if family == 'qwen3':
        return _qwen3_render_with_thinking(
            model, tokenizer, conversation,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            thinking_config=thinking_config,
        )

    if family is None or family == 'none':
        raise ValueError(
            f"thinking-on requested but model {model_name!r} has no "
            f"thinking_family set. Add 'thinking_family: qwen3' (or another "
            f"supported family) to its entry in model_config.yaml."
        )

    if family == 'gemma4':
        raise NotImplementedError(
            f"thinking_family 'gemma4' is recognized but the on-path is a stub. "
            f"Only the off-path (suppress thinking) is implemented today. "
            f"Add a _gemma4_render_with_thinking() adapter to enable thinking-on."
        )

    raise NotImplementedError(
        f"thinking_family {family!r} is recognized but not yet implemented. "
        f"Currently supported: qwen3 (on+off), gemma4 (off only)."
    )


def _render_thinking_off(
    family: str | None,
    tokenizer,
    conversation: list[dict],
    *,
    tools: list | None,
    add_generation_prompt: bool,
    continue_final_message: bool,
) -> str:
    """Standard apply_chat_template call with thinking explicitly off.

    For qwen3 family: passes enable_thinking=False so the chat template
    doesn't emit a <think>...</think> block on add_generation_prompt=True
    (and doesn't auto-inject an empty <think></think> on continue_final_message=True).
    For gemma4 family: passes enable_thinking=False to suppress Gemma 4's
    thought channel (different markers from Qwen3, but the chat-template kwarg
    is the same). Default is already off, but "ghost" thought channels can
    leak unless the off-path is explicit.
    For other families (or absent): just a plain apply_chat_template call.
    """
    kwargs = dict(
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
    )
    if tools is not None:
        kwargs['tools'] = tools
    if family in ('qwen3', 'gemma4'):
        kwargs['enable_thinking'] = False
    return tokenizer.apply_chat_template(conversation, **kwargs)


def _qwen3_render_with_thinking(
    model,
    tokenizer,
    conversation: list[dict],
    *,
    tools: list | None,
    add_generation_prompt: bool,
    continue_final_message: bool,
    thinking_config: ThinkingConfig,
) -> tuple[str, dict]:
    """Qwen3 thinking-on adapter.

    Strategy:
      1. If continue_final_message=True, pop the trailing assistant message
         into prefill_text; everything else stays in conv_for_render.
      2. Render conv_for_render with enable_thinking=True, add_generation_prompt=True.
         The output string ends just inside the <think> block (i.e. waiting for
         the model to fill in the trace).
      3. model.generate(...) until either </think> or the budget cap.
      4. Decode the generated trace. If we hit the budget without </think>,
         force-append </think> for downstream consistency.
      5. Return pre_think + trace_text + '\\n\\n' + prefill_text.
    """
    # Pop prefill if present.
    if continue_final_message and conversation and conversation[-1].get('role') == 'assistant':
        prefill_text = conversation[-1].get('content', '')
        conv_for_render = conversation[:-1]
    else:
        prefill_text = ''
        conv_for_render = conversation

    # Render up to start of <think>.
    pre_think_kwargs = dict(
        tokenize=False,
        enable_thinking=True,
        add_generation_prompt=True,
    )
    if tools is not None:
        pre_think_kwargs['tools'] = tools
    pre_think = tokenizer.apply_chat_template(conv_for_render, **pre_think_kwargs)

    # Resolve </think> as a single token id.
    end_think_ids = tokenizer.encode('</think>', add_special_tokens=False)
    if len(end_think_ids) != 1:
        raise RuntimeError(
            f"'</think>' does not tokenize to a single token (got {end_think_ids}). "
            f"This implementation assumes a Qwen3-style tokenizer that has "
            f"</think> as a special token."
        )
    end_think_id = end_think_ids[0]

    # Generate the thinking trace.
    inputs = tokenizer(pre_think, return_tensors='pt').to(model.device)
    max_new = thinking_config.budget if thinking_config.budget > 0 else thinking_config.safety_max_tokens

    eos_ids: list[int] = [end_think_id]
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)

    # Build generate kwargs. Only pass temperature when sampling — passing it
    # alongside do_sample=False triggers a transformers warning about ignored
    # generation flags, even though the value is semantically irrelevant.
    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_new,
        eos_token_id=eos_ids,
        pad_token_id=(tokenizer.pad_token_id
                      if tokenizer.pad_token_id is not None
                      else tokenizer.eos_token_id),
        use_cache=True,
    )
    if thinking_config.temperature > 0:
        generate_kwargs['do_sample'] = True
        generate_kwargs['temperature'] = thinking_config.temperature
    else:
        generate_kwargs['do_sample'] = False

    # CRITICAL: override model.config.use_cache for the duration of this call.
    # load_model_and_tokenizer sets it to False (correct default for the
    # forward-pass-only measurement code throughout the rest of the codebase),
    # which causes the model's *forward* to skip computing past_key_values
    # entirely. Without that, the generate(use_cache=True) kwarg is
    # functionally a no-op — each new token re-computes attention over the
    # full growing prefix (O(n²) per trace, ~100× slowdown for 500 tokens).
    old_use_cache = getattr(model.config, 'use_cache', None)
    model.config.use_cache = True
    try:
        with torch.inference_mode():
            gen_out = model.generate(**generate_kwargs)
    finally:
        model.config.use_cache = old_use_cache

    trace_ids = gen_out[0][inputs['input_ids'].shape[1]:].tolist()
    truncated = end_think_id not in trace_ids
    trace_text = tokenizer.decode(trace_ids, skip_special_tokens=False)
    if truncated:
        # Hit the budget cap without </think>; force-close so the post-trace
        # prompt is well-formed. The model's natural format is
        # `[reasoning]\n</think>\n\n[answer]` (verified empirically) — match
        # that by ensuring a newline before the appended </think>.
        if not trace_text.endswith('\n'):
            trace_text += '\n'
        trace_text = trace_text + tokenizer.decode([end_think_id], skip_special_tokens=False)

    final_prompt = pre_think + trace_text + '\n\n' + prefill_text

    return final_prompt, {
        'thinking_trace': trace_text,
        'trace_token_count': len(trace_ids),
        'truncated': truncated,
        'thinking_family': 'qwen3',
    }
