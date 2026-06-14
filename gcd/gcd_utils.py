"""
Utility functions for GCD Attack.
"""

import torch
from typing import Any, List, Optional, Set, Tuple, Union

_warned_victim_prompt_target_prefix_mismatch = False


def get_tokenizer_default_system_prompt(tokenizer: Any, model: Any = None) -> Optional[str]:
    """
    Return the model/tokenizer default system prompt if one is defined, else None.

    Checks common Hugging Face tokenizer and model config attributes.
    """
    if tokenizer is not None:
        for attr in ("default_system_prompt", "system_prompt", "default_system_message"):
            val = getattr(tokenizer, attr, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        init_kwargs = getattr(tokenizer, "init_kwargs", None)
        if isinstance(init_kwargs, dict):
            for key in ("default_system_prompt", "system_prompt", "default_system_message"):
                val = init_kwargs.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()

    if model is not None:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            for attr in ("default_system_prompt", "system_prompt", "default_system_message"):
                val = getattr(cfg, attr, None)
                if isinstance(val, str) and val.strip():
                    return val.strip()

    return None


def resolve_add_prefix_target(value: Any) -> Optional[str]:
    """Return CE bridge string between prompt and target, or None to omit."""
    if value is None:
        return None
    return str(value)


def attack_text_has_content(text: Any) -> bool:
    """True when ``text`` contains non-whitespace. For guards only; never mutates ``text``."""
    if text is None:
        return False
    return bool(str(text).strip())


def resolve_forbidden_token_string_ids(tokenizer: Any, token_str: str) -> List[int]:
    """Map a forbidden token config string to tokenizer id(s) (supports multi-char strings like ``\\n\\n``)."""
    if tokenizer is None or token_str is None:
        return []
    s = str(token_str)
    if not s:
        return []
    out: List[int] = []
    try:
        enc = tokenizer(s, add_special_tokens=False).get("input_ids", [])
        if enc:
            out.extend(int(x) for x in enc if int(x) >= 0)
    except Exception:
        pass
    if not out:
        try:
            tid = tokenizer.convert_tokens_to_ids(s)
            if tid is not None and int(tid) >= 0:
                out.append(int(tid))
        except Exception:
            pass
    return out


def _vocab_ids_containing_any_substring(tokenizer: Any, patterns: List[str]) -> List[int]:
    """Return vocab ids whose decoded text contains any of ``patterns`` as a substring."""
    if tokenizer is None or not patterns:
        return []
    out: List[int] = []
    try:
        vocab_size = int(getattr(tokenizer, "vocab_size", None) or len(tokenizer))
    except Exception:
        return out
    for tid in range(vocab_size):
        try:
            decoded = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        except Exception:
            continue
        if not decoded:
            continue
        for pat in patterns:
            if pat in decoded:
                out.append(int(tid))
                break
    return out


def resolve_forbidden_token_config_ids(tokenizer: Any, token_strings: List[str]) -> List[int]:
    """
    Resolve config ``forbidden_diffusion_tokens`` to tokenizer ids.

    Non-newline strings use exact tokenization. Newline patterns (``\\n``, ``\\n\\n``, etc.)
    also forbid every vocab token whose decoded text contains that substring, so compound
    tokens like ``':\\n\\n'`` are blocked and not only the standalone ``\\n\\n`` token.
    """
    if tokenizer is None or not token_strings:
        return []
    out: List[int] = []
    newline_patterns: List[str] = []
    for raw in token_strings:
        s = str(raw)
        if not s:
            continue
        if "\n" in s:
            newline_patterns.append(s)
        else:
            out.extend(resolve_forbidden_token_string_ids(tokenizer, s))
    if newline_patterns:
        out.extend(_vocab_ids_containing_any_substring(tokenizer, newline_patterns))
    seen: Set[int] = set()
    deduped: List[int] = []
    for tid in out:
        tid_int = int(tid)
        if tid_int not in seen:
            seen.add(tid_int)
            deduped.append(tid_int)
    return deduped


def align_victim_prompt_text_to_tokenizer(tokenizer: Any, prompt_text: str) -> str:
    """
    Decode the victim tokenization of ``prompt_text`` without ``.strip()``.

    Keeps saved attack prompts identical to the exact string implied by the token ids
    passed to ``generate()`` (no Python whitespace trimming).
    """
    if tokenizer is None:
        return str(prompt_text or "")
    text = str(prompt_text or "")
    if not attack_text_has_content(text):
        return text
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def victim_tokenize_prompt_with_target(
    tokenizer: Any,
    prompt_text: str,
    target_text: str,
    target_prefix: Optional[str] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Tokenize prompt and target jointly so boundary tokens match autoregressive generation.

    ``target_prefix`` (e.g. a single space for Mistral after ``[/INST]``) is inserted
    between ``prompt_text`` and ``target_text`` for joint tokenization only. Prefix/success
    checks should still use ``target_text`` without this bridge.

    Returns:
        prompt_ids: token ids for ``prompt_text`` alone
        full_ids: token ids for ``prompt_text + bridge + target_text`` (joint)
        target_ids: suffix of ``full_ids`` after ``prompt_ids`` (bridge + target tokens)
    """
    global _warned_victim_prompt_target_prefix_mismatch
    bridge = resolve_add_prefix_target(target_prefix) or ""
    combined_target = bridge + str(target_text or "")
    prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    full_ids = list(tokenizer(prompt_text + combined_target, add_special_tokens=False)["input_ids"])
    prompt_len = len(prompt_ids)
    if prompt_len <= len(full_ids) and full_ids[:prompt_len] == prompt_ids:
        target_ids = full_ids[prompt_len:]
        return prompt_ids, full_ids, target_ids

    if not _warned_victim_prompt_target_prefix_mismatch:
        _warned_victim_prompt_target_prefix_mismatch = True
        print(
            "[GCDAttack] WARNING: joint prompt+target tokenization prefix mismatch; "
            "falling back to separate tokenize+concat."
        )
    target_ids = list(tokenizer(combined_target, add_special_tokens=False)["input_ids"])
    full_ids = prompt_ids + target_ids
    return prompt_ids, full_ids, target_ids


def decode_victim_generation(
    tokenizer: Any,
    token_ids: Union[List[int], torch.Tensor],
    *,
    clean_up_tokenization_spaces: bool = True,
) -> str:
    """Decode newly generated victim tokens without stripping trailing whitespace."""
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
    )


def should_reduce_batch_size(exception: Exception) -> bool:
    """
    NanoGCG-style heuristic: detect OOM-like failures and allow halving the batch size.
    """
    _statements = [
        "CUDA out of memory.",
        "CUDA error: out of memory",
        "CUBLAS_STATUS_ALLOC_FAILED",
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",
        "DefaultCPUAllocator: can't allocate memory",
    ]
    try:
        if isinstance(exception, RuntimeError) and len(exception.args) == 1:
            msg = str(exception.args[0])
            return any(s in msg for s in _statements)
    except Exception:
        return False
    return False


def lcp_len(a: list, b: list) -> int:
    """Longest common prefix length of two python int lists."""
    n = min(len(a), len(b))
    i = 0
    # Tight loop (python) but sequences are usually not huge (few hundred tokens)
    while i < n and a[i] == b[i]:
        i += 1
    return i


def repeat_past_key_values(past_key_values, batch_size: int):
    """
    Repeat/expand a single-sample (batch=1) past_key_values to a larger batch size.

    Returns:
        repeated_past_key_values or None if unsupported (e.g., new Cache objects).
    """
    if past_key_values is None:
        return None
    if batch_size is None or int(batch_size) <= 0:
        return None

    # Common HF format: tuple(n_layers) of (k, v) each shaped [B, H, T, D] (or model-specific variants)
    if isinstance(past_key_values, (tuple, list)):
        repeated_layers = []
        for layer in past_key_values:
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                return None
            k, v = layer[0], layer[1]
            if (not torch.is_tensor(k)) or (not torch.is_tensor(v)):
                return None
            # Expand batch dimension (dim=0). Expand is zero-copy, which is ideal here.
            try:
                k_rep = k.expand(int(batch_size), *k.shape[1:])
                v_rep = v.expand(int(batch_size), *v.shape[1:])
            except Exception:
                # Some cache tensors may not be expandable; fall back to repeat.
                k_rep = k.repeat(int(batch_size), *([1] * (k.dim() - 1)))
                v_rep = v.repeat(int(batch_size), *([1] * (v.dim() - 1)))
            repeated_layers.append((k_rep, v_rep))
        return tuple(repeated_layers)

    # Newer Transformers may return a Cache object (DynamicCache, StaticCache, etc.).
    # Supporting safe "repeat" for those is model/version dependent; fall back gracefully.
    return None


def slice_past_key_values(past_key_values, seq_len: int, full_seq_len: int = None):
    """
    Slice a batch=1 past_key_values to the first `seq_len` cached positions.

    We try to locate the sequence-length dimension by matching `full_seq_len` (if provided),
    otherwise we assume common HF layout where K/V have shape [B, H, T, D] and slice dim=2.
    Returns None if unsupported.
    """
    if past_key_values is None:
        return None
    if seq_len is None or int(seq_len) < 0:
        return None
    seq_len = int(seq_len)
    if full_seq_len is not None:
        full_seq_len = int(full_seq_len)
        if seq_len >= full_seq_len:
            return past_key_values

    # Common HF format: tuple(n_layers) of (k, v) each shaped [B, H, T, D]
    if isinstance(past_key_values, (tuple, list)):
        sliced_layers = []
        for layer in past_key_values:
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                return None
            k, v = layer[0], layer[1]
            if (not torch.is_tensor(k)) or (not torch.is_tensor(v)):
                return None
            # Assume sequence length is at dim=2 (common for [B, H, T, D] layout)
            try:
                k_slice = k[:, :, :seq_len, :] if k.dim() >= 4 else k[:, :seq_len] if k.dim() >= 3 else k
                v_slice = v[:, :, :seq_len, :] if v.dim() >= 4 else v[:, :seq_len] if v.dim() >= 3 else v
                sliced_layers.append((k_slice, v_slice))
            except Exception:
                return None
        return tuple(sliced_layers)

    return None
