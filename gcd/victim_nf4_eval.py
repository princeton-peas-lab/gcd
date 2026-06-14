"""Post-process victim logits before optimization-time CE / perplexity (no weight changes)."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F


def simulate_low_precision_logits(
    logits: torch.Tensor,
    *,
    mode: str = "float8_e4m3",
) -> torch.Tensor:
    """
    Approximate lower-precision logits by dtype round-trip only.

    Default: float32 -> float8_e4m3fn -> float32 -> original dtype.
    Falls back to bfloat16 round-trip if float8 is unavailable.
    Does not touch model weights or change the forward pass itself.
    """
    if logits is None or not torch.is_tensor(logits):
        return logits

    orig_dtype = logits.dtype
    staging = logits.float()

    if mode in ("float8_e4m3", "float8", "fp8", "e4m3"):
        fp8 = getattr(torch, "float8_e4m3fn", None)
        if fp8 is not None:
            try:
                return staging.to(fp8).float().to(dtype=orig_dtype)
            except Exception:
                pass
        fp8 = getattr(torch, "float8_e5m2", None)
        if fp8 is not None:
            try:
                return staging.to(fp8).float().to(dtype=orig_dtype)
            except Exception:
                pass

    if mode in ("bfloat16", "bf16"):
        return staging.to(torch.bfloat16).float().to(dtype=orig_dtype)

    if mode in ("float16", "fp16"):
        return staging.to(torch.float16).float().to(dtype=orig_dtype)

    # Default fallback when float8 is unavailable.
    return staging.to(torch.bfloat16).float().to(dtype=orig_dtype)


def _recompute_causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def apply_low_precision_logits_to_outputs(
    outputs: Any,
    *,
    labels: Optional[torch.Tensor] = None,
    mode: str = "float8_e4m3",
) -> Any:
    """Mutate ``outputs.logits`` (and ``outputs.loss`` if labels given) in place."""
    logits = getattr(outputs, "logits", None)
    if logits is None or not torch.is_tensor(logits):
        return outputs

    outputs.logits = simulate_low_precision_logits(logits, mode=mode)

    if labels is not None and torch.is_tensor(labels):
        try:
            outputs.loss = _recompute_causal_lm_loss(outputs.logits, labels)
        except Exception:
            pass
    elif getattr(outputs, "loss", None) is not None:
        # Loss from the un-quantized logits would be misleading; drop it.
        outputs.loss = None

    return outputs
