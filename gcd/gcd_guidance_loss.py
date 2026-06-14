"""
Batch-wise normalization and weighted combination of guidance objectives.

Each objective (target CE, self-PPL, defence loss, etc.) is min–max scaled to [0, 1]
over the candidate batch, then summed with its coefficient. Raw per-objective tensors
are kept separate for logging.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from gcd.gcd_core import GCDAttack

GuidanceTerm = Tuple[torch.Tensor, float]

# Upper cap for exp(mean CE) when ``use_raw_ppl`` is enabled (matches logging in gcd_core).
_DEFAULT_MAX_SELF_PPL = 2000.0


def self_ppl_for_guidance(
    self_ppl_losses: torch.Tensor,
    *,
    use_raw_ppl: bool,
    max_ppl: float = _DEFAULT_MAX_SELF_PPL,
) -> torch.Tensor:
    """
    Map per-candidate self-PPL mean CE to the value used in guidance combine/normalize.

    When ``use_raw_ppl`` is True, use perplexity ``exp(CE)`` (capped). Otherwise use raw CE.
    """
    losses = self_ppl_losses.to(torch.float32)
    if not use_raw_ppl:
        return losses
    return torch.exp(losses).clamp(max=float(max_ppl))


def minmax_normalize_batch(losses: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Min–max scale ``losses`` to [0, 1] over the batch (lower raw loss -> lower value).

    If all values are equal (or the batch has one element), returns zeros so the term
    does not affect relative ranking within that step.
    """
    if losses is None:
        return losses

    # Guard against float16 overflows / bad CE values returning Inf or NaN in a batch.
    losses = torch.nan_to_num(losses.to(torch.float32), nan=0.0, posinf=2000.0, neginf=0.0)
    if losses.numel() == 0:
        return losses
    if losses.numel() == 1:
        return torch.zeros_like(losses)
    lo = losses.min()
    hi = losses.max()
    if float((hi - lo).item()) < eps:
        return torch.zeros_like(losses)
    return (losses - lo) / (hi - lo + eps)


def combine_guidance_losses(
    terms: Sequence[GuidanceTerm],
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Combine guidance objectives.

    Args:
        terms: ``(per_candidate_raw_loss, coefficient)`` pairs. Terms with ``coef <= 0``
            are skipped.
        normalize: If True, min–max normalize each term over the batch before weighting.
            If False, use the legacy raw weighted sum.

    Returns:
        Combined per-candidate guidance loss (lower is better).
    """
    active: List[GuidanceTerm] = [
        (losses.to(torch.float32), float(coef))
        for losses, coef in terms
        if losses is not None and float(coef) > 0.0
    ]
    if not active:
        if not terms:
            raise ValueError("combine_guidance_losses requires at least one term")
        ref = terms[0][0]
        return torch.zeros(ref.shape[0], device=ref.device, dtype=torch.float32)

    total = torch.zeros_like(active[0][0])
    for losses, coef in active:
        term = minmax_normalize_batch(losses) if normalize else losses
        total = total + float(coef) * term
    return total


def build_guidance_terms(
    attack: "GCDAttack",
    ce_losses: torch.Tensor,
    self_ppl_losses: torch.Tensor,
    extra_terms: Optional[Sequence[GuidanceTerm]] = None,
) -> List[GuidanceTerm]:
    """
    Standard guidance objectives for victim candidate evaluation.

    Target CE is always included (implicit coefficient 1.0). Optional terms are added
    when enabled on ``attack``. ``extra_terms`` can supply defence / transfer / etc.
    """
    terms: List[GuidanceTerm] = [(ce_losses, 1.0)]

    if bool(getattr(attack, "self_perplexity", False)):
        ppl_term = self_ppl_for_guidance(
            self_ppl_losses,
            use_raw_ppl=bool(getattr(attack, "use_raw_ppl", True)),
        )
        terms.append(
            (ppl_term, float(getattr(attack, "self_perplexity_coef", 0.0)))
        )

    if bool(getattr(attack, "self_perplexity_rpp", False)):
        # RPP is derived from self-PPL and suffix repetition; use precomputed tensor if set.
        rpp = getattr(attack, "_batch_self_ppl_rpp_losses", None)
        if rpp is not None and isinstance(rpp, torch.Tensor) and rpp.shape == ce_losses.shape:
            terms.append(
                (rpp, float(getattr(attack, "self_perplexity_rpp_coef", 0.0)))
            )

    if extra_terms:
        terms.extend(extra_terms)

    return terms
