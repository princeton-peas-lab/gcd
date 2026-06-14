"""Experiment configuration helpers shared by the attack pipeline."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from gcd.gcd_utils import attack_text_has_content

def _tunable_init_len_for_experiment(
    experiment_config: Dict[str, Any], num_tunable_tokens: int
) -> Tuple[int, bool, Optional[int]]:
    """
    When incremental_tunable_growth is enabled, the tunable tensor starts at the first
    block_vise_schedule width and grows each block; core cap is the sum of schedule widths
    (should match num_tunable_tokens).

    Returns:
        (initial_tensor_len, incremental_enabled, incremental_core_max_len or None)
    """
    nt = int(num_tunable_tokens)
    if not bool(experiment_config.get("incremental_tunable_growth", False)):
        return nt, False, None
    raw = experiment_config.get("block_vise_schedule")
    if not raw or not isinstance(raw, (list, tuple)) or len(raw) == 0:
        print(
            "[exp_utils] incremental_tunable_growth=True requires non-empty block_vise_schedule; "
            "using full num_tunable_tokens."
        )
        return nt, False, None
    try:
        pairs = [tuple(x) if isinstance(x, (list, tuple)) else x for x in raw]
        sched_sum = sum(int(a[0]) for a in pairs)
        first = int(pairs[0][0])
    except Exception as e:
        print(
            f"[exp_utils] incremental_tunable_growth: invalid block_vise_schedule ({e}); "
            "using full num_tunable_tokens."
        )
        return nt, False, None
    if first <= 0 or sched_sum <= 0:
        return nt, False, None
    if sched_sum != nt:
        print(
            f"[exp_utils] incremental_tunable_growth: block_vise_schedule token sum ({sched_sum}) != "
            f"num_tunable_tokens ({nt}); using schedule sum ({sched_sum}) as core cap."
        )
    print(
        f"[exp_utils] incremental_tunable_growth: initial tunable length={first}, "
        f"core cap={sched_sum} (grows each schedule block)."
    )
    return first, True, sched_sum


def _adapt_tune_tokens_resolve(
    experiment_config: Dict[str, Any],
    *,
    tokenizer,
    victim_tokenizer,
    no_diffusion: bool,
    adapt_tokenizers: bool,
    initial_query: str,
    goal: Optional[str],
) -> Tuple[int, Dict[str, Any]]:
    """
    When adapt_tune_tokens is True, set num_tunable_tokens to
    round(len(tokenizer(original_text)) * adapt_tune_tokens_coef), where original_text is
    _adapt_tune_tokens_source_text if set, else non-empty goal, else initial_query.

    Optionally syncs fill_max_tokens_per_step to the same value when
    adapt_tune_tokens_sync_fill_max is True (default).
    """
    base_nt = int(experiment_config.get("num_tunable_tokens", experiment_config.get("suffix_len", 40)))
    if not bool(experiment_config.get("adapt_tune_tokens", False)):
        return base_nt, experiment_config

    count_tok = victim_tokenizer if (no_diffusion and (not adapt_tokenizers)) else tokenizer
    src = str(experiment_config.get("_adapt_tune_tokens_source_text") or "").strip()
    if not src:
        g = str(goal if goal is not None else "").strip()
        src = g if g else str(initial_query or "")
    try:
        coef = float(experiment_config.get("adapt_tune_tokens_coef", 1.0))
    except Exception:
        coef = 1.0
    try:
        n_src = len(count_tok(src, add_special_tokens=False)["input_ids"])
    except Exception:
        try:
            n_src = len(count_tok.encode(src, add_special_tokens=False))
        except Exception:
            n_src = base_nt
    adapted = max(1, int(round(float(n_src) * coef)))
    cfg = dict(experiment_config)
    cfg["num_tunable_tokens"] = adapted
    if bool(cfg.get("adapt_tune_tokens_sync_fill_max", True)):
        cfg["fill_max_tokens_per_step"] = adapted
    _sync_note = (
        f", fill_max_tokens_per_step={adapted}"
        if bool(cfg.get("adapt_tune_tokens_sync_fill_max", True))
        else ""
    )
    print(
        f"[adapt_tune_tokens] source_tokens={n_src}, coef={coef} -> num_tunable_tokens={adapted}{_sync_note}"
    )
    return adapted, cfg


def _fixed_ids_source_text(initial_query: str, experiment_config: Dict[str, Any]) -> str:
    """Text tokenized into fixed user-prefix IDs (before tunable masks).

    When ``initial_query_prefix`` is injected as ``_tokenizer_fixed_prefix_text`` (e.g. MMLU exam
    boilerplate), only that prefix is fixed; ``initial_query`` is then the stem for logging / GSM8K parity.
    """
    pfx = str(experiment_config.get("_tokenizer_fixed_prefix_text", "") or "")
    return pfx if attack_text_has_content(pfx) else str(initial_query or "")


def _logged_initial_query_for_result(
    experiment_config: Dict[str, Any],
    goal: Optional[str],
    initial_query: str,
) -> str:
    """Full initial user request text for result JSON key ``initial_query``.

    Concatenates the same conceptual pieces as the victim user message before optimization
    replaces the tunable span: **fixed prefix** + **initial proposed stem** (goal) +
    **fixed suffix** (``fixed_user_suffix_after_tunable``, e.g. MMLU choices + ``Answer: ``).

    Fixed prefix is ``_tokenizer_fixed_prefix_text`` when set; otherwise it is the
    attack-time ``initial_query`` string (tokenized as the entire fixed block in
    ``_fixed_ids_source_text``). The stem prefers ``goal``, then ``_result_original_query``,
    then ``initial_query`` when a separate prefix is configured.
    """
    suffix = str(experiment_config.get("fixed_user_suffix_after_tunable", "") or "")
    pfx_cfg = str(experiment_config.get("_tokenizer_fixed_prefix_text", "") or "").strip()
    iq = str(initial_query or "").strip()
    if pfx_cfg:
        prefix = pfx_cfg
    else:
        prefix = iq

    stem = str(goal or "").strip()
    if not stem:
        stem = str(experiment_config.get("_result_original_query", "") or "").strip()
    if not stem and pfx_cfg:
        stem = iq

    if prefix == stem:
        return prefix + suffix
    if not stem:
        return prefix + suffix
    return prefix + stem + suffix
