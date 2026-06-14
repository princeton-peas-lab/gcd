"""
Judge user-prompt formatting and truncation (shared by dataset prep and run_experiment).

Uses only the tokenizer interface (encode/decode); no torch import required here.
"""

from pathlib import Path
from typing import List, Optional


DEFAULT_JUDGE_USER_TEMPLATE = (
    "[User Question]\n"
    "{instruction}\n\n"
    "[The Start of Assistant A's Answer]\n"
    "{output_a}\n"
    "[The End of Assistant A's Answer]\n\n"
    "[The Start of Assistant B's Answer]\n"
    "{output_b}\n"
    "[The End of Assistant B's Answer]\n\n"
    "Only output your final verdict by strictly following this format: "
    '"[[A]]" if assistant A is better, "[[B]]" if assistant B is better.'
)


def load_judge_user_template(template_path: Optional[str], project_root: Path) -> str:
    if template_path:
        p = Path(template_path)
    else:
        p = project_root / "prompts" / "judge_user_prompt_template.txt"

    if p.exists():
        with open(p, encoding="utf-8") as f:
            return f.read()
    else:
        print(f"[warn] Judge user template not found at {p}; using built-in default.")
        return DEFAULT_JUDGE_USER_TEMPLATE


def format_judge_user_prompt(
    template: str,
    instruction: str,
    output_a: str,
    output_b: str,
) -> str:
    return template.format(
        instruction=instruction,
        output_a=output_a,
        output_b=output_b,
        # Legacy / explicit keys for templates that use model names
        output_1=output_a,
        output_2=output_b,
        output_ref=output_b,
    )


def truncate_text_to_max_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """
    Keep the first ``max_tokens`` tokens (judge tokenizer); decode back to string.
    No-op if ``max_tokens`` <= 0 or text fits in the budget.
    """
    if max_tokens <= 0 or not text:
        return text
    try:
        enc = tokenizer(
            str(text),
            add_special_tokens=False,
            return_tensors="pt",
        )
        ids = enc["input_ids"][0]
        n = int(ids.shape[0])
        if n <= max_tokens:
            return str(text)
        trimmed = ids[:max_tokens]
        return tokenizer.decode(trimmed, skip_special_tokens=True)
    except Exception:
        return str(text)


def truncate_record_outputs_rebuild_judge_prompts(
    rec: dict,
    tokenizer,
    max_tokens: int,
    judge_template: str,
    keys: tuple = ("output_1", "output_2"),
) -> bool:
    """
    Truncate listed output fields in-place (token budget), then recompute
    ``initial_query`` / ``goal`` / ``judge_competitor_initial_query`` like
    :func:`update_judge_user_prompts_for_records`.

    Returns True if any listed field was truncated (prompts rebuilt only then).
    """
    if max_tokens <= 0 or not isinstance(rec, dict):
        return False
    if not (rec.get("output_1") or rec.get("output_2")):
        return False
    changed = False
    for key in keys:
        if key not in rec:
            continue
        raw = rec.get(key)
        if raw is None:
            continue
        s = str(raw)
        if not s.strip():
            continue
        enc = tokenizer(s, add_special_tokens=False, return_tensors="pt")
        n = int(enc["input_ids"].shape[1])
        if n > max_tokens:
            rec[key] = truncate_text_to_max_tokens(s, tokenizer, max_tokens)
            changed = True
    if changed:
        update_judge_user_prompts_for_records([rec], judge_template)
    return changed


def update_judge_user_prompts_for_records(records: List[dict], judge_template: str) -> None:
    """
    Set initial_query, goal, and judge_competitor_initial_query on each record.

    When output_ref is set (typical):
      Primary:    A = output_1, B = output_ref
      Competitor: A = output_2, B = output_ref
    When output_ref is missing (legacy):
      Primary:    A = output_1, B = output_2
      Competitor: A = output_2, B = output_1
    """
    for rec in records:
        if not isinstance(rec, dict):
            continue
        output_1 = rec.get("output_1") or ""
        output_2 = rec.get("output_2") or ""
        output_ref = rec.get("output_ref") or ""
        inst = str(rec.get("instruction", "") or "")

        out_b_primary = output_ref or output_2
        if output_1 and out_b_primary:
            primary_prompt = format_judge_user_prompt(
                judge_template,
                instruction=inst,
                output_a=output_1,
                output_b=out_b_primary,
            )
            rec["initial_query"] = primary_prompt
            rec["goal"] = primary_prompt

        out_b_competitor = output_ref or output_1
        if output_2 and out_b_competitor:
            rec["judge_competitor_initial_query"] = format_judge_user_prompt(
                judge_template,
                instruction=inst,
                output_a=output_2,
                output_b=out_b_competitor,
            )
