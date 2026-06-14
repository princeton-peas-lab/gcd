"""
Victim-model candidate evaluation for pure-diffusion GCD.
Dream fill + victim forward + cross-entropy (+ optional self-perplexity).
"""

from __future__ import annotations

import gc
import time
from typing import Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F

from gcd.gcd_guidance_loss import build_guidance_terms, combine_guidance_losses

if TYPE_CHECKING:
    from gcd.gcd_core import GCDAttack


class VictimEvaluator:
    """Evaluate candidate suffixes on the victim model."""

    def __init__(self, attack: "GCDAttack") -> None:
        self.attack = attack

    def eval_candidates(
        self,
        step_num: int,
        new_candidate_ids: torch.Tensor,
        cand_pos: Optional[torch.Tensor],
        precomputed_tokens: Optional[torch.Tensor],
        dream_logprobs: Optional[torch.Tensor],
        selected_positions_for_dream: Optional[torch.Tensor],
        seq_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_losses, ce_losses, self_ppl_losses, filled_ids_all
        """
        attack = self.attack
        n_cands = int(new_candidate_ids.shape[0])
        total_losses = torch.empty((n_cands,), device=device, dtype=torch.float32)
        ce_losses = torch.empty((n_cands,), device=device, dtype=torch.float32)
        self_ppl_losses = torch.zeros((n_cands,), device=device, dtype=torch.float32)
        self_ppl_rpp_losses = torch.zeros((n_cands,), device=device, dtype=torch.float32)
        filled_ids_all = torch.empty((n_cands, seq_len), device=device, dtype=torch.long)
        need_self_ppl = bool(getattr(attack, "self_perplexity", False)) or bool(
            getattr(attack, "self_perplexity_rpp", False)
        )
        need_self_ppl_rpp = bool(getattr(attack, "self_perplexity_rpp", False))
        need_p2div = bool(getattr(attack, "phase2_div_loss", False)) and bool(
            getattr(attack, "_phase2_div_bad_token_ids", []) or []
        )
        p2div_losses = (
            torch.zeros((n_cands,), device=device, dtype=torch.float32)
            if need_p2div
            else None
        )

        _victim_eval_start_time = time.time()

        def _accumulate_p2div(batch_slice: slice, post_prefix_logits: torch.Tensor) -> None:
            if not need_p2div or p2div_losses is None:
                return
            p2div_losses[batch_slice] = attack._phase2_div_unlikelihood_from_post_prefix_logits(
                post_prefix_logits
            ).to(device=device, dtype=torch.float32)

        def _tunable_texts_from_batch(batch_to_eval: torch.Tensor) -> list:
            dec_tok = attack.tokenizer
            texts = []
            for row in batch_to_eval:
                try:
                    texts.append(
                        dec_tok.decode(row.tolist(), skip_special_tokens=False)
                    )
                except Exception:
                    texts.append("")
            return texts

        def _eval_with_batch_size(bs: int) -> None:
            nonlocal total_losses, ce_losses, self_ppl_losses, self_ppl_rpp_losses, filled_ids_all
            bs = max(1, int(bs))
            with torch.no_grad():
                for i in range(0, n_cands, bs):
                    end_i = min(i + bs, n_cands)
                    batch_candidates_unfilled = new_candidate_ids[i:end_i]
                    cand_pos_batch = None
                    try:
                        if cand_pos is not None and cand_pos.numel() == n_cands:
                            cand_pos_batch = cand_pos[i:end_i]
                    except Exception:
                        cand_pos_batch = None

                    if attack.fill_during_eval:
                        if attack.use_precomputed_score and precomputed_tokens is not None:
                            batch_to_eval = batch_candidates_unfilled.clone()
                            mask_locs = batch_to_eval == attack.mask_token_id
                            if dream_logprobs is not None:
                                probs = torch.exp(dream_logprobs)
                                if mask_locs.any():
                                    batch_to_eval = attack._smart_fill_masks_with_precomputed(
                                        batch_to_eval, precomputed_tokens, probs
                                    )
                            else:
                                filler = precomputed_tokens.unsqueeze(0).expand(batch_to_eval.size(0), -1)
                                batch_to_eval[mask_locs] = filler[mask_locs]
                        else:
                            dream_input_batch = attack._dream_input_for_eval_fill(batch_candidates_unfilled)
                            attack._maybe_print_template_diff(dream_input_batch)
                            attack._maybe_print_dream_fill_input_each_step(step_num, dream_input_batch)

                            allowed_mask = None
                            if getattr(attack, "fill_only_neighbouring", False) and cand_pos_batch is not None:
                                allowed_mask = torch.zeros_like(dream_input_batch, dtype=torch.bool)
                                tunable_start = attack._dream_tunable_start_idx()
                                n_size = int(seq_len * float(getattr(attack, "fill_neighbouring_size", 0.15)))
                                h_size = n_size // 2
                                for idx_in_batch, p in enumerate(cand_pos_batch):
                                    p_i = int(p.item())
                                    start_p = max(0, p_i - h_size)
                                    end_p = min(seq_len, p_i + h_size + 1)
                                    allowed_mask[
                                        idx_in_batch,
                                        tunable_start + start_p : tunable_start + end_p,
                                    ] = True
                            elif (
                                getattr(attack, "fill_only_sampled", False)
                                and selected_positions_for_dream is not None
                            ):
                                allowed_mask = torch.zeros_like(dream_input_batch, dtype=torch.bool)
                                tunable_start = attack._dream_tunable_start_idx()
                                allowed_mask[:, tunable_start + selected_positions_for_dream] = True

                            if attack.use_llada:
                                filled_dream_output = attack._llada_fill_masks(
                                    dream_input_batch,
                                    steps=attack.dream_eval_steps,
                                    mask_token_id=attack.mask_token_id,
                                    generation_logits_hook_func=attack._dream_generation_logits_hook,
                                    allowed_mask=allowed_mask,
                                )
                            else:
                                filled_dream_output = attack._dream_diffusion_generate(
                                    inputs=dream_input_batch,
                                    steps=attack.dream_eval_steps,
                                    mask_token_id=attack.mask_token_id,
                                    generation_logits_hook_func=attack._dream_generation_logits_hook,
                                    allowed_mask=allowed_mask,
                                )
                            tunable_start = attack._dream_tunable_start_idx()
                            batch_to_eval = filled_dream_output[:, tunable_start : tunable_start + seq_len]
                    else:
                        batch_to_eval = batch_candidates_unfilled

                    filled_ids_all[i:end_i] = batch_to_eval

                    if bool(attack.to_text_before_eval) or bool(
                        getattr(attack, "retokenize_before_victim_loss", False)
                    ):
                        pp_out: list = []
                        if need_self_ppl:
                            _comp = attack._victim_loss_from_text(
                                batch_to_eval,
                                return_components=True,
                                post_prefix_logits_out=pp_out if need_p2div else None,
                            )
                            ce_ps = _comp[1].to(torch.float32)
                            self_ppl_ps = _comp[4].to(torch.float32)
                            self_ppl_rpp_ps = (
                                _comp[5].to(torch.float32)
                                if need_self_ppl_rpp
                                else torch.zeros_like(ce_ps)
                            )
                        else:
                            ce_ps = attack._victim_ce_loss_for_batch(
                                batch_to_eval,
                                post_prefix_logits_out=pp_out if need_p2div else None,
                            )
                            self_ppl_ps = torch.zeros_like(ce_ps)
                            self_ppl_rpp_ps = torch.zeros_like(ce_ps)
                        if need_p2div and pp_out:
                            _accumulate_p2div(slice(i, end_i), pp_out[0])
                        ce_losses[i:end_i] = ce_ps
                        self_ppl_losses[i:end_i] = self_ppl_ps
                        self_ppl_rpp_losses[i:end_i] = self_ppl_rpp_ps
                        continue

                    # Embed path: clip OOB IDs to victim vocab here, not before the
                    # to_text_before_eval check, so the warning never fires when
                    # to_text_before_eval=True.
                    batch_to_eval_safe = attack._victim_safe_ids(batch_to_eval)

                    if (
                        attack._victim_chat_prefix_embeds is not None
                        and attack._victim_chat_suffix_embeds is not None
                        and attack._victim_chat_target_embeds is not None
                        and attack._victim_chat_target_ids is not None
                    ):
                        full_batch = torch.cat(
                            [
                                attack._victim_chat_prefix_embeds.expand(batch_to_eval_safe.shape[0], -1, -1),
                                attack.target_embedding_layer(batch_to_eval_safe),
                                attack._victim_chat_suffix_embeds.expand(batch_to_eval_safe.shape[0], -1, -1),
                                attack._victim_chat_target_embeds.expand(batch_to_eval_safe.shape[0], -1, -1),
                            ],
                            dim=1,
                        )
                        tgt_emb = attack._victim_chat_target_embeds
                        tgt_ids = attack._victim_chat_target_ids
                    else:
                        full_batch = torch.cat(
                            [
                                attack.system_embeds.expand(batch_to_eval_safe.shape[0], -1, -1),
                                attack.fixed_user_embeds.expand(batch_to_eval_safe.shape[0], -1, -1),
                                attack.target_embedding_layer(batch_to_eval_safe),
                                attack.assist_embeds.expand(batch_to_eval_safe.shape[0], -1, -1),
                                attack.target_embeds.expand(batch_to_eval_safe.shape[0], -1, -1),
                            ],
                            dim=1,
                        )
                        tgt_emb = attack.target_embeds
                        tgt_ids = attack.target_ids

                    outputs = attack._victim_forward(inputs_embeds=full_batch)
                    start = full_batch.shape[1] - tgt_emb.shape[1] - 1
                    end = start + tgt_emb.shape[1]
                    target_logits = outputs.logits[:, start:end, :]
                    _accumulate_p2div(slice(i, end_i), target_logits[:, -1, :])

                    ce_ps = F.cross_entropy(
                        target_logits.transpose(1, 2),
                        tgt_ids.expand(batch_to_eval_safe.shape[0], -1),
                        reduction="none",
                    ).mean(dim=1)

                    self_ppl_ps = torch.zeros_like(ce_ps)
                    if need_self_ppl:
                        if attack._victim_chat_prefix_embeds is not None:
                            ppl_start = int(attack._victim_chat_prefix_embeds.shape[1])
                        else:
                            ppl_start = int(attack.system_embeds.shape[1]) + int(
                                attack.fixed_user_embeds.shape[1]
                            )
                        # CE only over the tunable suffix: logits at ppl_start-1.. predict tunable tokens.
                        ppl_logits = outputs.logits[
                            :, ppl_start - 1 : ppl_start + seq_len - 1, :
                        ]
                        ppl_targets = batch_to_eval_safe
                        self_ppl_ps = F.cross_entropy(
                            ppl_logits.reshape(-1, ppl_logits.size(-1)),
                            ppl_targets.reshape(-1),
                            reduction="none",
                        ).view(batch_to_eval_safe.shape[0], -1).mean(dim=1)

                    self_ppl_rpp_ps = torch.zeros_like(ce_ps)
                    if need_self_ppl_rpp:
                        self_ppl_rpp_ps = attack._self_ppl_rpp_from_ce_and_texts(
                            self_ppl_ps, _tunable_texts_from_batch(batch_to_eval)
                        )

                    ce_losses[i:end_i] = ce_ps.to(torch.float32)
                    self_ppl_losses[i:end_i] = self_ppl_ps.to(torch.float32)
                    self_ppl_rpp_losses[i:end_i] = self_ppl_rpp_ps.to(torch.float32)

                    del outputs
                    del target_logits

        if bool(getattr(attack, "optimize_batch_size", False)):
            bs = (
                int(attack._tuned_eval_batch_size)
                if getattr(attack, "_tuned_eval_batch_size", None) is not None
                else int(attack.eval_batch_size)
            )
            bs = max(1, min(int(bs), n_cands if n_cands > 0 else 1))
            while True:
                try:
                    _eval_with_batch_size(bs)
                    attack._tuned_eval_batch_size = int(bs)
                    attack._last_eval_batch_size_used = int(bs)
                    if not bool(getattr(attack, "_printed_tuned_batch_size", False)):
                        print(f"[GCDAttack] optimize_batch_size=True used eval_batch_size={int(bs)}")
                        attack._printed_tuned_batch_size = True
                    break
                except Exception as e:
                    if attack._should_reduce_batch_size(e) and int(bs) > 1:
                        bs = max(1, int(bs) // 2)
                        attack._tuned_eval_batch_size = int(bs)
                        attack._last_eval_batch_size_used = int(bs)
                        gc.collect()
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        print(
                            f"[GCDAttack] optimize_batch_size=True OOM -> "
                            f"decreasing eval_batch_size to {int(bs)}"
                        )
                        continue
                    raise
        else:
            attack._last_eval_batch_size_used = int(attack.eval_batch_size)
            _eval_with_batch_size(int(attack.eval_batch_size))

        if n_cands > 0:
            attack._batch_self_ppl_rpp_losses = (
                self_ppl_rpp_losses if need_self_ppl_rpp else None
            )
            attack._last_self_ppl_rpp_losses = (
                self_ppl_rpp_losses if need_self_ppl_rpp else None
            )
            attack._last_phase2_div_losses = p2div_losses
            try:
                attack._defence_log_best_idx = int(torch.argmin(total_losses).item())
            except Exception:
                attack._defence_log_best_idx = 0
            extra_terms = self._optional_defence_guidance_terms(attack, filled_ids_all, device)
            attack._defence_log_best_idx = None
            p2div_terms = self._optional_phase2_div_guidance_terms(attack, filled_ids_all, device)
            if p2div_terms:
                extra_terms = (extra_terms or []) + p2div_terms
            guidance_terms = build_guidance_terms(
                attack, ce_losses, self_ppl_losses, extra_terms=extra_terms
            )
            total_losses[:] = combine_guidance_losses(
                guidance_terms,
                normalize=bool(getattr(attack, "normalize_guidance_losses", True)),
            )
        else:
            attack._batch_self_ppl_rpp_losses = None
            attack._last_self_ppl_rpp_losses = None
            attack._last_phase2_div_losses = None

        attack._step_time_victim_eval = time.time() - _victim_eval_start_time
        return total_losses, ce_losses, self_ppl_losses, filled_ids_all

    def _optional_defence_guidance_terms(
        self,
        attack: "GCDAttack",
        filled_ids_all: torch.Tensor,
        device: torch.device,
    ):
        """
        Defence evasion loss (optional). Returns extra guidance terms or None.

        Runs LlamaGuard CE on every candidate using a tuned per-step batch size.
        On OOM the batch size is halved and the pass is retried (same strategy as
        ``optimize_batch_size`` for the victim evaluator).  A successful batch size
        is remembered across steps so the overhead is paid only once.
        """
        if not bool(getattr(attack, "defence_evasion", False)):
            return None
        if getattr(attack, "defence_model", None) is None:
            return None
        coef = float(getattr(attack, "alpha_def", 0.0))
        if coef <= 0.0:
            return None

        n_cands = int(filled_ids_all.shape[0])
        if n_cands == 0:
            return None

        # Determine starting chunk size.  Use the previously successful size when
        # available so we don't re-probe every step.
        max_chunk = int(getattr(attack, "defence_eval_batch_size", 16) or 16)
        tuned = getattr(attack, "_tuned_defence_guidance_batch_size", None)
        chunk = int(tuned) if tuned is not None else max(1, min(max_chunk, n_cands))

        prompt_texts = attack._defence_prompt_texts_from_filled_ids(filled_ids_all)
        attack._defence_ce_prompt_logged_this_step = False

        while chunk >= 1:
            try:
                parts = []
                for i in range(0, n_cands, chunk):
                    attack._defence_ce_log_chunk_start = int(i)
                    sub_texts = prompt_texts[i : i + chunk]
                    sub_loss = attack._compute_defence_loss_cached(sub_texts).to(
                        device=device, dtype=torch.float32
                    )
                    parts.append(sub_loss)
                defence_losses = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
                attack._last_batch_defence_losses = defence_losses.detach()
                # Remember the successful chunk size for subsequent steps.
                attack._tuned_defence_guidance_batch_size = chunk
                return [(defence_losses, coef)]
            except Exception as exc:
                if not attack._should_reduce_batch_size(exc) or chunk <= 1:
                    attack._last_batch_defence_losses = None
                    if not getattr(attack, "_warned_defence_guidance_failure", False):
                        attack._warned_defence_guidance_failure = True
                        print(
                            f"{attack._lp()}[defence] Batch guidance loss failed ({exc!r}); "
                            "defence CE will be omitted from this step's candidate scoring."
                        )
                    return None
                new_chunk = max(1, chunk // 2)
                print(
                    f"{attack._lp()}[defence] OOM with chunk={chunk} ({exc.__class__.__name__}); "
                    f"retrying with chunk={new_chunk}."
                )
                chunk = new_chunk
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        attack._last_batch_defence_losses = None
        return None

    def _optional_phase2_div_guidance_terms(
        self,
        attack: "GCDAttack",
        filled_ids_all: torch.Tensor,
        device: torch.device,
    ):
        """Phase-2 unlikelihood on rejected first-continuation tokens (from CE forward logits)."""
        if not bool(getattr(attack, "phase2_div_loss", False)):
            return None
        bad = list(getattr(attack, "_phase2_div_bad_token_ids", []) or [])
        if not bad:
            return None
        coef = float(getattr(attack, "phase2_div_loss_coef", 0.0))
        if coef <= 0.0:
            return None
        div_losses = getattr(attack, "_last_phase2_div_losses", None)
        if div_losses is None:
            return None
        try:
            return [(div_losses.to(device=device, dtype=torch.float32), coef)]
        except Exception:
            return None
