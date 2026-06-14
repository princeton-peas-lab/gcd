"""
Dream model integration methods.
"""

import gc
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


class GcddreamMixin:
    """Mixin class for dream model integration methods."""

    def _wrap_tunable_with_format(self, tunable_ids: torch.Tensor) -> torch.Tensor:
        """
        Wrap tunable token ids with the prompt format template.
        """
        if not getattr(self, 'prompt_format_diffusion', False):
            return tunable_ids

        B = tunable_ids.shape[0]
        device = tunable_ids.device
        parts = []

        prefix_ids = getattr(self, '_prompt_format_prefix_ids', None)
        suffix_ids = getattr(self, '_prompt_format_suffix_ids', None)

        if prefix_ids is not None:
            parts.append(prefix_ids.expand(B, -1).to(device))
        parts.append(tunable_ids)
        if suffix_ids is not None:
            parts.append(suffix_ids.expand(B, -1).to(device))

        return torch.cat(parts, dim=1) if len(parts) > 1 else tunable_ids

    def _dream_tunable_start_idx(self) -> int:
        """
        Starting index (in Dream input ids) of the tunable span we slice out after diffusion filling.
        """
        fmt_prefix_len = int(getattr(self, "_prompt_format_prefix_len", 0) or 0) if getattr(
            self, "prompt_format_diffusion", False
        ) else 0
        include_fixed = True
        if getattr(self, "prompt_format_diffusion", False):
            include_fixed = bool(getattr(self, "prompt_format_include_fixed_user", True))
        fixed_len = int(self.fixed_user_ids.shape[1]) if include_fixed else 0
        return int(self.dream_prefix_ids.shape[1] + fmt_prefix_len + fixed_len)

    def _dream_input_for_eval_fill(self, batch_candidates_unfilled: torch.Tensor) -> torch.Tensor:
        """
        Build Dream diffusion input ids for eval-time filling.

        Layout: dream_prefix_ids + [optional fixed_user_ids +] tunable_ids + eos
        When prompt_format_diffusion is enabled, the tunable span is wrapped in a guiding template.
        """
        B = int(batch_candidates_unfilled.shape[0])

        if getattr(self, 'prompt_format_diffusion', False):
            if bool(getattr(self, "prompt_format_include_fixed_user", True)):
                batch_with_fixed = torch.cat(
                    [self.fixed_user_ids.expand(B, -1), batch_candidates_unfilled],
                    dim=1,
                )
            else:
                batch_with_fixed = batch_candidates_unfilled
            batch_candidates_unfilled = self._wrap_tunable_with_format(batch_with_fixed)
            return torch.cat(
                [
                    self.dream_prefix_ids.expand(B, -1),
                    batch_candidates_unfilled,
                    self.dream_eos_id.expand(B, -1),
                ],
                dim=1,
            )

        return torch.cat(
            [
                self.dream_prefix_ids.expand(B, -1),
                self.fixed_user_ids.expand(B, -1),
                batch_candidates_unfilled,
                self.dream_eos_id.expand(B, -1),
            ],
            dim=1,
        )

    def _dream_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Build an explicit all-ones attention mask for Dream diffusion_generate."""
        return torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

    def _dream_diffusion_generate(
        self,
        inputs: torch.Tensor,
        steps: int,
        mask_token_id: int,
        generation_logits_hook_func,
        allowed_mask=None,
        temperature: Optional[float] = None,
    ):
        """
        Wrapper around Dream diffusion_generate with micro-batching and adaptive OOM recovery.

        ``temperature``: if None, uses ``self.diffusion_temperature`` from config
        (e.g. ``diffusion_temperature: 0.0`` for greedy filling).
        """
        eff_temp = (
            float(temperature)
            if temperature is not None
            else float(getattr(self, "diffusion_temperature", 1.0))
        )
        B = int(inputs.shape[0])

        optimize_bs = bool(getattr(self, "optimize_batch_size", False))
        tuned_bs = getattr(self, "_tuned_dream_fill_batch_size", None)
        configured_bs = getattr(self, "dream_fill_eval_batch_size", None)

        try:
            if tuned_bs is not None:
                bs = int(tuned_bs)
            elif configured_bs is not None:
                bs = int(configured_bs)
            else:
                bs = None
        except Exception:
            bs = None

        fill_max = getattr(self, "fill_max_tokens_per_step", None)

        def _call(chunk: torch.Tensor, chunk_allowed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            gen_kwargs = dict(
                inputs=chunk,
                attention_mask=self._dream_attention_mask(chunk),
                steps=steps,
                mask_token_id=mask_token_id,
                generation_logits_hook_func=generation_logits_hook_func,
                alg=getattr(self, "dream_alg", "origin"),
                temperature=eff_temp,
            )
            if fill_max is not None:
                gen_kwargs["fill_max_tokens_per_step"] = int(fill_max)
            if chunk_allowed_mask is not None:
                gen_kwargs["allowed_mask"] = chunk_allowed_mask
            return self.dream_model.diffusion_generate(**gen_kwargs)

        def _is_oom_or_cuda_error(e: Exception) -> bool:
            msg = str(e).lower()
            if "invalid configuration argument" in msg:
                return True
            if hasattr(self, "_should_reduce_batch_size"):
                return self._should_reduce_batch_size(e)
            return False

        def _clear_cache():
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        if bs is None or bs <= 0 or B <= bs:
            try:
                return _call(inputs, allowed_mask)
            except RuntimeError as e:
                if optimize_bs and _is_oom_or_cuda_error(e) and B > 1:
                    new_bs = max(1, B // 2)
                    self._tuned_dream_fill_batch_size = new_bs
                    if not bool(getattr(self, "_printed_tuned_dream_batch_size", False)):
                        self._printed_tuned_dream_batch_size = True
                        print(f"[GCDAttack] optimize_batch_size OOM in Dream fill -> dream_fill_batch_size={new_bs}")
                    _clear_cache()
                    return self._dream_diffusion_generate(
                        inputs,
                        steps=steps,
                        mask_token_id=mask_token_id,
                        generation_logits_hook_func=generation_logits_hook_func,
                        allowed_mask=allowed_mask,
                        temperature=temperature,
                    )

                msg = str(e).lower()
                if ("invalid configuration argument" in msg) and (B > 1):
                    mid = B // 2
                    left = self._dream_diffusion_generate(
                        inputs[:mid],
                        steps=steps,
                        mask_token_id=mask_token_id,
                        generation_logits_hook_func=generation_logits_hook_func,
                        allowed_mask=allowed_mask[:mid] if allowed_mask is not None else None,
                        temperature=temperature,
                    )
                    right = self._dream_diffusion_generate(
                        inputs[mid:],
                        steps=steps,
                        mask_token_id=mask_token_id,
                        generation_logits_hook_func=generation_logits_hook_func,
                        allowed_mask=allowed_mask[mid:] if allowed_mask is not None else None,
                        temperature=temperature,
                    )
                    return torch.cat([left, right], dim=0)
                if ("invalid configuration argument" in msg) and (B == 1):
                    if not bool(getattr(self, "_diffusion_fail_warned", False)):
                        self._diffusion_fail_warned = True
                        print(
                            "[GCDAttack] WARNING: Dream diffusion_generate failed with CUDA 'invalid configuration argument' "
                            "even at batch=1. Falling back to NO diffusion filling."
                        )
                    return inputs
                raise

        outs = []
        i = 0
        current_bs = bs
        while i < B:
            chunk = inputs[i : i + current_bs]
            chunk_allowed = (
                allowed_mask[i : i + current_bs] if allowed_mask is not None else None
            )
            try:
                outs.append(_call(chunk, chunk_allowed))
                i += current_bs
            except RuntimeError as e:
                if _is_oom_or_cuda_error(e) and current_bs > 1:
                    current_bs = max(1, current_bs // 2)
                    if optimize_bs:
                        self._tuned_dream_fill_batch_size = current_bs
                        print(f"[GCDAttack] optimize_batch_size OOM in Dream fill -> dream_fill_batch_size={current_bs}")
                    _clear_cache()
                    continue
                elif _is_oom_or_cuda_error(e) and current_bs == 1:
                    if not bool(getattr(self, "_diffusion_fail_warned", False)):
                        self._diffusion_fail_warned = True
                        print("[GCDAttack] WARNING: Dream diffusion_generate failed even at batch=1. Falling back to NO diffusion filling for this chunk.")
                    outs.append(chunk)
                    i += current_bs
                else:
                    raise
        return torch.cat(outs, dim=0)

    def _dream_generation_logits_hook(self, step, x, logits):
        """Logits hook for Dream diffusion sampling to forbid certain token ids."""
        if self._dream_forbidden_token_tensor is None:
            return logits
        if logits is None:
            return logits
        try:
            logits.index_fill_(-1, self._dream_forbidden_token_tensor, -float("inf"))
        except Exception:
            pass
        return logits

    def _dream_offsets_from_current_ids(self, text: str, ids_override: Optional[List[int]] = None) -> List[Tuple[int, int]]:
        """
        Construct per-position character offsets for the current Dream tunable ids.
        """
        ids = ids_override if ids_override is not None else self.tunable_ids[0].tolist()
        offsets: List[Tuple[int, int]] = []
        prev = ""
        for j in range(len(ids)):
            cur = self.tokenizer.decode(ids[: j + 1], skip_special_tokens=False)
            offsets.append((len(prev), len(cur)))
            prev = cur
        if len(prev) != len(text):
            L = len(text)
            offsets = [(max(0, min(s, L)), max(0, min(e, L))) for (s, e) in offsets]
        return offsets

    def _compute_block_scores(self) -> torch.Tensor:
        """
        Compute Dream-only block scores by masking each block and running Dream.
        Returns block_scores: shape [n_blocks], float32.
        """
        device = self.dream_model.device
        seq_len = int(self.tunable_ids.shape[1])
        tunable_start_idx = int(self._dream_tunable_start_idx())
        bs = max(1, int(getattr(self, "block_size", 5)))
        n_blocks = (seq_len + bs - 1) // bs

        if getattr(self, 'prompt_format_diffusion', False):
            if bool(getattr(self, "prompt_format_include_fixed_user", True)):
                tunable_with_fixed = torch.cat([self.fixed_user_ids, self.tunable_ids], dim=1)
            else:
                tunable_with_fixed = self.tunable_ids
            tunable_for_scoring = self._wrap_tunable_with_format(tunable_with_fixed)
            base_ids = torch.cat([self.dream_prefix_ids, tunable_for_scoring, self.dream_eos_id], dim=1)
        else:
            base_ids = torch.cat([self.dream_prefix_ids, self.fixed_user_ids, self.tunable_ids, self.dream_eos_id], dim=1)

        masked_batch = base_ids.repeat(n_blocks, 1)
        for b in range(n_blocks):
            start = b * bs
            end = min(seq_len, (b + 1) * bs)
            if end > start:
                masked_batch[b, tunable_start_idx + start: tunable_start_idx + end] = self.mask_token_id

        logits = self.dream_model(input_ids=masked_batch).logits  # [B, T, V]
        k = max(1, min(int(getattr(self, "block_mean_compute_top_k", 32)), logits.size(-1)))

        block_scores = torch.empty((n_blocks,), device=device, dtype=torch.float32)
        for b in range(n_blocks):
            start = b * bs
            end = min(seq_len, (b + 1) * bs)
            if end <= start:
                block_scores[b] = -float("inf")
                continue
            pos_indices = torch.arange(start, end, device=device)
            logit_positions = (tunable_start_idx + pos_indices - 1).clamp(min=0)
            rel = logits[b, logit_positions, :]  # [block_len, V]
            topv, _ = torch.topk(rel, k=k, dim=-1)
            block_scores[b] = topv.mean(dim=-1).mean()
        return block_scores

    def _maybe_print_template_diff(self, dream_input_batch: torch.Tensor):
        """Print one example of the Dream filling template. Printed at most once."""
        if (not self.print_template_diff) or self._printed_template_diff:
            return

    def _maybe_print_dream_fill_input_each_step(self, step_num: int, dream_input_batch: torch.Tensor):
        """Print the full backend fill input (decoded) once per optimization step."""
        if not bool(self.print_dream_fill_input_each_step):
            return
        try:
            step_num_i = int(step_num)
        except Exception:
            step_num_i = step_num
        try:
            if self._last_printed_fill_step == step_num_i:
                return
        except Exception:
            pass
        try:
            if dream_input_batch is None or dream_input_batch.numel() == 0:
                return
            ids0 = dream_input_batch[0].detach().cpu().tolist()
            txt = self.tokenizer.decode(ids0, skip_special_tokens=False)
            print(f"\n{self._lp()}[GCDAttack] ===== Backend fill input (step={step_num_i}, B={int(dream_input_batch.shape[0])}, L={int(dream_input_batch.shape[1])}) =====")
            print(txt)
            print(f"{self._lp()}[GCDAttack] ===== end fill input =====\n")
            self._last_printed_fill_step = step_num_i
        except Exception:
            return

    def _maybe_print_dream_score_input_each_step(self, step_num: int, base_ids: torch.Tensor):
        """Print the full backend scoring input (decoded) at step 0 only."""
        if not bool(self.print_dream_score_input_each_step):
            return
        try:
            step_num_i = int(step_num)
        except Exception:
            step_num_i = step_num
        try:
            if self._last_printed_score_step is not None:
                return
        except Exception:
            return
        if step_num_i != 0:
            return
        try:
            if base_ids is None or base_ids.numel() == 0:
                return
            ids0 = base_ids[0].detach().cpu().tolist()
            txt0 = self.tokenizer.decode(ids0, skip_special_tokens=False)
            print(f"\n{self._lp()}[GCDAttack] ===== Dream SCORE INPUT (step={step_num_i}, B={int(base_ids.shape[0])}, L={int(base_ids.shape[1])}) =====")
            print(txt0)
            print("<|im_end|>")
            print(f"{self._lp()}[GCDAttack] ===== end SCORE INPUT =====\n")
            self._last_printed_score_step = step_num_i
        except Exception:
            return

    def get_dream_scores(self, mask_p=0.5, return_logprobs: bool = False, return_probs: bool = False, selected_positions=None):
        """
        Get Dream model scores for candidate tokens.

        Args:
            mask_p: Probability of masking tokens for base computation
            return_logprobs: Whether to return log probabilities
            return_probs: Whether to return probabilities (for smart mask filling)
            selected_positions: Optional 1D tensor of position indices to compute scores for.

        Returns:
            scores: [seq_len, vocab_size]
            precomputed_best_tokens: [seq_len]
            (optional) logprobs or probs
        """
        len_tunable = int(self.tunable_ids.shape[1])
        device = self.dream_model.device
        tunable_start_idx = int(self._dream_tunable_start_idx())

        if selected_positions is not None:
            selected_positions = selected_positions.to(device)
            selected_positions = selected_positions[(selected_positions >= 0) & (selected_positions < len_tunable)]
            if len(selected_positions) == 0:
                selected_positions = None
            else:
                selected_positions = torch.unique(selected_positions)

        # Build base ids (non-amortized, Dream causal)
        if getattr(self, 'prompt_format_diffusion', False):
            if bool(getattr(self, "prompt_format_include_fixed_user", True)):
                tunable_with_fixed = torch.cat([self.fixed_user_ids, self.tunable_ids], dim=1)
            else:
                tunable_with_fixed = self.tunable_ids
            tunable_for_scoring = self._wrap_tunable_with_format(tunable_with_fixed)
            base_ids = torch.cat([self.dream_prefix_ids, tunable_for_scoring, self.dream_eos_id], dim=1)
        else:
            base_ids = torch.cat([self.dream_prefix_ids, self.fixed_user_ids, self.tunable_ids, self.dream_eos_id], dim=1)

        try:
            step_num = self._current_step_num if getattr(self, "_current_step_num", None) is not None else -1
        except Exception:
            step_num = -1
        self._maybe_print_dream_score_input_each_step(step_num, base_ids)

        # Dream (causal): logits at position i predict token i+1 -> use [start-1 : start+len-1]
        current_outputs = self.dream_model(input_ids=base_ids).logits
        current_tunable_logits = current_outputs[:, tunable_start_idx - 1 : tunable_start_idx + len_tunable - 1, :]
        current_probs_dist = F.softmax(current_tunable_logits.squeeze(0), dim=-1)
        actual_ids = self.tunable_ids[0]
        token_probs = current_probs_dist[torch.arange(len_tunable, device=device), actual_ids]

        num_to_mask = int(mask_p * len_tunable)
        masked_base_ids = base_ids.clone()
        already_masked_positions = torch.tensor([], dtype=torch.long, device=device)
        if num_to_mask > 0:
            _, indices_to_mask = torch.topk(token_probs, k=num_to_mask, largest=False)
            masked_base_ids[0, tunable_start_idx + indices_to_mask] = self.mask_token_id
            already_masked_positions = indices_to_mask

        if selected_positions is not None and self.pre_compute_mask:
            already_masked_selected = selected_positions[torch.isin(selected_positions, already_masked_positions)]
            need_masking_selected = selected_positions[~torch.isin(selected_positions, already_masked_positions)]

            vocab_size = None
            relevant_logits = None

            if len(already_masked_selected) > 0:
                masked_base_logits = self.dream_model(input_ids=masked_base_ids).logits
                masked_base_tunable_logits = masked_base_logits[:, tunable_start_idx - 1 : tunable_start_idx + len_tunable - 1, :].squeeze(0)
                vocab_size = masked_base_tunable_logits.shape[-1]
                relevant_logits = torch.zeros((len_tunable, vocab_size), device=device, dtype=masked_base_tunable_logits.dtype)
                relevant_logits[already_masked_selected] = masked_base_tunable_logits[already_masked_selected]

            if len(need_masking_selected) > 0:
                n_need_masking = len(need_masking_selected)
                masked_batch = masked_base_ids.repeat(n_need_masking, 1)
                need_masking_tunable_indices = tunable_start_idx + need_masking_selected
                masked_batch[torch.arange(n_need_masking), need_masking_tunable_indices] = self.mask_token_id
                logits = self.dream_model(input_ids=masked_batch).logits
                relevant_logits_need_masking = logits[torch.arange(n_need_masking), need_masking_tunable_indices - 1, :]

                if vocab_size is None:
                    vocab_size = relevant_logits_need_masking.shape[-1]
                    relevant_logits = torch.zeros((len_tunable, vocab_size), device=device, dtype=relevant_logits_need_masking.dtype)
                relevant_logits[need_masking_selected] = relevant_logits_need_masking

            if relevant_logits is None:
                n_selected = len(selected_positions)
                masked_batch = masked_base_ids.repeat(n_selected, 1)
                selected_tunable_indices = tunable_start_idx + selected_positions
                masked_batch[torch.arange(n_selected), selected_tunable_indices] = self.mask_token_id
                logits = self.dream_model(input_ids=masked_batch).logits
                relevant_logits_selected = logits[torch.arange(n_selected), selected_tunable_indices - 1, :]
                vocab_size = relevant_logits_selected.shape[-1]
                relevant_logits = torch.zeros((len_tunable, vocab_size), device=device, dtype=relevant_logits_selected.dtype)
                relevant_logits[selected_positions] = relevant_logits_selected

        elif self.pre_compute_mask:
            already_masked_mask = torch.isin(torch.arange(len_tunable, device=device), already_masked_positions)
            need_masking_positions = torch.arange(len_tunable, device=device)[~already_masked_mask]

            vocab_size = None
            relevant_logits = None

            if len(already_masked_positions) > 0:
                masked_base_logits = self.dream_model(input_ids=masked_base_ids).logits
                masked_base_tunable_logits = masked_base_logits[:, tunable_start_idx - 1 : tunable_start_idx + len_tunable - 1, :].squeeze(0)
                vocab_size = masked_base_tunable_logits.shape[-1]
                relevant_logits = torch.zeros((len_tunable, vocab_size), device=device, dtype=masked_base_tunable_logits.dtype)
                relevant_logits[already_masked_positions] = masked_base_tunable_logits[already_masked_positions]

            if len(need_masking_positions) > 0:
                n_need_masking = len(need_masking_positions)
                masked_batch = masked_base_ids.repeat(n_need_masking, 1)
                tunable_range_indices = tunable_start_idx + need_masking_positions
                masked_batch[torch.arange(n_need_masking), tunable_range_indices] = self.mask_token_id
                logits = self.dream_model(input_ids=masked_batch).logits
                relevant_logits_need_masking = logits[torch.arange(n_need_masking), tunable_range_indices - 1, :]

                if vocab_size is None:
                    vocab_size = relevant_logits_need_masking.shape[-1]
                    relevant_logits = torch.zeros((len_tunable, vocab_size), device=device, dtype=relevant_logits_need_masking.dtype)
                relevant_logits[need_masking_positions] = relevant_logits_need_masking

            if relevant_logits is None:
                masked_batch = masked_base_ids.repeat(len_tunable, 1)
                tunable_range_indices = torch.arange(tunable_start_idx, tunable_start_idx + len_tunable, device=device)
                masked_batch[torch.arange(len_tunable), tunable_range_indices] = self.mask_token_id
                logits = self.dream_model(input_ids=masked_batch).logits
                relevant_logits = logits[torch.arange(len_tunable), tunable_range_indices - 1, :]
        else:
            logits = self.dream_model(input_ids=masked_base_ids).logits
            relevant_logits = logits[:, tunable_start_idx - 1 : tunable_start_idx + len_tunable - 1, :].squeeze(0)

        precomputed_best_tokens = torch.argmax(relevant_logits, dim=-1)
        if self.substract_current:
            current_ids = self.tunable_ids[0]
            current_token_scores = relevant_logits[torch.arange(len_tunable, device=device), current_ids].unsqueeze(1)
            scores = relevant_logits - current_token_scores
        else:
            scores = relevant_logits

        probs = None
        if return_probs:
            probs = F.softmax(relevant_logits.to(torch.float32), dim=-1)

        if not return_logprobs:
            if return_probs:
                return scores, precomputed_best_tokens, probs
            return scores, precomputed_best_tokens

        logprobs = F.log_softmax(relevant_logits.to(torch.float32), dim=-1)
        if return_probs:
            return scores, precomputed_best_tokens, logprobs, probs
        return scores, precomputed_best_tokens, logprobs

    def _smart_fill_masks_with_precomputed(
        self,
        batch_to_eval: torch.Tensor,
        precomputed_tokens: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fill mask positions using precomputed tokens, but when multiple consecutive mask positions
        recommend the same token, only fill the position with the highest probability.
        """
        device = batch_to_eval.device
        B = batch_to_eval.shape[0]

        for b in range(B):
            mask_locs = (batch_to_eval[b] == self.mask_token_id)
            if not mask_locs.any():
                continue

            mask_positions = torch.where(mask_locs)[0]
            if len(mask_positions) == 0:
                continue

            sorted_positions = torch.sort(mask_positions)[0]
            mask_blocks = []
            current_block = [sorted_positions[0].item()]

            for i in range(1, len(sorted_positions)):
                if sorted_positions[i].item() == sorted_positions[i-1].item() + 1:
                    current_block.append(sorted_positions[i].item())
                else:
                    mask_blocks.append(torch.tensor(current_block, device=device, dtype=torch.long))
                    current_block = [sorted_positions[i].item()]
            mask_blocks.append(torch.tensor(current_block, device=device, dtype=torch.long))

            filler = precomputed_tokens.clone()

            for block_positions in mask_blocks:
                if len(block_positions) == 0:
                    continue

                block_recommended = precomputed_tokens[block_positions]
                block_probs = probs[block_positions]
                block_token_probs = block_probs[torch.arange(len(block_positions), device=device), block_recommended]
                unique_tokens = torch.unique(block_recommended)

                for token in unique_tokens:
                    token_mask = (block_recommended == token)
                    positions_with_token = block_positions[token_mask]

                    if len(positions_with_token) > 1:
                        probs_for_token = block_token_probs[token_mask]
                        best_pos_idx = torch.argmax(probs_for_token)
                        best_pos = positions_with_token[best_pos_idx]
                        filler[positions_with_token] = self.mask_token_id
                        filler[best_pos] = token

            batch_to_eval[b, mask_locs] = filler[mask_locs]

        return batch_to_eval
