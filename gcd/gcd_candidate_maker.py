"""
Candidate generation for pure-diffusion GCD (grad_coef=0).
"""

from __future__ import annotations

import random
from typing import List, Optional, Set, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from gcd.gcd_core import GCDAttack


def exact_fraction_count(total: int, fraction: float) -> int:
    """Return exactly round(total * fraction) items, clamped to [1, total] (or 0 if total <= 0)."""
    total = int(total)
    if total <= 0:
        return 0
    f = float(fraction)
    if f <= 0.0:
        return 1
    if f >= 1.0:
        return total
    n = int(round(float(total) * f))
    return max(1, min(total, n))


class CandidateGenerator:
    """Position selection, top-k pair extraction, and deduplicated candidate sampling."""

    def __init__(self, attack: "GCDAttack") -> None:
        self.attack = attack

    def _random_pos_reference_seq_len(self, seq_len: int) -> int:
        r = getattr(self.attack, "random_pos_reference_len", None)
        if r is None:
            return int(seq_len)
        try:
            ri = int(r)
        except (TypeError, ValueError):
            return int(seq_len)
        return ri if ri > 0 else int(seq_len)

    def _exact_position_select_count(self, num_available: int) -> int:
        """Exactly floor(num_available * random_pos_p) positions (at least 1 when num_available > 0)."""
        p = float(getattr(self.attack, "random_pos_p", 0.25))
        return exact_fraction_count(int(num_available), p)

    def _sample_exact_position_indices(
        self,
        pool_indices: torch.Tensor,
        n_select: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample exactly n_select indices from pool without replacement."""
        n_pool = int(pool_indices.numel())
        if n_pool <= 0:
            return pool_indices
        n_select = max(1, min(n_pool, int(n_select)))
        perm = torch.randperm(n_pool, device=device)[:n_select]
        return pool_indices[perm]

    def restrict_pairs_to_positions(
        self,
        seq_len: int,
        device: torch.device,
        all_pos: torch.Tensor,
        candidate_tokens: torch.Tensor,
        pair_scores: torch.Tensor,
        selected_positions: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep only (position, token) pairs whose position is in selected_positions."""
        if selected_positions is None or len(selected_positions) == 0 or all_pos.numel() == 0:
            return all_pos, candidate_tokens, pair_scores
        pos_mask = torch.zeros((seq_len,), device=device, dtype=torch.bool)
        pos_mask[selected_positions.to(device)] = True
        keep = pos_mask[all_pos]
        return all_pos[keep], candidate_tokens[keep], pair_scores[keep]

    def exact_candidate_count_from_pairs(self, n_pairs: int) -> int:
        """Exactly round(n_pairs * candidate_batch_pct) candidates to evaluate."""
        pct = float(getattr(self.attack, "candidate_batch_pct", 1.0))
        return exact_fraction_count(int(n_pairs), pct)

    def select_positions(
        self,
        step_num: int,
        seq_len: int,
        device: torch.device,
        unfrozen_positions_effective: torch.Tensor,
        num_available: int,
    ) -> Optional[torch.Tensor]:
        """Exact-fraction position subsampling for Dream scoring and candidate search."""
        attack = self.attack
        if not getattr(attack, "select_random_pos", False):
            return None

        if getattr(attack, "pos_geom_select", False):
            positions = torch.arange(seq_len, device=device, dtype=torch.float32)
            dist_matrix = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))
            dist_matrix.fill_diagonal_(float("inf"))
            l_i = torch.min(dist_matrix, dim=1)[0]

            if getattr(attack, "consider_start_and_end_fill", False):
                dist_to_start_ghost = positions + 1
                dist_to_end_ghost = seq_len - positions
                l_i = torch.minimum(l_i, torch.minimum(dist_to_start_ghost, dist_to_end_ghost))

            coef_i = torch.pow(float(getattr(attack, "p_geom", 0.5)), l_i)
            coef_sum = coef_i.sum()
            if coef_sum > 1e-12:
                probs = coef_i / coef_sum
            else:
                probs = torch.ones(seq_len, device=device) / seq_len

            n_select = self._exact_position_select_count(num_available)

            unfrozen_positions = (
                attack._get_unfrozen_positions(step_num, seq_len)
                if getattr(attack, "block_vise_generation", False)
                else None
            )
            if unfrozen_positions is not None or int(num_available) != int(seq_len):
                probs_unfrozen = probs[unfrozen_positions_effective]
                probs_sum = probs_unfrozen.sum()
                if probs_sum > 1e-12:
                    probs_unfrozen = probs_unfrozen / probs_sum
                else:
                    probs_unfrozen = torch.ones(num_available, device=device) / num_available
                selected_indices_unfrozen = torch.multinomial(probs_unfrozen, n_select, replacement=False)
                selected = unfrozen_positions_effective[selected_indices_unfrozen]
            else:
                selected = torch.multinomial(probs, n_select, replacement=False)

            if len(selected) > 0:
                if unfrozen_positions is not None or int(num_available) != int(seq_len):
                    print(
                        f"[SelectGeomPos] step={step_num} selected {len(selected)}/{num_available} "
                        f"allowed positions (out of {seq_len} total) using geometric distribution "
                        f"(p_geom={float(getattr(attack, 'p_geom', 0.5)):.3f})"
                    )
                else:
                    print(
                        f"[SelectGeomPos] step={step_num} selected {len(selected)}/{seq_len} "
                        f"positions using geometric distribution "
                        f"(p_geom={float(getattr(attack, 'p_geom', 0.5)):.3f})"
                    )
            return selected

        n_select = self._exact_position_select_count(num_available)
        p = float(getattr(attack, "random_pos_p", 0.25))

        unfrozen_positions = (
            attack._get_unfrozen_positions(step_num, seq_len)
            if getattr(attack, "block_vise_generation", False)
            else None
        )
        if unfrozen_positions is not None or int(num_available) != int(seq_len):
            selected = self._sample_exact_position_indices(
                unfrozen_positions_effective, n_select, device
            )
            if len(selected) > 0:
                print(
                    f"[SelectRandomPos] step={step_num} selected exactly {len(selected)}/{num_available} "
                    f"allowed positions (random_pos_p={p:.3g}, out of {seq_len} total) "
                    f"for dream scores and candidate search"
                )
        else:
            pool = torch.arange(seq_len, device=device, dtype=torch.long)
            selected = self._sample_exact_position_indices(pool, n_select, device)
            if len(selected) > 0:
                print(
                    f"[SelectRandomPos] step={step_num} selected exactly {len(selected)}/{seq_len} "
                    f"positions (random_pos_p={p:.3g}) for dream scores and candidate search"
                )
        return selected

    def apply_filling_schedule_restrictions(
        self,
        step_num: int,
        seq_len: int,
        device: torch.device,
        final_scores: torch.Tensor,
    ) -> None:
        """Restrict coordinate moves to a random subset of masked positions when behind schedule."""
        attack = self.attack
        if not (getattr(attack, "filling_schedule", False) and (not attack.no_diffusion)):
            return

        target_filled = int((step_num + 1) * seq_len / int(getattr(attack, "filling_schedule_steps", 256)))
        target_filled = min(target_filled, seq_len)
        current_ids = attack.tunable_ids[0]
        masked_positions = torch.where(current_ids == attack.mask_token_id)[0]
        num_masked = len(masked_positions)
        num_filled = seq_len - num_masked

        if num_filled >= target_filled or num_masked <= 0:
            return

        random_pos_p = float(getattr(attack, "random_pos_p", 0.1))
        u_pos = torch.rand(num_masked, device=device)
        selected_mask = u_pos < random_pos_p
        if not selected_mask.any():
            selected_mask[random.randint(0, num_masked - 1)] = True
        filling_schedule_allowed_pos = masked_positions[selected_mask]
        if len(filling_schedule_allowed_pos) == 0:
            return

        print(
            f"[FillingSchedule] step={step_num} behind schedule ({num_filled}/{target_filled} filled), "
            f"restricting to {len(filling_schedule_allowed_pos)}/{num_masked} masked positions"
        )
        outside = torch.ones((seq_len,), device=device, dtype=torch.bool)
        outside[filling_schedule_allowed_pos] = False
        if bool(torch.any(outside).item()):
            outside_pos = torch.where(outside)[0]
            final_scores[outside_pos, :] = -float("inf")
            final_scores[outside_pos, current_ids[outside_pos]] = 0.0

    def topk_pairs(
        self,
        final_scores: torch.Tensor,
        current_top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-k (position, token) pairs via torch.topk (replaces NEC refinement)."""
        attack = self.attack
        mat = final_scores
        k = min(int(current_top_k), mat.shape[1])
        if bool(getattr(attack, "top_k_total", False)):
            flat = mat.reshape(-1)
            k_total = min(int(current_top_k), flat.numel())
            topk_vals, topk_flat_idx = torch.topk(flat, k=k_total, largest=True)
            all_pos = topk_flat_idx // mat.shape[1]
            candidate_tokens = topk_flat_idx % mat.shape[1]
            return all_pos, candidate_tokens, topk_vals
        topk_vals, topk_idx = torch.topk(mat, k=k, dim=1, largest=True)
        seq_len = mat.shape[0]
        all_pos = torch.arange(seq_len, device=mat.device).unsqueeze(1).expand(seq_len, k).reshape(-1)
        return all_pos, topk_idx.reshape(-1), topk_vals.reshape(-1)

    def generate_candidates(
        self,
        count: int,
        all_pos: torch.Tensor,
        candidate_tokens: torch.Tensor,
        pair_scores: Optional[torch.Tensor] = None,
        existing_eval_keys: Optional[Set[Tuple[int, ...]]] = None,
        use_history: bool = True,
        blocked_pairs: Optional[Set[Tuple[int, int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate up to count unique evaluation candidates (deduplicated)."""
        attack = self.attack
        device = all_pos.device
        seq_len = attack.tunable_ids.shape[1]

        if all_pos.numel() == 0 or count <= 0:
            return (
                torch.empty((0, seq_len), device=device, dtype=torch.long),
                torch.empty((0,), device=device, dtype=torch.long),
                torch.empty((0,), device=device, dtype=torch.long),
            )

        valid_indices: List[int] = []
        current_ids_list = attack.tunable_ids[0].tolist()
        for i in range(len(all_pos)):
            p_idx, t_idx = int(all_pos[i].item()), int(candidate_tokens[i].item())
            if blocked_pairs is not None and (p_idx, t_idx) in blocked_pairs:
                continue
            if use_history and attack.never_repeat:
                orig_tok = current_ids_list[p_idx]
                current_ids_list[p_idx] = t_idx
                if tuple(current_ids_list) in attack.history:
                    current_ids_list[p_idx] = orig_tok
                    continue
                current_ids_list[p_idx] = orig_tok
            valid_indices.append(i)

        if not valid_indices:
            return (
                torch.empty((0, seq_len), device=device, dtype=torch.long),
                torch.empty((0,), device=device, dtype=torch.long),
                torch.empty((0,), device=device, dtype=torch.long),
            )

        valid_idx_t = torch.tensor(valid_indices, device=device, dtype=torch.long)

        if pair_scores is not None and getattr(attack, "prob_based_sampling", False):
            scores = pair_scores[valid_idx_t].to(torch.float32)
            scores = scores - scores.max()
            temp = float(attack.sampling_temperature) if attack.sampling_temperature is not None else 0.6
            if temp <= 0:
                temp = 1e-6
            probs = torch.softmax(scores / temp, dim=0)
            shuffled_indices = valid_idx_t[
                torch.multinomial(probs, num_samples=len(valid_indices), replacement=False)
            ]
        else:
            shuffled_indices = valid_idx_t[torch.randperm(len(valid_indices), device=device)]

        found_candidates: List[torch.Tensor] = []
        found_pos: List[int] = []
        found_tok: List[int] = []
        found_eval_keys = existing_eval_keys if existing_eval_keys is not None else set()

        for idx in shuffled_indices:
            if len(found_candidates) >= count:
                break
            p_idx, t_idx = int(all_pos[idx].item()), int(candidate_tokens[idx].item())
            cand = attack.tunable_ids[0].clone()
            cand[p_idx] = t_idx
            if getattr(attack, "remove_str_dublicate_opt", True):
                eval_key = attack._get_eval_id_key(cand)
                if eval_key in found_eval_keys:
                    continue
                found_eval_keys.add(eval_key)
            found_candidates.append(cand)
            found_pos.append(p_idx)
            found_tok.append(t_idx)
            if use_history and attack.never_repeat:
                attack.history.add(tuple(cand.tolist()))

        if len(found_candidates) < count:
            print(f"[Warning] only {len(found_candidates)}/{count} unique candidates found")

        if not found_candidates:
            return (
                torch.empty((0, seq_len), device=device, dtype=torch.long),
                torch.empty((0,), device=device, dtype=torch.long),
                torch.empty((0,), device=device, dtype=torch.long),
            )

        return (
            torch.stack(found_candidates, dim=0),
            torch.tensor(found_pos, device=device, dtype=torch.long),
            torch.tensor(found_tok, device=device, dtype=torch.long),
        )

    def filter_candidates_by_selected_positions(
        self,
        step_num: int,
        seq_len: int,
        device: torch.device,
        new_candidate_ids: torch.Tensor,
        cand_pos: torch.Tensor,
        cand_tok: torch.Tensor,
        selected_positions_for_dream: Optional[torch.Tensor],
        extra_tensors: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Safety filter: drop candidates that flip outside selected_positions (should be a no-op)."""
        attack = self.attack
        if (
            not getattr(attack, "select_random_pos", False)
            or selected_positions_for_dream is None
            or len(selected_positions_for_dream) == 0
            or new_candidate_ids.shape[0] <= 0
        ):
            return new_candidate_ids, cand_pos, cand_tok, extra_tensors or {}

        n_orig = int(new_candidate_ids.shape[0])
        pos_mask = torch.zeros((seq_len,), device=device, dtype=torch.bool)
        pos_mask[selected_positions_for_dream.to(device)] = True
        keep_mask = pos_mask[cand_pos]
        if not keep_mask.any():
            return new_candidate_ids, cand_pos, cand_tok, extra_tensors or {}
        n_kept = int(keep_mask.sum().item())
        if n_kept == n_orig:
            return new_candidate_ids, cand_pos, cand_tok, extra_tensors or {}
        evaluated_indices = torch.where(keep_mask)[0]
        new_candidate_ids = new_candidate_ids[evaluated_indices]
        cand_pos = cand_pos[evaluated_indices]
        cand_tok = cand_tok[evaluated_indices]
        extra = {}
        if extra_tensors:
            for key, val in extra_tensors.items():
                if val is None:
                    extra[key] = None
                elif torch.is_tensor(val):
                    extra[key] = val[evaluated_indices]
                else:
                    extra[key] = val
        print(
            f"[SelectRandomPos] step={step_num} safety-filtered {n_orig} -> {n_kept} "
            f"candidates (outside selected positions)"
        )
        return new_candidate_ids, cand_pos, cand_tok, extra
