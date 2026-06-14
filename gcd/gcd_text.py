"""
Text processing and tokenization utilities.
"""

import torch
from typing import List, Tuple, Optional, Set, Dict, Any


class GcdtextMixin:
    """Mixin class for text processing and tokenization utilities."""

    def _extract_prompt_from_format_text(self, wrapped_text: str) -> str:
        """
        Extract the actual prompt text from the wrapped format text.

        When prompt_format_diffusion is enabled, the text looks like:
            "Sure, here's your desired prompt: '{actual_prompt}'"
        This method extracts just the '{actual_prompt}' portion.

        The input may be either the raw diffusion output (starts with prefix_text)
        or the full victim [INST] block (contains the instruction template BEFORE the
        format wrapper).  We use ``find`` rather than ``startswith`` so that extraction
        succeeds in both cases, keeping guard evaluation consistent between the per-step
        tracking path and the pool-entry ``_defence_safe_for_pool_entry`` path.
        """
        if not getattr(self, 'prompt_format_diffusion', False):
            return wrapped_text

        prefix_text = getattr(self, '_prompt_format_prefix_text', None)
        suffix_text = getattr(self, '_prompt_format_suffix_text', None)

        if prefix_text is None and suffix_text is None:
            return wrapped_text

        result = wrapped_text
        if prefix_text:
            idx = result.find(prefix_text)
            if idx >= 0:
                result = result[idx + len(prefix_text):]
        if suffix_text and result.endswith(suffix_text):
            result = result[:-len(suffix_text)]
        return result

    def _victim_system_content_for_messages(self) -> Optional[str]:
        """System message content for victim chat template, or None to omit the system role."""
        if bool(getattr(self, "default_system_prompt", False)):
            sys_text = getattr(self, "_system_text", None)
            if sys_text is None:
                return None
            sys_text = str(sys_text).strip()
            return sys_text if sys_text else None
        sys_text = getattr(self, "_system_text", None)
        if sys_text is None:
            return "You are a helpful assistant."
        sys_text = str(sys_text)
        if not sys_text.strip():
            return None
        return sys_text

    def _build_victim_chat_messages(self, user_content: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        sys_content = self._victim_system_content_for_messages()
        if sys_content is not None:
            messages.append({"role": "system", "content": sys_content})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _apply_victim_chat_template(self, user_content: str, add_generation_prompt: bool = True) -> str:
        vt = self.victim_tokenizer
        messages = self._build_victim_chat_messages(user_content)
        try:
            return vt.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except TypeError:
            return vt.apply_chat_template(messages, tokenize=False)

    def _victim_prompt_text_for_user_content(
        self, user_content: str, add_generation_prompt: bool = True
    ) -> str:
        """Build victim-view prompt text for an explicit user message (fixed + tunable text)."""
        vt = self.victim_tokenizer
        if vt is not None and hasattr(vt, "apply_chat_template"):
            try:
                return self._apply_victim_chat_template(
                    user_content, add_generation_prompt=add_generation_prompt
                )
            except Exception:
                sys_content = self._victim_system_content_for_messages()
                if sys_content is not None:
                    fallback_user = f"{sys_content}\n\n{user_content}"
                else:
                    fallback_user = user_content
                msgs = [{"role": "user", "content": fallback_user}]
                try:
                    return vt.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=add_generation_prompt
                    )
                except TypeError:
                    return vt.apply_chat_template(msgs, tokenize=False)
                except Exception:
                    pass
        sys_content = self._victim_system_content_for_messages()
        if bool(getattr(self, "default_system_prompt", False)) and sys_content is None:
            return f"{user_content}{self._suffix_str if add_generation_prompt else ''}"
        prompt = f"{self._prefix_str}{user_content}"
        if add_generation_prompt:
            prompt = f"{prompt}{self._suffix_str}"
        return prompt

    def _victim_user_message_shift_bounds(self, prompt_token_len: int) -> Tuple[int, int]:
        """
        Shift-index span [start, end) for self-perplexity over the full user message
        (fixed user text + tunable), excluding system/template prefix and assistant gen suffix.
        """
        if self.victim_tokenizer is None:
            return 0, max(0, prompt_token_len - 1)
        try:
            if getattr(self, "_victim_user_msg_shift_start", None) is None:
                vt = self.victim_tokenizer
                if self._should_use_victim_chat_template():
                    # Use a marker to find the exact token position where user content
                    # starts and how long the user-turn closing tag is (e.g. <|eot_id|>
                    # for Llama-3).  The old heuristic used len(empty_prompt_ids) which
                    # includes the closing tag, causing an off-by-one: shift_start was 1
                    # too high (predicting t2 instead of t1) and shift_end was 1 too high
                    # (including the closing tag token instead of t_last).
                    _marker = "XPPLMARKERX"
                    _marker_ids = list(vt(_marker, add_special_tokens=False)["input_ids"])
                    _marker_prompt = self._victim_prompt_text_for_user_content(
                        _marker, add_generation_prompt=False
                    )
                    _marker_prompt_ids = list(
                        vt(_marker_prompt, add_special_tokens=False)["input_ids"]
                    )
                    _user_start = None
                    for _i in range(len(_marker_prompt_ids) - len(_marker_ids) + 1):
                        if _marker_prompt_ids[_i : _i + len(_marker_ids)] == _marker_ids:
                            _user_start = _i
                            break
                    if _user_start is not None:
                        user_token_start = _user_start
                        _closing_tag_len = (
                            len(_marker_prompt_ids) - _user_start - len(_marker_ids)
                        )
                    else:
                        # Fallback: old heuristic (off-by-one for templates with closing tag)
                        _empty_prompt = self._victim_prompt_text_for_user_content(
                            "", add_generation_prompt=False
                        )
                        user_token_start = len(
                            vt(_empty_prompt, add_special_tokens=False)["input_ids"]
                        )
                        _closing_tag_len = 0
                    probe_user = str(getattr(self, "_fixed_user_text", "") or "")[:32] or "."
                    no_gen = self._victim_prompt_text_for_user_content(
                        probe_user, add_generation_prompt=False
                    )
                    with_gen = self._victim_prompt_text_for_user_content(
                        probe_user, add_generation_prompt=True
                    )
                    # gen_suffix_len = tokens added by add_generation_prompt (assistant header)
                    # + closing_tag_len (e.g. <|eot_id|>) so that shift_end correctly
                    # excludes the closing tag from the scored span.
                    gen_suffix_len = max(
                        0,
                        len(vt(with_gen, add_special_tokens=False)["input_ids"])
                        - len(vt(no_gen, add_special_tokens=False)["input_ids"]),
                    ) + _closing_tag_len
                else:
                    user_token_start = len(
                        vt(self._prefix_str, add_special_tokens=False)["input_ids"]
                    )
                    gen_suffix_len = len(
                        vt(self._suffix_str, add_special_tokens=False)["input_ids"]
                    )
                self._victim_user_msg_token_start = int(user_token_start)
                self._victim_user_msg_gen_suffix_len = int(gen_suffix_len)
                # Start from the SECOND user-content token (skip t1) so that the
                # first token — predicted with minimal context — doesn't dominate PPL.
                self._victim_user_msg_shift_start = int(user_token_start)
            shift_start = int(self._victim_user_msg_shift_start)
            shift_end = int(prompt_token_len) - int(self._victim_user_msg_gen_suffix_len) - 1
            shift_end = max(shift_start, shift_end)
            return shift_start, shift_end
        except Exception:
            return 0, max(0, prompt_token_len - 1)

    def _build_victim_prompt_text(self, tunable_text: str, run_phase2: bool = False) -> str:
        """
        Build the victim-view prompt text for a single tunable suffix string.
        """
        if getattr(self, 'prompt_format_diffusion', False):
            tunable_text = self._extract_prompt_from_format_text(tunable_text)
        user_content = f"{self._fixed_user_text}{tunable_text}"
        return self._victim_prompt_text_for_user_content(user_content, add_generation_prompt=True)

    def _tunable_text_for_victim(self) -> str:
        """
        Decode current Dream tunable ids into text for victim-side operations.
        """
        if self._skip_mask_for_victim:
            ids = [tid for tid in self.tunable_ids[0].tolist() if tid != int(self.mask_token_id)]
            return self.tokenizer.decode(ids, skip_special_tokens=False) if len(ids) > 0 else ""
        # Decode with Dream tokenizer directly — all Dream IDs are valid in Dream's
        # vocab, so _victim_safe_ids (which clips to victim vocab size) must not be
        # applied here.
        return self.tokenizer.decode(self.tunable_ids[0].tolist(), skip_special_tokens=False)

    def _safe_decode_ids(self, ids: List[int]) -> str:
        """
        Robust decoding for logging/debugging.
        Removes null bytes and control characters that might cause truncation in logs.
        """
        try:
            s = self.tokenizer.decode(ids, skip_special_tokens=False)
            if s is not None:
                return s.replace("\0", "[NULL]")
        except Exception:
            pass

        toks = None
        try:
            try:
                toks = self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
            except TypeError:
                toks = self.tokenizer.convert_ids_to_tokens(ids)
        except Exception:
            toks = None

        if not isinstance(toks, list) or len(toks) == 0:
            return ""

        out_parts: List[str] = []
        for tid, t in zip(ids, toks):
            if isinstance(t, str):
                out_parts.append(t.replace("\0", "[NULL]"))
            elif t is None:
                out_parts.append(f"<|id:{int(tid)}|>")
            else:
                out_parts.append(str(t))

        try:
            if hasattr(self.tokenizer, "convert_tokens_to_string"):
                s = str(self.tokenizer.convert_tokens_to_string(out_parts))
                return s.replace("\0", "[NULL]")
        except Exception:
            pass
        return "".join(out_parts).replace("\0", "[NULL]")

    def _decode_for_victim(self, ids: Optional[torch.Tensor] = None, text: Optional[str] = None) -> str:
        """
        Decode token IDs or text to a string as seen by the victim (handling masks).
        """
        if text is not None:
            if not self.fill_mask:
                return text
            if self.fill_mask_value == "":
                return text.replace(self._mask_token_text, "")
            return text.replace(self._mask_token_text, self.fill_mask_value)

        if ids is None:
            return ""
        if ids.dim() > 1:
            ids = ids[0]
        row = ids.tolist()
        if getattr(self, "_skip_mask_for_victim", False):
            filtered = [tid for tid in row if tid != self.mask_token_id]
            return self._safe_decode_ids(filtered)
        # Decode with Dream tokenizer directly — all Dream IDs are valid in Dream's
        # vocab, so _victim_safe_ids must not be applied here.
        return self._safe_decode_ids(ids.tolist())

    def _get_eval_id_key(self, ids: torch.Tensor) -> Tuple[int, ...]:
        """
        Convert a sequence of token IDs into a tuple that uniquely identifies
        the resulting string when masks are removed.
        """
        row = ids.tolist()
        if self._skip_mask_for_victim:
            return tuple(tid for tid in row if tid != self.mask_token_id)
        if self.fill_mask and self._fill_mask_token_id is not None:
            return tuple(self._fill_mask_token_id if tid == self.mask_token_id else tid for tid in row)
        return tuple(row)
