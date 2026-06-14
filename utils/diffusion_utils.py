"""
Diffusion generation utilities for Dream models.
Contains DreamGenerationMixin, DreamGenerationConfig, DreamModelOutput, and sampling functions.
"""

import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

logger = logging.get_logger(__name__)


def top_p_logits(logits, top_p=None):
    """Apply top-p (nucleus) sampling to logits."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    """Apply top-k sampling to logits."""
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
    """
    Sample tokens from logits with various strategies.
    
    Args:
        logits: Token logits [batch_size, vocab_size] or [vocab_size]
        temperature: Sampling temperature (0.0 = greedy)
        top_p: Top-p (nucleus) sampling threshold
        top_k: Top-k sampling threshold
        margin_confidence: Use margin confidence (top1 - top2)
        neg_entropy: Use negative entropy as confidence
    
    Returns:
        confidence: Confidence scores
        x0: Sampled token indices
    """
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


class LLaDAGenerationMixin:
    """Mixin class for LLaDA model diffusion generation."""

    @staticmethod
    def add_gumbel_noise(logits, temperature):
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (- torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    @staticmethod
    def get_num_transfer_tokens(mask_index, steps, fill_max_tokens_per_step=None):
        mask_num = mask_index.sum(dim=1, keepdim=True)
        if fill_max_tokens_per_step is not None:
            try:
                max_total = int(steps) * int(fill_max_tokens_per_step)
                if max_total < 0:
                    max_total = 0
                cap = torch.full_like(mask_num, max_total)
                mask_num = torch.minimum(mask_num, cap)
            except Exception:
                pass
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, :remainder[i]] += 1
        return num_transfer_tokens

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: torch.Tensor,
        steps: int = 128,
        mask_token_id: int = 156895,
        generation_logits_hook_func = None,
        allowed_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        LLaDA-specific diffusion generation. 
        Batched implementation of the provided sample code.
        """
        model = self
        x = inputs.clone()
        device = x.device
        batch_size, total_length = x.shape
        
        temperature = float(kwargs.get("temperature", 0.0))
        cfg_scale = float(kwargs.get("cfg_scale", 0.0))
        remasking = kwargs.get("remasking", "low_confidence")
        fill_max_tokens_per_step = kwargs.get("fill_max_tokens_per_step", None)
        
        mask_id = mask_token_id
        prompt_index = (x != mask_id)

        # allowed_mask should be boolean [B, L]
        if allowed_mask is not None:
            initial_mask_index = (x == mask_id) & allowed_mask
        else:
            initial_mask_index = (x == mask_id)
        
        num_transfer_tokens = self.get_num_transfer_tokens(
            initial_mask_index,
            steps,
            fill_max_tokens_per_step=fill_max_tokens_per_step,
        )

        cfg = getattr(model, "config", None)
        model_type = str(getattr(cfg, "model_type", "") or "").lower()
        use_llada2_block_mask = model_type == "llada2_moe" or "llada2" in model_type

        def _forward_logits(ids: torch.Tensor):
            if not use_llada2_block_mask:
                return model(ids).logits
            b, seq_len = ids.shape[:2]
            dt = next(model.parameters()).dtype
            attn = torch.zeros(
                b, 1, seq_len, seq_len, device=ids.device, dtype=dt
            )
            return model(input_ids=ids, attention_mask=attn).logits

        for i in range(steps):
            mask_index = (x == mask_id)
            if allowed_mask is not None:
                mask_index = mask_index & allowed_mask
                
            if not mask_index.any():
                break
                
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = _forward_logits(x_)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = _forward_logits(x)

            # Apply logits hook if provided (e.g. for forbidden tokens)
            if generation_logits_hook_func is not None:
                logits = generation_logits_hook_func(i, x, logits)

            logits_with_noise = self.add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -torch.inf if x0_p.is_floating_point() else -1000000)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(batch_size):
                k = min(num_transfer_tokens[j, i].item(), mask_index[j].sum().item())
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

        return x


@dataclass
class DreamModelOutput(ModelOutput):
    """Output from Dream model generation."""
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DreamGenerationConfig(GenerationConfig):
    """Generation configuration for Dream models with diffusion-specific parameters."""
    
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        """Validate configuration parameters."""
        pass


class DreamGenerationMixin:
    """Mixin class for Dream model diffusion generation."""

    @staticmethod
    def _per_seq_transfer_counts(
        mask_index: torch.Tensor,
        step_i: int,
        steps: int,
        s: float,
        t: float,
        fill_max_tokens_per_step: Optional[int] = None,
    ) -> torch.Tensor:
        """Number of mask positions to transfer at step ``step_i`` for each batch row."""
        mask_count = mask_index.sum(dim=1)
        if fill_max_tokens_per_step is not None:
            try:
                max_total = int(steps) * int(fill_max_tokens_per_step)
                if max_total >= 0:
                    mask_count = torch.minimum(
                        mask_count,
                        torch.full_like(mask_count, max_total),
                    )
            except Exception:
                pass
        if step_i < steps - 1:
            num_transfer = (mask_count.to(dtype=torch.float32) * (1.0 - float(s) / float(t))).long()
        else:
            num_transfer = mask_count.long()
        return torch.minimum(num_transfer, mask_count)

    @staticmethod
    def _batched_confidence_transfer(
        x: torch.LongTensor,
        mask_index: torch.Tensor,
        x0: torch.LongTensor,
        confidence: torch.Tensor,
        num_transfer: torch.Tensor,
        mask_token_id: int,
        alg_temp: Optional[float] = None,
    ) -> torch.LongTensor:
        """Confidence-ranked mask transfer with a potentially different count per batch row."""
        batch_size = x.size(0)
        num_transfer = num_transfer.clamp(min=0)
        max_k = int(num_transfer.max().item())
        if max_k <= 0:
            return x

        full_confidence = torch.full_like(x, -torch.inf, dtype=confidence.dtype)
        full_confidence[mask_index] = confidence

        if alg_temp is None or alg_temp == 0:
            _, transfer_cols = torch.topk(full_confidence, k=max_k, dim=-1)
        else:
            transfer_cols = torch.zeros(
                (batch_size, max_k), dtype=torch.long, device=x.device
            )
            for b in range(batch_size):
                k_b = int(num_transfer[b].item())
                if k_b <= 0:
                    continue
                probs = F.softmax(full_confidence[b] / alg_temp, dim=-1)
                transfer_cols[b, :k_b] = torch.multinomial(probs, num_samples=k_b)

        x_proposed = torch.full_like(x, mask_token_id)
        x_proposed[mask_index] = x0.clone()

        batch_rows = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, max_k)
        rank_ok = torch.arange(max_k, device=x.device).unsqueeze(0) < num_transfer.unsqueeze(1)
        rows = batch_rows[rank_ok]
        cols = transfer_cols[rank_ok]
        x[rows, cols] = x_proposed[rows, cols]
        return x

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            try:
                generation_config = DreamGenerationConfig.from_model_config(self.config)
            except (AttributeError, TypeError):
                # Fallback to default config if from_model_config doesn't work
                generation_config = DreamGenerationConfig()
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        """
        Generate sequences using diffusion process.
        
        Args:
            inputs: Input token IDs [batch_size, seq_len]
            generation_config: Generation configuration
            **kwargs: Additional generation parameters
        
        Returns:
            Generated sequences or DreamModelOutput
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)
        allowed_mask = kwargs.pop("allowed_mask", None)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func,
            allowed_mask=allowed_mask,
        )
        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        allowed_mask: Optional[torch.Tensor] = None,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        """
        Core diffusion sampling loop.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            generation_config: Generation configuration
            generation_tokens_hook_func: Hook function for tokens
            generation_logits_hook_func: Hook function for logits
        
        Returns:
            Generated sequences or DreamModelOutput
        """
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        fill_max_tokens_per_step = getattr(generation_config, "fill_max_tokens_per_step", None)

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        if allowed_mask is not None:
            try:
                allowed_mask = allowed_mask.to(x.device).bool()
                if allowed_mask.shape[1] < x.shape[1]:
                    allowed_mask = F.pad(allowed_mask, (0, x.shape[1] - allowed_mask.shape[1]), value=False)
                elif allowed_mask.shape[1] > x.shape[1]:
                    allowed_mask = allowed_mask[:, : x.shape[1]]
            except Exception:
                allowed_mask = None

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)
        # If requested, cap the total number of fillable masks per sample.
        if fill_max_tokens_per_step is not None:
            try:
                max_total = int(steps) * int(fill_max_tokens_per_step)
                if max_total < 0:
                    max_total = 0
                base_mask = (x == mask_token_id)
                if allowed_mask is not None:
                    base_mask = base_mask & allowed_mask
                if max_total == 0:
                    allowed_mask = torch.zeros_like(base_mask, dtype=torch.bool)
                else:
                    if (base_mask.sum(dim=1) > max_total).any():
                        capped_mask = torch.zeros_like(base_mask, dtype=torch.bool)
                        for b in range(base_mask.size(0)):
                            pos = torch.where(base_mask[b])[0]
                            if pos.numel() == 0:
                                continue
                            k = min(int(max_total), int(pos.numel()))
                            capped_mask[b, pos[:k]] = True
                        allowed_mask = capped_mask
            except Exception:
                pass

        for i in range(steps):
            mask_index = (x == mask_token_id)
            if allowed_mask is not None:
                mask_index = mask_index & allowed_mask
            logits = self(x, attention_mask, tok_idx).logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)

            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]

            if alg == 'origin':
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                x[mask_index] = x0.clone()
            else:
                if alg == 'maskgit_plus':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                elif alg == 'topk_margin':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
                elif alg == 'entropy':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")
                num_transfer = DreamGenerationMixin._per_seq_transfer_counts(
                    mask_index,
                    step_i=i,
                    steps=steps,
                    s=s,
                    t=t,
                    fill_max_tokens_per_step=fill_max_tokens_per_step,
                )
                if int(num_transfer.max().item()) > 0:
                    x = DreamGenerationMixin._batched_confidence_transfer(
                        x,
                        mask_index,
                        x0,
                        confidence,
                        num_transfer,
                        mask_token_id,
                        alg_temp=alg_temp,
                    )

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)

            if histories is not None:
                histories.append(x.clone())

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x

