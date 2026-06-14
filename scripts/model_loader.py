"""Model loading utilities for GCD experiments."""

import os
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.diffusion_utils import DreamGenerationMixin, LLaDAGenerationMixin

# Primary HF cache (also written to HF_HOME so transformers/hub pick it up).
HF_CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# Ordered list of directories searched when resolving a model repo ID to a
# local snapshot.  Populated from the GCD_HF_CACHE_DIRS env var
# (colon-separated) which config_utils sets from the config's
# huggingface.cache_dirs list.  Falls back to HF_CACHE_DIR alone.
_raw_extra = os.environ.get("GCD_HF_CACHE_DIRS", "")
HF_CACHE_DIRS: List[str] = [d for d in _raw_extra.split(":") if d.strip()] or [HF_CACHE_DIR]

_BNB_AVAILABLE = False
try:
    import bitsandbytes  # noqa: F401

    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    _BNB_AVAILABLE = True
except Exception:
    pass


def _make_bnb_config(use_quantization: bool, compute_dtype=None):
    """Return a BitsAndBytesConfig for 4-bit NF4 loading, or None if unavailable."""
    if compute_dtype is None:
        compute_dtype = torch.bfloat16
    if not (use_quantization and _BNB_AVAILABLE):
        if use_quantization and not _BNB_AVAILABLE:
            print(
                "[Warning] use_quantization=True but bitsandbytes is not available "
                "in this environment. Loading models in full precision instead."
            )
        return None
    try:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    except Exception as e:
        print(f"[Warning] BitsAndBytesConfig construction failed ({e}). Loading in full precision.")
        return None


def _repo_id_to_cache_folder_name(repo_id: str) -> str:
    return "models--" + str(repo_id).replace("/", "--")


def _find_offline_cached_snapshot_dirs(cache_dir: str, repo_id: str) -> List[Path]:
    """
    Locate locally cached model directories for *repo_id*.

    Supports the standard HuggingFace hub cache layout
    (``models--ORG--NAME/snapshots/<rev>/``) and flat downloads written directly
    under ``hub/models--ORG--NAME/`` (e.g. via ``hf download --local-dir``).
    """
    folder_name = _repo_id_to_cache_folder_name(repo_id)
    base = Path(cache_dir)
    found: List[Path] = []
    for root in (base, base / "hub"):
        repo_dir = root / folder_name
        if not repo_dir.is_dir():
            continue
        snapshots_dir = repo_dir / "snapshots"
        if snapshots_dir.is_dir():
            for snap in sorted(snapshots_dir.iterdir()):
                if snap.is_dir() and (snap / "config.json").exists():
                    found.append(snap.resolve())
        elif (repo_dir / "config.json").exists():
            found.append(repo_dir.resolve())
    return found


def _resolve_pretrained_path(model_id_or_path: str, *, offline_mode: bool) -> str:
    """Resolve a HF model/tokenizer identifier to a local snapshot directory.

    Search order:
    1. Literal path on disk (absolute or relative).
    2. Each directory in HF_CACHE_DIRS (first match wins).
    3. If online: return the repo ID as-is for normal HF hub download.
    4. If offline: raise a clear error listing all searched directories.
    """
    if not isinstance(model_id_or_path, str) or len(model_id_or_path) == 0:
        return model_id_or_path

    p = Path(model_id_or_path).expanduser()
    if p.exists():
        return str(p.resolve())

    # Search all configured cache directories in order.
    for cache_dir in HF_CACHE_DIRS:
        cached = _find_offline_cached_snapshot_dirs(cache_dir, model_id_or_path)
        if cached:
            print(f"[model_loader] Found '{model_id_or_path}' in cache: {cached[0]}")
            return str(cached[0])

    if not offline_mode:
        return model_id_or_path

    # Offline and not found anywhere — try snapshot_download with local_files_only
    # as a last resort (covers non-standard hub layouts).
    for cache_dir in HF_CACHE_DIRS:
        try:
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(
                repo_id=model_id_or_path,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            return str(local_dir)
        except Exception:
            continue

    searched = ", ".join(HF_CACHE_DIRS)
    raise RuntimeError(
        f"Offline mode is enabled, but '{model_id_or_path}' was not found in any "
        f"of the configured cache directories: [{searched}].\n"
        f"Run 'bash download_models.sh' to download models, or add the directory "
        f"containing the model to 'huggingface.cache_dirs' in configs/config.yaml."
    )


def _same_pretrained_checkpoint(a_raw: Any, b_raw: Any, *, offline_mode: bool) -> bool:
    """Whether two YAML model entries refer to the same Hugging Face snapshot."""
    if a_raw is None or b_raw is None:
        return False
    sa = str(a_raw).strip()
    sb = str(b_raw).strip()
    if not sa or not sb:
        return False
    if sa == sb:
        return True
    try:
        pa = _resolve_pretrained_path(sa, offline_mode=offline_mode)
        pb = _resolve_pretrained_path(sb, offline_mode=offline_mode)
    except Exception:
        return False
    if pa == pb:
        return True
    ppa = Path(pa).expanduser()
    ppb = Path(pb).expanduser()
    try:
        if ppa.exists() and ppb.exists() and ppa.resolve() == ppb.resolve():
            return True
    except Exception:
        pass

    def _looks_like_hub_repo_only(x: str) -> bool:
        s = x.strip().replace("\\", "/")
        if "/" not in s:
            return False
        try:
            expanded = Path(s).expanduser()
            return not (expanded.exists() and expanded.is_dir())
        except OSError:
            return True

    if (
        _looks_like_hub_repo_only(pa)
        and _looks_like_hub_repo_only(pb)
        and pa.strip().lower() == pb.strip().lower()
    ):
        return True
    return False


def _tokenizer_pad_and_patch(tok) -> None:
    """Pad token + safe convert_tokens_to_string."""
    if tok is None:
        return
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        print(f"Set tokenizer.pad_token to eos_token ({tok.eos_token})")
    if getattr(tok, "_gcgd_none_token_patch", False):
        return
    orig = getattr(tok, "convert_tokens_to_string", None)
    if not callable(orig):
        return
    unk = getattr(tok, "unk_token", "") or ""

    def _wrapped(tokens):
        try:
            fixed = [t if isinstance(t, str) else unk for t in tokens]
        except Exception:
            fixed = []
            for t in tokens:
                fixed.append(t if isinstance(t, str) else unk)
        return orig(fixed)

    tok.convert_tokens_to_string = _wrapped
    tok._gcgd_none_token_patch = True


def _patch_tokenizer_convert_tokens_to_string(tok):
    _tokenizer_pad_and_patch(tok)


def load_models(
    dream_model_path: str = "Dream-org/Dream-v0-Instruct-7B",
    llada_model_path: str = "GSAI-ML/LLaDA-8B-Base",
    victim_model_path: str = "Qwen/Qwen2.5-7B-Instruct",
    use_quantization: bool = True,
    quant_judge: bool = False,
    quantize_diffusion: bool = False,
    dream_gguf_file: Optional[str] = None,
    llada_gguf_file: Optional[str] = None,
    dream_tokenizer_path: Optional[str] = None,
    llada_tokenizer_path: Optional[str] = None,
    device: str = "cuda",
    gpu_id: int = 0,
    offline_mode: bool = False,
    use_llada: bool = False,
    skip_backend_model: bool = False,
    victim_model_f16: bool = False,
):
    """Load scoring/filling model (Dream or LLaDA) + victim model."""
    visible_gpus = torch.cuda.device_count() if (device == "cuda" and torch.cuda.is_available()) else 0

    if device == "cuda" and torch.cuda.is_available():
        if gpu_id >= 0 and gpu_id < visible_gpus:
            torch.cuda.set_device(gpu_id)
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            print(f"Warning: GPU {gpu_id} not available (visible={visible_gpus}), using default GPU")

    backend_name = "LLaDA" if use_llada else "Dream"
    backend_path = llada_model_path if use_llada else dream_model_path

    gguf_file = llada_gguf_file if use_llada else dream_gguf_file
    tokenizer_path_override = llada_tokenizer_path if use_llada else dream_tokenizer_path

    if gguf_file is not None:
        if tokenizer_path_override is not None:
            tokenizer_path = tokenizer_path_override
        else:
            tokenizer_path = backend_path
            if "GGUF" in backend_path.upper() or backend_path.endswith("-GGUF"):
                if "/" in backend_path:
                    model_part = backend_path.split("/")[-1]
                    if model_part.endswith("-GGUF"):
                        original_model = model_part[:-5]
                        if "Dream-org" in original_model or "Dream_org" in original_model:
                            if "Dream-org_" in original_model:
                                tokenizer_path = f"Dream-org/{original_model.replace('Dream-org_', '')}"
                            elif "Dream_org_" in original_model:
                                tokenizer_path = f"Dream-org/{original_model.replace('Dream_org_', '')}"
                            else:
                                tokenizer_path = (
                                    original_model.replace("_", "/")
                                    if "_" in original_model
                                    else f"Dream-org/{original_model}"
                                )
                        elif "LLaDA" in original_model or "LLADA" in original_model:
                            tokenizer_path = (
                                original_model.replace("_", "/")
                                if "_" in original_model
                                else original_model
                            )
                        else:
                            tokenizer_path = (
                                original_model.replace("_", "/")
                                if "_" in original_model
                                else original_model
                            )
        print(f"Loading {backend_name} model from GGUF repository: {backend_path}")
        print(f"Loading {backend_name} tokenizer from: {tokenizer_path}")
    else:
        tokenizer_path = backend_path
        print(f"Loading {backend_name} model from {backend_path}...")

    print(f"Offline mode: {offline_mode}")
    if visible_gpus > 0:
        print(f"Visible GPUs: {visible_gpus}")
    print(f"Cache directory: {HF_CACHE_DIR}")

    bnb_config = _make_bnb_config(use_quantization)
    diffusion_bnb_config = _make_bnb_config(quantize_diffusion)

    backend_path = _resolve_pretrained_path(backend_path, offline_mode=offline_mode)
    tokenizer_path = _resolve_pretrained_path(tokenizer_path, offline_mode=offline_mode)
    victim_model_path = _resolve_pretrained_path(victim_model_path, offline_mode=offline_mode)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=offline_mode,
    )

    if device == "cuda" and torch.cuda.is_available():
        device_map = {"": gpu_id}
    else:
        device_map = "auto"

    dream_model = None
    if skip_backend_model:
        print(
            f"Skipping {backend_name} model weight load (skip_backend_model=True). "
            "Backend tokenizer is still loaded."
        )
    else:
        use_causal_lm = use_llada and ("LLaDA2.0" in backend_path or "llada2" in backend_path.lower())

        if gguf_file is not None:
            print(f"Loading {backend_name} model from GGUF file: {gguf_file}")
            loader_cls = AutoModelForCausalLM if use_causal_lm else AutoModel
            dream_model = loader_cls.from_pretrained(
                backend_path,
                gguf_file=gguf_file,
                device_map=device_map,
                trust_remote_code=True,
                local_files_only=offline_mode,
            )
        else:
            if quantize_diffusion:
                print(f"Loading {backend_name} model with 4-bit quantization...")
            else:
                print(f"Loading {backend_name} model without quantization...")
            loader_cls = AutoModelForCausalLM if use_causal_lm else AutoModel
            dream_model = loader_cls.from_pretrained(
                backend_path,
                quantization_config=diffusion_bnb_config,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                local_files_only=offline_mode,
            )
        dream_model.eval()

    if use_llada:
        if hasattr(tokenizer, "mask_token_id") and tokenizer.mask_token_id is not None:
            MASK_ID = int(tokenizer.mask_token_id)
        elif "LLaDA-MoE" in backend_path:
            MASK_ID = 156895
        else:
            MASK_ID = 126336
        if dream_model is not None:
            actual_dream = dream_model.module if hasattr(dream_model, "module") else dream_model
            actual_dream.diffusion_generate = types.MethodType(
                LLaDAGenerationMixin.diffusion_generate, actual_dream
            )
            actual_dream.add_gumbel_noise = LLaDAGenerationMixin.add_gumbel_noise
            actual_dream.get_num_transfer_tokens = LLaDAGenerationMixin.get_num_transfer_tokens
    else:
        MASK_ID = 151666
        if dream_model is not None:
            actual_dream = dream_model.module if hasattr(dream_model, "module") else dream_model
            actual_dream.diffusion_generate = types.MethodType(
                DreamGenerationMixin.diffusion_generate, actual_dream
            )
            actual_dream._sample = types.MethodType(DreamGenerationMixin._sample, actual_dream)

    v_use_quant = (use_quantization or quant_judge) and (not victim_model_f16)
    v_dtype = torch.float16 if victim_model_f16 else torch.bfloat16

    print(f"Loading victim model from {victim_model_path}...")
    if v_use_quant:
        print(f"Loading victim model with 4-bit NF4 quantization (compute dtype={v_dtype})...")
    else:
        print(f"Loading victim model in full {v_dtype} (no quantization)...")

    if "llada" in str(victim_model_path).lower():
        raise ValueError("victim_model cannot be an LLaDA checkpoint.")

    v_bnb_config = _make_bnb_config(v_use_quant, compute_dtype=v_dtype)

    victim_load_kwargs = dict(
        device_map=device_map,
        torch_dtype=v_dtype,
        trust_remote_code=True,
        local_files_only=offline_mode,
    )
    if v_bnb_config is not None:
        victim_load_kwargs["quantization_config"] = v_bnb_config

    victim_model = AutoModelForCausalLM.from_pretrained(
        victim_model_path,
        **victim_load_kwargs,
    )
    victim_model.eval()

    victim_tokenizer = AutoTokenizer.from_pretrained(
        victim_model_path,
        trust_remote_code=True,
        local_files_only=offline_mode,
    )

    _patch_tokenizer_convert_tokens_to_string(tokenizer)
    _patch_tokenizer_convert_tokens_to_string(victim_tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if victim_tokenizer.pad_token is None:
        victim_tokenizer.pad_token = victim_tokenizer.eos_token

    print("Models loaded successfully!")
    return dream_model, victim_model, tokenizer, victim_tokenizer, MASK_ID


def _resolve_phase2_judge_use_quantization(
    judge_model_path: str,
    use_quantization: Optional[bool],
) -> bool:
    """Decide whether to apply BitsAndBytes NF4 when loading the Phase-2 judge."""
    if use_quantization is not None:
        return bool(use_quantization)
    path_lower = str(judge_model_path or "").lower()
    if "fp8" in path_lower:
        return False
    return True


def load_phase2_judge_model(
    judge_model_path: str,
    *,
    use_quantization: Optional[bool] = True,
    device: str = "cuda",
    gpu_id: int = 0,
    offline_mode: bool = False,
):
    """Load a causal LM + tokenizer for in-attack Phase-2 LLM judging."""
    if device == "cuda" and torch.cuda.is_available():
        if gpu_id >= 0 and gpu_id < torch.cuda.device_count():
            torch.cuda.set_device(gpu_id)

    judge_model_path = _resolve_pretrained_path(judge_model_path, offline_mode=offline_mode)
    if "llada" in str(judge_model_path).lower():
        raise ValueError("phase2 judge model cannot be an LLaDA checkpoint.")

    if device == "cuda" and torch.cuda.is_available():
        device_map = {"": gpu_id}
    else:
        device_map = "auto"

    use_quantization = _resolve_phase2_judge_use_quantization(
        judge_model_path, use_quantization
    )
    bnb_config = _make_bnb_config(use_quantization, compute_dtype=torch.bfloat16)
    dtype = torch.bfloat16

    print(f"Loading Phase-2 judge model from {judge_model_path}...")
    if use_quantization:
        print(f"Loading Phase-2 judge model with 4-bit NF4 quantization (compute dtype={dtype})...")
    else:
        print(f"Loading Phase-2 judge model in native {dtype} (no BitsAndBytes NF4)...")

    judge_load_kwargs = dict(
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=offline_mode,
    )
    if bnb_config is not None:
        judge_load_kwargs["quantization_config"] = bnb_config

    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_model_path,
        **judge_load_kwargs,
    )
    judge_model.eval()

    judge_tokenizer = AutoTokenizer.from_pretrained(
        judge_model_path,
        trust_remote_code=True,
        local_files_only=offline_mode,
    )
    _patch_tokenizer_convert_tokens_to_string(judge_tokenizer)
    if judge_tokenizer.pad_token is None:
        judge_tokenizer.pad_token = judge_tokenizer.eos_token

    print("Phase-2 judge model loaded successfully.")
    return judge_model, judge_tokenizer


def load_defence_model(
    defence_model_path: str,
    *,
    use_quantization: bool = True,
    device: str = "cuda",
    gpu_id: int = 0,
    offline_mode: bool = False,
):
    """Load Llama Guard (or other defence classifier) for optimization-time defence loss."""
    if device == "cuda" and torch.cuda.is_available():
        if gpu_id >= 0 and gpu_id < torch.cuda.device_count():
            torch.cuda.set_device(gpu_id)

    defence_model_path = _resolve_pretrained_path(defence_model_path, offline_mode=offline_mode)
    if device == "cuda" and torch.cuda.is_available():
        device_map = {"": gpu_id}
    else:
        device_map = "auto"

    bnb_config = _make_bnb_config(use_quantization, compute_dtype=torch.bfloat16)
    dtype = torch.bfloat16

    if bnb_config is not None:
        print(f"Loading defence model from {defence_model_path} with 4-bit quantization...")
    else:
        print(f"Loading defence model from {defence_model_path} in full {dtype} (no quantization)...")
    defence_tokenizer = AutoTokenizer.from_pretrained(
        defence_model_path,
        use_fast=False,
        truncation_side="left",
        padding_side="left",
        trust_remote_code=True,
        local_files_only=offline_mode,
    )
    _patch_tokenizer_convert_tokens_to_string(defence_tokenizer)
    if defence_tokenizer.pad_token is None:
        defence_tokenizer.pad_token = defence_tokenizer.eos_token

    load_kwargs = dict(
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=offline_mode,
    )
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config

    defence_model = AutoModelForCausalLM.from_pretrained(
        defence_model_path,
        **load_kwargs,
    )
    defence_model.eval()
    for param in defence_model.parameters():
        param.requires_grad = False
    print("Defence model loaded successfully.")
    return defence_model, defence_tokenizer
