"""Configuration utilities for GCD experiments."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: F401

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries.

    A value of ``null`` in YAML (``None`` in Python) unsets a key inherited from
    ``base`` — e.g. ``slurm: {partition: null}`` clears an inherited partition.
    Set ``slurm.default_gpu80_constraint: false`` to omit the script's default
    ``gpu80`` / ``h200`` constraint when neither partition nor constraint is set.
    """
    merged = base.copy()
    for key, value in override.items():
        if value is None:
            merged.pop(key, None)
            continue
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_experiment_config(
    config: Dict[str, Any],
    experiment_name: str,
    visited: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Resolve inheritance for an experiment configuration."""
    if visited is None:
        visited = []

    if experiment_name in visited:
        raise ValueError(f"Circular inheritance detected: {' -> '.join(visited + [experiment_name])}")

    visited.append(experiment_name)

    experiments = config.get("experiments", {})
    if experiment_name not in experiments:
        if experiment_name == "default":
            return {}
        raise ValueError(f"Experiment '{experiment_name}' not found in config.")

    exp_cfg = experiments[experiment_name]
    if not isinstance(exp_cfg, dict):
        return {}

    parent_name = exp_cfg.get("inherit") or exp_cfg.get("base")
    if parent_name:
        parent_cfg = resolve_experiment_config(config, parent_name, visited)
        return deep_merge(parent_cfg, exp_cfg)

    return exp_cfg


def setup_hf_environment(config_path: str = None):
    """Set up HuggingFace environment variables from config."""
    default_cache = os.path.expanduser("~/.cache/huggingface")
    offline_mode = False
    default_wandb_base = os.path.expanduser("~/.cache/wandb")

    cache_dirs: List[str] = []

    if config_path and Path(config_path).exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            global_config = config.get("global", {})
            hf_config = global_config.get("environment", {}).get("huggingface", {})

            config_root = Path(config_path).resolve().parent.parent

            def _resolve_dir(raw: str) -> str:
                """Resolve a cache dir path, expanding relative paths against config root."""
                if not raw:
                    return ""
                expanded = os.path.expanduser(raw)
                if os.path.isabs(expanded):
                    return expanded
                # Relative path — resolve against the repo root (parent of configs/)
                return str((config_root / expanded.lstrip("./")).resolve())

            # Support both a single cache_dir and a list of cache_dirs.
            raw_dirs: List[str] = []
            if "cache_dirs" in hf_config:
                raw_dirs = [d for d in (hf_config["cache_dirs"] or []) if d]
            if "cache_dir" in hf_config and hf_config["cache_dir"]:
                primary = hf_config["cache_dir"]
                if primary not in raw_dirs:
                    raw_dirs.insert(0, primary)
            if not raw_dirs:
                raw_dirs = [default_cache]

            cache_dirs = [_resolve_dir(d) for d in raw_dirs if d]
            cache_dir = cache_dirs[0] if cache_dirs else default_cache

            offline_mode = hf_config.get("offline_mode", False)
            scratch_dir = global_config.get("scratch_dir", None)
            if scratch_dir:
                default_wandb_base = str(Path(scratch_dir) / "wandb")
        except Exception:
            cache_dir = default_cache
            cache_dirs = [default_cache]
    else:
        cache_dir = default_cache
        cache_dirs = [default_cache]

    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_dir, "datasets"))

    # Expose the full ordered search list so model_loader picks it up even
    # when it is imported after this function runs.
    os.environ["GCD_HF_CACHE_DIRS"] = ":".join(cache_dirs)

    if offline_mode:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    os.makedirs(cache_dir, exist_ok=True)

    wandb_dir = os.environ.get("WANDB_DIR", str(Path(default_wandb_base) / "runs"))
    wandb_cache_dir = os.environ.get("WANDB_CACHE_DIR", str(Path(default_wandb_base) / "cache"))
    wandb_config_dir = os.environ.get("WANDB_CONFIG_DIR", str(Path(default_wandb_base) / "config"))
    wandb_data_dir = os.environ.get("WANDB_DATA_DIR", str(Path(default_wandb_base) / "data"))
    wandb_artifact_dir = os.environ.get("WANDB_ARTIFACT_DIR", str(Path(default_wandb_base) / "artifacts"))
    wandb_tmp_dir = os.environ.get("TMPDIR", str(Path(default_wandb_base) / "tmp"))

    os.environ.setdefault("WANDB_DIR", wandb_dir)
    os.environ.setdefault("WANDB_CACHE_DIR", wandb_cache_dir)
    os.environ.setdefault("WANDB_CONFIG_DIR", wandb_config_dir)
    os.environ.setdefault("WANDB_DATA_DIR", wandb_data_dir)
    os.environ.setdefault("WANDB_ARTIFACT_DIR", wandb_artifact_dir)
    os.environ.setdefault("TMPDIR", wandb_tmp_dir)

    for p in (wandb_dir, wandb_cache_dir, wandb_config_dir, wandb_data_dir, wandb_artifact_dir, wandb_tmp_dir):
        try:
            os.makedirs(p, exist_ok=True)
        except Exception:
            pass

    return cache_dir, offline_mode, cache_dirs
