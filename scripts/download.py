#!/usr/bin/env python3
"""
Download models and dependencies for GCD experiments.
Reads configuration from configs/config.yaml and downloads all required models.
Modified to support AdvPrefix fields: uncensored_models and judge_model.
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
from huggingface_hub import snapshot_download, hf_hub_download

# Default config file path
DEFAULT_CONFIG = "configs/config_dif_3.yaml"


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_cache_directories(config: dict) -> tuple:
    """Set up HuggingFace cache directories from config."""
    global_config = config.get("global", {})
    hf_config = global_config.get("environment", {}).get("huggingface", {})
    
    # Get cache directory from config or use default
    hf_cache_dir = hf_config.get("cache_dir", "/scratch/gpfs/KOROLOVA/huggingface")
    datasets_cache_dir = os.path.join(hf_cache_dir, "datasets")
    
    # Force online mode for downloads
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    
    # Set other environment variables if not set
    os.environ.setdefault("HF_HOME", hf_cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", datasets_cache_dir)
    
    # Create directories
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.makedirs(datasets_cache_dir, exist_ok=True)
    
    print(f"HuggingFace cache directory: {hf_cache_dir}")
    print(f"Datasets cache directory: {datasets_cache_dir}")
    
    return hf_cache_dir, datasets_cache_dir


def download_model(repo_id: str, cache_dir: str) -> None:
    """Download a model from HuggingFace Hub."""
    print(f"\n{'='*60}")
    print(f"Downloading model: {repo_id}")
    print(f"Cache directory: {cache_dir}")
    print(f"{'='*60}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_dir_use_symlinks=False,
            local_files_only=False,
        )
        print(f"✓ Successfully downloaded {repo_id}")
    except Exception as e:
        print(f"✗ Error downloading {repo_id}: {e}")
        raise


def download_gguf_file(repo_id: str, filename: str, cache_dir: str) -> None:
    """Download a specific GGUF file from HuggingFace Hub."""
    print(f"\n{'='*60}")
    print(f"Downloading GGUF file: {filename} from {repo_id}")
    print(f"Cache directory: {cache_dir}")
    print(f"{'='*60}")
    
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            local_files_only=False,
        )
        print(f"✓ Successfully downloaded {filename} from {repo_id}")
    except Exception as e:
        print(f"✗ Error downloading {filename} from {repo_id}: {e}")
        raise


def get_models_from_config(config: dict, experiment_name: str = None) -> tuple:
    """
    Extract model names and GGUF files from config.
    
    Args:
        config: Configuration dictionary
        experiment_name: Optional experiment name
    
    Returns:
        Tuple of (list of model repository IDs to download, list of (repo_id, filename) tuples for GGUF files)
    """
    models_to_download = []
    gguf_files_to_download = []
    
    # Global models
    models_config = config.get("models", {})
    for k in ["dream_model", "llada_model", "victim_model"]:
        if models_config.get(k):
            models_to_download.append(models_config[k])
    
    # Global GGUF files
    if models_config.get("dream_gguf_file"):
        if models_config.get("dream_model"):
            gguf_files_to_download.append((models_config["dream_model"], models_config["dream_gguf_file"]))
    if models_config.get("llada_gguf_file"):
        if models_config.get("llada_model"):
            gguf_files_to_download.append((models_config["llada_model"], models_config["llada_gguf_file"]))

    # Defence evasion models (global)
    defaults = config.get("defaults", {})
    if defaults.get("defence_model_name"):
        models_to_download.append(defaults["defence_model_name"])
    
    # Perplexity model (global)
    if defaults.get("perplexity_model"):
        models_to_download.append(defaults["perplexity_model"])

    # Experiment specific models
    experiments = config.get("experiments", {})
    
    def add_exp_models(exp_cfg):
        if not isinstance(exp_cfg, dict):
            return
        # Traditional GCD models
        m_cfg = exp_cfg.get("models", {})
        for k in ["dream_model", "llada_model", "victim_model"]:
            if m_cfg.get(k):
                models_to_download.append(m_cfg[k])
        
        # GGUF files for experiment-specific models
        if m_cfg.get("dream_gguf_file") and m_cfg.get("dream_model"):
            gguf_files_to_download.append((m_cfg["dream_model"], m_cfg["dream_gguf_file"]))
        if m_cfg.get("llada_gguf_file") and m_cfg.get("llada_model"):
            gguf_files_to_download.append((m_cfg["llada_model"], m_cfg["llada_gguf_file"]))
        
        # Defence evasion models (per experiment)
        if exp_cfg.get("defence_model_name"):
            models_to_download.append(exp_cfg["defence_model_name"])
        
        # Perplexity model (per experiment)
        if exp_cfg.get("perplexity_model"):
            models_to_download.append(exp_cfg["perplexity_model"])
        
        # AdvPrefix models
        if exp_cfg.get("victim_model"):
            models_to_download.append(exp_cfg["victim_model"])
        if exp_cfg.get("judge_model"):
            models_to_download.append(exp_cfg["judge_model"])
        if exp_cfg.get("uncensored_models"):
            models_to_download.extend(exp_cfg["uncensored_models"])

    if experiment_name:
        if experiment_name in experiments:
            add_exp_models(experiments[experiment_name])
        else:
            print(f"Warning: Experiment {experiment_name} not found.")
    else:
        # Check all experiments
        for exp_name, exp_cfg in experiments.items():
            add_exp_models(exp_cfg)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_models = []
    for model in models_to_download:
        if model and isinstance(model, str) and model not in seen:
            seen.add(model)
            unique_models.append(model)
    
    # Remove duplicate GGUF files
    seen_gguf = set()
    unique_gguf = []
    for repo_id, filename in gguf_files_to_download:
        key = (repo_id, filename)
        if key not in seen_gguf:
            seen_gguf.add(key)
            unique_gguf.append(key)
    
    return unique_models, unique_gguf


def main():
    parser = argparse.ArgumentParser(
        description="Download models for GCD and AdvPrefix experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to config file (default: {DEFAULT_CONFIG})"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Specific experiment name (default: download models for all experiments)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading"
    )
    
    args = parser.parse_args()
    
    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    
    print("="*60)
    print("GCD / AdvPrefix Model Downloader")
    print("="*60)
    print(f"Config file: {config_path}")
    if args.experiment:
        print(f"Experiment: {args.experiment}")
    else:
        print("Experiment: All experiments")
    print("="*60)
    
    # Load config
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Set up cache directories
    try:
        cache_dir, datasets_cache_dir = setup_cache_directories(config)
    except Exception as e:
        print(f"Error setting up cache directories: {e}")
        sys.exit(1)
    
    # Get models and GGUF files to download
    models_to_download, gguf_files_to_download = get_models_from_config(config, args.experiment)
    
    if not models_to_download and not gguf_files_to_download:
        print("\nNo models found in config file!")
        sys.exit(1)
    
    if models_to_download:
        print(f"\nModels to download ({len(models_to_download)}):")
        for i, model in enumerate(models_to_download, 1):
            print(f"  {i}. {model}")
    
    if gguf_files_to_download:
        print(f"\nGGUF files to download ({len(gguf_files_to_download)}):")
        for i, (repo_id, filename) in enumerate(gguf_files_to_download, 1):
            print(f"  {i}. {filename} from {repo_id}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would download the above models and GGUF files.")
        return
    
    # Download models
    print("\n" + "="*60)
    print("Starting downloads...")
    print("="*60)
    
    failed_downloads = []
    for model in models_to_download:
        try:
            download_model(model, cache_dir)
        except Exception as e:
            print(f"Failed to download {model}: {e}")
            failed_downloads.append(model)
    
    # Download GGUF files
    failed_gguf_downloads = []
    for repo_id, filename in gguf_files_to_download:
        try:
            download_gguf_file(repo_id, filename, cache_dir)
        except Exception as e:
            print(f"Failed to download {filename} from {repo_id}: {e}")
            failed_gguf_downloads.append((repo_id, filename))
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    successful_models = len(models_to_download) - len(failed_downloads)
    successful_gguf = len(gguf_files_to_download) - len(failed_gguf_downloads)
    
    if models_to_download:
        print(f"Models - Successful: {successful_models}/{len(models_to_download)}")
        if failed_downloads:
            print(f"Models - Failed: {len(failed_downloads)}")
            for model in failed_downloads:
                print(f"  - {model}")
    
    if gguf_files_to_download:
        print(f"GGUF files - Successful: {successful_gguf}/{len(gguf_files_to_download)}")
        if failed_gguf_downloads:
            print(f"GGUF files - Failed: {len(failed_gguf_downloads)}")
            for repo_id, filename in failed_gguf_downloads:
                print(f"  - {filename} from {repo_id}")
    
    if failed_downloads or failed_gguf_downloads:
        sys.exit(1)
    else:
        print("\n✓ All models and GGUF files downloaded successfully!")
        print(f"\nYou can now run experiments with offline mode enabled.")
        print(f"Cache location: {cache_dir}")


if __name__ == "__main__":
    main()
