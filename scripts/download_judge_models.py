#!/usr/bin/env python3
"""
Download the models needed for the LLM-as-a-Judge manipulation experiments.

Run this on a login node (which has internet access) BEFORE submitting any
compute jobs.  All models are saved to the shared HuggingFace cache at
/scratch/gpfs/KOROLOVA/huggingface so that compute nodes can load them offline.

Usage (from project root):
    python scripts/download_judge_models.py

Or with a custom cache directory:
    python scripts/download_judge_models.py --cache /path/to/hf/cache

Models downloaded
─────────────────
  google/gemma-7b-it               ← judge model (MISSING, must download)
  Qwen/Qwen2.5-7B-Instruct         ← LLM_1 / favored model  (already cached, skipped)
  meta-llama/Meta-Llama-3-8B-Instruct  ← LLM_2 / competitor  (already cached, skipped)
  Dream-org/Dream-v0-Instruct-7B   ← diffusion model          (already cached, skipped)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

HF_CACHE_DEFAULT = "/scratch/gpfs/KOROLOVA/huggingface"

# Models required for the judge-manipulation experiments.
# Annotated with their role so you can skip any you don't need.
MODELS = [
    # ── Must download ─────────────────────────────────────────────────────────
    ("google/gemma-3-4b-it",                  "Judge model (Gemma-3 4B instruct)"),
    # ── Already cached on this cluster — listed for completeness ─────────────
    ("Qwen/Qwen2.5-7B-Instruct",              "LLM_1: favored model for output_1"),
    ("meta-llama/Meta-Llama-3-8B-Instruct",   "LLM_2: competitor model for output_2"),
    ("Dream-org/Dream-v0-Instruct-7B",        "Diffusion model (system-prompt optimizer)"),
]


def _find_hf_token() -> Optional[str]:
    """
    Return the stored HuggingFace token, searching the standard locations
    in priority order.  We read from the HOME-based paths directly so that
    overriding HF_HOME (to redirect the *cache*) does not lose the credentials
    stored by `huggingface-cli login`.
    """
    home = Path.home()
    candidates = [
        home / ".cache" / "huggingface" / "token",   # huggingface-cli login default
        home / ".huggingface" / "token",              # older / alternative location
    ]
    for p in candidates:
        if p.exists():
            tok = p.read_text(encoding="utf-8").strip()
            if tok:
                return tok
    # Also check the HUGGING_FACE_HUB_TOKEN / HF_TOKEN env vars
    for env_key in ("HUGGING_FACE_HUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        tok = os.environ.get(env_key, "").strip()
        if tok:
            return tok
    return None


def check_cached(repo_id: str, cache_dir: str) -> bool:
    """
    Return True only if the snapshot directory exists AND contains at least one
    model-weight file (safetensors / pytorch_model.bin / etc.).
    A directory with only a README is NOT considered cached.
    """
    try:
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, local_files_only=True)
        # Verify that actual weight files are present
        weight_extensions = {".safetensors", ".bin", ".pt", ".ckpt", ".msgpack", ".h5"}
        p = Path(local)
        for f in p.iterdir():
            # Follow symlinks (HF cache uses symlinks from snapshot → blobs)
            target = f.resolve() if f.is_symlink() else f
            if target.suffix in weight_extensions and target.stat().st_size > 1_000_000:
                return True
        return False  # directory exists but no weight files
    except Exception:
        return False


def download_model(repo_id: str, cache_dir: str, token: Optional[str] = None) -> None:
    from huggingface_hub import snapshot_download
    print(f"  Downloading {repo_id} → {cache_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_files_only=False,
        token=token,
    )
    print(f"  ✓ Done: {repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Download judge-manipulation models to the HF cache."
    )
    parser.add_argument(
        "--cache",
        default=HF_CACHE_DEFAULT,
        help=f"HuggingFace cache directory (default: {HF_CACHE_DEFAULT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the model is already cached.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without downloading.",
    )
    args = parser.parse_args()

    cache_dir = args.cache
    os.makedirs(cache_dir, exist_ok=True)

    # Force online mode for downloads.
    # Important: set HF_HOME AFTER reading the token so credentials stored in
    # ~/.cache/huggingface/token by `huggingface-cli login` are still found.
    token = _find_hf_token()
    if token:
        print(f"HuggingFace token : found ({token[:8]}...)")
    else:
        print("HuggingFace token : NOT FOUND — gated models will fail. "
              "Run: huggingface-cli login")

    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

    print(f"HuggingFace cache : {cache_dir}\n")
    print(f"{'Model':<50} {'Status'}")
    print("-" * 70)

    to_download = []
    for repo_id, description in MODELS:
        if not args.force and check_cached(repo_id, cache_dir):
            print(f"  {'CACHED':<8} {repo_id:<50}  ({description})")
        else:
            status = "DRY-RUN" if args.dry_run else "WILL DOWNLOAD"
            print(f"  {status:<8} {repo_id:<50}  ({description})")
            to_download.append((repo_id, description))

    if not to_download:
        print("\nAll models are already cached. Nothing to download.")
        return

    if args.dry_run:
        print(f"\n[dry-run] Would download {len(to_download)} model(s).")
        return

    print(f"\nDownloading {len(to_download)} model(s)...\n")
    errors = []
    for repo_id, description in to_download:
        print(f"[{description}]")
        try:
            download_model(repo_id, cache_dir, token=token)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            errors.append((repo_id, e))

    print("\n" + "=" * 70)
    if errors:
        print(f"Completed with {len(errors)} error(s):")
        for repo_id, e in errors:
            print(f"  ✗ {repo_id}: {e}")
        sys.exit(1)
    else:
        print(f"All {len(to_download)} model(s) downloaded successfully.")
        print("\nYou can now run compute jobs with offline_mode: true in your config.")


if __name__ == "__main__":
    main()
