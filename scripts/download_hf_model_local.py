#!/usr/bin/env python3
"""
Download a Hugging Face Hub model snapshot into an explicit folder for YAML ``models.*`` paths.

``run_experiment._resolve_pretrained_path()`` passes through any existing filesystem path unchanged,
so ``AutoModelForCausalLM.from_pretrained("/abs/path/...")`` works without relying on HF cache layout.

Recommended layout (matches many cluster setups):
  SCRATCH_ROOT/hf_models/ORG__MODELNAME   (slashes in Hub id replaced by "__")

Examples:
  python scripts/download_hf_model_local.py Qwen/Qwen2.5-7B-Instruct \\
      --local-dir /scratch/gpfs/KOROLOVA/bt4811/hf_models/Qwen2.5-7B-Instruct

CLI equivalent (modern ``hf``, not deprecated ``huggingface-cli``):
  hf download REPO_ID --local-dir /path/to/dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo_id", help="Hub repo id, e.g. Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Directory for the snapshot (created). Omit to use project_dir/hf_models/REPO with __ instead of /.",
    )
    args = ap.parse_args()
    repo = str(args.repo_id).strip()

    if args.local_dir is not None:
        out_dir = args.local_dir.expanduser()
    else:
        safe = repo.replace("/", "__")
        out_dir = _project_root() / "hf_models" / safe

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Missing huggingface_hub. Install: pip install -U huggingface_hub", file=sys.stderr)
        raise SystemExit(1)

    path = snapshot_download(repo_id=repo, local_dir=str(out_dir))
    path = Path(path).resolve()
    print(path, flush=True)
    print(
        "\n# Add under models: in configs/bench_manipulation.yaml\n"
        f'bench_verifier_model: "{path}"\n',
        file=sys.stderr,
        end="",
    )


if __name__ == "__main__":
    main()
