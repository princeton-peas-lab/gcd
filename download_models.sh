#!/usr/bin/env bash
# download_models.sh — Download all models required by GCD into ./models/
#
# Usage:
#   bash download_models.sh [--dir PATH]
#
# Options:
#   --dir PATH      Target cache directory  (default: ./models)
#
# All models are downloaded in the standard HuggingFace hub cache layout so
# that the code can find them automatically when HF_HOME is set to the same
# directory.  After downloading, set:
#
#   export HF_HOME="$(pwd)/models"
#
# or update configs/config.yaml:
#   global.environment.huggingface.cache_dir: "./models"
#   global.environment.huggingface.offline_mode: true

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
CACHE_DIR="./models"

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)         CACHE_DIR="$2"; shift ;;
    --dir=*)       CACHE_DIR="${1#--dir=}" ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
  shift
done

mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"

# Prefer huggingface-cli (ships with transformers / huggingface_hub).
# Fall back to the Python API if the CLI is absent.
if command -v huggingface-cli &>/dev/null; then
  _download() { huggingface-cli download "$1" --cache-dir "$CACHE_DIR"; }
elif python -c "from huggingface_hub import snapshot_download" &>/dev/null; then
  _download() {
    python - "$1" "$CACHE_DIR" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], cache_dir=sys.argv[2])
PYEOF
  }
else
  echo "ERROR: neither huggingface-cli nor huggingface_hub Python package found." >&2
  echo "Install with:  pip install huggingface_hub" >&2
  exit 1
fi

echo "Downloading models into: $(realpath "$CACHE_DIR")"
echo ""

# ── Helper ────────────────────────────────────────────────────────────────────
download_model() {
  local repo_id="$1"
  local label="$2"
  echo "━━━ $label ($repo_id) ━━━"
  if _download "$repo_id"; then
    echo "✓ $label downloaded successfully."
  else
    echo "✗ Failed to download $label. Check your HF token or network." >&2
  fi
  echo ""
}

echo "=== Attack models ==="
echo ""

# Diffusion model (Dream) — required for all experiments
download_model "Dream-org/Dream-v0-Instruct-7B" "Dream diffusion model"

# Victim models — download all three; the experiment config selects which one
download_model "mistralai/Mistral-7B-Instruct-v0.3"      "Mistral victim"
download_model "Qwen/Qwen2.5-7B-Instruct"                "Qwen2.5 victim"
download_model "meta-llama/Meta-Llama-3-8B-Instruct"     "Llama-3-8B victim"

# Phase-2 LLM judge (used inside the attack loop for all final_exp_* configs)
# Note: Qwen2.5-7B-Instruct is already downloaded above as a victim model.

# Optional: Llama Guard defence (only for *_def experiment variants)
download_model "meta-llama/Llama-Guard-3-1B" "Llama Guard defence"

echo "================================================================"
echo "All downloads complete."
echo ""
echo "Next step — add to your shell or update configs/config.yaml:"
echo "  export HF_HOME=\"$(realpath "$CACHE_DIR")\""
echo ""
echo "Or set in config.yaml:"
echo "  global.environment.huggingface.cache_dir: \"$(realpath "$CACHE_DIR")\""
echo "  global.environment.huggingface.offline_mode: true"
