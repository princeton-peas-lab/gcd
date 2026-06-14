#!/bin/bash
# GCD Environment Initialization Script
# This script sets up the environment and runs the experiment

set -e  # Exit on error

# Configuration (can be overridden by environment variables)
NETID="${NETID:-bt4811}"
HOME_DIR="${HOME_DIR:-/home/bt4811}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/gpfs/KOROLOVA/bt4811}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/gpfs/KOROLOVA/bt4811/gcg-diffusion}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/scratch/gpfs/KOROLOVA/huggingface}"
CONDA_ENV="${CONDA_ENV:-gcg-diffusion}"

# Print configuration
echo "=========================================="
echo "GCD Environment Setup"
echo "=========================================="
echo "Project Directory: $PROJECT_DIR"
echo "Conda Environment: $CONDA_ENV"
echo "HuggingFace Cache: $HF_CACHE_DIR"
echo "=========================================="

# Load required modules
echo "Loading modules..."
module load proxy/default
module load anaconda3/2025.6

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Verify conda activation
if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]; then
    echo "ERROR: Failed to activate conda environment $CONDA_ENV"
    echo "Current environment: $CONDA_DEFAULT_ENV"
    exit 1
fi

# Set up environment variables
export HF_HOME="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"

# Create cache directory if it doesn't exist
mkdir -p "$HF_CACHE_DIR"

# Change to project directory
cd "$PROJECT_DIR"
echo "Working directory: $(pwd)"

# Verify Python and packages
echo "Python version: $(python --version)"
echo "Python path: $(which python)"

# Check if required packages are available
echo "Checking key packages..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo "WARNING: PyTorch not found"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" || echo "WARNING: Transformers not found"
python -c "import wandb; print(f'WandB: {wandb.__version__}')" || echo "WARNING: WandB not found"

# GPU check
if command -v nvidia-smi &> /dev/null; then
    echo "=========================================="
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo "=========================================="
fi

# Run the experiment command
# If arguments are provided, use them; otherwise use defaults
if [ $# -eq 0 ]; then
    echo "=========================================="
    echo "Running experiment with default arguments:"
    echo "=========================================="
    python scripts/run_experiment.py \
        --config configs/config.yaml \
        --examples data/examples.json \
        --experiment-name test \
        --experiment exp_1 \
        --num-examples 8 \
        --gpu-id 0
else
    echo "=========================================="
    echo "Running experiment with provided arguments:"
    echo "=========================================="
    python scripts/run_experiment.py "$@"
fi

echo "=========================================="
echo "Experiment completed!"
echo "=========================================="

