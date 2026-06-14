# GCD

Pure-diffusion adversarial suffix attacks on instruction-tuned LLMs.

This repository implements **GCD**: a jailbreak attack that optimizes an adversarial user-message suffix so a **victim LLM** begins its assistant reply with a specified **target prefix**. Candidate tokens are proposed entirely by a **Dream diffusion model**, producing semantically coherent natural-language prompts.

---

## Table of contents

- [Getting started](#getting-started)
- [Implementation](#implementation)
- [Attack pipeline (two-phase PCG)](#attack-pipeline-two-phase-pcg)
- [Repository layout](#repository-layout)
- [CLI reference](#cli-reference)
- [Configuration](#configuration)
- [Data format](#data-format)
- [Results](#results)
- [Monitoring](#monitoring)
- [Extending the codebase](#extending-the-codebase)

---

## Getting started

### 1. Clone the repository

```bash
git clone <repo-url> gcd
cd gcd
```

### 2. Create the environment

**Option A — conda (recommended for SLURM / HPC clusters)**

```bash
conda create -n gcd python=3.11 -y
conda activate gcd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate bitsandbytes datasets wandb pyyaml simple-slurm tqdm huggingface_hub
pip install peft python-dotenv seaborn tensorboard langdetect
```

**Option B — uv (fast resolver, good for local experiments)**

```bash
uv venv .venv --python 3.11 --seed
source .venv/bin/activate
uv pip install torch torchvision --torch-backend=cu128
uv pip install transformers accelerate bitsandbytes datasets wandb pyyaml simple-slurm tqdm huggingface_hub
uv pip install peft python-dotenv seaborn tensorboard langdetect
```

> **Tip:** Check `requirements_conda.txt` for the exact package versions used in the paper experiments.

### 3. Download models

All models are downloaded into `./models/` using the standard HuggingFace hub cache layout.
A [Hugging Face account](https://huggingface.co) and access token are required for gated models
(Llama, Mistral).  Log in once with:

```bash
huggingface-cli login   # paste your HF token when prompted
```

Then run the download script:

```bash
# Download attack models (~30 GB: Dream, Mistral, Qwen, Llama, Llama Guard)
bash download_models.sh

# Download to a custom directory (e.g. a shared HPC cache)
bash download_models.sh --dir /shared/hf_cache
```

The script downloads:

| Model | HuggingFace ID | Role |
|---|---|---|
| Dream | `Dream-org/Dream-v0-Instruct-7B` | Diffusion / candidate proposer |
| Mistral victim | `mistralai/Mistral-7B-Instruct-v0.3` | Victim LLM |
| Qwen victim / judge | `Qwen/Qwen2.5-7B-Instruct` | Victim LLM + Phase-2 judge |
| Llama victim | `meta-llama/Meta-Llama-3-8B-Instruct` | Victim LLM |
| Llama Guard | `meta-llama/Llama-Guard-3-1B` | Defence model (`*_def` experiments) |

### 4. Configure paths

Open `configs/config.yaml` and update the two settings at the top of the `global:` block:

```yaml
global:
  project_dir: "/absolute/path/to/this/repo"   # ← set to your checkout
  environment:
    conda_env: "gcd"                             # ← name of your conda env
    huggingface:
      cache_dir: "./models"          # primary download target (HF_HOME is set to this)
      cache_dirs:                    # ordered search path when loading models
        - "./models"                 # check local downloads first
        - "/path/to/existing/cache"  # ← add any pre-existing HF cache here
      offline_mode: true             # set to false if compute nodes have internet
```

The code searches `cache_dirs` in order and uses the first directory that contains the model — so you can point at an existing shared HPC cache without re-downloading anything.

You can also override the search path at runtime:

```bash
export GCD_HF_CACHE_DIRS="/path/to/cache1:/path/to/cache2"
export HF_HOME="/path/to/cache1"
```

### 5. Run an attack

**Locally (single GPU, no SLURM):**

```bash
# Llama-3-8B victim, 1 example, disable W&B
python scripts/run_experiment.py \
  --config configs/config.yaml \
  --examples data/data_subsample.json \
  --experiment-name test_local \
  --experiment llama \
  --num-examples 1 \
  --gpu-id 0 \
  --no-wandb
```

Swap `--experiment` to use a different victim model (see [Experiment keys](#experiment-keys-confignano-yaml)):

| `--experiment` flag | Victim |
|---|---|
| `mistral` | Mistral-7B |
| `qwen` | Qwen2.5-7B |
| `llama` | Llama-3-8B |
| `llama_def` | Llama-3-8B + Llama Guard defence |

**Via SLURM (array job):**

```bash
# Submit a SLURM array — shards examples across N jobs with M GPUs each
python scripts/run_slurm.py \
  --config configs/config.yaml \
  --examples data/data_subsample.json \
  --experiment-name my_llama_run \
  --experiment llama

# Monitor jobs
squeue -u $USER
tail -f experiments/my_llama_run_*/slurm/*.out
```

SLURM resource settings (GPUs, time, partition, constraint) are read from the experiment's
`slurm:` block in the config.  Override any value at submission time:

```bash
python scripts/run_slurm.py \
  --config configs/config.yaml \
  --examples data/data_subsample.json \
  --experiment-name my_run \
  --experiment llama \
  --override slurm.n_jobs=4 slurm.time=02:00:00 slurm.partition=gpu
```

---

## Implementation

GCD iterates over the adversarial suffix at each step:

1. **Dream scoring** — the diffusion model scores token replacements at masked suffix positions.
2. **Candidate sampling** — top-k position/token pairs are sampled into a candidate batch.
3. **Victim evaluation** — each candidate suffix is scored by cross-entropy on the target response prefix in the victim model.
4. **Commit** — the candidate with the lowest combined loss is kept (CE + batch-normalized self-PPL when `self_perplexity: true`).

Key source files:

- `gcd/gcd_core.py` — `GCDAttack.step()`, one optimization step
- `gcd/gcd_candidate_maker.py` — `CandidateGenerator`, position selection and candidate sampling
- `gcd/gcd_loss_evaluator.py` — `VictimEvaluator`, Dream fill + victim forward + combined guidance loss
- `scripts/attack_pipeline.py` — per-example orchestration, W&B logging, two-phase PCG, checkpoint resume

---

## Attack pipeline (two-phase PCG)

All primary configs (`mistral`, `qwen`, `llama`) use a **two-phase prefix-continuation (PCG)** pipeline.

### Phase 1 — prefix optimization

Up to `main_optimization_num_steps` (800) steps optimizing the suffix until the victim emits the target prefix. Stops cleanly when no prefix is found (`prefix_continuation_no_fallback: true`).

### Phase 2 — continuation + judge pool

After the first prefix match, up to `optimization_gen_check_steps` (400) additional steps. Candidates are staged; once a batch is full (batch size 16), the victim runs **batched greedy decode** for 512 tokens on all staged prompts at once. Each 512-token continuation is scored by a local Qwen Phase-2 judge (`fine_grained_legal_circumvention` mode, min score 4). Up to 8 passing responses are kept in `choose_best_n_verified_pool`.

### Llama Guard defence variants (`mistral_def`, `qwen_def`, `llama_def`)

These configs add Llama Guard (`meta-llama/Llama-Guard-3-1B`) mixed into the guidance loss at `alpha_def: 0.2`. The attack stops when the defence model flags the prompt unsafe (`req_safe_stop: true`). Each pool entry records `fresh_judge_prediction` (the Guard label at collection time).

---

## Repository layout

```
gcd/                          # Attack core
  gcd_core.py                 # GCDAttack class (__init__, step, run)
  gcd_attack_methods.py       # Shared mixins (Dream scores, caches, helpers)
  gcd_candidate_maker.py      # CandidateGenerator
  gcd_loss_evaluator.py       # VictimEvaluator
  gcd_dream.py                # Dream model interaction mixin
  gcd_victim.py               # Victim CE / generation mixin
  gcd_text.py                 # Text encoding/decoding helpers
  gcd_attack.py               # Re-export shim: from gcd_core import GCDAttack

scripts/
  run_experiment.py           # Main entry point (argparse, model load, example loop)
  attack_pipeline.py          # Per-example attack orchestration + result saving
  run_slurm.py                # SLURM array / multi-GPU submission
  config_utils.py             # YAML merge, experiment inherit resolution, HF env
  model_loader.py             # Dream + victim model loading (4-bit optional)
  eval_utils.py               # Prefix/tail checks, PCG refusal helpers
  pcg_phase2_judge.py         # Phase-2 LLM judge

configs/
  config.yaml                 # All experiment configs

data/
  data_subsample.json         # 48-example primary eval set
  data_full.json              # 98-example set (same schema)
  harmful-behaviors-dataset-full.json  # HarmBench-style raw export

utils/
  diffusion_utils.py          # DreamGenerationMixin

plots/                        # Analysis scripts

experiments/                  # Output directory (created at runtime)
  <experiment-name>_<job-id>/
    example_N_result.json
    all_results_shardK.json
    summary_shardK.json
    slurm/
```

---

## CLI reference

### `scripts/run_experiment.py`

| Flag | Description |
|---|---|
| `--config` | Path to YAML config (required) |
| `--examples` | Path to examples JSON (required) |
| `--experiment-name` | Run label; used in output directory name (required) |
| `--experiment` | Experiment key under `experiments:` in YAML |
| `--experiment-id` | Override job ID (default: `SLURM_JOB_ID` or random local ID) |
| `--start-example` | First example index (default: 0) |
| `--num-examples` | Limit number of examples |
| `--shard-id` / `--num-shards` | Contiguous sharding within selected range |
| `--gpu-id` | CUDA device index (default: 0) |
| `--wandb-group` | W&B group name for multi-shard runs |
| `--no-wandb` | Disable Weights & Biases |
| `--override KEY=VALUE` | Override nested config keys at runtime |

Override examples:

```bash
--override candidate_batch_pct=0.2
--override self_perplexity_coef='[0.2, 0.2]'
--override slurm.n_jobs=4
```

### `scripts/run_slurm.py`

Wraps `run_experiment.py` with SLURM resource allocation from the experiment's `slurm:` block (`n_jobs`, `gpus_per_job`, `tasks_per_job`, `time`, `constraint`, etc.). Supports skipping already-finished examples when result JSONs exist.

---

## Configuration

All experiment parameters live in `configs/config.yaml`.

### Structure

```yaml
global:
  project_dir: "/path/to/this/repo"  # absolute path to the repo
  environment:
    conda_env: gcd          # name of your conda / venv environment
    huggingface:
      cache_dir: "./models" # relative to project_dir, or absolute; set HF_HOME to override
      offline_mode: false   # set to true after all models are downloaded

experiments:
  my_experiment:
    inherit: parent_experiment    # optional: deep-merge parent config
    grad_coef: 0.0
    num_steps: 800
    models:
      dream_model: "Dream-org/Dream-v0-Instruct-7B"
      victim_model: "meta-llama/Meta-Llama-3-8B-Instruct"
      use_quantization: true
    slurm:
      mode_split: "examples"
      n_jobs: 12
      gpus_per_job: 4
```

Child experiments inherit all parent keys; child values override. Resolved by `scripts/config_utils.resolve_experiment_config()`.

### Important knobs

| Parameter | Typical value | Meaning |
|---|---|---|
| `grad_coef` | `0.0` | No victim-gradient mixing (pure diffusion) |
| `dream_alg` | `"entropy"` | Dream filling / scoring algorithm |
| `diffusion_temperature` | `0.0` | Dream sampling temperature |
| `dream_eval_steps` | `1` | MDLM denoising steps per eval fill |
| `main_optimization_num_steps` | `800` | Phase 1 step budget |
| `optimization_gen_check_steps` | `400` | Phase 2 step budget after first prefix |
| `num_tunable_tokens` | `80` | Suffix length in tokens |
| `top_k_start` / `top_k_end` | `32` | Candidate pool size per position |
| `candidate_batch_pct` | `0.1` | Fraction of candidates evaluated per step |
| `eval_batch_size` | `512` | Victim eval micro-batch size |
| `self_perplexity` | `true` | Add batch-normalized suffix self-PPL to CE for candidate ranking |
| `prefix_continuation_full_gen_max_new_tokens` | `512` | Batched greedy continuation length (Phase 2) |
| `prefix_continuation_phase2_collect_pool_size` | `8` | Max entries in `choose_best_n_verified_pool` |

### Swapping models

Edit the experiment's `models:` block:

```yaml
models:
  dream_model: "Dream-org/Dream-v0-Instruct-7B"
  victim_model: "meta-llama/Meta-Llama-3-8B-Instruct"
  use_quantization: true
```

### Experiment keys (`config.yaml`)

| Key | Notes |
|---|---|
| `mistral` | Mistral-7B-Instruct victim |
| `qwen` | Qwen2.5-7B-Instruct victim |
| `qwen_gpu80` | Same as `qwen` but forced to `gpu80` partition (no ailab) |
| `llama` | Llama-3-8B-Instruct victim |
| `llama_test` | Single-job smoke test (1 GPU, 20 min) |
| `mistral_def` | Mistral + Llama Guard defence during attack |
| `qwen_def` | Qwen + Llama Guard defence during attack |
| `llama_def` | Llama + Llama Guard defence during attack |

---

## Data format

Each attack dataset is a JSON **list** of objects with this schema (see `data/data_subsample.json`):

```json
[
  {
    "goal": "Harmful request text shown to the diffusion model",
    "initial_query": "Optional prefix prepended to the tunable suffix",
    "target_behavior": "Sure, here's ...\n\n",
    "forbidden_prompt": "Alternate phrasing of the harmful request"
  }
]
```

| Field | Role |
|---|---|
| `goal` | Substituted into `instruction_template` as `{goal}` for Dream |
| `initial_query` | Fixed user-message prefix (before tunable suffix) |
| `target_behavior` | Prefix the victim must emit; CE loss target |
| `forbidden_prompt` | Used when `goal` is empty |

Shipped files:

| File | Examples | Notes |
|---|---|---|
| `data/data_subsample.json` | 48 | Primary eval subset |
| `data/data_full.json` | 98 | Larger set, same schema |

---

## Results

Per-example output: `experiments/<experiment-name>_<job-id>/example_N_result.json`

Key fields:

| Field | Description |
|---|---|
| `example_id` | Index in the examples file |
| `initial_query`, `target_behavior` | Input behavior |
| `best_suffix_filled` | Decoded adversarial suffix after Dream fill |
| `best_response` | Best generation from Phase-2 batched eval |
| `num_steps` | Steps actually run |
| `success` / `success_so_far` | Attack success flags |
| `choose_best_n_verified_pool` | Up to 8 verified 512-token continuations |
| `config` | Resolved experiment config snapshot |

Shard aggregates:

```
all_results_shard0.json   # Combined results for shard 0
summary_shard0.json       # Success rate, timing summary
```

Runs resume automatically from an existing `example_N_result.json` checkpoint when `"final": false`.

---

## Monitoring

### Weights & Biases

Enabled by default unless `--no-wandb`. Project name comes from `wandb_project` in the experiment config. Multi-shard runs should use distinct runs grouped via `--wandb-group`.

### SLURM logs

Stdout/stderr are copied under `experiments/<name>_<id>/slurm/`. The runner prints resolved config knobs at startup to simplify debugging OOMs and misconfigurations.

---

## Extending the codebase

| Goal | File(s) |
|---|---|
| Add a config flag | `configs/config.yaml`, wire in `attack_pipeline.py` → `GCDAttack(...)` |
| Change optimization step | `gcd/gcd_core.py` (`step`) |
| Change candidate sampling | `gcd/gcd_candidate_maker.py` |
| Change victim scoring | `gcd/gcd_loss_evaluator.py`, `gcd/gcd_victim.py` |
| Change Dream fill/score | `gcd/gcd_dream.py`, `utils/diffusion_utils.py` |
| Change success criteria | `scripts/eval_utils.py`, `scripts/attack_pipeline.py` |
| Change model loading | `scripts/model_loader.py` |

External code can import:

```python
from gcd.gcd_attack import GCDAttack
```
