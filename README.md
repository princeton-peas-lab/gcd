# GCD

Implementation and evaluation harness for **Greedy Coordinate Diffusion (GCD)**, an ICML 2026 jailbreak attack. GCD uses a masked diffusion language model (Dream-v0-Instruct-7B) to iteratively optimize an adversarial suffix that causes a victim LLM to begin its reply with a specified harmful target prefix.

---

## Overview

The attack optimizes a tunable suffix token sequence appended to the user message, using a multi-objective guidance loss:

| Term | Config key | Description |
|---|---|---|
| Target CE | — | Cross-entropy toward the target response prefix in the victim model |
| Self-PPL | `self_perplexity_coef` | Fluency regularizer: batch-normalized victim perplexity of the suffix |
| Phase-2 Diversity | `phase2_div_loss_coef` | Unlikelihood penalty on bad continuation tokens (Phase 2 only) |
| Defence | `alpha_def` | Llama Guard guidance loss for `*_def` experiment variants |

At each step, Dream scores token replacements at masked suffix positions; the top-k candidates are evaluated against the victim; the best is committed. The attack runs for up to `num_steps` steps or until a stopping criterion fires.

---

## Repository Structure

```
gcd-attack/
├── gcd/                         # Attack core
│   ├── gcd_core.py              # GCDAttack class (init, step, run)
│   ├── gcd_attack_methods.py    # Shared mixins (Dream scores, caches, helpers)
│   ├── gcd_candidate_maker.py   # CandidateGenerator: position sampling + top-k
│   ├── gcd_loss_evaluator.py    # VictimEvaluator: Dream fill + victim forward + combined loss
│   ├── gcd_dream.py             # Dream model interaction mixin
│   ├── gcd_victim.py            # Victim CE / generation mixin
│   ├── gcd_text.py              # Text encoding/decoding helpers
│   └── gcd_attack.py            # Re-export shim: from gcd_core import GCDAttack
│
├── scripts/
│   ├── run_experiment.py        # Main entry point (argparse, model load, example loop)
│   ├── attack_pipeline.py       # Per-example orchestration, W&B logging, two-phase PCG, resume
│   ├── run_slurm.py             # SLURM array / multi-GPU submission
│   ├── config_utils.py          # YAML merge, experiment inherit resolution, HF env setup
│   ├── model_loader.py          # Dream + victim model loading (4-bit optional)
│   ├── eval_utils.py            # Prefix/tail checks, PCG refusal helpers
│   └── pcg_phase2_judge.py      # Phase-2 LLM judge (Qwen)
│
├── configs/
│   └── config.yaml              # All experiment configs (global settings + per-experiment blocks)
│
├── data/
│   ├── data_subsample.json      # 48-example primary eval set
│   ├── data_full.json           # 98-example set (same schema)
│   └── harmful-behaviors-dataset-full.json  # HarmBench-style raw export
│
├── utils/
│   └── diffusion_utils.py       # DreamGenerationMixin
│
├── plots/                       # Analysis and plotting scripts
│
└── experiments/                 # Output directory (created at runtime)
    └── <experiment-name>_<job-id>/
        ├── example_N_result.json
        ├── all_results_shardK.json
        ├── summary_shardK.json
        └── slurm/
```

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url> gcd-attack
cd gcd-attack
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

A [Hugging Face account](https://huggingface.co) and access token are required for gated models (Llama, Mistral). Log in once:

```bash
huggingface-cli login   # paste your HF token when prompted
```

Then run the download script:

```bash
# Download all attack models (~30 GB: Dream, Mistral, Qwen, Llama, Llama Guard)
bash download_models.sh

# Download to a custom directory (e.g. a shared HPC cache)
bash download_models.sh --dir /shared/hf_cache
```

| Model | HuggingFace ID | Role |
|---|---|---|
| Dream | `Dream-org/Dream-v0-Instruct-7B` | Diffusion model / candidate proposer |
| Mistral victim | `mistralai/Mistral-7B-Instruct-v0.3` | Victim LLM |
| Qwen victim / judge | `Qwen/Qwen2.5-7B-Instruct` | Victim LLM + Phase-2 judge |
| Llama victim | `meta-llama/Meta-Llama-3-8B-Instruct` | Victim LLM |
| Llama Guard | `meta-llama/Llama-Guard-3-1B` | Defence model (`*_def` experiments) |

### 4. Configure paths

Open `configs/config.yaml` and set the two values at the top of the `global:` block:

```yaml
global:
  project_dir: "/absolute/path/to/this/repo"   # ← set to your checkout
  environment:
    conda_env: "gcd"                             # ← name of your conda env
    huggingface:
      cache_dir: "/models"           # primary download target; HF_HOME is set to this
      cache_dirs:                    # ordered search path when loading any model
        - "/models"                  # default — where download_models.sh places files
        - "/path/to/existing/cache"  # add any pre-existing HF cache here
      offline_mode: true             # set to false if compute nodes have internet
```

#### Model search path (`cache_dirs`)

`cache_dirs` is the ordered list of directories the loader searches when resolving a HuggingFace repo ID to a local path. The **first** directory that contains the model wins — no re-download occurs. Both settings live in the top-level `global:` block of `configs/config.yaml`:

| Setting | Purpose |
|---|---|
| `global.environment.huggingface.cache_dirs` | Ordered list of local directories searched when loading any model |
| `global.environment.huggingface.cache_dir` | Primary download destination; also exported as `HF_HOME` |

The list ships with a single default entry (`/models/`). Add any pre-existing shared caches below it:

```yaml
cache_dirs:
  - "/models"                         # local downloads (default)
  - "/scratch/gpfs/shared/hf"         # example: shared HPC cache
  - "/home/user/.cache/huggingface"   # example: personal HF cache
```

You can also override the search path at runtime without editing the config:

```bash
export GCD_HF_CACHE_DIRS="/path/to/cache1:/path/to/cache2"
export HF_HOME="/path/to/cache1"
```

---

## Running Experiments

### Quick local run (single GPU, no SLURM)

```bash
python scripts/run_experiment.py \
  --config configs/config.yaml \
  --examples data/data_subsample.json \
  --experiment-name test_local \
  --experiment llama \
  --num-examples 1 \
  --gpu-id 0 \
  --no-wandb
```

Swap `--experiment` to target a different victim model:

| `--experiment` | Victim |
|---|---|
| `mistral` | Mistral-7B-Instruct-v0.3 |
| `qwen` | Qwen2.5-7B-Instruct |
| `llama` | Llama-3-8B-Instruct |
| `mistral_def` | Mistral + Llama Guard defence |
| `qwen_def` | Qwen + Llama Guard defence |
| `llama_def` | Llama + Llama Guard defence |

### SLURM submission (array job)

```bash
python scripts/run_slurm.py \
  --config configs/config.yaml \
  --examples data/data_subsample.json \
  --experiment-name my_llama_run \
  --experiment llama

# Monitor jobs
squeue -u $USER
tail -f experiments/my_llama_run_*/slurm/*.out
```

SLURM resource settings (`n_jobs`, `gpus_per_job`, `time`, `partition`, `constraint`) are read from the experiment's `slurm:` block. Override any value at submission time:

```bash
python scripts/run_slurm.py \
  --config configs/config.yaml \
  --examples data/data_subsample.json \
  --experiment-name my_run \
  --experiment llama \
  --override slurm.n_jobs=4 slurm.time=02:00:00 slurm.partition=gpu
```

### Override config values at runtime

```bash
python scripts/run_experiment.py \
  --config configs/config.yaml \
  --examples data/data_subsample.json \
  --experiment-name debug_run \
  --experiment llama \
  --override candidate_batch_pct=0.1 \
  --override self_perplexity_coef='[0.05, 0.05]' \
  --override slurm.n_jobs=4
```

---

## Attack Pipeline (Two-Phase PCG)

All primary configs (`mistral`, `qwen`, `llama`) use a **two-phase prefix-continuation (PCG)** pipeline.

### Phase 1 — prefix optimization

Up to `main_optimization_num_steps` (800) steps optimizing the suffix until the victim emits the target prefix. Stops cleanly when no prefix is found (`prefix_continuation_no_fallback: true`).

### Phase 2 — continuation + judge pool

After the first prefix match, up to `optimization_gen_check_steps` (400) additional steps. Candidates are staged; once a batch is full (batch size 16), the victim runs batched greedy decode for 512 tokens on all staged prompts. Each continuation is scored by a local Qwen Phase-2 judge (`fine_grained_legal_circumvention` mode, min score 4). Up to 8 passing responses are kept in `choose_best_n_verified_pool`.

### Llama Guard defence variants (`mistral_def`, `qwen_def`, `llama_def`)

These configs mix Llama Guard (`meta-llama/Llama-Guard-3-1B`) into the guidance loss at `alpha_def: 0.2`. The attack only stops when the defence model flags the output safe (`req_safe_stop: true`). Each pool entry records `fresh_judge_prediction` (the Guard label at collection time).

---

## Key Hyperparameters

From `configs/config.yaml` (`experiments._base`):

| Key | Default | Description |
|---|---|---|
| `num_steps` | `800` | Total optimization steps (Phase 1) |
| `num_tunable_tokens` | `80` | Adversarial suffix length in tokens |
| `top_k_start` / `top_k_end` | `32` | Token proposals per position |
| `random_pos_p` | `0.25` | Fraction of suffix positions sampled per step |
| `candidate_batch_pct` | `0.15` | Fraction of top-k candidates evaluated per step |
| `eval_batch_size` | `512` | Max candidates per victim eval chunk (auto-halved on OOM) |
| `self_perplexity` | `true` | Add batch-normalized suffix self-PPL to CE for candidate ranking |
| `self_perplexity_coef` | `[0.2, 0.2]` | Weight of the self-PPL term in guidance loss |
| `use_raw_ppl` | `false` | Use raw exp(CE) instead of log-CE for self-PPL scoring |
| `phase2_div_loss_coef` | `0.2` | Phase-2 diversity penalty weight |
| `main_optimization_num_steps` | `800` | Phase 1 step budget |
| `optimization_gen_check_steps` | `400` | Phase 2 step budget after first prefix match |
| `prefix_continuation_full_gen_max_new_tokens` | `512` | Batched greedy continuation length (Phase 2) |
| `prefix_continuation_phase2_collect_pool_size` | `8` | Max entries in `choose_best_n_verified_pool` |

### Speed vs. readability trade-off

Two knobs have the largest impact on the balance between attack speed and the human-readability / fluency of the resulting adversarial prompts:

**Prioritize speed** — evaluate fewer candidates per step, use weak PPL regularization:

| Key | Recommended range |
|---|---|
| `candidate_batch_pct` | `0.1` |
| `self_perplexity_coef` | `[0.0, 0.0]` – `[0.1, 0.1]` |

**Prioritize human readability / fluency** — evaluate a wider candidate set with stronger PPL regularization, producing more semantically coherent adversarial prompts:

| Key | Recommended range |
|---|---|
| `candidate_batch_pct` | `0.15` – `0.2` |
| `self_perplexity_coef` | `[0.1, 0.1]` – `[0.2, 0.2]` |

> Increasing `candidate_batch_pct` lets GCD evaluate more of the top-k position/token pairs at each step, giving the selection step a richer pool to find the most fluent candidate — without narrowing `top_k` itself. Reducing it trades that exploration breadth for throughput, potentially making perplexity convergense more unstable, while generally sufficient for the main adversarial attack objective.

---

## Configuration

All experiment parameters live in `configs/config.yaml`.

### Structure

```yaml
global:
  project_dir: "/path/to/this/repo"
  environment:
    conda_env: gcd
    huggingface:
      cache_dir: "/models"
      cache_dirs:
        - "/models"
      offline_mode: true

experiments:
  _base:                              # shared defaults inherited by all experiments
    num_steps: 800
    candidate_batch_pct: 0.15
    models:
      dream_model: "Dream-org/Dream-v0-Instruct-7B"
      victim_model: "mistralai/Mistral-7B-Instruct-v0.3"
      use_quantization: true
    slurm:
      n_jobs: 12
      gpus_per_job: 4

  mistral:
    inherit: _base                    # deep-merge _base; child values override
    random_pos_reference_len: 80

  qwen:
    inherit: mistral
    models:
      victim_model: "Qwen/Qwen2.5-7B-Instruct"
```

Child experiments inherit all parent keys; child values override. Resolution is handled by `scripts/config_utils.resolve_experiment_config()`.

### Experiment keys

| Key | Victim | Notes |
|---|---|---|
| `mistral` | Mistral-7B-Instruct-v0.3 | Primary baseline |
| `qwen` | Qwen2.5-7B-Instruct | ailab partition |
| `llama` | Llama-3-8B-Instruct | ailab partition |
| `mistral_def` | Mistral + Llama Guard | Defence evasion variant |
| `qwen_def` | Qwen + Llama Guard | Defence evasion variant |
| `llama_def` | Llama + Llama Guard | Defence evasion variant |

### Swapping models

Edit the `models:` block of any experiment:

```yaml
models:
  dream_model: "Dream-org/Dream-v0-Instruct-7B"
  victim_model: "meta-llama/Meta-Llama-3-8B-Instruct"
  use_quantization: true
```

---

## Data Format

Each dataset is a JSON list of objects:

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

| File | Examples | Notes |
|---|---|---|
| `data/data_subsample.json` | 48 | Primary eval subset |
| `data/data_full.json` | 98 | Larger set, same schema |

---

## Output Format

Per-example output: `experiments/<experiment-name>_<job-id>/example_N_result.json`

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

Enabled by default unless `--no-wandb`. Project name comes from `wandb_project` in the experiment config. Multi-shard runs should pass `--wandb-group` to group all shards under one run.

### SLURM logs

Stdout/stderr are written under `experiments/<name>_<id>/slurm/`. The runner prints the resolved config at startup to help debug OOMs and misconfigurations.

---

## CLI Reference

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

### `scripts/run_slurm.py`

Wraps `run_experiment.py` with SLURM resource allocation from the experiment's `slurm:` block (`n_jobs`, `gpus_per_job`, `tasks_per_job`, `time`, `constraint`, etc.). Supports skipping already-finished examples when result JSONs exist.

---

## Extending the Codebase

| Goal | File(s) |
|---|---|
| Add a config flag | `configs/config.yaml`, wire in `attack_pipeline.py` → `GCDAttack(...)` |
| Change optimization step | `gcd/gcd_core.py` (`step`) |
| Change candidate sampling | `gcd/gcd_candidate_maker.py` |
| Change victim scoring | `gcd/gcd_loss_evaluator.py`, `gcd/gcd_victim.py` |
| Change Dream fill/score | `gcd/gcd_dream.py`, `utils/diffusion_utils.py` |
| Change success criteria | `scripts/eval_utils.py`, `scripts/attack_pipeline.py` |
| Change model loading | `scripts/model_loader.py` |

External code can import the attack class directly:

```python
from gcd.gcd_attack import GCDAttack
```

---

## Citation

```bibtex
@misc{turbal2026greedycoordinatediffusioneffective,
      title={Greedy Coordinate Diffusion: Effective and Semantically Coherent Adversarial Attacks via Diffusion Guidance}, 
      author={Bohdan Turbal and Blossom Metevier and Max Springer and Aleksandra Korolova},
      year={2026},
      eprint={2606.15531},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.15531}, 
}
```
