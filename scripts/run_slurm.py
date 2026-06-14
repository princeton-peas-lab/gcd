#!/usr/bin/env python3
"""
SLURM job submission wrapper for GCD experiments.
Splits examples across GPUs when split_jobs=True.
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from simple_slurm import Slurm

from config_utils import deep_merge


PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "configs" / "config.yaml"
DEFAULT_SLURM_LOG_DIR = PROJECT_DIR / "slurm"
DEFAULT_SLURM_LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> dict:
    """Load configuration from YAML file."""
    with path.open("r") as handle:
        return yaml.safe_load(handle) or {}


def load_examples(examples_path: Path) -> List[dict]:
    """Load examples from JSON file."""
    import json
    with examples_path.open("r") as handle:
        return json.load(handle)


def _resolve_experiment_config(cfg: dict, experiment_name: str, visited: Optional[List[str]] = None) -> Dict[str, Any]:
    """Same semantics as run_experiment.resolve_experiment_config (YAML inherit/base)."""
    if visited is None:
        visited = []
    if experiment_name in visited:
        raise ValueError(f"Circular inheritance: {' -> '.join(visited + [experiment_name])}")
    visited = visited + [experiment_name]
    experiments = cfg.get("experiments", {}) or {}
    if experiment_name not in experiments:
        if experiment_name == "default":
            return {}
        raise ValueError(f"Experiment '{experiment_name}' not found in config.")
    exp_cfg = experiments[experiment_name]
    if not isinstance(exp_cfg, dict):
        return {}
    parent_name = exp_cfg.get("inherit") or exp_cfg.get("base")
    if parent_name:
        parent_cfg = _resolve_experiment_config(cfg, str(parent_name), visited)
        return deep_merge(parent_cfg, exp_cfg)
    return dict(exp_cfg)


def is_finished(result_path: Path) -> bool:
    """Return True if a result JSON exists and indicates a finished run."""
    import json
    if not result_path.exists():
        return False
    try:
        with result_path.open("r") as handle:
            data = json.load(handle)
    except Exception:
        return False
    if data.get("error"):
        return False
    # Backward compatibility: if finished_run is missing, assume the run finished.
    if "finished_run" not in data:
        return True
    return bool(data.get("finished_run", False))


def select_experiment_config(cfg: dict, experiment_key: str) -> dict:
    """Merge defaults with experiment-specific config (experiment overrides defaults)."""
    defaults = cfg.get("defaults", {}) or {}
    experiments = cfg.get("experiments", {}) or {}
    if experiment_key not in experiments:
        avail = sorted(experiments.keys()) if isinstance(experiments, dict) else []
        preview = ", ".join(avail[:20])
        more = "" if len(avail) <= 20 else f", ... (+{len(avail) - 20} more)"
        print(
            f"[run_slurm] WARNING: experiment key {experiment_key!r} not under 'experiments:' — "
            f"using defaults only (SLURM resource hints may not match the real run).\n"
            f"  Available: [{preview}{more}]"
        )
        return {**defaults}
    try:
        resolved = _resolve_experiment_config(cfg, experiment_key)
    except ValueError as e:
        print(f"[run_slurm] WARNING: could not resolve experiment {experiment_key!r}: {e}")
        return {**defaults}
    if isinstance(resolved, dict):
        resolved = {k: v for k, v in resolved.items() if k not in ("inherit", "base")}
    return {**defaults, **resolved}


def _format_value_for_suffix(value: Any) -> str:
    """Compact, filesystem-safe token for an override value."""
    if isinstance(value, bool):
        token = "true" if value else "false"
    elif isinstance(value, int):
        token = str(value)
    elif isinstance(value, float):
        token = f"{value:g}"
    elif isinstance(value, str):
        token = value
    elif isinstance(value, (list, tuple)):
        token = "-".join(_format_value_for_suffix(v) for v in value)
    elif isinstance(value, dict):
        token = "-".join(
            f"{k}:{_format_value_for_suffix(v)}" for k, v in value.items()
        )
    else:
        token = str(value)

    token = token.replace(".", "p").replace("/", "_").replace(" ", "")
    token = "".join(ch if (ch.isalnum() or ch in {"-", "_", ":"}) else "_" for ch in token)
    token = token.strip("_")
    return token or "val"


def _expand_override_variants(slurm_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand sweep variants from slurm config.

    Supported schema:
      slurm:
        override_combinations:
          - {candidate_batch_pct: 0.1, self_perplexity_coef: [0.0, 0.0]}
          - {candidate_batch_pct: 0.25, self_perplexity_coef: [0.0, 0.0]}

      slurm:
        override_grid:
          candidate_batch_pct: [0.1, 0.25, 0.5, 1.0]
          self_perplexity_coef:
            - [0.0, 0.0]
            - [0.1, 0.1]
    """
    combos_raw = slurm_config.get("override_combinations", None)
    if combos_raw is not None:
        if not isinstance(combos_raw, list):
            raise ValueError("slurm.override_combinations must be a list of dicts.")
        variants: List[Dict[str, Any]] = []
        for i, item in enumerate(combos_raw):
            if not isinstance(item, dict):
                raise ValueError(
                    f"slurm.override_combinations[{i}] must be a dict, got {type(item).__name__}"
                )
            variants.append(dict(item))
        return variants

    grid_raw = slurm_config.get("override_grid", None)
    if grid_raw is not None:
        if not isinstance(grid_raw, dict) or not grid_raw:
            raise ValueError("slurm.override_grid must be a non-empty dict.")
        keys = list(grid_raw.keys())
        value_axes: List[List[Any]] = []
        for k in keys:
            vals = grid_raw.get(k)
            if not isinstance(vals, list) or len(vals) == 0:
                raise ValueError(f"slurm.override_grid['{k}'] must be a non-empty list.")
            value_axes.append(vals)

        variants = []
        for prod_vals in itertools.product(*value_axes):
            variants.append({k: v for k, v in zip(keys, prod_vals)})
        return variants

    return [{}]


def _build_variant_experiment_name(base_name: str, overrides: Dict[str, Any]) -> str:
    if not overrides:
        return base_name
    parts = []
    for key, value in overrides.items():
        key_token = key.split(".")[-1]
        val_token = _format_value_for_suffix(value)
        parts.append(f"{key_token}-{val_token}")
    suffix = "__".join(parts)
    out = f"{base_name}__{suffix}"
    # Keep names manageable for slurm/job names/path lengths.
    return out[:200]


def build_command(
    config_path: Path,
    examples_path: Path,
    experiment_name: str,
    experiment_key: str,
    experiment_id: str,
    start_example: int,
    num_examples: int,
    gpu_id: int,
    mode_split: str,
    no_wandb: bool = False,
    use_accelerate: bool = False,
    num_processes: int = 1,
    overrides: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the command to run the experiment."""
    
    if use_accelerate:
        base = f"accelerate launch --multi_gpu --num_processes {num_processes}"
        cmd_parts = [base, f"{PROJECT_DIR / 'scripts' / 'run_experiment.py'}"]
    else:
        cmd_parts = [f"python -u {PROJECT_DIR / 'scripts' / 'run_experiment.py'}"]
    
    cmd_parts.extend([
        f"--config {config_path}",
        f"--examples {examples_path}",
        f"--experiment-name {experiment_name}",
        f"--experiment {experiment_key}",
        f"--experiment-id {experiment_id}",
        f"--start-example {start_example}",
        f"--num-examples {num_examples}",
        f"--gpu-id {gpu_id}",
        f"--mode-split {mode_split}",
    ])
    
    if no_wandb:
        cmd_parts.append("--no-wandb")

    if overrides:
        for key, value in overrides.items():
            override_payload = f"{key}={json.dumps(value, separators=(',', ':'))}"
            override_payload = override_payload.replace("\\", "\\\\").replace('"', '\\"')
            cmd_parts.append(f'--override "{override_payload}"')
    
    return " \\\n    ".join(cmd_parts)


def submit_jobs(
    config_path: Path,
    examples_path: Path,
    experiment_name: str,
    experiment_key: str,
    no_wandb: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
    _expanded_sweep: bool = False,
):
    """Submit SLURM jobs for the experiment."""
    
    cfg = load_config(config_path)
    examples = load_examples(examples_path)
    
    experiment_config = select_experiment_config(cfg, experiment_key)
    # Per-experiment slurm config (preferred). Fallback to legacy top-level cfg["slurm"] if present.
    slurm_config = experiment_config.get("slurm", {}) or cfg.get("slurm", {}) or {}
    process_odd_even = bool(slurm_config.get("process_odd_even", experiment_config.get("process_odd_even", False)))
    try:
        check_missing = int(slurm_config.get("check_missing", experiment_config.get("check_missing", 0)))
    except Exception:
        check_missing = 0

    # Expand sweep variants once at top level. Each variant is submitted as a separate experiment
    # with an auto-generated suffix in experiment_name.
    if not _expanded_sweep:
        variants = _expand_override_variants(slurm_config)
        if len(variants) == 1 and not variants[0]:
            overrides = {}
        else:
            print(f"[run_slurm] Sweep variants to submit: {len(variants)}")
            for idx, variant in enumerate(variants, start=1):
                variant_name = _build_variant_experiment_name(experiment_name, variant)
                print(
                    f"[run_slurm] Submitting variant {idx}/{len(variants)}: "
                    f"experiment_name={variant_name}, overrides={variant}"
                )
                submit_jobs(
                    config_path=config_path,
                    examples_path=examples_path,
                    experiment_name=variant_name,
                    experiment_key=experiment_key,
                    no_wandb=no_wandb,
                    overrides=variant,
                    _expanded_sweep=True,
                )
            return

    mode_split = str(slurm_config.get("mode_split", "examples")).strip().lower()
    n_jobs = max(1, int(slurm_config.get("n_jobs", 1)))
    tasks_per_job = max(1, int(slurm_config.get("tasks_per_job", slurm_config.get("ntasks", 1))))
    gpus_per_job = max(1, int(slurm_config.get("gpus_per_job", 1)))
    # If any srun task exits non-zero, Slurm commonly aborts the whole step and SIGKILLs the rest.
    # For debugging, you can set this to 0 to keep other tasks running even if one fails.
    kill_on_bad_exit = int(slurm_config.get("kill_on_bad_exit", experiment_config.get("kill_on_bad_exit", 0)))
    # Helps debugging: split stdout/stderr into per-task files instead of one interleaved log.
    per_task_logs = bool(slurm_config.get("per_task_logs", True))
    # Prefix each output line with task id when using a shared log (or just generally helpful).
    srun_label = bool(slurm_config.get("srun_label", True))
    # Independent tasks: --mpi=pmi2 avoids PMI interpreting one task exit as "job done" and killing others.
    # Override with srun_mpi: "none" in config if your cluster does not use PMI.
    srun_mpi = str(slurm_config.get("srun_mpi", slurm_config.get("mpi", "pmi2"))).strip().lower()
    
    total_examples = len(examples)
    
    print(f"Total examples: {total_examples}")
    print(f"Experiment key: {experiment_key}")
    print(f"Mode split: {mode_split}")
    print(f"n_jobs (separate sbatch): {n_jobs}")
    print(f"tasks_per_job: {tasks_per_job}")
    print(f"GPUs per job: {gpus_per_job}")
    print(f"kill_on_bad_exit: {kill_on_bad_exit} (0 = do not kill other tasks if one fails)")
    print(f"srun_mpi: {srun_mpi} (default pmi2; use 'none' if cluster does not use PMI)")
    if overrides:
        print(f"Override values for this variant: {overrides}")

    # ------------------------------------------------------------------
    # Shared experiment id across ALL submitted jobs/tasks in this call.
    #
    # Why: by default, run_experiment.py uses SLURM_JOB_ID which differs per sbatch,
    # which spreads results across many folders. Here we generate a shared id once
    # so that all shards save into ONE shared folder.
    #
    # Folder naming in run_experiment.py is:
    #   experiments/{experiment_name}_{experiment_id}
    #
    # We include the config key in experiment_id for traceability:
    #   experiment_id = "{experiment_key}_{run_uid}"
    # ------------------------------------------------------------------
    run_uid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shared_experiment_id = f"{experiment_key}_{run_uid}"
    experiment_results_dir = PROJECT_DIR / "experiments" / f"{experiment_name}_{shared_experiment_id}"
    experiment_slurm_log_dir = experiment_results_dir / "slurm"
    experiment_slurm_log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run_slurm] Shared experiment id: {shared_experiment_id}")
    print(f"[run_slurm] Results folder: {experiment_results_dir}")
    print(f"[run_slurm] Slurm log folder: {experiment_slurm_log_dir}")
    
    if mode_split not in {"model", "examples", "inside_job_coherent"}:
        raise ValueError(
            f"Invalid mode_split={mode_split!r}. Must be 'model', 'examples', or 'inside_job_coherent'."
        )

    if mode_split == "inside_job_coherent" and tasks_per_job != 1:
        print(
            f"[slurm] mode_split=inside_job_coherent requires tasks_per_job=1 (got {tasks_per_job}). Overriding to 1."
        )
        tasks_per_job = 1

    if mode_split == "examples" and tasks_per_job != gpus_per_job:
        # For example-parallel inside a single sbatch allocation, we want one task per GPU.
        # This maps cleanly to SLURM's task model and avoids needing accelerate.
        raise ValueError("For mode_split='examples', set tasks_per_job == gpus_per_job (one task per GPU).")

    # Split across n_jobs (each sbatch submission). Within each sbatch, tasks_per_job workers run in parallel.
    #
    # - mode_split="examples": workers are distinct (one per GPU) and shard the example range via --shard-id/--num-shards.
    # - mode_split="model": a single worker can see all GPUs and shard model weights (device_map="auto").
    #
    # For mode_split="examples", we deliberately pass the full example range to all jobs and rely on global sharding to
    # avoid overlap. This keeps numbering consistent and avoids complex start/stop arithmetic in bash.
    job_specs: List[int] = list(range(1, n_jobs + 1))
    
    # SLURM settings
    global_settings = cfg.get("global", {})
    
    # Check for h200 option
    use_h200 = slurm_config.get("h200", False)
    
    # Build slurm_kwargs based on h200 setting
    slurm_kwargs = dict(
        job_name=f"gcd_diff_{experiment_name}",
        nodes=slurm_config.get("nodes", 1),
        ntasks=tasks_per_job,
        cpus_per_task=slurm_config.get("cpus_per_task", 8),
        mem=slurm_config.get("mem", "128G"),
        time=slurm_config.get("time", "24:00:00"),
        gres=f"gpu:{gpus_per_job}",
        # One file per job for all outputs (stdout/stderr). Job id is injected by SLURM via %j.
        output=str(experiment_slurm_log_dir / f"gcd_diff_{experiment_name}_%j.out"),
        error=str(experiment_slurm_log_dir / f"gcd_diff_{experiment_name}_%j.out"),
        mail_type=slurm_config.get("mail_type", "END,FAIL"),
        mail_user=slurm_config.get("mail_user", "bt4811@princeton.edu"),
    )
    
    # Set partition and constraint based on h200 option
    partition = slurm_config.get("partition")
    constraint = slurm_config.get("constraint")
    
    if use_h200:
        # When targeting H200s, many configs inherit a default constraint like "gpu80".
        # If we're submitting to a specific H200 partition (e.g. "ailab"), that inherited
        # constraint becomes wrong and can make Slurm report "Requested node configuration is not available".
        #
        # Policy:
        # - If a partition is specified, do NOT forward the default "gpu80" constraint.
        # - If a non-default constraint is explicitly set (e.g. "h200"), keep it.
        if partition:
            slurm_kwargs["partition"] = partition
            if constraint and str(constraint).strip().lower() not in {"gpu80"}:
                slurm_kwargs["constraint"] = constraint
        else:
            # No partition specified, use constraint to select H200 GPUs
            if constraint:
                slurm_kwargs["constraint"] = constraint
            elif slurm_config.get("default_gpu80_constraint", True):
                slurm_kwargs["constraint"] = "h200"
    else:
        # For non-h200, set partition if specified, otherwise use constraint.
        # When a partition is set, suppress the inherited default "gpu80" constraint
        # to avoid "Requested node configuration is not available" errors.
        if partition:
            slurm_kwargs["partition"] = partition
            if constraint and str(constraint).strip().lower() not in {"gpu80"}:
                slurm_kwargs["constraint"] = constraint
        elif constraint:
            slurm_kwargs["constraint"] = constraint
        elif not partition and slurm_config.get("default_gpu80_constraint", True):
            # Default constraint only if no partition is set
            slurm_kwargs["constraint"] = "gpu80"

    print(
        f"SLURM partition: {slurm_kwargs.get('partition', '(account default)')}, "
        f"constraint: {slurm_kwargs.get('constraint', 'none')}"
    )
    
    # Environment setup
    env = global_settings.get("environment", {})
    modules = env.get("modules", [])
    conda_env = env.get("conda_env", "gcg-diffusion")
    hf_cache = env.get("huggingface", {}).get("cache_dir", "/scratch/gpfs/KOROLOVA/huggingface")
    wandb_offline = env.get("wandb", {}).get("offline_mode", False)
    # Prefer writing W&B files to scratch to avoid HOME quota issues.
    scratch_dir = global_settings.get("scratch_dir", None) or "/scratch/gpfs/KOROLOVA/bt4811"
    wandb_base = Path(scratch_dir) / "wandb"
    wandb_run_dir = wandb_base / "runs"
    wandb_cache_dir = wandb_base / "cache"
    wandb_config_dir = wandb_base / "config"
    wandb_data_dir = wandb_base / "data"
    wandb_artifacts_dir = wandb_base / "artifacts"
    wandb_tmp_dir = wandb_base / "tmp"
    
    # Submit jobs
    total_shards = max(1, n_jobs * tasks_per_job)
    submitted_job_ids: List[str] = []
    override_check_lines = ""
    if overrides:
        for key, value in overrides.items():
            payload = f"{key}={json.dumps(value, separators=(',', ':'))}"
            payload = payload.replace("\\", "\\\\").replace('"', '\\"')
            override_check_lines += f'  --override "{payload}" \\\n'

    for job_idx in job_specs:
        slurm = Slurm(**slurm_kwargs)
        
        # Build environment setup
        env_setup = f"""
module purge
"""
        for module in modules:
            env_setup += f"module load {module}\n"
        
        env_setup += f"""
source ~/.bashrc
conda activate {conda_env}

# Force W&B to write *everything* under scratch (runs, cache, artifacts staging) to avoid HOME quota.
# - WANDB_DIR controls run directory (logs, media)
# - WANDB_CACHE_DIR / WANDB_DATA_DIR influence caching + artifact staging paths
# - TMPDIR is used by Python tempfile, which W&B uses for staging
export WANDB_DIR="{wandb_run_dir}"
export WANDB_CACHE_DIR="{wandb_cache_dir}"
export WANDB_CONFIG_DIR="{wandb_config_dir}"
export WANDB_DATA_DIR="{wandb_data_dir}"
export WANDB_ARTIFACT_DIR="{wandb_artifacts_dir}"
export TMPDIR="{wandb_tmp_dir}"
export PYTHONUNBUFFERED="1"
export HF_HOME="{hf_cache}"
export HUGGINGFACE_HUB_CACHE="{hf_cache}"
export HF_DATASETS_CACHE="{hf_cache}/datasets"
"""
        
        if process_odd_even:
            env_setup += """
wait_for_minute_parity() {
  target="$1"
  label="$2"
  while true; do
    m=$(date +%M)
    mod=$((10#$m % 4))
    if [ "$mod" -eq "$target" ]; then
      echo "[OddEven] ${label}: minute=${m} mod4 OK"
      break
    fi
    echo "[OddEven] ${label}: waiting for mod4 ${target} (now=${m})"
    s=$(date +%S)
    sleep $((60 - 10#$s))
  done
}
"""
        
        if wandb_offline:
            env_setup += 'export WANDB_MODE="offline"\n'
        
        if env.get("huggingface", {}).get("offline_mode", False):
            env_setup += """
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
"""
        
        env_setup += f"""
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" "$TMPDIR"

cd "{PROJECT_DIR}"

"""
        # Build command(s)
        if mode_split == "examples":
            # Example-parallel within allocation: run N tasks, each task uses one GPU and a distinct shard id.
            # Global shard id = (job_idx-1)*tasks_per_job + SLURM_PROCID
            base_shard = (job_idx - 1) * tasks_per_job

            # All tasks receive the full example range; sharding happens inside run_experiment.py.
            # Note: build_command returns a string that we reuse inside the srun template.
            # For mode_split="examples", we use python directly on each GPU.
            base_cmd = build_command(
                config_path=config_path,
                examples_path=examples_path,
                experiment_name=experiment_name,
                experiment_key=experiment_key,
                experiment_id=shared_experiment_id,
                start_example=0,
                num_examples=total_examples,
                gpu_id=0,
                mode_split=mode_split,
                no_wandb=no_wandb,
                use_accelerate=False,
                overrides=overrides,
            )

            # Note: we set CUDA_VISIBLE_DEVICES per-task using SLURM_LOCALID (0..tasks_per_job-1).
            # Each task passes shard-id and num-shards so there is no overlap across all jobs/tasks.
            srun_out = ""
            if per_task_logs:
                # %j = job id, %t = task id (rank within this srun step)
                srun_out = (
                    f' --output="{experiment_slurm_log_dir / f"gcd_diff_{experiment_name}_%j_task%t.out"}"'
                    f' --error="{experiment_slurm_log_dir / f"gcd_diff_{experiment_name}_%j_task%t.out"}"'
                )
            srun_lbl = " --label" if srun_label else ""
            # Barrier: each task runs its work, then waits until all tasks have finished before exiting.
            barrier_base = PROJECT_DIR / "experiments" / f"{experiment_name}_{shared_experiment_id}" / "_srun_barrier"
            pre_wait = "wait_for_minute_parity 0 start" if process_odd_even else ""
            post_wait = "wait_for_minute_parity 2 exit" if process_odd_even else ""
            command = f"""
{pre_wait}
srun --unbuffered{srun_lbl}{srun_out} --mpi={srun_mpi} --kill-on-bad-exit={kill_on_bad_exit} -n {tasks_per_job} --ntasks-per-node {tasks_per_job} bash -lc '
export CUDA_VISIBLE_DEVICES="$SLURM_LOCALID"
BARRIER_DIR="{barrier_base}/$SLURM_JOB_ID"
N_TASKS={tasks_per_job}
mkdir -p "$BARRIER_DIR"
trap '"'"'touch "$BARRIER_DIR/task_$SLURM_PROCID.done" 2>/dev/null; sync'"'"' EXIT
{base_cmd} --num-shards {total_shards} --shard-id $(({base_shard} + $SLURM_PROCID)) --wandb-group "{experiment_name}_{shared_experiment_id}"
ec=$?
touch "$BARRIER_DIR/task_$SLURM_PROCID.done" && sync
echo "[Barrier] task $SLURM_PROCID work done, waiting for all $N_TASKS tasks..."
while [ $(ls "$BARRIER_DIR"/task_*.done 2>/dev/null | wc -l) -lt $N_TASKS ]; do sleep 5; done
echo "[Barrier] all tasks done, exiting with $ec"
exit $ec
'
{post_wait}
"""
        elif mode_split == "inside_job_coherent":
            # Inside-job coherent mode:
            # one multi-GPU distributed run where ranks cooperate on the same attacks.
            command = build_command(
                config_path=config_path,
                examples_path=examples_path,
                experiment_name=experiment_name,
                experiment_key=experiment_key,
                experiment_id=shared_experiment_id,
                start_example=0,
                num_examples=total_examples,
                gpu_id=0,
                mode_split=mode_split,
                no_wandb=no_wandb,
                use_accelerate=True,
                num_processes=gpus_per_job,
                overrides=overrides,
            )
            if process_odd_even:
                command = f"wait_for_minute_parity 0 start\n{command}\nwait_for_minute_parity 2 exit"
        else:
            # Model-sharding: single process sees all allocated GPUs.
            command = build_command(
                config_path=config_path,
                examples_path=examples_path,
                experiment_name=experiment_name,
                experiment_key=experiment_key,
                experiment_id=shared_experiment_id,
                start_example=0,
                num_examples=total_examples,
                gpu_id=0,
                mode_split=mode_split,
                no_wandb=no_wandb,
                use_accelerate=False,
                overrides=overrides,
            )
            if process_odd_even:
                command = f"wait_for_minute_parity 0 start\n{command}\nwait_for_minute_parity 2 exit"
        
        full_command = env_setup + command
        
        job_id = slurm.sbatch(full_command)
        submitted_job_ids.append(str(job_id))
        print(f"Submitted job {job_id} (job_idx={job_idx}, mode_split={mode_split}, gpus_per_job={gpus_per_job}, tasks_per_job={tasks_per_job})")

    # ------------------------------------------------------------------
    # Optional: submit follow-up jobs to check and re-run missing examples.
    # check_missing = number of passes. Each pass is ONE job that:
    # 1) Scans the whole result folder for unfinished examples
    # 2) Redistributes those missing examples across this job's GPUs (num_shards=tasks_per_job)
    # Pass 2 starts after pass 1 completes (and so on).
    # ------------------------------------------------------------------
    if check_missing > 0 and submitted_job_ids:
        dependency_chain = list(submitted_job_ids)
        for pass_idx in range(1, check_missing + 1):
            dependency = f"afterany:{':'.join(dependency_chain)}"
            # One job per pass: shard missing examples across this job's tasks (GPUs).
            total_shards = max(1, tasks_per_job)
            pass_job_ids: List[str] = []

            check_kwargs = dict(slurm_kwargs)
            check_kwargs["job_name"] = f"gcd_diff_{experiment_name}_check_missing_{pass_idx}"
            check_kwargs["ntasks"] = tasks_per_job
            check_kwargs["gres"] = f"gpu:{gpus_per_job}"
            slurm_check = Slurm(**check_kwargs, dependency=dependency)

            srun_out = ""
            if per_task_logs:
                srun_out = (
                    f' --output="{experiment_slurm_log_dir / f"gcd_diff_{experiment_name}_check_missing_%j_task%t.out"}"'
                    f' --error="{experiment_slurm_log_dir / f"gcd_diff_{experiment_name}_check_missing_%j_task%t.out"}"'
                )
            srun_lbl = " --label" if srun_label else ""
            barrier_base = PROJECT_DIR / "experiments" / f"{experiment_name}_{shared_experiment_id}" / "_srun_barrier"
            check_cmd = f"""
srun --unbuffered{srun_lbl}{srun_out} --mpi={srun_mpi} --kill-on-bad-exit={kill_on_bad_exit} -n {tasks_per_job} --ntasks-per-node {tasks_per_job} bash -lc '
export CUDA_VISIBLE_DEVICES="$SLURM_LOCALID"
BARRIER_DIR="{barrier_base}/$SLURM_JOB_ID"
N_TASKS={tasks_per_job}
mkdir -p "$BARRIER_DIR"
trap '"'"'touch "$BARRIER_DIR/task_$SLURM_PROCID.done" 2>/dev/null; sync'"'"' EXIT
python -u {PROJECT_DIR / 'scripts' / 'check_missing.py'} \\
  --config {config_path} \\
  --examples {examples_path} \\
  --experiment-name {experiment_name} \\
  --experiment {experiment_key} \\
  --experiment-id {shared_experiment_id} \\
  --check-pass {pass_idx} \\
  --resume \\
  --num-shards {total_shards} \\
  --shard-id $SLURM_PROCID \\
  --gpu-id 0 \\
{override_check_lines}
  {"--no-wandb" if no_wandb else ""}
ec=$?
touch "$BARRIER_DIR/task_$SLURM_PROCID.done" && sync
echo "[Barrier] task $SLURM_PROCID work done, waiting for all $N_TASKS tasks..."
while [ $(ls "$BARRIER_DIR"/task_*.done 2>/dev/null | wc -l) -lt $N_TASKS ]; do sleep 5; done
echo "[Barrier] all tasks done, exiting with $ec"
exit $ec
'
"""
            full_check_command = env_setup + check_cmd
            check_job_id = slurm_check.sbatch(full_check_command)
            pass_job_ids.append(str(check_job_id))
            print(
                f"Submitted check_missing job {check_job_id} "
                f"(pass {pass_idx}/{check_missing}, {tasks_per_job} GPUs, scans whole result folder; dependency=afterany:{':'.join(dependency_chain)})"
            )

            dependency_chain.extend(pass_job_ids)


def submit_post_fix_jobs(
    config_path: Path,
    examples_path: Path,
    experiment_name: str,
    experiment_key: str,
    target_folder: Path,
    n_jobs_fix: int = 1,
    no_wandb: bool = False,
):
    """Submit SLURM jobs to re-run missing/unfinished examples in target_folder."""
    cfg = load_config(config_path)
    examples = load_examples(examples_path)
    total_examples = len(examples)

    if not target_folder.exists():
        raise FileNotFoundError(f"target_folder not found: {target_folder}")

    target_folder = target_folder.resolve()
    target_slurm_log_dir = target_folder / "slurm"
    target_slurm_log_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{experiment_name}_"
    if not target_folder.name.startswith(prefix) and not target_folder.name.startswith("submission_"):
        raise ValueError(
            f"target_folder must be named like '{prefix}<experiment_id>' or 'submission_...'. Got: {target_folder.name}"
        )
    if target_folder.name.startswith("submission_"):
        # Extract experiment_id assuming format submission_<experiment_name>_<experiment_id>
        # but run_slurm doesn't strictly need to parse the ID out if check_missing.py handles it.
        # Actually check_missing.py likely takes --experiment-id.
        # Let's try to extract it.
        # Typical format: submission_victim_mistral_7b_instruct_v03_exp_v_1_4_0223_131427
        # Experiment name: victim_mistral_7b_instruct_v03_exp_v_1_4
        # Suffix is _0223_131427 (the ID).
        # We can try replacing "submission_" + experiment_name + "_" with nothing.
        
        # However, experiment_name might be slightly different or partial.
        # Safer: just assume the suffix after the last few underscores is the ID if we need it.
        # But actually, check_missing.py uses --experiment-id mostly for wandb grouping.
        # Let's just extract the suffix after the experiment name if present.
        
        expected_start = f"submission_{experiment_name}_"
        if target_folder.name.startswith(expected_start):
             experiment_id = target_folder.name[len(expected_start):]
        else:
             # Fallback or loose match
             experiment_id = target_folder.name
             print(f"Warning: could not strictly parse experiment_id from {target_folder.name} using {experiment_name}")
    else:
        experiment_id = target_folder.name[len(prefix):]

    missing_ids: List[int] = []
    for i in range(total_examples):
        result_path = target_folder / f"example_{i}_result.json"
        if not is_finished(result_path):
            missing_ids.append(i)

    missing_pct = (100.0 * len(missing_ids) / max(1, total_examples))
    print(f"[post_fixes] Missing examples: {len(missing_ids)}/{total_examples} ({missing_pct:.2f}%).")
    print(f"[post_fixes] Missing example ids: {missing_ids}")

    if not missing_ids:
        print("[post_fixes] No missing examples found. Exiting.")
        return

    # Per-experiment slurm config (preferred). Fallback to legacy top-level cfg["slurm"] if present.
    experiment_config = select_experiment_config(cfg, experiment_key)
    slurm_config = experiment_config.get("slurm", {}) or cfg.get("slurm", {}) or {}
    process_odd_even = bool(slurm_config.get("process_odd_even", experiment_config.get("process_odd_even", False)))
    gpus_per_job = max(1, int(slurm_config.get("gpus_per_job", 1)))
    tasks_per_job = max(1, int(slurm_config.get("tasks_per_job", slurm_config.get("ntasks", gpus_per_job))))
    if tasks_per_job != gpus_per_job:
        print(
            f"[post_fixes] tasks_per_job ({tasks_per_job}) != gpus_per_job ({gpus_per_job}); "
            f"using gpus_per_job for tasks."
        )
        tasks_per_job = gpus_per_job
    kill_on_bad_exit = int(slurm_config.get("kill_on_bad_exit", experiment_config.get("kill_on_bad_exit", 0)))
    per_task_logs = bool(slurm_config.get("per_task_logs", True))
    srun_label = bool(slurm_config.get("srun_label", True))
    srun_mpi = str(slurm_config.get("srun_mpi", slurm_config.get("mpi", "pmi2"))).strip().lower()

    n_jobs_fix = max(1, int(n_jobs_fix))

    # SLURM settings
    global_settings = cfg.get("global", {})
    use_h200 = slurm_config.get("h200", False)
    slurm_kwargs = dict(
        job_name=f"gcd_diff_{experiment_name}_post_fixes",
        nodes=slurm_config.get("nodes", 1),
        ntasks=tasks_per_job,
        cpus_per_task=slurm_config.get("cpus_per_task", 8),
        mem=slurm_config.get("mem", "128G"),
        time=slurm_config.get("time", "24:00:00"),
        gres=f"gpu:{gpus_per_job}",
        output=str(target_slurm_log_dir / f"gcd_diff_{experiment_name}_postfix_%j.out"),
        error=str(target_slurm_log_dir / f"gcd_diff_{experiment_name}_postfix_%j.out"),
        mail_type=slurm_config.get("mail_type", "END,FAIL"),
        mail_user=slurm_config.get("mail_user", "bt4811@princeton.edu"),
    )

    partition = slurm_config.get("partition")
    constraint = slurm_config.get("constraint")
    if use_h200:
        if partition:
            slurm_kwargs["partition"] = partition
            if constraint and str(constraint).strip().lower() not in {"gpu80"}:
                slurm_kwargs["constraint"] = constraint
        else:
            if constraint:
                slurm_kwargs["constraint"] = constraint
            elif slurm_config.get("default_gpu80_constraint", True):
                slurm_kwargs["constraint"] = "h200"
    else:
        if partition:
            slurm_kwargs["partition"] = partition
        if constraint:
            slurm_kwargs["constraint"] = constraint
        elif not partition and slurm_config.get("default_gpu80_constraint", True):
            slurm_kwargs["constraint"] = "gpu80"

    env = global_settings.get("environment", {})
    modules = env.get("modules", [])
    conda_env = env.get("conda_env", "gcg-diffusion")
    hf_cache = env.get("huggingface", {}).get("cache_dir", "/scratch/gpfs/KOROLOVA/huggingface")
    wandb_offline = env.get("wandb", {}).get("offline_mode", False)
    scratch_dir = global_settings.get("scratch_dir", None) or "/scratch/gpfs/KOROLOVA/bt4811"
    wandb_base = Path(scratch_dir) / "wandb"
    wandb_run_dir = wandb_base / "runs"
    wandb_cache_dir = wandb_base / "cache"
    wandb_config_dir = wandb_base / "config"
    wandb_data_dir = wandb_base / "data"
    wandb_artifacts_dir = wandb_base / "artifacts"
    wandb_tmp_dir = wandb_base / "tmp"

    total_shards = max(1, n_jobs_fix * tasks_per_job)

    for job_idx in range(1, n_jobs_fix + 1):
        base_shard = (job_idx - 1) * tasks_per_job
        if base_shard >= len(missing_ids):
            print(
                f"[post_fixes] Skipping job {job_idx}/{n_jobs_fix}: "
                f"no missing examples in shard range starting at {base_shard}."
            )
            continue
        slurm = Slurm(**slurm_kwargs)

        env_setup = f"""
module purge
"""
        for module in modules:
            env_setup += f"module load {module}\n"

        env_setup += f"""
source ~/.bashrc
conda activate {conda_env}
export WANDB_DIR="{wandb_run_dir}"
export WANDB_CACHE_DIR="{wandb_cache_dir}"
export WANDB_CONFIG_DIR="{wandb_config_dir}"
export WANDB_DATA_DIR="{wandb_data_dir}"
export WANDB_ARTIFACT_DIR="{wandb_artifacts_dir}"
export TMPDIR="{wandb_tmp_dir}"
export PYTHONUNBUFFERED="1"
export HF_HOME="{hf_cache}"
export HUGGINGFACE_HUB_CACHE="{hf_cache}"
export HF_DATASETS_CACHE="{hf_cache}/datasets"
export RESUME_FROM_INTERMEDIATE="1"
"""

        if process_odd_even:
            env_setup += """
wait_for_minute_parity() {
  target="$1"
  label="$2"
  while true; do
    m=$(date +%M)
    mod=$((10#$m % 4))
    if [ "$mod" -eq "$target" ]; then
      echo "[OddEven] ${label}: minute=${m} mod4 OK"
      break
    fi
    echo "[OddEven] ${label}: waiting for mod4 ${target} (now=${m})"
    s=$(date +%S)
    sleep $((60 - 10#$s))
  done
}
"""

        if wandb_offline:
            env_setup += 'export WANDB_MODE="offline"\n'
        if env.get("huggingface", {}).get("offline_mode", False):
            env_setup += """
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
"""

        env_setup += f"""
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" "$TMPDIR"

cd "{PROJECT_DIR}"
"""

        srun_out = ""
        if per_task_logs:
            srun_out = (
                f' --output="{target_slurm_log_dir / f"gcd_diff_{experiment_name}_postfix_%j_task%t.out"}"'
                f' --error="{target_slurm_log_dir / f"gcd_diff_{experiment_name}_postfix_%j_task%t.out"}"'
            )
        srun_lbl = " --label" if srun_label else ""
        pre_wait = "wait_for_minute_parity 0 start\n" if process_odd_even else ""
        post_wait = "\nwait_for_minute_parity 2 exit" if process_odd_even else ""

        barrier_base = target_folder / "_srun_barrier"
        command = f"""
{pre_wait}
srun --unbuffered{srun_lbl}{srun_out} --mpi={srun_mpi} --kill-on-bad-exit={kill_on_bad_exit} -n {tasks_per_job} --ntasks-per-node {tasks_per_job} bash -lc '
export CUDA_VISIBLE_DEVICES="$SLURM_LOCALID"
BARRIER_DIR="{barrier_base}/$SLURM_JOB_ID"
N_TASKS={tasks_per_job}
mkdir -p "$BARRIER_DIR"
trap '"'"'touch "$BARRIER_DIR/task_$SLURM_PROCID.done" 2>/dev/null; sync'"'"' EXIT
python -u {PROJECT_DIR / 'scripts' / 'check_missing.py'} \\
  --config {config_path} \\
  --examples {examples_path} \\
  --experiment-name {experiment_name} \\
  --experiment {experiment_key} \\
  --experiment-id {experiment_id} \\
  --results-dir {target_folder} \\
  --check-pass 0 \\
  --resume \\
  --num-shards {total_shards} \\
  --shard-id $(({base_shard} + $SLURM_PROCID)) \\
  --gpu-id 0 \\
  {"--no-wandb" if no_wandb else ""}
ec=$?
touch "$BARRIER_DIR/task_$SLURM_PROCID.done" && sync
echo "[Barrier] task $SLURM_PROCID work done, waiting for all $N_TASKS tasks..."
while [ $(ls "$BARRIER_DIR"/task_*.done 2>/dev/null | wc -l) -lt $N_TASKS ]; do sleep 5; done
echo "[Barrier] all tasks done, exiting with $ec"
exit $ec
'
{post_wait}
"""
        full_command = env_setup + command
        job_id = slurm.sbatch(full_command)
        print(
            f"[post_fixes] Submitted fix job {job_id} (job {job_idx}/{n_jobs_fix}, "
            f"tasks_per_job={tasks_per_job}, gpus_per_job={gpus_per_job})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Submit GCD experiments via SLURM"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to experiment config YAML (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=PROJECT_DIR / "data" / "examples.json",
        help="Path to examples JSON file (default: data/examples.json)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Experiment name (used for results directory and job naming)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment config key from configs/config.yaml experiments (e.g., exp_1). Defaults to configs/config.yaml 'experiment'."
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )
    parser.add_argument(
        "--target-folder",
        type=Path,
        default=None,
        help="Target results folder for post-fix runs (e.g., experiments/exp_name_<id>)"
    )
    parser.add_argument(
        "--post-fixes",
        action="store_true",
        help="Run in post-fix mode: detect missing/unfinished examples and re-run them."
    )
    parser.add_argument(
        "--n-jobs-fix",
        type=int,
        default=1,
        help="Number of SLURM jobs to use for post-fix runs (default: 1)."
    )
    
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        return 1
    
    if not args.examples.exists():
        print(f"Error: Examples file not found: {args.examples}")
        return 1
    
    cfg = load_config(args.config)
    experiment_key = args.experiment if args.experiment else cfg.get("experiment", "default")

    if args.post_fixes:
        if args.target_folder is None:
            raise ValueError("--post-fixes requires --target-folder to be set.")
        submit_post_fix_jobs(
            config_path=args.config,
            examples_path=args.examples,
            experiment_name=args.experiment_name,
            experiment_key=experiment_key,
            target_folder=args.target_folder,
            n_jobs_fix=args.n_jobs_fix,
            no_wandb=args.no_wandb,
        )
    else:
        submit_jobs(
            config_path=args.config,
            examples_path=args.examples,
            experiment_name=args.experiment_name,
            experiment_key=experiment_key,
            no_wandb=args.no_wandb
        )
    
    return 0


if __name__ == "__main__":
    exit(main())

