#!/usr/bin/env python3
"""
Main experiment runner for GCD attacks.
Handles model loading, attack execution, result saving, and wandb logging.
"""

import argparse
import copy
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

_config_path = None
for i, arg in enumerate(sys.argv):
    if arg == "--config" and i + 1 < len(sys.argv):
        _config_path = sys.argv[i + 1]
        break

from scripts.config_utils import (  # noqa: E402
    deep_merge,
    resolve_experiment_config,
    setup_hf_environment,
)

HF_CACHE_DIR, OFFLINE_MODE, HF_CACHE_DIRS_LIST = setup_hf_environment(_config_path)

for _pkg in ("tensorflow", "keras"):
    try:
        __import__(_pkg)
    except Exception:
        sys.modules[_pkg] = None
del _pkg

import torch  # noqa: E402
import wandb  # noqa: E402

from scripts import model_loader
from scripts.attack_pipeline import run_single_attack
from scripts.exp_utils import _logged_initial_query_for_result
from scripts.model_loader import _same_pretrained_checkpoint, load_defence_model, load_models, load_phase2_judge_model

model_loader.HF_CACHE_DIR = HF_CACHE_DIR
model_loader.HF_CACHE_DIRS = HF_CACHE_DIRS_LIST

def main():
    parser = argparse.ArgumentParser(description="Run GCD attack experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--examples", type=str, required=True, help="Path to examples JSON file")
    parser.add_argument("--experiment-name", type=str, required=True, help="Experiment name")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment config to use (overrides config file, e.g., 'exp_1' or 'default')")
    parser.add_argument("--experiment-id", type=str, default=None, help="Experiment ID (default: SLURM_JOB_ID or timestamp)")
    parser.add_argument("--start-example", type=int, default=0, help="Start example index")
    parser.add_argument("--num-examples", type=int, default=None, help="Number of examples to process")
    parser.add_argument("--shard-id", type=int, default=None, help="Shard id for distributed example processing (0..num_shards-1).")
    parser.add_argument("--num-shards", type=int, default=None, help="Total number of shards for distributed example processing.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID for this job")
    parser.add_argument(
        "--mode-split",
        type=str,
        default=None,
        choices=["model", "examples"],
        help="Override slurm mode_split from config (model/examples/inside_job_coherent)",
    )
    parser.add_argument("--wandb-group", type=str, default=None, help="Optional W&B group for multi-shard runs.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override experiment config value after defaults+experiment merge. "
            "Supports dotted keys for nested dicts, e.g. "
            "--override candidate_batch_pct=0.25 "
            "--override self_perplexity_coef='[0.1, 0.1]' "
            "--override slurm.mode_split='\"examples\"'"
        ),
    )
    
    args = parser.parse_args()
    # NOTE: We intentionally allow wandb for all shards (one run per shard) and group them via --wandb-group.
    # Multiple processes writing to the same W&B run concurrently is unsafe; grouping gives a clean UX.
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load examples
    with open(args.examples, 'r') as f:
        examples = json.load(f)
    
    # Determine experiment ID
    experiment_id = args.experiment_id or os.environ.get("SLURM_JOB_ID") or f"local_{int(torch.randint(0, 1000000, (1,)).item())}"
    
    # Create results directory (use absolute path)
    project_dir = Path(__file__).resolve().parent.parent
    results_dir = project_dir / "experiments" / f"{args.experiment_name}_{experiment_id}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which examples to process
    start_idx = args.start_example
    end_idx = start_idx + args.num_examples if args.num_examples else len(examples)
    base_examples = examples[start_idx:end_idx]
    shard_global_start = 0

    # Optional sharding over the selected range.
    # Shards are contiguous for simplicity and reproducibility.
    #
    # IMPORTANT: Use a remainder-based split instead of a fixed ceil() shard_size.
    # A fixed ceil() can produce empty trailing shards even when num_shards <= num_examples
    # (e.g., 20 examples, 12 shards -> shard_size=2 -> shards 10-11 empty). This can
    # cause SLURM `srun` steps to be cancelled if some tasks exit early while others run.
    if args.shard_id is not None or args.num_shards is not None:
        if args.shard_id is None or args.num_shards is None:
            raise ValueError("If using sharding, both --shard-id and --num-shards must be provided.")
        if args.num_shards <= 0:
            raise ValueError("--num-shards must be > 0")
        if args.shard_id < 0 or args.shard_id >= args.num_shards:
            raise ValueError("--shard-id must be in [0, num_shards)")

        n = len(base_examples)
        s = int(args.num_shards)
        # Balanced contiguous partitioning:
        # - first `rem = n % s` shards get (base + 1) examples
        # - remaining shards get `base = n // s` examples
        base = n // s
        rem = n % s

        if args.shard_id < rem:
            shard_start = args.shard_id * (base + 1)
            shard_end = shard_start + (base + 1)
        else:
            shard_start = rem * (base + 1) + (args.shard_id - rem) * base
            shard_end = shard_start + base

        shard_start = min(max(0, int(shard_start)), n)
        shard_end = min(max(int(shard_end), shard_start), n)
        examples_to_process = base_examples[shard_start:shard_end]
        shard_global_start = shard_start
        print(
            f"Sharding enabled: shard {args.shard_id}/{args.num_shards} "
            f"-> local indices [{shard_start}, {shard_end}) "
            f"out of {len(base_examples)} selected examples."
        )
    else:
        examples_to_process = base_examples
    
    print(f"Processing examples {start_idx} to {end_idx-1} on GPU {args.gpu_id}")
    print(f"Results will be saved to: {results_dir}")
    
    # Get experiment configuration (merge defaults with experiment-specific config)
    # Command-line argument overrides config file
    experiment_name = args.experiment if args.experiment else config.get("experiment", "default")
    defaults = config.get("defaults", {})
    experiments = config.get("experiments", {})
    
    # Get experiment-specific config or use defaults
    if experiment_name in experiments:
        # Resolve `inherit:` / `base:` so child experiments actually receive parent keys
        # (shallow {**defaults, **exp} alone ignores inheritance).
        exp_config = resolve_experiment_config(config, experiment_name)
        if isinstance(exp_config, dict):
            exp_config = {k: v for k, v in exp_config.items() if k not in ("inherit", "base")}
        experiment_config = {**defaults, **exp_config}
    else:
        if experiment_name != "default":
            available = sorted(list(experiments.keys())) if isinstance(experiments, dict) else []
            preview = ", ".join(available[:25])
            more = "" if len(available) <= 25 else f", ... (+{len(available) - 25} more)"
            print(
                f"Warning: Experiment '{experiment_name}' not found in config. Using defaults.\n"
                f"Available experiments: [{preview}{more}]\n"
                f"Tip: either pass `--experiment default` or add '{experiment_name}:' under the YAML 'experiments:' section."
            )
        experiment_config = defaults.copy()
    
    print(f"Using experiment configuration: {experiment_name}")

    # Lightweight, high-signal config summary (helps debug SLURM logs / OOMs)
    try:
        resolved_eval_batch_size = experiment_config.get("eval_batch_size", 256)
        resolved_top_k_start = experiment_config.get("top_k_start", 64)
        resolved_top_k_end = experiment_config.get("top_k_end", 8)
        resolved_candidate_batch_pct = experiment_config.get("candidate_batch_pct", 1.0)
        resolved_to_text_before_eval = experiment_config.get("to_text_before_eval", True)
        resolved_no_gradient = experiment_config.get("no_gradient", True)
        print(
            "Resolved attack config (key knobs): "
            f"eval_batch_size={resolved_eval_batch_size}, "
            f"top_k_start={resolved_top_k_start}, top_k_end={resolved_top_k_end}, "
            f"candidate_batch_pct={resolved_candidate_batch_pct}, "
            f"to_text_before_eval={resolved_to_text_before_eval}, "
            f"no_gradient={resolved_no_gradient}"
        )
    except Exception:
        # Never fail the run due to logging
        pass

    # Apply explicit config overrides from CLI (after defaults+experiment merge).
    if args.override:
        def _set_nested(cfg: dict, dotted_key: str, value):
            parts = [p for p in str(dotted_key).split(".") if p]
            if not parts:
                raise ValueError(f"Invalid override key: {dotted_key!r}")
            cur = cfg
            for p in parts[:-1]:
                if p not in cur or not isinstance(cur[p], dict):
                    cur[p] = {}
                cur = cur[p]
            cur[parts[-1]] = value

        for item in args.override:
            if "=" not in item:
                raise ValueError(
                    f"Invalid --override '{item}'. Expected KEY=VALUE format."
                )
            key, raw_value = item.split("=", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            if not key:
                raise ValueError(f"Invalid --override '{item}'. Empty key.")
            try:
                parsed_value = yaml.safe_load(raw_value)
            except Exception:
                parsed_value = raw_value
            _set_nested(experiment_config, key, parsed_value)

        try:
            printable_overrides = {s.split("=", 1)[0].strip(): yaml.safe_load(s.split("=", 1)[1].strip()) for s in args.override if "=" in s}
        except Exception:
            printable_overrides = args.override
        print(f"Applied CLI overrides: {printable_overrides}")

    # Determine slurm mode_split (can be overridden by CLI)
    slurm_cfg = experiment_config.get("slurm", {}) if isinstance(experiment_config.get("slurm", {}), dict) else {}
    mode_split = args.mode_split if args.mode_split else slurm_cfg.get("mode_split", "examples")
    mode_split = str(mode_split).strip().lower()

    is_main_process = True

    # Resolve model config: allow per-experiment override of top-level models.*
    root_models_cfg = config.get("models", {}) if isinstance(config.get("models", {}), dict) else {}
    exp_models_cfg = experiment_config.get("models", {}) if isinstance(experiment_config.get("models", {}), dict) else {}
    model_config = {**root_models_cfg, **exp_models_cfg}
    try:
        print(
            "Resolved model config: "
            f"dream_model={model_config.get('dream_model')}, "
            f"llada_model={model_config.get('llada_model')}, "
            f"victim_model={model_config.get('victim_model')}, "
            f"bench_verifier_model={model_config.get('bench_verifier_model')}, "
            f"use_quantization={model_config.get('use_quantization', True)}"
        )
    except Exception:
        pass
    
    # Initialize wandb
    wandb_run = None
    if not args.no_wandb and config.get("wandb", {}).get("enabled", False) and is_main_process:
        # Allow per-experiment override of wandb.project via experiment_config.
        # This keeps backwards compatibility with the global top-level config["wandb"]["project"].
        wandb_project = experiment_config.get("wandb_project", None)
        if not wandb_project:
            wandb_project = config["wandb"].get("project", "gcg-diffusion")

        # Name/grouping: if sharded, create one run per shard but share a group for easy viewing.
        if args.shard_id is not None and args.num_shards is not None:
            wandb_group = args.wandb_group or f"{args.experiment_name}_{experiment_id}"
            wandb_name = f"{args.experiment_name}_{experiment_id}_shard{args.shard_id}"
        else:
            wandb_group = args.wandb_group
            wandb_name = f"{args.experiment_name}_{experiment_id}_gpu{args.gpu_id}"

        wandb.init(
            project=wandb_project,
            name=wandb_name,
            group=wandb_group,
            config={
                "experiment_name": args.experiment_name,
                "experiment_id": experiment_id,
                "gpu_id": args.gpu_id,
                "start_example": start_idx,
                "num_examples": len(examples_to_process),
                "config_name": experiment_name,
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "dream_model": model_config.get("dream_model"),
                "victim_model": model_config.get("victim_model"),
                "use_quantization": model_config.get("use_quantization"),
                **experiment_config
            },
            entity=config["wandb"].get("entity", None),
            mode=config["wandb"].get("mode", "online")
        )
        wandb_run = wandb.run
    
    # Get offline mode from config
    global_config = config.get("global", {})
    hf_config = global_config.get("environment", {}).get("huggingface", {})
    offline_mode = hf_config.get("offline_mode", False)
    
    # Load models (from merged model_config)
    use_llada = bool(experiment_config.get("use_llada", False))
    # Optimization-only mode
    no_diffusion = bool(experiment_config.get("no_diffusion", False))
    adapt_tokenizers = bool(experiment_config.get("adapt_tokenizers", False))
    skip_backend_model = bool(no_diffusion) and (not bool(adapt_tokenizers))
    
    victim_model_f16 = bool(experiment_config.get("victim_model_f16", False))
    quantize_diffusion = bool(experiment_config.get("quantize_diffusion", False))
    
    # Get GGUF file paths for Dream/LLaDA models (if specified)
    dream_gguf_file = model_config.get("dream_gguf_file", None)
    llada_gguf_file = model_config.get("llada_gguf_file", None)
    # Get tokenizer paths (if specified separately for GGUF models)
    dream_tokenizer_path = model_config.get("dream_tokenizer_path", None)
    llada_tokenizer_path = model_config.get("llada_tokenizer_path", None)

    dream_model, victim_model, tokenizer, victim_tokenizer, mask_token_id = load_models(
        dream_model_path=model_config.get("dream_model", "Dream-org/Dream-v0-Instruct-7B"),
        llada_model_path=model_config.get("llada_model", "GSAI-ML/LLaDA-8B-Base"),
        victim_model_path=model_config.get("victim_model", "Qwen/Qwen2.5-7B-Instruct"),
        use_quantization=model_config.get("use_quantization", True),
        quant_judge=bool(model_config.get("quant_judge", False)),
        quantize_diffusion=quantize_diffusion,
        dream_gguf_file=dream_gguf_file,
        llada_gguf_file=llada_gguf_file,
        dream_tokenizer_path=dream_tokenizer_path,
        llada_tokenizer_path=llada_tokenizer_path,
        gpu_id=args.gpu_id,
        offline_mode=offline_mode,
        use_llada=use_llada,
        skip_backend_model=skip_backend_model,
        victim_model_f16=victim_model_f16,
    )

    defence_model = None
    defence_tokenizer = None
    defence_evasion = experiment_config.get("defence_evasion", None)
    defence_model_name = experiment_config.get("defence_model_name", None)
    if defence_evasion:
        if not defence_model_name:
            raise ValueError(
                f"defence_evasion={defence_evasion!r} requires defence_model_name in experiment config."
            )
        _defence_quant_cfg = experiment_config.get("defence_use_quantization", None)
        if _defence_quant_cfg is None:
            _defence_quant_cfg = model_config.get("use_quantization", True)
        defence_model, defence_tokenizer = load_defence_model(
            str(defence_model_name),
            use_quantization=bool(_defence_quant_cfg),
            gpu_id=args.gpu_id,
            offline_mode=offline_mode,
        )
        print(
            f"Defence guidance enabled: {defence_evasion!r}, "
            f"alpha_def={float(experiment_config.get('alpha_def', 0.0))}, "
            f"model={defence_model_name}"
        )
    
    # Add mask_token_id to experiment config
    experiment_config["mask_token_id"] = mask_token_id

    phase2_judge_model = None
    phase2_judge_tokenizer = None
    if bool(experiment_config.get("prefix_continuation_phase2_llm_judge", False)):
        _p2_judge_path = (
            experiment_config.get("prefix_continuation_phase2_judge_model")
            or model_config.get("phase2_judge_model")
            or model_config.get("judge_model")
            or "Qwen/Qwen2.5-7B-Instruct"
        )
        _victim_path = model_config.get("victim_model", "Qwen/Qwen2.5-7B-Instruct")
        if _same_pretrained_checkpoint(_p2_judge_path, _victim_path, offline_mode=offline_mode):
            phase2_judge_model = victim_model
            phase2_judge_tokenizer = victim_tokenizer
            print(
                f"[Phase-2 LLM judge] Reusing victim model/tokenizer ({_p2_judge_path}) "
                "for short-continuation rubric scoring."
            )
        else:
            _judge_quant_cfg = experiment_config.get(
                "prefix_continuation_phase2_judge_use_quantization", None
            )
            if _judge_quant_cfg is None:
                _judge_quant_cfg = model_config.get("phase2_judge_use_quantization")
            phase2_judge_model, phase2_judge_tokenizer = load_phase2_judge_model(
                str(_p2_judge_path),
                use_quantization=_judge_quant_cfg,
                gpu_id=args.gpu_id,
                offline_mode=offline_mode,
            )

    try:
        max_token_response = int(experiment_config.get("max_token_response", 0) or 0)
    except Exception:
        max_token_response = 0
    if max_token_response > 0:
        if victim_tokenizer is None:
            print(
                f"[max_token_response={max_token_response}] Skipped truncating output_1/output_2: "
                "victim_tokenizer is None."
            )
        else:
            _proj = Path(__file__).resolve().parent.parent
            _judge_tmpl = load_judge_user_template(
                experiment_config.get("judge_user_prompt_template_path"),
                _proj,
            )
            _truncated_examples = 0
            for _ex in examples_to_process:
                if truncate_record_outputs_rebuild_judge_prompts(
                    _ex, victim_tokenizer, max_token_response, _judge_tmpl
                ):
                    _truncated_examples += 1
            print(
                f"[max_token_response={max_token_response}] "
                f"Truncated output_1/output_2 and rebuilt judge prompts for "
                f"{_truncated_examples}/{len(examples_to_process)} examples "
                f"(victim/judge tokenizer)."
            )

    # Run experiments
    # (no tqdm: tqdm carriage returns can hide/overwrite print output in SLURM logs)
    all_results = []
    total_local = len(examples_to_process)
    joint_multi_target = bool(experiment_config.get("multi_target_joint", False))
    try:
        joint_multi_target_n = max(1, int(experiment_config.get("multi_target_joint_n", 1)))
    except Exception:
        joint_multi_target_n = 1
    joint_drop_last = bool(experiment_config.get("multi_target_joint_drop_last", False))
    work_items: List[Dict[str, Any]] = []
    if joint_multi_target and joint_multi_target_n > 1:
        i = 0
        while i < total_local:
            grp = examples_to_process[i : i + joint_multi_target_n]
            if len(grp) < joint_multi_target_n and joint_drop_last:
                break
            anchor_gid = start_idx + shard_global_start + i
            joint_targets = []
            for j, ex in enumerate(grp):
                gid = start_idx + shard_global_start + i + j
                joint_targets.append({
                    "label": f"trg_{j+1}",
                    "global_example_id": int(gid),
                    "initial_query": ex.get("initial_query", ""),
                    "goal": ex.get("goal", ex.get("forbidden_prompt", ex.get("initial_query", ""))),
                    "target_behavior": ex.get("target_behavior", ""),
                    "target_behavior_before_llm_suffix": ex.get("target_behavior_before_llm_suffix", None),
                    "judge_competitor_initial_query": str(
                        ex.get("judge_competitor_initial_query", "") or ""
                    ),
                })
            work_items.append({
                "example_id": int(anchor_gid),
                "anchor_example": grp[0],
                "multi_targets": joint_targets,
            })
            i += joint_multi_target_n
    else:
        for idx, example in enumerate(examples_to_process):
            example_id = start_idx + shard_global_start + idx
            work_items.append({
                "example_id": int(example_id),
                "anchor_example": example,
                "multi_targets": None,
            })
    def _fmt_seconds(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    total_work = len(work_items)
    exp_wall_start = time.time()
    per_example_times: List[float] = []

    for idx, work in enumerate(work_items):
        example_id = int(work["example_id"])
        example = work["anchor_example"]
        _multi_targets = work.get("multi_targets", None)
        try:
            if _multi_targets:
                print(
                    f"\n[Progress] Starting group {idx + 1}/{total_work} "
                    f"(global example_id={example_id}, n_targets={len(_multi_targets)})"
                )
            else:
                print(f"\n[Progress] Starting example {idx + 1}/{total_work} (global example_id={example_id})")
            ex_wall_start = time.time()
            # Inject per-example judge competitor prompt into a shallow config copy
            # so _run_single_attack_core can read it without mutating the shared config.
            _ex_experiment_config = experiment_config
            _judge_comp_iq = str(example.get("judge_competitor_initial_query", "") or "")
            _usfx = str(example.get("user_suffix_after_tunable", "") or "").strip()
            if _usfx:
                if _ex_experiment_config is experiment_config:
                    _ex_experiment_config = dict(experiment_config)
                _ex_experiment_config["fixed_user_suffix_after_tunable"] = example["user_suffix_after_tunable"]
            _iqpfx = str(example.get("initial_query_prefix", "") or "").strip()
            if _iqpfx:
                if _ex_experiment_config is experiment_config:
                    _ex_experiment_config = dict(experiment_config)
                _ex_experiment_config["_tokenizer_fixed_prefix_text"] = _iqpfx
            _q_adapt = str(example.get("question", "") or "").strip()
            if _q_adapt:
                if _ex_experiment_config is experiment_config:
                    _ex_experiment_config = dict(experiment_config)
                _ex_experiment_config["_result_original_query"] = _q_adapt
                if bool(experiment_config.get("adapt_tune_tokens", False)):
                    _ex_experiment_config["_adapt_tune_tokens_source_text"] = _q_adapt
            result = run_single_attack(
                dream_model=dream_model,
                victim_model=victim_model,
                tokenizer=tokenizer,
                victim_tokenizer=victim_tokenizer,
                initial_query=example["initial_query"],
                target_behavior=example["target_behavior"],
                extended_target_behavior=example.get("extended_target_behavior", example.get("extended_target", None)),
                goal=example.get("goal", example.get("forbidden_prompt", example.get("initial_query", ""))),
                conts=example.get("conts", None),
                experiment_config=_ex_experiment_config,
                example_id=example_id,
                results_dir=results_dir,
                wandb_run=wandb_run,
                offline_mode=offline_mode,
                target_behavior_before_llm_suffix=example.get("target_behavior_before_llm_suffix", None),
                phase2_judge_model=phase2_judge_model,
                phase2_judge_tokenizer=phase2_judge_tokenizer,
                defence_model=defence_model,
                defence_tokenizer=defence_tokenizer,
            )
            ex_wall_sec = time.time() - ex_wall_start
            per_example_times.append(ex_wall_sec)
            result["attack_wall_time_sec"] = ex_wall_sec
            if _multi_targets:
                result["joint_multi_target"] = True
                result["joint_multi_target_n"] = len(_multi_targets)
                result["joint_multi_target_example_ids"] = [int(x.get("global_example_id", -1)) for x in _multi_targets]
            all_results.append(result)

            # Explicitly clear memory after each example to prevent leakage
            try:
                # Clear cache if runner is available (it's not here, but we can call it statically)
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

            # Print running ETA
            done = idx + 1
            avg_sec = sum(per_example_times) / max(1, len(per_example_times))
            remaining = total_work - done
            eta_sec = remaining * avg_sec
            total_elapsed = time.time() - exp_wall_start
            print(
                f"[Progress] Finished item {done}/{total_work} (global example_id={example_id}) "
                f"in {_fmt_seconds(ex_wall_sec)} | avg {_fmt_seconds(avg_sec)}/example | "
                f"elapsed {_fmt_seconds(total_elapsed)} | ETA {_fmt_seconds(eta_sec)}"
            )
            
            # Save intermediate result (atomic replace so partial files aren't left on dump errors)
            result_file = results_dir / f"example_{example_id}_result.json"
            _rf_tmp = result_file.with_suffix(result_file.suffix + ".tmp")
            with open(_rf_tmp, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            os.replace(_rf_tmp, result_file)

            # Log per-example timing and ETA to wandb
            if wandb_run:
                try:
                    wandb_run.log({
                        f"example_{example_id}/attack_wall_time_sec": float(ex_wall_sec),
                        "progress/examples_done": int(done),
                        "progress/examples_total": int(total_work),
                        "progress/avg_attack_wall_time_sec": float(avg_sec),
                        "progress/eta_sec": float(eta_sec),
                    })
                except Exception as e:
                    print(f"Warning: could not log timing/ETA to wandb: {e}")
            
            # Save JSON as artifact to WandB
            if wandb_run:
                try:
                    # Create artifact for this example's JSON result
                    # Use consistent naming: example_{example_id}_result
                    artifact = wandb.Artifact(
                        name=f"example_{example_id}_result",
                        type="experiment_result",
                        description=(
                            f"Result JSON for example {example_id} (initial_query: "
                            f"{str(result.get('initial_query', example['initial_query']) or '')[:50]}...)"
                        ),
                    )
                    artifact.add_file(str(result_file), name=f"example_{example_id}_result.json")
                    wandb_run.log_artifact(artifact)
                    print(f"✓ Saved example_{example_id}_result.json as WandB artifact")
                except Exception as artifact_error:
                    print(f"Warning: Could not save JSON as artifact to wandb: {artifact_error}")
                    traceback.print_exc()
                    print("JSON file is still saved locally.")
            
        except Exception as e:
            print(f"Error processing example {example_id}: {e}")
            traceback.print_exc()
            _err_goal = example.get("goal", example.get("forbidden_prompt", example.get("initial_query", "")))
            _err_cfg: Dict[str, Any] = {}
            _err_q = str(example.get("question", "") or "").strip()
            if _err_q:
                _err_cfg["_result_original_query"] = _err_q
            _err_iqpfx = str(example.get("initial_query_prefix", "") or "").strip()
            if _err_iqpfx:
                _err_cfg["_tokenizer_fixed_prefix_text"] = _err_iqpfx
            _err_usfx = str(example.get("user_suffix_after_tunable", "") or "").strip()
            if _err_usfx:
                _err_cfg["fixed_user_suffix_after_tunable"] = example["user_suffix_after_tunable"]
            all_results.append({
                "example_id": example_id,
                "error": str(e),
                "initial_query": _logged_initial_query_for_result(
                    _err_cfg,
                    _err_goal,
                    str(example.get("initial_query", "") or ""),
                ),
                "target_behavior": example.get("target_behavior", ""),
                "goal": example.get("goal", example.get("forbidden_prompt", example.get("initial_query", ""))),
            })

    # Save all results (avoid clobbering in multi-shard mode)
    if args.shard_id is not None and args.num_shards is not None:
        results_file = results_dir / f"all_results_shard{args.shard_id}.json"
    else:
        results_file = results_dir / "all_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Save all_results.json as artifact to WandB
    if wandb_run:
        try:
            artifact = wandb.Artifact(
                name=f"all_results_shard{args.shard_id}" if (args.shard_id is not None and args.num_shards is not None) else "all_results",
                type="experiment_results",
                description=f"Combined results for all examples (examples {start_idx} to {end_idx-1})"
            )
            artifact.add_file(str(results_file), name=results_file.name)
            wandb_run.log_artifact(artifact)
            print(f"✓ Saved {results_file.name} as WandB artifact")
        except Exception as artifact_error:
            print(f"Warning: Could not save all_results.json as artifact to wandb: {artifact_error}")
            print("JSON file is still saved locally.")

    # Save summary
    summary = {
        "experiment_name": args.experiment_name,
        "experiment_id": experiment_id,
        "gpu_id": args.gpu_id,
        "start_example": start_idx,
        "num_examples": len(examples_to_process),
        "num_work_items": len(all_results),
        "joint_multi_target": bool(experiment_config.get("multi_target_joint", False)),
        "joint_multi_target_n": int(experiment_config.get("multi_target_joint_n", 1)),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "successful": sum(1 for r in all_results if r.get("success", False)),
        "failed": sum(1 for r in all_results if not r.get("success", False)),
        "errors": sum(1 for r in all_results if "error" in r),
        "total": len(all_results)
    }

    if args.shard_id is not None and args.num_shards is not None:
        summary_file = results_dir / f"summary_shard{args.shard_id}.json"
    else:
        summary_file = results_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save summary.json as artifact to WandB
    if wandb_run:
        try:
            artifact = wandb.Artifact(
                name=f"summary_shard{args.shard_id}" if (args.shard_id is not None and args.num_shards is not None) else "summary",
                type="experiment_summary",
                description=f"Experiment summary for {args.experiment_name}"
            )
            artifact.add_file(str(summary_file), name=summary_file.name)
            wandb_run.log_artifact(artifact)
            print(f"✓ Saved {summary_file.name} as WandB artifact")
        except Exception as artifact_error:
            print(f"Warning: Could not save summary.json as artifact to wandb: {artifact_error}")
            print("JSON file is still saved locally.")

    print(f"\n{'='*60}")
    print(f"Experiment completed!")
    print(f"Results saved to: {results_dir}")
    print(f"Summary: {summary}")
    print(f"{'='*60}")

    # Final wandb log
    if wandb_run:
        wandb_run.log({
            "summary/successful": summary["successful"],
            "summary/failed": summary["failed"],
            "summary/errors": summary["errors"],
            "summary/total": summary["total"],
            "summary/success_rate": summary["successful"] / summary["total"] if summary["total"] > 0 else 0.0
        })
        wandb_run.finish()


if __name__ == "__main__":
    main()
