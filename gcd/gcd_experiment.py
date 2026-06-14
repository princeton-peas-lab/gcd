import gc
import torch
from typing import List, Tuple, Optional
from gcd.gcd_core import GCDAttack

def run_attack_experiment(
    victim_model,
    dream_model,
    tokenizer,
    initial_query,
    target_behavior,
    goal: Optional[str] = None,
    victim_tokenizer=None,
    no_diffusion: bool = False,
    no_gradient: bool = True,
    use_cache: bool = False,
    instruction_template="Write a question.",
    num_tunable_tokens=40,
    num_steps=512,
    p_multipl_token: int = 1,
    p=1.0,
    top_k_start=64,
    top_k_end=8,
    start_coeff=1.0,
    end_coeff=1.0,
    grad_coef=1.0,
    eval_batch_size=256,
    candidate_batch_pct=1.0,
    candidate_batch_pct_dec: float = 1.0,
    mask_p=0.0,
    only_ascii=True,
    never_repeat=True,
    substract_current=True,
    pre_compute_mask=True,
    top_k_total=False,
    fill_during_eval=True,
    dream_eval_steps=5,
    dream_alg: str = "origin",
    fill_max_tokens_per_step: Optional[int] = None,
    prob_sampling=False,
    prob_based_sampling: bool = False,
    sampling_temperature: float = 0.6,
    use_precomputed_score=True,
    combined_sim_select: bool = False,
    alpha_select: float = 0.5,
    soft_loss: bool = False,
    soft_loss_top_k: int = 256,
    soft_loss_gamma: float = 6.0,
    semi_soft_loss: bool = False,
    semi_soft_loss_k: int = 16,
    semi_soft_loss_tau: float = 1.0,
    semi_soft_loss_proj_dim: int = 256,
    semi_soft_loss_proj_seed: int = 0,
    soft_target_ce_loss: bool = False,
    soft_target_ce_k: int = 64,
    soft_target_ce_tau: float = 0.1,
    prune_merg_tokens: bool = False,
    prune_merg_tokens_inter: int = 32,
    prune_merg_tokens_pers: float = 0.25,
    free_after_change: int = 0,
    block_wise_filling: bool = False,
    block_size: int = 5,
    block_mean_compute_top_k: int = 32,
    uniform_block_sampling: bool = True,
    prob_based_block_selection: bool = False,
    steps_per_block: int = 5,
    free_block_after_change: int = 0,
    print_block_choice: bool = False,
    mask_token_id=151666,
    always_change=True,
    mask_exploration_boost=0.05,
    print_example=True,
    print_example_interval=1,
    breadth_k_search: Optional[str] = None,
    breadth_k_schedule: Optional[List[Tuple[int, int]]] = None,
    breadth_k_cand_coef: Optional[List[float]] = None,
    breadth_k_sync_after: Optional[List[int]] = None,
    remove_str_dublicate_opt: bool = True,
    remove_str_dublicate_opt_breadth: bool = True,
    append_tunable_suffix: bool = False,
    tunable_suffix_app: str = "",
    token_separator: Optional[str] = None
):
    """
    Run a single GCD attack experiment.
    
    Args:
        victim_model: Target LLM model
        dream_model: Dream model for guidance
        tokenizer: Tokenizer
        initial_query: Initial user query
        target_behavior: Target response to elicit
        instruction_template: Template for Dream model instruction
        num_tunable_tokens: Number of tokens to optimize
        num_steps: Number of optimization steps
        p: Schedule parameter for top_k decay
        top_k_start: Starting top_k value
        top_k_end: Ending top_k value
        start_coeff: Dream coefficient at start
        end_coeff: Dream coefficient at end
        grad_coef: Gradient coefficient
        eval_batch_size: Batch size for evaluation
        candidate_batch_pct: Percentage of candidates to evaluate
        mask_p: Mask probability for Dream scoring
        only_ascii: Only allow ASCII tokens
        never_repeat: Never repeat candidate sequences
        substract_current: Subtract current token scores in Dream
        pre_compute_mask: Pre-compute mask positions
        top_k_total: Use total top_k across all positions
        fill_during_eval: Fill masks during evaluation
        dream_eval_steps: Number of diffusion steps for evaluation
        prob_sampling: (legacy alias) Use probability sampling
        prob_based_sampling: If true, sample candidate flips proportional to their score (no repetition)
        sampling_temperature: Softmax temperature for prob_based_sampling (lower=more greedy)
        use_precomputed_score: Use precomputed scores for filling
        combined_sim_select: If true, select best candidate by normalize(CE) + alpha_select * normalize(-logprob)
        alpha_select: Weight for the -logprob term in combined_sim_select
        free_after_change: If >0, after a position changes it is frozen (cannot change) for this many steps
        block_wise_filling: If true, restrict changes to one block for steps_per_block steps
        block_size: Number of tunable positions per block
        block_mean_compute_top_k: For block selection, mean top-k token scores per position
        uniform_block_sampling: If true, choose block uniformly; else choose highest score block
        prob_based_block_selection: If uniform_block_sampling is false, sample blocks proportional to score (softmax)
        steps_per_block: How many optimization steps to perform before selecting a new block
        free_block_after_change: If >0, don't reselect a block until this many subsequent block selections
        print_block_choice: If true, log which block is being rewritten
        mask_token_id: Mask token ID
        always_change: Always change tokens each step
        mask_exploration_boost: Boost for mask exploration (not used in current implementation)
        print_example: Print example generation from victim model at each step
        print_example_interval: Print example every N steps (only used if print_example: true)
    
    Returns:
        Dictionary with suffix, response, and success status
    """
    gc.collect()
    torch.cuda.empty_cache()

    resolved_goal = goal if isinstance(goal, str) and goal else initial_query

    try:
        formatted_instruction = instruction_template.format(
            query=initial_query,
            target=target_behavior,
            goal=resolved_goal,
        )
    except:
        formatted_instruction = instruction_template

    print(f"\nRUNNING EXPERIMENT")
    print(f"Template: {formatted_instruction}")
    print(f"Fast Fill (Precomputed): {use_precomputed_score}")

    # In no_diffusion + no-adaptation mode, operate in victim-tokenizer id space.
    suffix_tokenizer = victim_tokenizer if (no_diffusion and (victim_tokenizer is not None)) else tokenizer
    fixed_ids = suffix_tokenizer(initial_query, return_tensors="pt", add_special_tokens=False).input_ids

    # Initialize suffix
    if no_diffusion and (victim_tokenizer is not None):
        try:
            x_ids = suffix_tokenizer("x", add_special_tokens=False)["input_ids"]
            init_id = int(x_ids[0]) if isinstance(x_ids, list) and len(x_ids) > 0 else (suffix_tokenizer.eos_token_id or 0)
        except Exception:
            init_id = int(suffix_tokenizer.eos_token_id) if suffix_tokenizer.eos_token_id is not None else 0
        tunable_ids = torch.full((1, int(num_tunable_tokens)), int(init_id), dtype=torch.long)
    else:
        # Initialize with masks
        mask_token = tokenizer.decode([mask_token_id]) if mask_token_id < tokenizer.vocab_size else "<|mask|>"
        initial_tunable_str = mask_token * num_tunable_tokens
        tunable_ids = tokenizer(initial_tunable_str, return_tensors="pt", add_special_tokens=False).input_ids

    # Basic forbidden tokens
    forbidden_ids = [tokenizer.convert_tokens_to_ids("<|mask|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    forbidden_ids = [fid for fid in forbidden_ids if fid is not None]

    runner = GCDAttack(
        target_llm=victim_model,
        dream_model=dream_model,
        tokenizer=tokenizer,
        victim_tokenizer=(victim_tokenizer if victim_tokenizer is not None else tokenizer),
        no_diffusion=no_diffusion,
        no_gradient=no_gradient,
        use_cache=use_cache,
        p_multipl_token=p_multipl_token,
        target_response=target_behavior,
        goal=resolved_goal,
        fixed_user_ids=fixed_ids,
        tunable_ids=tunable_ids,
        forbidden_suffix_tokens=forbidden_ids,
        num_steps=num_steps,
        p=p,
        start_coeff=start_coeff,
        end_coeff=end_coeff,
        top_k_gradients=top_k_start,
        top_k_gradients_end=top_k_end,
        eval_batch_size=eval_batch_size,
        candidate_batch_pct=candidate_batch_pct,
        candidate_batch_pct_dec=candidate_batch_pct_dec,
        grad_coef=grad_coef,
        never_repeat=never_repeat,
        pre_compute_mask=pre_compute_mask,
        only_ascii=only_ascii,
        instruction_text=formatted_instruction,
        substract_current=substract_current,
        mask_p=mask_p,
        top_k_total=top_k_total,
        fill_during_eval=fill_during_eval,
        dream_eval_steps=dream_eval_steps,
        dream_alg=dream_alg,
        fill_max_tokens_per_step=fill_max_tokens_per_step,
        prob_sampling=prob_sampling,
        prob_based_sampling=prob_based_sampling,
        sampling_temperature=sampling_temperature,
        use_precomputed_score=use_precomputed_score,
        combined_sim_select=combined_sim_select,
        alpha_select=alpha_select,
        soft_loss=soft_loss,
        soft_loss_top_k=soft_loss_top_k,
        soft_loss_gamma=soft_loss_gamma,
        semi_soft_loss=semi_soft_loss,
        semi_soft_loss_k=semi_soft_loss_k,
        semi_soft_loss_tau=semi_soft_loss_tau,
        semi_soft_loss_proj_dim=semi_soft_loss_proj_dim,
        semi_soft_loss_proj_seed=semi_soft_loss_proj_seed,
        soft_target_ce_loss=soft_target_ce_loss,
        soft_target_ce_k=soft_target_ce_k,
        soft_target_ce_tau=soft_target_ce_tau,
        prune_merg_tokens=prune_merg_tokens,
        prune_merg_tokens_inter=prune_merg_tokens_inter,
        prune_merg_tokens_pers=prune_merg_tokens_pers,
        free_after_change=free_after_change,
        block_wise_filling=block_wise_filling,
        block_size=block_size,
        block_mean_compute_top_k=block_mean_compute_top_k,
        uniform_block_sampling=uniform_block_sampling,
        prob_based_block_selection=prob_based_block_selection,
        steps_per_block=steps_per_block,
        free_block_after_change=free_block_after_change,
        print_block_choice=print_block_choice,
        mask_token_id=mask_token_id,
        always_change=always_change,
        mask_exploration_boost=mask_exploration_boost,
        print_example=print_example,
        print_example_interval=print_example_interval,
        breadth_k_search=breadth_k_search,
        breadth_k_schedule=breadth_k_schedule,
        breadth_k_cand_coef=breadth_k_cand_coef,
        breadth_k_sync_after=breadth_k_sync_after,
        remove_str_dublicate_opt=remove_str_dublicate_opt,
        remove_str_dublicate_opt_breadth=remove_str_dublicate_opt_breadth,
        append_tunable_suffix=append_tunable_suffix,
        tunable_suffix_app=tunable_suffix_app,
        token_separator=token_separator,
        negative_reward_refusal=bool(experiment_config.get("negative_reward_refusal", False)),
        refusal_coef=float(experiment_config.get("refusal_coef", 0.5)),
        refusals=experiment_config.get("refusals", None),
    )

    final_tunable_ids = runner.run()

    # Final verification against the victim model
    suffix_text = tokenizer.decode(final_tunable_ids[0], skip_special_tokens=False)
    if runner.no_gradient:
        vt = runner.victim_tokenizer
        # When prompt_format_diffusion is enabled, extract the actual prompt
        actual_suffix_text = suffix_text
        if getattr(runner, 'prompt_format_diffusion', False):
            actual_suffix_text = runner._extract_prompt_from_format_text(suffix_text)
        user_content = f"{runner._fixed_user_text}{actual_suffix_text}"
        if hasattr(vt, "apply_chat_template"):
            prompt_text = runner._apply_victim_chat_template(user_content)
        else:
            prompt_text = f"{runner._prefix_str}{runner._fixed_user_text}{actual_suffix_text}{runner._suffix_str}"

        full_prompt_ids = vt(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(victim_model.device)
        with torch.no_grad():
            gen_output = victim_model.generate(
                input_ids=full_prompt_ids,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=(vt.eos_token_id if vt.eos_token_id is not None else 0),
            )
        gen_text = vt.decode(gen_output[0][full_prompt_ids.shape[1]:], skip_special_tokens=True)
    else:
        full_prompt_ids = torch.cat([
            runner.system_ids,
            runner.fixed_user_ids,
            final_tunable_ids,
            runner.assist_ids
        ], dim=1).to(victim_model.device)

        with torch.no_grad():
            gen_output = victim_model.generate(
                input_ids=full_prompt_ids,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_text = tokenizer.decode(gen_output[0][full_prompt_ids.shape[1]:], skip_special_tokens=True)

    print(f"\nFinal Discovered Suffix:\n{suffix_text}")
    print(f"\nVictim Response:\n{gen_text}")

    return {"suffix": suffix_text, "response": gen_text}

