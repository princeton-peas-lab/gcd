#!/usr/bin/env python3
"""
Script to compute three types of perplexity metrics for sequences:
1. Usual perplexity (dropping masked tokens)
2. Full PPPL (iteratively mask each non-mask token and compute probability)
3. Simple/fast PPL (one forward pass, compute PPPL for non-mask tokens)

Usage:
    python compute_perplexity_metrics.py \
        --model_path "Dream-org/Dream-v0-Instruct-7B" \
        --input_file sequences.txt \
        --mask_token_id 151666 \
        --device cuda
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional
import json
from pathlib import Path


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    offline_mode: bool = False,
    mask_token_id: Optional[int] = None,
):
    """Load model and tokenizer."""
    print(f"Loading model and tokenizer from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=offline_mode,
    )
    
    # Determine mask token ID
    if mask_token_id is None:
        if hasattr(tokenizer, "mask_token_id") and tokenizer.mask_token_id is not None:
            mask_token_id = int(tokenizer.mask_token_id)
        else:
            # Try common mask token IDs
            try:
                mask_token_id = tokenizer.convert_tokens_to_ids("<|mask|>")
            except:
                mask_token_id = 151666  # Dream default
                print(f"Warning: Could not find mask token, using default: {mask_token_id}")
    
    # Load model
    try:
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=offline_mode,
        ).to(device)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=offline_mode,
        ).to(device)
    
    model.eval()
    print(f"Model loaded on {device}, mask_token_id={mask_token_id}")
    
    return model, tokenizer, mask_token_id


def usual_perplexity(
    input_ids: torch.Tensor,
    model: torch.nn.Module,
    mask_token_id: int,
    device: str = "cuda",
) -> float:
    """
    Compute usual perplexity, excluding mask tokens.
    PPL = exp(mean(-log p(x_i | x_<i))) for non-mask tokens
    """
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # [1, L, V]
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()  # [1, L-1, V]
        shift_labels = input_ids[:, 1:].contiguous()  # [1, L-1]
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits.to(torch.float32), dim=-1)  # [1, L-1, V]
        
        # Get log probabilities for actual tokens
        batch_size, seq_len = shift_labels.shape
        log_probs_selected = log_probs.view(-1, log_probs.size(-1))[
            torch.arange(batch_size * seq_len, device=shift_labels.device),
            shift_labels.view(-1)
        ].view(batch_size, seq_len)  # [1, L-1]
        
        # Exclude mask tokens
        non_mask_mask = (shift_labels != mask_token_id)  # [1, L-1]
        
        if non_mask_mask.any():
            # Compute negative log-likelihood (mean over non-mask tokens only)
            nll = -log_probs_selected[non_mask_mask].mean()
            ppl = torch.exp(nll).item()
        else:
            ppl = float('inf')
    
    return ppl


def full_pppl(
    input_ids: torch.Tensor,
    model: torch.nn.Module,
    mask_token_id: int,
    device: str = "cuda",
) -> float:
    """
    Compute full pseudo-perplexity by iteratively masking each non-mask token
    and computing its probability.
    PPPL = exp(mean(-log p(x_i | context with x_i=mask))) for non-mask tokens
    
    For each non-mask token at position i:
    1. Mask position i
    2. Get logits at position i (predicting the masked token)
    3. Compute log probability of the actual token
    """
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]
    
    # Find all non-mask token positions
    non_mask_positions = []
    for i in range(seq_len):
        if input_ids[0, i].item() != mask_token_id:
            non_mask_positions.append(i)
    
    if len(non_mask_positions) == 0:
        return float('inf')
    
    log_probs_list = []
    
    with torch.no_grad():
        for pos in non_mask_positions:
            # Create a copy with this position masked
            masked_input = input_ids.clone()
            masked_input[0, pos] = mask_token_id
            
            # Forward pass
            outputs = model(input_ids=masked_input)
            logits = outputs.logits  # [1, L, V]
            
            # Get log probability of the actual token at the masked position
            # Strategy: Try both approaches and use the one that makes sense
            actual_token = input_ids[0, pos].item()
            log_prob = None
            
            # Approach 1: For masked models, logits at position pos predict token at pos
            if pos < logits.shape[1]:
                log_probs = F.log_softmax(logits[0, pos, :].to(torch.float32), dim=-1)
                log_prob = log_probs[actual_token].item()
            
            # Approach 2: For causal models, logits at pos-1 predict token at pos
            # Use this if approach 1 didn't work or if pos > 0
            if log_prob is None and pos > 0 and (pos - 1) < logits.shape[1]:
                log_probs = F.log_softmax(logits[0, pos - 1, :].to(torch.float32), dim=-1)
                log_prob = log_probs[actual_token].item()
            
            if log_prob is not None:
                log_probs_list.append(log_prob)
    
    if len(log_probs_list) == 0:
        return float('inf')
    
    # Compute mean negative log-likelihood
    nll = -sum(log_probs_list) / len(log_probs_list)
    pppl = torch.exp(torch.tensor(nll)).item()
    
    return pppl


def simple_fast_pppl(
    input_ids: torch.Tensor,
    model: torch.nn.Module,
    mask_token_id: int,
    device: str = "cuda",
) -> float:
    """
    Compute simple/fast pseudo-perplexity with one forward pass.
    PPPL = exp(mean(-log p(x_i | x_<i))) for non-mask tokens
    This is similar to usual perplexity but computed on the sequence as-is.
    """
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # [1, L, V]
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()  # [1, L-1, V]
        shift_labels = input_ids[:, 1:].contiguous()  # [1, L-1]
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits.to(torch.float32), dim=-1)  # [1, L-1, V]
        
        # Get log probabilities for actual tokens
        batch_size, seq_len = shift_labels.shape
        log_probs_selected = log_probs.view(-1, log_probs.size(-1))[
            torch.arange(batch_size * seq_len, device=shift_labels.device),
            shift_labels.view(-1)
        ].view(batch_size, seq_len)  # [1, L-1]
        
        # Exclude mask tokens
        non_mask_mask = (shift_labels != mask_token_id)  # [1, L-1]
        
        if non_mask_mask.any():
            # Compute negative log-likelihood (mean over non-mask tokens only)
            nll = -log_probs_selected[non_mask_mask].mean()
            pppl = torch.exp(nll).item()
        else:
            pppl = float('inf')
    
    return pppl


def process_sequences(
    sequences: List[str],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    mask_token_id: int,
    device: str = "cuda",
) -> List[dict]:
    """
    Process a list of sequences and compute all three perplexity metrics.
    
    Args:
        sequences: List of sequences (can be text strings or token ID lists)
        model: The model to use
        tokenizer: The tokenizer to use
        mask_token_id: ID of the mask token
        device: Device to run on
    
    Returns:
        List of dictionaries with results for each sequence
    """
    results = []
    
    for idx, seq in enumerate(sequences):
        print(f"\nProcessing sequence {idx + 1}/{len(sequences)}...")
        
        # Tokenize sequence
        if isinstance(seq, str):
            # If it's a string, tokenize it
            input_ids = tokenizer(seq, return_tensors="pt", add_special_tokens=False)["input_ids"]
        elif isinstance(seq, list):
            # If it's a list of token IDs
            input_ids = torch.tensor([seq], dtype=torch.long)
        else:
            print(f"Warning: Skipping sequence {idx + 1}, unknown format")
            continue
        
        input_ids = input_ids.to(device)
        
        # Decode for display (excluding mask tokens)
        decoded = tokenizer.decode(
            [t for t in input_ids[0].tolist() if t != mask_token_id],
            skip_special_tokens=True
        )
        
        print(f"  Sequence (first 100 chars): {decoded[:100]}...")
        print(f"  Length: {input_ids.shape[1]} tokens")
        
        # Count mask tokens
        num_masks = (input_ids == mask_token_id).sum().item()
        num_non_masks = input_ids.shape[1] - num_masks
        print(f"  Mask tokens: {num_masks}, Non-mask tokens: {num_non_masks}")
        
        # Compute metrics
        print("  Computing usual perplexity...")
        usual_ppl = usual_perplexity(input_ids, model, mask_token_id, device)
        
        print("  Computing full PPPL (iterative masking)...")
        full_pppl_val = full_pppl(input_ids, model, mask_token_id, device)
        
        print("  Computing simple/fast PPPL...")
        simple_pppl = simple_fast_pppl(input_ids, model, mask_token_id, device)
        
        result = {
            "sequence_id": idx + 1,
            "sequence_length": int(input_ids.shape[1]),
            "num_mask_tokens": int(num_masks),
            "num_non_mask_tokens": int(num_non_masks),
            "usual_perplexity": float(usual_ppl),
            "full_pppl": float(full_pppl_val),
            "simple_fast_pppl": float(simple_pppl),
            "sequence_preview": decoded[:200],
        }
        
        print(f"  Results:")
        print(f"    Usual PPL: {usual_ppl:.4f}")
        print(f"    Full PPPL: {full_pppl_val:.4f}")
        print(f"    Simple/Fast PPPL: {simple_pppl:.4f}")
        
        results.append(result)
    
    return results


def load_sequences_from_file(file_path: str) -> List[str]:
    """Load sequences from a file. Supports JSON, JSONL, or plain text (one per line)."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    sequences = []
    
    if file_path.suffix == ".json":
        with open(file_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                sequences = data
            elif isinstance(data, dict) and "sequences" in data:
                sequences = data["sequences"]
            else:
                raise ValueError("JSON file must contain a list or dict with 'sequences' key")
    
    elif file_path.suffix == ".jsonl":
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                if "sequence" in data:
                    sequences.append(data["sequence"])
                elif "text" in data:
                    sequences.append(data["text"])
                else:
                    sequences.append(str(data))
    
    else:
        # Plain text file, one sequence per line
        with open(file_path, "r") as f:
            sequences = [line.strip() for line in f if line.strip()]
    
    return sequences


def main():
    parser = argparse.ArgumentParser(
        description="Compute perplexity metrics for sequences"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model (HuggingFace model ID or local path)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to input file with sequences (JSON, JSONL, or text)",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        help="Sequences to process (as strings)",
    )
    parser.add_argument(
        "--mask_token_id",
        type=int,
        default=None,
        help="Mask token ID (default: auto-detect from tokenizer)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--offline_mode",
        action="store_true",
        help="Use offline mode for model loading",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file to save results (JSON format)",
    )
    
    args = parser.parse_args()
    
    # Load sequences
    if args.input_file:
        sequences = load_sequences_from_file(args.input_file)
        print(f"Loaded {len(sequences)} sequences from {args.input_file}")
    elif args.sequences:
        sequences = args.sequences
        print(f"Processing {len(sequences)} sequences from command line")
    else:
        raise ValueError("Must provide either --input_file or --sequences")
    
    if len(sequences) == 0:
        raise ValueError("No sequences to process")
    
    # Load model
    model, tokenizer, mask_token_id = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        offline_mode=args.offline_mode,
        mask_token_id=args.mask_token_id,
    )
    
    # Process sequences
    results = process_sequences(
        sequences,
        model,
        tokenizer,
        mask_token_id,
        device=args.device,
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Seq ID':<8} {'Usual PPL':<12} {'Full PPPL':<12} {'Simple PPPL':<12}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['sequence_id']:<8} "
            f"{r['usual_perplexity']:<12.4f} "
            f"{r['full_pppl']:<12.4f} "
            f"{r['simple_fast_pppl']:<12.4f}"
        )
    print("=" * 80)
    
    # Save results if requested
    if args.output_file:
        output_path = Path(args.output_file)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
