#!/usr/bin/env python3
"""
Simple perplexity calculator for a list of strings.

Perplexity measures how "surprised" a language model is by the text.
Lower perplexity = more fluent/expected text.

Formula: PPL = exp(CE_loss) where CE_loss is mean cross-entropy per token.

=== IMPORTANT: Two Types of Perplexity in GCD ===

1. `calculate_perplexity` (logged as `example_X/perplexity` in WandB):
   - Uses external model (GPT-2 by default)
   - Computed ONLY on the tunable tokens (adversarial suffix) in ISOLATION
   - No context from system prompt or user query
   - This script replicates this calculation
   
2. `self_perplexity` (logged as `example_X/self_perplexity` in WandB):
   - Uses the victim model (e.g., Llama-3.1-8B-Instruct)
   - Computed on tunable tokens WITH full context conditioning
   - P(suffix_token_i | system_prompt, user_query, suffix_tokens_0..i-1)
   - Context makes the suffix appear more "expected", so lower perplexity

Usage:
    python scripts/compute_perplexity_simple.py
    python scripts/compute_perplexity_simple.py --model gpt2-medium
    python scripts/compute_perplexity_simple.py --texts "Hello world" "This is a test"
"""

import argparse
import torch
import math
from typing import List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_perplexity_single(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> dict:
    """
    Compute perplexity for a single text string.
    
    Returns dict with:
        - perplexity: exp(mean_ce_loss)
        - ce_loss: mean cross-entropy loss per token
        - num_tokens: number of tokens in the text
    """
    if not text or not text.strip():
        return {"perplexity": float("inf"), "ce_loss": float("inf"), "num_tokens": 0}
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(device)
    
    num_tokens = input_ids.shape[1]
    if num_tokens < 2:
        return {"perplexity": float("inf"), "ce_loss": float("inf"), "num_tokens": num_tokens}
    
    with torch.no_grad():
        # Method 1: Use model's built-in loss computation
        # When labels=input_ids, model internally shifts by 1 and computes CE loss
        outputs = model(input_ids, labels=input_ids)
        ce_loss = outputs.loss.item()
        perplexity = math.exp(ce_loss)
    
    return {
        "perplexity": perplexity,
        "ce_loss": ce_loss,
        "num_tokens": num_tokens,
    }


def compute_perplexity_single_manual(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> dict:
    """
    Compute perplexity manually (step-by-step) for educational purposes.
    
    This shows exactly what's happening under the hood.
    """
    if not text or not text.strip():
        return {"perplexity": float("inf"), "ce_loss": float("inf"), "num_tokens": 0}
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(device)  # [1, seq_len]
    
    num_tokens = input_ids.shape[1]
    if num_tokens < 2:
        return {"perplexity": float("inf"), "ce_loss": float("inf"), "num_tokens": num_tokens}
    
    with torch.no_grad():
        # Get logits from model
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]
        
        # For language modeling, we predict token[i+1] from token[i]
        # So we shift: logits[:-1] predicts labels[1:]
        shift_logits = logits[:, :-1, :].contiguous()  # [1, seq_len-1, vocab_size]
        shift_labels = input_ids[:, 1:].contiguous()    # [1, seq_len-1]
        
        # Compute cross-entropy loss per token
        # CE = -log(P(correct_token))
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),  # [seq_len-1, vocab_size]
            shift_labels.view(-1)                           # [seq_len-1]
        )  # [seq_len-1]
        
        # Mean CE loss
        ce_loss = per_token_loss.mean().item()
        
        # Perplexity = exp(mean_ce_loss)
        perplexity = math.exp(ce_loss)
    
    return {
        "perplexity": perplexity,
        "ce_loss": ce_loss,
        "num_tokens": num_tokens,
        "num_predicted_tokens": num_tokens - 1,  # First token has no prediction
        "per_token_losses": per_token_loss.cpu().tolist(),
    }


def compute_perplexity_batch(
    texts: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int = 8,
) -> List[dict]:
    """
    Compute perplexity for a batch of texts efficiently.
    """
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize with padding
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Shift for LM prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()
            
            # Per-token CE loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(shift_labels.size())  # [batch, seq_len-1]
            
            # Average over valid (non-padding) tokens per sample
            valid_tokens = shift_mask.sum(dim=1).clamp(min=1)
            sample_ce_loss = (loss * shift_mask).sum(dim=1) / valid_tokens
            sample_ppl = torch.exp(sample_ce_loss)
        
        # Collect results
        for j, text in enumerate(batch_texts):
            results.append({
                "text": text[:50] + "..." if len(text) > 50 else text,
                "perplexity": sample_ppl[j].item(),
                "ce_loss": sample_ce_loss[j].item(),
                "num_tokens": int(valid_tokens[j].item()) + 1,
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute perplexity for text strings")
    parser.add_argument("--model", type=str, default="gpt2", 
                        help="Model name (e.g., gpt2, gpt2-medium, gpt2-large)")
    parser.add_argument("--texts", nargs="+", type=str, default=None,
                        help="Text strings to evaluate")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    # Default test texts if none provided
    if args.texts is None:
        args.texts = [
            "The quick brown fox jumps over the lazy dog.",
            "asdf jkl; qwer uiop zxcv bnm,",
            "Hello! How are you doing today?",
            "!@#$%^&*() random symbols everywhere !!!",
            "The capital of France is Paris.",
            "quantum entanglement superposition decoherence",
            "Sure, here is a tutorial on how to make a cake.",
            "I cannot help with that request.",
        ]
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model: {args.model}")
    print(f"Device: {device}")
    print("-" * 60)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\n" + "=" * 60)
    print("SINGLE TEXT EVALUATION (detailed)")
    print("=" * 60)
    
    for text in args.texts:
        result = compute_perplexity_single(text, model, tokenizer, device)
        print(f"\nText: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"  Perplexity: {result['perplexity']:.4f}")
        print(f"  CE Loss:    {result['ce_loss']:.4f}")
        print(f"  Tokens:     {result['num_tokens']}")
    
    print("\n" + "=" * 60)
    print("BATCH EVALUATION")
    print("=" * 60)
    
    batch_results = compute_perplexity_batch(args.texts, model, tokenizer, device)
    
    print(f"\n{'Text':<55} {'PPL':>10} {'CE Loss':>10} {'Tokens':>8}")
    print("-" * 85)
    for r in batch_results:
        print(f"{r['text']:<55} {r['perplexity']:>10.2f} {r['ce_loss']:>10.4f} {r['num_tokens']:>8}")
    
    print("\n" + "=" * 60)
    print("MANUAL COMPUTATION (educational)")
    print("=" * 60)
    
    # Show detailed manual computation for first text
    text = args.texts[0]
    result = compute_perplexity_single_manual(text, model, tokenizer, device)
    print(f"\nText: {text}")
    print(f"Perplexity: {result['perplexity']:.4f}")
    print(f"CE Loss (mean): {result['ce_loss']:.4f}")
    print(f"Total tokens: {result['num_tokens']}")
    print(f"Predicted tokens: {result['num_predicted_tokens']} (first token is context only)")
    
    if 'per_token_losses' in result:
        tokens = tokenizer.encode(text)
        print(f"\nPer-token losses (token -> loss):")
        for i, loss in enumerate(result['per_token_losses'][:10]):  # First 10
            tok_str = tokenizer.decode([tokens[i+1]]).replace('\n', '\\n')
            print(f"  {i+1}: '{tok_str}' -> {loss:.4f} (PPL={math.exp(loss):.2f})")
        if len(result['per_token_losses']) > 10:
            print(f"  ... ({len(result['per_token_losses']) - 10} more tokens)")


if __name__ == "__main__":
    main()
