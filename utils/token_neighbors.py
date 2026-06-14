import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from pathlib import Path
import os

def resolve_model_path(model_id: str, offline_mode: bool, cache_dir: str) -> str:
    """Resolve a model ID to a local path if in offline mode."""
    if not model_id or not offline_mode:
        return model_id
    
    p = Path(model_id)
    if p.exists():
        return str(p.absolute())

    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        return str(local_dir)
    except Exception:
        return model_id

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description="Find nearest neighbors for tokens in embedding space.")
    parser.add_argument("--json", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Victim model ID.")
    parser.add_argument("--k", type=int, default=32, help="Number of neighbors to find.")
    parser.add_argument("--cache-dir", type=str, default="/scratch/gpfs/KOROLOVA/huggingface", help="HF cache directory.")
    parser.add_argument("--offline", action="store_true", default=True, help="Force offline mode.")
    parser.add_argument("--cos_sim", type=str2bool, default=True, help="Use cosine similarity (default: True). If False, uses L2 distance.")
    args = parser.parse_args()

    # Resolve path
    model_path = resolve_model_path(args.model, args.offline, args.cache_dir)
    print(f"Loading model and tokenizer from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=args.offline)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype=torch.float16, 
        local_files_only=args.offline
    )
    
    # Get embeddings
    embeddings = model.get_input_embeddings().weight.data.clone() # (vocab_size, hidden_dim)
    if args.cos_sim:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    vocab_size = embeddings.shape[0]

    # Load JSON and extract text
    with open(args.json, 'r') as f:
        data = json.load(f)
    
    for item_idx, item in enumerate(data):
        target_text = item.get("target_behavior", "")
        if not target_text:
            continue
            
        print(f"Target {item_idx + 1}: \"{target_text}\"")
        
        # Tokenize and show tokens
        token_ids = tokenizer.encode(target_text, add_special_tokens=False)
        tokens_str = [tokenizer.decode([tid]) for tid in token_ids]
        
        print(f"Tokenization: {tokens_str}")
        print(f"Metric: {'Cosine Similarity' if args.cos_sim else 'L2 Distance'}")
        print("-" * 20)

        for tid, tstr in zip(token_ids, tokens_str):
            query_emb = embeddings[tid].unsqueeze(0) # (1, hidden_dim)
            
            if args.cos_sim:
                # Calculate cosine similarity (normalized dot product)
                scores = torch.matmul(embeddings, query_emb.T).squeeze() # (vocab_size)
                top_vals, top_indices = torch.topk(scores, k=args.k + 1)
            else:
                # Calculate L2 distance
                dists = torch.norm(embeddings - query_emb, p=2, dim=1)
                top_vals, top_indices = torch.topk(dists, k=args.k + 1, largest=False)
            
            neighbors = []
            for i in range(len(top_indices)):
                idx = top_indices[i].item()
                if idx == tid:
                    continue
                
                neigh_str = tokenizer.decode([idx])
                score = top_vals[i].item()
                neighbors.append(f"'{neigh_str}' ({score:.3f})")
                
                if len(neighbors) >= args.k:
                    break
            
            print(f"Token: '{tstr}' (ID: {tid})")
            print(f"Neighbors: {', '.join(neighbors)}")
            print(".")
        
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
