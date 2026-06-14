"""
Run the victim model (e.g., Llama-3.1-8B-Instruct) on prompts and generate responses.

For Google Colab / Jupyter Notebook usage - just paste this cell and run!
Configure the variables in the "CONFIGURATION" section below.
"""

import json
import torch
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# =============================================================================
# CONFIGURATION - Edit these variables as needed
# =============================================================================

# Model settings
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
USE_QUANTIZATION = True  # Set to False to disable 4-bit quantization (uses more VRAM)

# Input settings - choose ONE of the following:
# Option 1: Single prompt (set to None to use Option 2)
SINGLE_PROMPT = None  # e.g., "How do I make a cake?"

# Option 2: List of prompts
PROMPTS = [
    "How do I make a cake?",
    "What is the capital of France?",
]

# Option 3: Load from JSON file (set to None to use Option 1 or 2)
INPUT_FILE = None  # e.g., "data/harmbench_examples_s_mm.json"
PROMPT_KEY = "initial_query"  # Key to extract prompts from JSON

# Generation settings
SYSTEM_PROMPT = "You are a helpful assistant."
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True  # Set to False for greedy decoding

# Output settings
OUTPUT_FILE = None  # e.g., "results.json" - set to save results

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_model_and_tokenizer(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    use_quantization: bool = True,
):
    """Load the model and tokenizer."""
    print(f"Loading model: {model_name}")
    print(f"Quantization: {use_quantization}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    
    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate a response from the model given a prompt."""
    
    # Format with chat template
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # Apply chat template
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        print(f"Warning: Could not apply chat template: {e}")
        formatted_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return response.strip()


def load_prompts_from_file(file_path: str, prompt_key: str = "initial_query") -> List[Dict]:
    """Load prompts from a JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    
    prompts = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                prompt = item.get(prompt_key, "")
                target = item.get("target_behavior", item.get("target", ""))
                prompts.append({"prompt": prompt, "target": target})
            elif isinstance(item, str):
                prompts.append({"prompt": item, "target": ""})
    elif isinstance(data, dict):
        prompt = data.get(prompt_key, "")
        target = data.get("target_behavior", data.get("target", ""))
        prompts.append({"prompt": prompt, "target": target})
    
    return prompts


# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Load model
model, tokenizer = load_model_and_tokenizer(
    model_name=MODEL_NAME,
    use_quantization=USE_QUANTIZATION,
)

# Collect prompts
if INPUT_FILE:
    prompt_list = load_prompts_from_file(INPUT_FILE, PROMPT_KEY)
elif SINGLE_PROMPT:
    prompt_list = [{"prompt": SINGLE_PROMPT, "target": ""}]
else:
    prompt_list = [{"prompt": p, "target": ""} for p in PROMPTS]

print(f"\nRunning {len(prompt_list)} prompt(s)...\n")
print("=" * 80)

results = []

for i, item in enumerate(prompt_list):
    prompt = item["prompt"]
    target = item.get("target", "")
    
    print(f"\n[Example {i + 1}/{len(prompt_list)}]")
    print(f"Prompt: {prompt[:300]}{'...' if len(prompt) > 300 else ''}")
    if target:
        print(f"Target: {target[:150]}{'...' if len(target) > 150 else ''}")
    
    # Generate response
    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=DO_SAMPLE,
    )
    
    print(f"\nResponse:\n{response}")
    print("-" * 80)
    
    results.append({
        "prompt": prompt,
        "target": target,
        "response": response,
        "model": MODEL_NAME,
    })

# Save results if requested
if OUTPUT_FILE:
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_FILE}")

print("\nDone!")
