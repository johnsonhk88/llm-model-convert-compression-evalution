#!/usr/bin/env python3
import os
import gc
import math
import pathlib
import argparse
import logging
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Suppress warnings and extra logging
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.getLogger("llmcompressor").setLevel(logging.WARNING)


def get_folder_size(path):
    """Calculates total folder size in bytes."""
    p = pathlib.Path(path)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def format_size(nbytes):
    """Formats bytes into a human-readable string."""
    if nbytes < 1024**2:
        return f"{nbytes/1024:.1f} KB"
    if nbytes < 1024**3:
        return f"{nbytes/1024**2:.1f} MB"
    return f"{nbytes/1024**3:.2f} GB"


def compare_sizes(model_dir, output_dir):
    """Prints a file size comparison between both model directories."""
    size_orig = get_folder_size(model_dir)
    size_q = get_folder_size(output_dir)
    reduction = (1 - size_q / size_orig) * 100 if size_orig > 0 else 0

    print("\n=============================================")
    print("Model Size Comparison")
    print("=============================================")
    print(f"Original (BF16):    {format_size(size_orig)}")
    print(f"Quantized (W4A16):  {format_size(size_q)}")
    print(f"Reduction:          {reduction:.0f}%")
    print("=============================================\n")


def test_generation(model_path, tokenizer, prompt, model_label):
    """Loads a model and generates text from a prompt, auto-selecting GPU if available."""
    print(f"Loading {model_label} model from {model_path}...")
    
    # Auto-select GPU ("auto") if available, otherwise fall back to CPU
    device_layout = "auto" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device_layout, dtype=torch.bfloat16
    )
    
    # Ensure inputs are explicitly pushed to the model's active device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=60, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    
    print(f"\n--- {model_label} Model Response ---")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")
    
    # Clean up RAM and VRAM memory completely
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def calculate_perplexity(model_path, tokenizer, dataset, max_tokens=5000, stride=512):
    """Loads model and calculates its perplexity over a test dataset, auto-selecting GPU if available."""
    print(f"Loading model for perplexity evaluation: {model_path}...")
    
    device_layout = "auto" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device_layout, dtype=torch.bfloat16
    )
    
    encodings = tokenizer(
        "\n\n".join(dataset["text"]),
        return_tensors="pt", truncation=True, max_length=max_tokens,
    )
    input_ids = encodings.input_ids
    nlls, prev_end = [], 0

    # Track target device for tracking dataset processing loops
    device = model.device

    for begin_loc in range(0, input_ids.size(1), stride):
        end_loc = min(begin_loc + stride, input_ids.size(1))
        trg_len = end_loc - prev_end
        
        # Explicitly pass current batch slice to GPU acceleration
        input_slice = input_ids[:, begin_loc:end_loc].to(device)
        target_slice = input_slice.clone()
        target_slice[:, :-trg_len] = -100
        
        with torch.no_grad():
            loss = model(input_slice, labels=target_slice).loss
            nlls.append(loss * trg_len)
        prev_end = end_loc

    ppl = math.exp(torch.stack(nlls).sum().item() / prev_end)
    
    # Clean up RAM and VRAM memory completely
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return ppl


def main():
    parser = argparse.ArgumentParser(description="Compare a base LLM model against its quantized version.")
    parser.add_argument("base_model", type=str, help="Path or HuggingFace ID for the base model")
    parser.add_argument("quant_model", type=str, help="Path or HuggingFace ID for the quantized model")
    parser.add_argument("--prompt", type=str, default="Machine learning is a branch of", help="Text prompt to test generation")
    args = parser.parse_args()

    # Guardrails to check local path existence early
    if not os.path.exists(args.base_model):
        print(f"\n[ERROR] Base model path does not exist: {os.path.abspath(args.base_model)}")
        return
    if not os.path.exists(args.quant_model):
        print(f"\n[ERROR] Quantized model path does not exist: {os.path.abspath(args.quant_model)}")
        return

    # Logging current hardware mode
    if torch.cuda.is_available():
        print(f"--> Hardware Detected: {torch.cuda.get_device_name(0)} (GPU Acceleration Enabled)")
    else:
        print("--> Hardware Detected: CPU Only")

    # 1. Compare Sizes
    compare_sizes(args.base_model, args.quant_model)

    # Initialize shared tokenizer
    print(f"Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # 2. Run Generations
    test_generation(args.base_model, tokenizer, args.prompt, "Base")
    test_generation(args.quant_model, tokenizer, args.prompt, "Quantized")

    # 3. Evaluate Perplexity
    print("Loading WikiText test split...")
    test_data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    print(f"Loaded {len(test_data)} test samples.\n")

    quant_ppl = calculate_perplexity(args.quant_model, tokenizer, test_data)
    base_ppl = calculate_perplexity(args.base_model, tokenizer, test_data)

    # Print Final Perplexity Report
    print("\n========================================")
    print("Perplexity Comparison (Lower is Better)")
    print("========================================")
    print(f"Base (BF16):      {base_ppl:.2f}")
    print(f"Quantized (W4A16): {quant_ppl:.2f}")
    print(f"Difference:       {quant_ppl - base_ppl:+.2f} ({(quant_ppl/base_ppl - 1)*100:+.1f}%)")
    print("========================================\n")


if __name__ == "__main__":
    main()