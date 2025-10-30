#!/usr/bin/env python3
"""
HellaSwag Evaluation for GPT-OSS-8B MoE

Evaluates model on HellaSwag common-sense reasoning benchmark.
Compatible with FSDP-sharded models.

Usage:
    # Standalone evaluation
    python scripts/evaluate_hellaswag.py --checkpoint_dir out/8b_moe_run1 --data_dir data/edu_fineweb10B_o200k

    # Or integrate into training loop (see train.py)
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.model import Transformer, ModelConfig, gpt_oss_moe_8b_config


def load_tokenizer(data_dir: str):
    """Load tokenizer from meta.json"""
    import tiktoken
    
    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    tok_name = meta.get("tokenizer", "o200k_base")
    try:
        enc = tiktoken.get_encoding(tok_name)
    except Exception:
        enc = tiktoken.get_encoding("o200k_base")
        tok_name = "o200k_base"
    
    vocab_size = int(meta.get("vocab_size", getattr(enc, "n_vocab", 201_088)))
    return enc, tok_name, vocab_size


def load_model_from_checkpoint(checkpoint_dir: str, data_dir: str, device: str = "cuda"):
    """Load model from checkpoint"""
    import glob
    
    # Find checkpoint files
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "ckpt_rank*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    # Load rank 0 checkpoint for config
    ckpt_path = os.path.join(checkpoint_dir, "ckpt_rank00000.pt")
    if not os.path.exists(ckpt_path):
        # Try any checkpoint
        ckpt_path = sorted(ckpt_files)[0]
    
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Reconstruct config
    cfg_dict = payload.get("model_config_dict", {})
    if not cfg_dict:
        # Fallback to default config
        cfg = gpt_oss_moe_8b_config()
    else:
        cfg = ModelConfig(**cfg_dict)
    
    # Load tokenizer to get vocab size
    _, _, vocab_size = load_tokenizer(data_dir)
    cfg.vocab_size = vocab_size
    
    # Build model
    model = Transformer(cfg)
    
    # Load state dict (for non-FSDP checkpoint loading, we'd need to handle sharding)
    # For now, assume single-GPU evaluation or use FSDP consolidation
    # Note: Full FSDP checkpoint loading requires FSDP wrapping - see train.py
    
    print(f"[HellaSwag] Model loaded: {cfg.num_hidden_layers} layers, {cfg.num_local_experts} experts")
    print(f"[HellaSwag] Warning: Full FSDP checkpoint loading not implemented in this script")
    print(f"[HellaSwag] Use export_to_safetensors.py first to create a consolidated checkpoint")
    
    return model, cfg


def format_hellaswag_prompt(context: str, ending: str) -> str:
    """Format HellaSwag example as completion prompt"""
    # Format: context + " " + ending
    prompt = context.strip() + " " + ending.strip()
    return prompt


def compute_continuation_probability(
    model: Transformer,
    tokenizer,
    context: str,
    ending: str,
    device: str,
    max_length: int = 512,
) -> float:
    """Compute log probability of ending given context"""
    model.eval()
    
    # Format prompt
    prompt = format_hellaswag_prompt(context, ending)
    
    # Tokenize
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_length:
        # Truncate context if needed
        context_tokens = tokenizer.encode(context)
        ending_tokens = tokenizer.encode(ending)
        # Keep full ending, truncate context
        max_context = max_length - len(ending_tokens) - 1
        if max_context > 0:
            context_tokens = context_tokens[-max_context:]
            tokens = context_tokens + ending_tokens
        else:
            # Ending too long, truncate
            tokens = tokens[-max_length:]
    
    # Convert to tensor
    input_ids = torch.tensor([tokens], device=device)
    
    with torch.no_grad():
        # Get logits
        logits, _ = model(input_ids, labels=None)
        
        # Compute log probability of continuation
        # Log prob = sum of log probs of each token in ending
        log_probs = F.log_softmax(logits[0], dim=-1)
        
        # Sum log probs for tokens in ending
        # We already have the full sequence, so sum log probs of tokens after context
        context_len = len(tokenizer.encode(context))
        if context_len < len(tokens):
            continuation_log_probs = log_probs[context_len-1:-1]  # Start from context end
            continuation_indices = input_ids[0, context_len:]
            
            # Gather log probs for actual tokens
            log_prob_sum = 0.0
            for i, token_id in enumerate(continuation_indices):
                if i < len(continuation_log_probs):
                    log_prob_sum += continuation_log_probs[i, token_id].item()
        else:
            # Context already includes ending, use whole sequence
            continuation_indices = input_ids[0, 1:]
            log_prob_sum = 0.0
            for i, token_id in enumerate(continuation_indices):
                if i < len(log_probs) - 1:
                    log_prob_sum += log_probs[i, token_id].item()
    
    return log_prob_sum


def evaluate_hellaswag(
    model: Transformer,
    tokenizer,
    device: str = "cuda",
    split: str = "validation",
    num_examples: int = None,
    max_length: int = 512,
) -> Dict[str, float]:
    """
    Evaluate model on HellaSwag dataset
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        device: Device to run on
        split: Dataset split ("validation" or "test")
        num_examples: Number of examples to evaluate (None = all)
        max_length: Maximum sequence length
    
    Returns:
        Dictionary with accuracy metrics
    """
    print(f"[HellaSwag] Loading {split} split...")
    
    # Load HellaSwag dataset
    try:
        dataset = load_dataset("Rowan/hellaswag", split=split)
    except Exception as e:
        print(f"[HellaSwag] Error loading dataset: {e}")
        print("[HellaSwag] Install with: pip install datasets")
        raise
    
    if num_examples:
        dataset = dataset.select(range(min(num_examples, len(dataset))))
    
    print(f"[HellaSwag] Evaluating on {len(dataset)} examples...")
    
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    for i, example in enumerate(dataset):
        if (i + 1) % 100 == 0:
            print(f"[HellaSwag] Progress: {i+1}/{len(dataset)} ({100*(i+1)/len(dataset):.1f}%)")
        
        context = example["ctx"]
        endings = example["endings"]
        label = example["label"]  # Correct answer index (0-3)
        
        # Compute log probability for each ending
        log_probs = []
        for ending in endings:
            try:
                log_prob = compute_continuation_probability(
                    model, tokenizer, context, ending, device, max_length
                )
                log_probs.append(log_prob)
            except Exception as e:
                print(f"[HellaSwag] Error processing example {i}: {e}")
                log_probs.append(float("-inf"))
        
        # Select ending with highest log probability
        predicted = int(np.argmax(log_probs))
        
        if predicted == label:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }
    
    print(f"\n[HellaSwag] Results:")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"  Random baseline: 0.25 (25%)")
    
    return results


def evaluate_hellaswag_simple(
    model: Transformer,
    tokenizer,
    device: str = "cuda",
    split: str = "validation",
    num_examples: int = 1000,
    max_length: int = 512,
) -> Dict[str, float]:
    """
    Simple HellaSwag evaluation that works with model directly (no checkpoint loading).
    Used for evaluation during training.
    
    Args:
        model: Model instance (can be FSDP-wrapped)
        tokenizer: Tokenizer instance
        device: Device to run on
        split: Dataset split
        num_examples: Number of examples to evaluate
        max_length: Maximum sequence length
    
    Returns:
        Dictionary with accuracy metrics
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")
    
    print(f"[HellaSwag] Loading {split} split...")
    
    # Load HellaSwag dataset
    dataset = load_dataset("Rowan/hellaswag", split=split)
    
    if num_examples:
        dataset = dataset.select(range(min(num_examples, len(dataset))))
    
    print(f"[HellaSwag] Evaluating on {len(dataset)} examples...")
    
    # Unwrap FSDP if needed
    eval_model = model.module if hasattr(model, "module") else model
    eval_model.to(device)
    eval_model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, example in enumerate(dataset):
            if (i + 1) % 100 == 0:
                print(f"[HellaSwag] Progress: {i+1}/{len(dataset)} ({100*(i+1)/len(dataset):.1f}%)")
            
            context = example["ctx"]
            endings = example["endings"]
            label = example["label"]  # Correct answer index (0-3)
            
            # Compute log probability for each ending
            log_probs = []
            for ending in endings:
                try:
                    log_prob = compute_continuation_probability(
                        eval_model, tokenizer, context, ending, device, max_length
                    )
                    log_probs.append(log_prob)
                except Exception as e:
                    if i < 5:  # Only print first few errors
                        print(f"[HellaSwag] Error processing example {i}: {e}")
                    log_probs.append(float("-inf"))
            
            # Select ending with highest log probability
            predicted = int(np.argmax(log_probs))
            
            if predicted == label:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on HellaSwag")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Directory containing checkpoints")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Data directory with meta.json")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--split", type=str, default="validation",
                       choices=["validation", "test"],
                       help="Dataset split")
    parser.add_argument("--num_examples", type=int, default=None,
                       help="Number of examples to evaluate (None = all)")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("HellaSwag Evaluation")
    print("=" * 70)
    
    # Load tokenizer
    print("[HellaSwag] Loading tokenizer...")
    tokenizer, tok_name, vocab_size = load_tokenizer(args.data_dir)
    print(f"[HellaSwag] Tokenizer: {tok_name}, Vocab: {vocab_size}")
    
    # Load model
    print("[HellaSwag] Loading model...")
    print("[HellaSwag] NOTE: This script requires FSDP checkpoint consolidation.")
    print("[HellaSwag] For standalone evaluation, use export_to_safetensors.py first.")
    print("[HellaSwag] For training-time evaluation, use --eval_hellaswag flag in train.py")
    
    # For now, return early with instructions
    print("\n[HellaSwag] To enable full evaluation:")
    print("  1. Export FSDP checkpoint: python scripts/export_to_safetensors.py")
    print("  2. Load consolidated checkpoint in this script")
    print("  3. Run evaluation")
    print("\nOR use during training:")
    print("  python train.py ... --eval_hellaswag --eval_hellaswag_interval 5000")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

