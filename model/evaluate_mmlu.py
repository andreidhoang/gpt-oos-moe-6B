#!/usr/bin/env python3
"""
MMLU Evaluation for GPT-OSS-8B MoE

Evaluates model on MMLU (Massive Multitask Language Understanding) benchmark.
MMLU tests knowledge across 57 subjects with ~16,000 multiple-choice questions.
Compatible with FSDP-sharded models.

Usage:
    # Standalone evaluation (single subject)
    python scripts/evaluate_mmlu.py --checkpoint_dir out/8b_moe_run1 --data_dir data/edu_fineweb10B_o200k --subject high_school_mathematics

    # Evaluate multiple subjects
    python scripts/evaluate_mmlu.py --checkpoint_dir out/8b_moe_run1 --data_dir data/edu_fineweb10B_o200k --all_subjects

    # Or integrate into training loop (see train.py)
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

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
    
    print(f"[MMLU] Model loaded: {cfg.num_hidden_layers} layers, {cfg.num_local_experts} experts")
    print(f"[MMLU] Warning: Full FSDP checkpoint loading not implemented in this script")
    print(f"[MMLU] Use export_to_safetensors.py first to create a consolidated checkpoint")
    
    return model, cfg


def format_mmlu_prompt(question: str, choices: Dict[str, str]) -> str:
    """
    Format MMLU question as completion prompt.
    
    Format:
    Question: [question text]
    A) [choice A]
    B) [choice B]
    C) [choice C]
    D) [choice D]
    Answer:
    """
    prompt = f"Question: {question.strip()}\n"
    prompt += f"A) {choices['A'].strip()}\n"
    prompt += f"B) {choices['B'].strip()}\n"
    prompt += f"C) {choices['C'].strip()}\n"
    prompt += f"D) {choices['D'].strip()}\n"
    prompt += "Answer:"
    return prompt


def compute_choice_probability(
    model: Transformer,
    tokenizer,
    question: str,
    choices: Dict[str, str],
    choice_letter: str,
    device: str,
    max_length: int = 512,
) -> float:
    """
    Compute log probability of a specific choice being the answer.
    
    This works by computing the log probability of the full prompt ending with
    the choice text, similar to HellaSwag's continuation probability.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        question: Question text
        choices: Dictionary mapping 'A', 'B', 'C', 'D' to choice texts
        choice_letter: Which choice to score ('A', 'B', 'C', or 'D')
        device: Device to run on
        max_length: Maximum sequence length
    
    Returns:
        Log probability of the choice
    """
    model.eval()
    
    # Format prompt with the choice as the answer
    # Format: "Question: [q] A) [A] B) [B] C) [C] D) [D] Answer: [choice_text]"
    choice_text = choices[choice_letter]
    prompt = format_mmlu_prompt(question, choices)
    prompt_with_answer = prompt + " " + choice_text.strip()
    
    # Tokenize
    tokens = tokenizer.encode(prompt_with_answer)
    if len(tokens) > max_length:
        # Truncate question/choices if needed, but try to keep answer
        prompt_tokens = tokenizer.encode(prompt)
        answer_tokens = tokenizer.encode(choice_text)
        
        max_prompt_len = max_length - len(answer_tokens) - 1
        if max_prompt_len > 0:
            prompt_tokens = prompt_tokens[-max_prompt_len:]
            tokens = prompt_tokens + answer_tokens
        else:
            # Answer too long, truncate everything
            tokens = tokens[-max_length:]
    
    # Convert to tensor
    input_ids = torch.tensor([tokens], device=device)
    
    with torch.no_grad():
        # Get logits
        logits, _ = model(input_ids, labels=None)
        
        # Compute log probability of the answer (choice text)
        log_probs = F.log_softmax(logits[0], dim=-1)
        
        # Find where answer starts in prompt
        prompt_tokens = tokenizer.encode(prompt)
        answer_tokens = tokenizer.encode(choice_text)
        
        if len(prompt_tokens) < len(tokens):
            # Answer is at the end
            answer_start_idx = len(prompt_tokens)
            log_prob_sum = 0.0
            
            for i, token_id in enumerate(answer_tokens):
                if answer_start_idx + i < len(log_probs):
                    # Use log prob at position (answer_start_idx + i - 1) to predict token at position (answer_start_idx + i)
                    pred_idx = answer_start_idx + i - 1
                    if pred_idx >= 0 and pred_idx < len(log_probs):
                        log_prob_sum += log_probs[pred_idx, token_id].item()
            
            return log_prob_sum
        else:
            # Context already includes answer, compute from sequence
            answer_start_idx = len(prompt_tokens) - len(answer_tokens)
            if answer_start_idx < 0:
                answer_start_idx = max(0, len(tokens) - len(answer_tokens))
            
            log_prob_sum = 0.0
            for i, token_id in enumerate(answer_tokens):
                pred_idx = answer_start_idx + i - 1
                if pred_idx >= 0 and pred_idx < len(log_probs):
                    log_prob_sum += log_probs[pred_idx, token_id].item()
            
            return log_prob_sum


def evaluate_mmlu(
    model: Transformer,
    tokenizer,
    device: str = "cuda",
    subject: str = "high_school_mathematics",
    split: str = "test",
    num_examples: Optional[int] = None,
    max_length: int = 512,
) -> Dict[str, float]:
    """
    Evaluate model on MMLU dataset for a specific subject.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        device: Device to run on
        subject: MMLU subject (e.g., "high_school_mathematics")
        split: Dataset split ("test" or "validation")
        num_examples: Number of examples to evaluate (None = all)
        max_length: Maximum sequence length
    
    Returns:
        Dictionary with accuracy metrics
    """
    print(f"[MMLU] Loading subject: {subject} ({split} split)...")
    
    # Load MMLU dataset
    try:
        dataset = load_dataset("cais/mmlu", subject, split=split)
    except Exception as e:
        print(f"[MMLU] Error loading dataset: {e}")
        print("[MMLU] Install with: pip install datasets")
        print(f"[MMLU] Available subjects: See https://huggingface.co/datasets/cais/mmlu")
        raise
    
    if num_examples:
        dataset = dataset.select(range(min(num_examples, len(dataset))))
    
    print(f"[MMLU] Evaluating on {len(dataset)} examples...")
    
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    for i, example in enumerate(dataset):
        if (i + 1) % 50 == 0:
            print(f"[MMLU] Progress: {i+1}/{len(dataset)} ({100*(i+1)/len(dataset):.1f}%)")
        
        question = example["question"]
        
        # MMLU format: choices is a list, answer is index (0-3)
        choices_list = example["choices"]
        choices = {
            "A": choices_list[0],
            "B": choices_list[1],
            "C": choices_list[2],
            "D": choices_list[3],
        }
        correct_answer_idx = example["answer"]  # 0, 1, 2, or 3
        correct_answer_letter = ["A", "B", "C", "D"][correct_answer_idx]
        
        # Compute log probability for each choice
        log_probs = {}
        for letter in ["A", "B", "C", "D"]:
            try:
                log_prob = compute_choice_probability(
                    model, tokenizer, question, choices, letter, device, max_length
                )
                log_probs[letter] = log_prob
            except Exception as e:
                if i < 5:  # Only print first few errors
                    print(f"[MMLU] Error processing choice {letter} for example {i}: {e}")
                log_probs[letter] = float("-inf")
        
        # Select choice with highest log probability
        predicted = max(log_probs, key=log_probs.get)
        
        if predicted == correct_answer_letter:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "subject": subject,
    }
    
    print(f"\n[MMLU] Results for {subject}:")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"  Random baseline: 0.25 (25%)")
    
    return results


def evaluate_mmlu_simple(
    model: Transformer,
    tokenizer,
    device: str = "cuda",
    subject: str = "high_school_mathematics",
    num_examples: int = 100,
    max_length: int = 512,
) -> Dict[str, float]:
    """
    Simple MMLU evaluation that works with model directly (no checkpoint loading).
    Used for evaluation during training.
    
    Args:
        model: Model instance (can be FSDP-wrapped)
        tokenizer: Tokenizer instance
        device: Device to run on
        subject: MMLU subject to evaluate
        num_examples: Number of examples to evaluate
        max_length: Maximum sequence length
    
    Returns:
        Dictionary with accuracy metrics
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")
    
    print(f"[MMLU] Loading subject: {subject}...")
    
    # Load MMLU dataset
    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"[MMLU] Error loading dataset: {e}")
        raise
    
    if num_examples:
        dataset = dataset.select(range(min(num_examples, len(dataset))))
    
    print(f"[MMLU] Evaluating on {len(dataset)} examples...")
    
    # Unwrap FSDP if needed
    eval_model = model.module if hasattr(model, "module") else model
    eval_model.to(device)
    eval_model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, example in enumerate(dataset):
            if (i + 1) % 50 == 0:
                print(f"[MMLU] Progress: {i+1}/{len(dataset)} ({100*(i+1)/len(dataset):.1f}%)")
            
            question = example["question"]
            choices_list = example["choices"]
            choices = {
                "A": choices_list[0],
                "B": choices_list[1],
                "C": choices_list[2],
                "D": choices_list[3],
            }
            correct_answer_idx = example["answer"]
            correct_answer_letter = ["A", "B", "C", "D"][correct_answer_idx]
            
            # Compute log probability for each choice
            log_probs = {}
            for letter in ["A", "B", "C", "D"]:
                try:
                    log_prob = compute_choice_probability(
                        eval_model, tokenizer, question, choices, letter, device, max_length
                    )
                    log_probs[letter] = log_prob
                except Exception as e:
                    if i < 5:  # Only print first few errors
                        print(f"[MMLU] Error processing choice {letter} for example {i}: {e}")
                    log_probs[letter] = float("-inf")
            
            # Select choice with highest log probability
            predicted = max(log_probs, key=log_probs.get)
            
            if predicted == correct_answer_letter:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "subject": subject,
    }
    
    return results


def evaluate_mmlu_multiple_subjects(
    model: Transformer,
    tokenizer,
    device: str = "cuda",
    subjects: Optional[List[str]] = None,
    num_examples_per_subject: int = 100,
    max_length: int = 512,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on multiple MMLU subjects.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        device: Device to run on
        subjects: List of subjects to evaluate (None = default diverse subset)
        num_examples_per_subject: Number of examples per subject
        max_length: Maximum sequence length
    
    Returns:
        Dictionary with results per subject and overall summary
    """
    if subjects is None:
        # Default: evaluate on a diverse subset covering different domains
        subjects = [
            # Mathematics
            "elementary_mathematics",
            "high_school_mathematics",
            "college_mathematics",
            # Science
            "high_school_biology",
            "college_biology",
            "high_school_chemistry",
            "college_chemistry",
            "high_school_physics",
            "college_physics",
            # Computer Science
            "machine_learning",
            "high_school_computer_science",
            # Social Sciences
            "high_school_geography",
            "high_school_government_and_politics",
            # Humanities
            "high_school_us_history",
            "high_school_world_history",
            "philosophy",
        ]
    
    print(f"[MMLU] Evaluating on {len(subjects)} subjects...")
    print(f"[MMLU] Subjects: {', '.join(subjects)}")
    
    results = {}
    all_correct = 0
    all_total = 0
    
    for subject_idx, subject in enumerate(subjects):
        print(f"\n{'='*70}")
        print(f"Evaluating Subject {subject_idx+1}/{len(subjects)}: {subject}")
        print(f"{'='*70}")
        
        try:
            subject_results = evaluate_mmlu(
                model,
                tokenizer,
                device=device,
                subject=subject,
                num_examples=num_examples_per_subject,
                max_length=max_length,
            )
            
            results[subject] = subject_results
            all_correct += subject_results["correct"]
            all_total += subject_results["total"]
        except Exception as e:
            print(f"[MMLU] Error evaluating {subject}: {e}")
            results[subject] = {
                "accuracy": 0.0,
                "correct": 0,
                "total": 0,
                "subject": subject,
                "error": str(e),
            }
    
    overall_accuracy = all_correct / all_total if all_total > 0 else 0.0
    
    results["overall"] = {
        "accuracy": overall_accuracy,
        "correct": all_correct,
        "total": all_total,
        "subjects_evaluated": len(subjects),
    }
    
    print(f"\n{'='*70}")
    print(f"MMLU Overall Results:")
    print(f"  Accuracy: {overall_accuracy:.4f} ({all_correct}/{all_total})")
    print(f"  Subjects evaluated: {len(subjects)}")
    print(f"  Random baseline: 0.25 (25%)")
    print(f"{'='*70}")
    
    # Print per-subject summary
    print("\nPer-Subject Results:")
    print("-" * 70)
    for subject, result in results.items():
        if subject != "overall":
            acc = result.get("accuracy", 0.0)
            correct = result.get("correct", 0)
            total = result.get("total", 0)
            print(f"  {subject:40s} {acc:.4f} ({correct}/{total})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on MMLU benchmark")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                       help="Directory containing checkpoints (for standalone evaluation)")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Data directory with meta.json")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--subject", type=str, default="high_school_mathematics",
                       help="MMLU subject to evaluate (see https://huggingface.co/datasets/cais/mmlu)")
    parser.add_argument("--split", type=str, default="test",
                       choices=["test", "validation"],
                       help="Dataset split")
    parser.add_argument("--num_examples", type=int, default=None,
                       help="Number of examples to evaluate (None = all)")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--all_subjects", action="store_true",
                       help="Evaluate on multiple subjects (diverse subset)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MMLU Evaluation")
    print("=" * 70)
    
    # Load tokenizer
    print("[MMLU] Loading tokenizer...")
    tokenizer, tok_name, vocab_size = load_tokenizer(args.data_dir)
    print(f"[MMLU] Tokenizer: {tok_name}, Vocab: {vocab_size}")
    
    # Load model if checkpoint_dir provided
    if args.checkpoint_dir:
        print("[MMLU] Loading model...")
        print("[MMLU] NOTE: This script requires FSDP checkpoint consolidation.")
        print("[MMLU] For standalone evaluation, use export_to_safetensors.py first.")
        print("[MMLU] For training-time evaluation, use --eval_mmlu flag in train.py")
        
        # For now, return early with instructions
        print("\n[MMLU] To enable full evaluation:")
        print("  1. Export FSDP checkpoint: python scripts/export_to_safetensors.py")
        print("  2. Load consolidated checkpoint in this script")
        print("  3. Run evaluation")
        print("\nOR use during training:")
        print("  python train.py ... --eval_mmlu --eval_mmlu_interval 5000")
        
        return 0
    
    # If no checkpoint_dir, we'd need model passed in (for training-time eval)
    # This is handled by evaluate_mmlu_simple() function
    print("\n[MMLU] This script is designed for standalone evaluation.")
    print("[MMLU] For training-time evaluation, use evaluate_mmlu_simple() function.")
    print("[MMLU] See train.py for integration example.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

