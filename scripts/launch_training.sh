#!/usr/bin/env python3
"""
Simple multi-GPU launch script for training
"""
import subprocess
import sys
import os

def main():
    # Detect number of GPUs automatically
    try:
        import torch
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs detected")
    except:
        # Fallback: use environment variable or default to 8
        num_gpus = int(os.environ.get("NUM_GPUS", "8"))
    
    # Allow override via environment variable
    if "NUM_GPUS" in os.environ:
        num_gpus = int(os.environ["NUM_GPUS"])
    
    # Training configuration
    batch_size = 2  # Safe default, increase to 4-6 if no OOM
    block_size = 2048  # Increased from 1024, try 4096 if memory allows
    grad_accum_steps = 32
    total_tokens = 10_000_000_000  # 10B tokens to train
    
    # Calculate tokens per iteration
    tokens_per_iter = batch_size * block_size * grad_accum_steps * num_gpus
    
    # Calculate max_iters needed to train 10B tokens
    # Add 10% buffer for data loading overlap and rounding
    calculated_max_iters = int(total_tokens / tokens_per_iter * 1.1)
    
    # Use provided max_iters or calculated value
    max_iters = int(os.environ.get("MAX_ITERS", calculated_max_iters))
    
    cmd = [
        sys.executable, "-m", "torch.distributed.launch",
        f"--nproc_per_node={num_gpus}",  # Use detected or specified number of GPUs
        "--master_addr=localhost",
        "--master_port=29500",
        "model/train.py",
        "--data_dir", "data/edu_fineweb_10BT",
        "--out_dir", "out/6b_moe_run1",
        "--batch_size", str(batch_size),
        "--block_size", str(block_size),
        "--max_iters", str(max_iters),
        "--grad_accum_steps", str(grad_accum_steps),
        "--lr", "3e-4",
        "--min_lr", "3e-5",
        "--warmup_iters", "2000",
        "--lr_decay_iters", "20000",
        "--weight_decay", "0.1",
        "--beta1", "0.9",
        "--beta2", "0.95",
        "--grad_clip", "1.0",
        "--eval_interval", "500",
        "--eval_iters", "100",
        "--save_every", "1000",
        "--keep_last_n", "3",
        "--sample_every", "1000",
        "--sample_tokens", "256",
        "--log_interval", "10",
        "--log_router_stats",
        "--dtype", "bfloat16",
        "--seed", "1337"
    ]
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    print("=" * 70)
    print("Launching Multi-GPU Training")
    print("=" * 70)
    print(f"Number of GPUs: {num_gpus}")
    print(f"Training Configuration:")
    print(f"  Batch size per GPU: {batch_size}")
    print(f"  Block size: {block_size}")
    print(f"  Gradient accumulation: {grad_accum_steps}")
    print(f"  Tokens per iteration: {tokens_per_iter:,}")
    print(f"  Total tokens to train: {total_tokens:,} ({total_tokens/1e9:.1f}B)")
    print(f"  Calculated max_iters: {calculated_max_iters:,}")
    print(f"  Max iterations (with buffer): {max_iters:,}")
    print(f"  Estimated epochs: {max_iters * tokens_per_iter / total_tokens:.2f}")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)
    print("Environment Variables:")
    print("  NUM_GPUS=5 python scripts/launch_training.sh  # Override GPU count")
    print("  MAX_ITERS=15000 python scripts/launch_training.sh  # Override iterations")
    print("=" * 70)
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()

