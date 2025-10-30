#!/usr/bin/env python3
"""
Simple multi-GPU launch script for training
"""
import subprocess
import sys
import os

def main():
    cmd = [
        sys.executable, "-m", "torch.distributed.launch",
        "--nproc_per_node=8",
        "--master_addr=localhost",
        "--master_port=29500",
        "model/train.py",
        "--data_dir", "data/edu_fineweb_10BT",
        "--out_dir", "out/8b_moe_run1",
        "--batch_size", "2",
        "--block_size", "1024",
        "--max_iters", "20027",
        "--grad_accum_steps", "32",
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
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()

