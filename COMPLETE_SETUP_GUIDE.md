# Complete Setup & Training Guide
## GPT-OSS-MoE-8B: From Zero to Trained Model

This guide provides **all commands needed** to set up, train, monitor, and deploy an 8B parameter Mixture-of-Experts language model from scratch.

---

## Table of Contents

1. [Prerequisites & System Requirements](#1-prerequisites--system-requirements)
2. [Initial Setup](#2-initial-setup)
3. [Data Preparation](#3-data-preparation)
4. [Pre-Training Validation](#4-pre-training-validation)
5. [Training Commands](#5-training-commands)
6. [Monitoring & Logging](#6-monitoring--logging)
7. [Important Runtime Commands](#7-important-runtime-commands)
8. [Post-Training](#8-post-training)
9. [Troubleshooting Commands](#9-troubleshooting-commands)
10. [Complete CLI Reference](#10-complete-cli-reference)
11. [Advanced Topics](#11-advanced-topics)

---

## 1. Prerequisites & System Requirements

### Hardware Requirements

```
┌──────────────────────────────────────────────────────────────┐
│                     GPU REQUIREMENTS                          │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│ MINIMUM (Testing):                                            │
│   • 1x NVIDIA A100 40GB or RTX 4090 24GB                     │
│   • 256GB RAM                                                 │
│   • 2TB SSD storage                                           │
│   • Training time: ~60-90 days for 100B tokens               │
│                                                                │
│ RECOMMENDED (Production):                                     │
│   • 8x NVIDIA A100 80GB                                       │
│   • 1TB RAM                                                   │
│   • 10TB NVMe SSD                                             │
│   • Training time: ~7-14 days for 100B tokens                │
│                                                                │
│ OPTIMAL (H100):                                               │
│   • 8x NVIDIA H100 NVL 94GB                                   │
│   • 1TB RAM                                                   │
│   • 10TB NVMe SSD                                             │
│   • Training time: ~3-5 days for 100B tokens                 │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### Storage Breakdown

```bash
# Total storage needed: ~1.5TB minimum, 10TB+ recommended

Raw data:              ~200GB   # Compressed text files
Tokenized data:        ~500GB   # .npy shards or .bin files
Checkpoints:           ~250GB   # 5 checkpoints × 50GB each
Logs & samples:        ~50GB    # Training logs, text samples
Working buffer:        ~500GB   # Temporary files, safety margin
─────────────────────────────
Total:                 ~1.5TB
```

### Software Requirements

```bash
# Operating System
Ubuntu 22.04 LTS (or any Linux with CUDA support)

# CUDA & Drivers
CUDA 12.1+ (12.3+ for FlashAttention 3 on H100)
NVIDIA Driver 530+ (545+ for H100)

# Python
Python 3.10 or 3.11
```

### Cost Estimates (Cloud Training)

```
Provider: Lambda Labs / RunPod / Vast.ai

8x A100 80GB:
  • Rate: ~$2.00/GPU-hour × 8 = $16/hour
  • 100B tokens: ~100 hours = $1,600
  • 10B tokens: ~10 hours = $160

8x H100 NVL:
  • Rate: ~$2.49/GPU-hour × 8 = $20/hour
  • 100B tokens: ~48 hours = $960 (faster!)
  • 10B tokens: ~5 hours = $100
```

---

## 2. Initial Setup

### Step 2.1: Clone Repository

```bash
# Navigate to your projects directory
cd ~/projects

# Clone the repository
git clone https://github.com/yourusername/gpt-oss-MoE-8B.git
cd gpt-oss-MoE-8B

# Verify directory structure
ls -la
# Expected: model/ data/ scripts/ docs/ configs/ README.md
```

### Step 2.2: Create Python Environment

#### Option A: Using venv (Recommended)

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Option B: Using conda

```bash
# Create conda environment
conda create -n gpt-moe python=3.10 -y

# Activate environment
conda activate gpt-moe
```

### Step 2.3: Install PyTorch with CUDA

```bash
# For CUDA 12.1 (A100, RTX 4090)
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Expected output:
# PyTorch: 2.3.0+cu121
# CUDA available: True
# CUDA version: 12.1
```

### Step 2.4: Install Core Dependencies

```bash
# Install required packages
pip install \
  tiktoken==0.7.0 \
  wandb==0.17.0 \
  pyyaml==6.0 \
  numpy==1.26.4 \
  tqdm==4.66.4 \
  safetensors==0.4.3

# Or install from requirements.txt
pip install -r requirements.txt
```

### Step 2.5: Install FlashAttention (Optional but Recommended)

```bash
# For A100 / RTX 4090 (FlashAttention 2)
pip install flash-attn --no-build-isolation

# For H100 (FlashAttention 3 - CUDA 12.3+ required)
pip install flash-attn>=2.5.0 --no-build-isolation

# Verify installation
python -c "import flash_attn; print(f'FlashAttention version: {flash_attn.__version__}')"
```

**Note**: FlashAttention provides ~2-3x speedup for attention computation. If installation fails, training will fall back to standard PyTorch attention (slower but functional).

### Step 2.6: Verify GPU Setup

```bash
# Check GPUs
nvidia-smi

# Test PyTorch GPU access
python << EOF
import torch

# Check GPU count
print(f"Number of GPUs: {torch.cuda.device_count()}")

# List all GPUs
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

# Test tensor creation on GPU
x = torch.randn(1000, 1000, device='cuda')
print(f"\n✓ GPU test successful!")
EOF
```

### Step 2.7: Setup Weights & Biases (Optional)

```bash
# Install wandb (already in requirements.txt)
pip install wandb

# Login to W&B
wandb login

# Paste your API key from: https://wandb.ai/authorize
```

---

## 3. Data Preparation

### Option A: Use Existing FineWeb-EDU Data (Quick Start)

```bash
# This project includes a reference to pre-tokenized FineWeb-EDU data
# Located at: ../gpt-oss-pretrain/build-nanogpt/edu_fineweb10B_o200k

# Verify data exists
ls -lh ../gpt-oss-pretrain/build-nanogpt/edu_fineweb10B_o200k/

# Expected files:
# *_train_*.npy  (training shards)
# *_val_*.npy    (validation shards)
# meta.json      (tokenizer metadata)
```

### Option B: Prepare Your Own Data

#### Step B.1: Collect Raw Text Data

```bash
# Create data directory
mkdir -p data/raw

# Download FineWeb-EDU sample (example)
python << EOF
from datasets import load_dataset

# Download 10B token sample
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")

# Save to text files
with open("data/raw/fineweb_edu.txt", "w", encoding="utf-8") as f:
    for i, example in enumerate(ds):
        f.write(example["text"] + "\n\n")
        if i % 10000 == 0:
            print(f"Processed {i:,} documents...")

print("✓ Data download complete!")
EOF
```

#### Step B.2: Tokenize Data

```bash
# Create tokenization script
cat > scripts/prepare_data.py << 'EOF'
#!/usr/bin/env python3
"""Tokenize raw text data for training"""

import argparse
import glob
import json
import numpy as np
import os
import tiktoken
from tqdm import tqdm

def tokenize_file(input_file, tokenizer, output_dir, split="train"):
    """Tokenize a single file"""
    print(f"Tokenizing {input_file}...")

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Tokenize
    tokens = tokenizer.encode(text)
    tokens_np = np.array(tokens, dtype=np.uint32)

    # Save
    output_file = os.path.join(output_dir, f"tokens_{split}.npy")
    np.save(output_file, tokens_np)

    return len(tokens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="o200k_base")
    parser.add_argument("--train_split", type=float, default=0.95)
    args = parser.parse_args()

    # Load tokenizer
    enc = tiktoken.get_encoding(args.tokenizer)
    vocab_size = enc.n_vocab

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Tokenize all files
    input_files = glob.glob(os.path.join(args.input_dir, "*.txt"))
    all_tokens = []

    for input_file in tqdm(input_files):
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = enc.encode(text)
        all_tokens.extend(tokens)

    # Convert to numpy
    all_tokens = np.array(all_tokens, dtype=np.uint32)

    # Split train/val
    split_idx = int(len(all_tokens) * args.train_split)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    # Save
    train_file = os.path.join(args.output_dir, "tokens_train.npy")
    val_file = os.path.join(args.output_dir, "tokens_val.npy")

    np.save(train_file, train_tokens)
    np.save(val_file, val_tokens)

    # Save metadata
    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": vocab_size,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
    }

    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Tokenization complete!")
    print(f"  Train: {len(train_tokens):,} tokens ({len(train_tokens)/1e9:.2f}B)")
    print(f"  Val:   {len(val_tokens):,} tokens ({len(val_tokens)/1e9:.2f}B)")
    print(f"  Vocab: {vocab_size:,}")

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/prepare_data.py

# Run tokenization
python scripts/prepare_data.py \
  --input_dir data/raw \
  --output_dir data/tokenized \
  --tokenizer o200k_base \
  --train_split 0.95
```

#### Step B.3: Verify Tokenized Data

```bash
# Check output files
ls -lh data/tokenized/

# Expected:
# tokens_train.npy  (e.g., 10GB for 10B tokens)
# tokens_val.npy    (e.g., 500MB for 500M tokens)
# meta.json

# Inspect metadata
cat data/tokenized/meta.json

# Test loading
python << EOF
import numpy as np
import json

# Load metadata
with open("data/tokenized/meta.json") as f:
    meta = json.load(f)
print(f"Tokenizer: {meta['tokenizer']}")
print(f"Vocab size: {meta['vocab_size']:,}")
print(f"Train tokens: {meta['train_tokens']:,}")
print(f"Val tokens: {meta['val_tokens']:,}")

# Load first 100 tokens
tokens = np.load("data/tokenized/tokens_train.npy")
print(f"\nFirst 10 tokens: {tokens[:10]}")
print(f"✓ Data loads successfully!")
EOF
```

---

## 4. Pre-Training Validation

### Run Validation Script

```bash
# Validate entire setup before training
python scripts/validate_setup.py \
  --data_dir data/tokenized

# This checks:
# ✓ Data format and tokenizer compatibility
# ✓ Model can build and initialize
# ✓ Forward pass works
# ✓ Backward pass works
# ✓ CUDA memory is sufficient
# ✓ Router aux loss configuration

# Expected output:
# [validate] Loading tokenizer...
# [validate] Tokenizer: o200k_base, vocab=201088
# [validate] Building model...
# [validate] Model parameters: 8.0B (2.2B active/token)
# [validate] Testing forward pass...
# [validate] Testing backward pass...
# [validate] ✓ All validation checks passed!
```

### Quick GPU Memory Test

```bash
# Test if your GPU can fit the model
python << EOF
import torch
from model.model import Transformer, gpt_oss_moe_8b_config

# Build model
cfg = gpt_oss_moe_8b_config()
cfg.vocab_size = 201088

device = "cuda"
model = Transformer(cfg).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params/1e9:.2f}B")

# Check memory
memory_gb = torch.cuda.memory_allocated() / 1e9
print(f"Model memory: {memory_gb:.2f} GB")

# Test forward pass
batch_size = 4
seq_len = 512
x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)

logits, outputs = model(x, labels=x)
print(f"Forward pass successful!")
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

print(f"\n✓ GPU memory test passed!")
EOF
```

---

## 5. Training Commands

### 5.1: Single GPU Training (Testing)

```bash
# Quick test run (10 iterations)
python model/train.py \
  --data_dir data/tokenized \
  --out_dir out/test_run \
  --batch_size 2 \
  --block_size 512 \
  --max_iters 10 \
  --grad_accum_steps 1 \
  --eval_interval -1 \
  --sample_every -1 \
  --save_every -1 \
  --log_interval 1 \
  --dtype bfloat16

# Expected output:
# [train] Building model on meta device...
# [params] Total: 8.00B parameters
# [train] Wrapping model with FSDP...
# [train] Starting training...
# iter 000000 | loss 10.8234 | lr 1.50e-04 | norm 4.2341 | dt 2145ms | tok/s 487
# iter 000001 | loss 10.7123 | lr 3.00e-04 | norm 3.9876 | dt 2032ms | tok/s 503
# ...
```

### 5.2: Multi-GPU Training (8x GPUs - Production)

```bash
# Full production training command
torchrun \
  --standalone \
  --nproc_per_node=8 \
  model/train.py \
  --data_dir data/tokenized \
  --out_dir out/8b_moe_run1 \
  --model_config 8b \
  --batch_size 4 \
  --block_size 4096 \
  --max_iters 50000 \
  --grad_accum_steps 32 \
  --lr 3e-4 \
  --min_lr 3e-5 \
  --weight_decay 0.1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --grad_clip 1.0 \
  --warmup_iters 2000 \
  --lr_decay_iters 50000 \
  --dtype bfloat16 \
  --save_every 1000 \
  --keep_last_n 5 \
  --eval_interval 500 \
  --eval_iters 100 \
  --sample_every 500 \
  --sample_tokens 200 \
  --temperature 0.8 \
  --top_k 200 \
  --log_interval 10 \
  --log_router_stats \
  --seed 1337

# Training metrics:
# Effective batch: 4 × 4096 × 32 × 8 = 4.2M tokens per update
# Expected throughput: ~12K-16K tokens/sec (8x A100)
# Training time: ~7-14 days for 50K iterations (210B tokens)
```

### 5.3: H100 NVL Optimized Training (8x H100)

```bash
# Optimized for 8x H100 NVL 94GB
torchrun \
  --standalone \
  --nproc_per_node=8 \
  model/train.py \
  --data_dir data/tokenized \
  --out_dir out/8b_moe_h100_run1 \
  --batch_size 12 \
  --block_size 4096 \
  --max_iters 50000 \
  --grad_accum_steps 12 \
  --lr 3e-4 \
  --min_lr 3e-5 \
  --weight_decay 0.1 \
  --grad_clip 1.0 \
  --warmup_iters 2000 \
  --lr_decay_iters 50000 \
  --dtype bfloat16 \
  --save_every 1000 \
  --keep_last_n 5 \
  --eval_interval 500 \
  --eval_iters 100 \
  --sample_every 500 \
  --log_interval 10 \
  --log_router_stats \
  --wandb \
  --wandb_project gpt-oss-8b-h100 \
  --seed 1337

# Training metrics:
# Effective batch: 12 × 4096 × 12 × 8 = 4.7M tokens per update
# Expected throughput: ~25K-30K tokens/sec (8x H100)
# Training time: ~3-5 days for 50K iterations (235B tokens)
```

### 5.4: Long Context Training (8K tokens)

```bash
# Train with 8K context window
torchrun \
  --standalone \
  --nproc_per_node=8 \
  model/train.py \
  --data_dir data/tokenized \
  --out_dir out/8b_moe_longctx \
  --batch_size 2 \
  --block_size 8192 \
  --grad_accum_steps 64 \
  --max_iters 50000 \
  --dtype bfloat16 \
  --save_every 1000 \
  --eval_interval 500 \
  --log_interval 10 \
  --log_router_stats

# Note: Longer context requires more memory
# Batch size reduced from 4 to 2
# Grad accum increased to maintain effective batch size
```

### 5.5: Resume from Checkpoint

```bash
# Training automatically resumes if checkpoint exists
torchrun \
  --standalone \
  --nproc_per_node=8 \
  model/train.py \
  --data_dir data/tokenized \
  --out_dir out/8b_moe_run1 \
  ... [same args as before]

# Output will show:
# [train] Resuming from out/8b_moe_run1/ckpt_rank00000.pt
# [train] Resumed at iter 5000 (best val 5.2345)
# [train] Starting training...
# iter 005001 | ...
```

### 5.6: Background Training with tmux

```bash
# Create tmux session for long-running training
tmux new-session -s training

# Inside tmux, activate environment and start training
cd ~/projects/gpt-oss-MoE-8B
source venv/bin/activate

torchrun --standalone --nproc_per_node=8 model/train.py \
  ... [full args] \
  2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

# Detach from tmux: Ctrl+B, then D

# Reattach later:
tmux attach-session -t training

# List sessions:
tmux ls

# Kill session (after training completes):
tmux kill-session -t training
```

---

## 6. Monitoring & Logging

### 6.1: Real-Time Log Monitoring

```bash
# Watch training logs in real-time
tail -f logs/training_*.log

# Or with color highlighting
tail -f logs/training_*.log | grep --color=auto -E "loss|ERROR|WARNING|✓"

# Expected output:
# iter 001000 | loss 5.2345 | lr 3.00e-04 | norm 0.8542 | dt 542ms | tok/s 12340
# iter 001010 | loss 5.2123 | lr 3.00e-04 | norm 0.8234 | dt 538ms | tok/s 12450
# [eval] iter 001500 | val_loss 5.3456 | val_ppl 209.87
```

### 6.2: GPU Monitoring

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Compact view with key metrics
watch -n 1 'nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits'

# Expected output (per GPU):
# 0, NVIDIA A100-SXM4-80GB, 98, 95, 76234, 81920, 68, 380

# Log GPU stats to file
nvidia-smi dmon -s pucmt -o TD > logs/gpu_stats.log &
# Kill with: pkill nvidia-smi
```

### 6.3: Training Progress Tracker

```bash
# Create progress tracking script
cat > scripts/track_progress.py << 'EOF'
#!/usr/bin/env python3
import re
import sys
from datetime import datetime, timedelta

def track_progress(log_file, max_iters=50000):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Find latest iteration
    latest_iter = 0
    latest_loss = 0
    latest_throughput = 0

    for line in reversed(lines):
        match = re.search(r'iter (\d+) \| loss ([\d.]+).*tok/s ([\d.]+)', line)
        if match:
            latest_iter = int(match.group(1))
            latest_loss = float(match.group(2))
            latest_throughput = float(match.group(3))
            break

    # Calculate progress
    progress = (latest_iter / max_iters) * 100
    tokens_processed = latest_iter * 4194304  # 4.2M tokens per iter

    # Time estimates
    if latest_iter > 0 and latest_throughput > 0:
        remaining_iters = max_iters - latest_iter
        remaining_tokens = remaining_iters * 4194304
        remaining_hours = remaining_tokens / latest_throughput / 3600

        print("=" * 60)
        print("TRAINING PROGRESS")
        print("=" * 60)
        print(f"\nCurrent Status:")
        print(f"  Iteration:     {latest_iter:,} / {max_iters:,} ({progress:.1f}%)")
        print(f"  Loss:          {latest_loss:.4f}")
        print(f"  Throughput:    {latest_throughput:,.0f} tokens/sec")
        print(f"  Tokens:        {tokens_processed/1e9:.2f}B processed")
        print(f"\nTime Estimate:")
        print(f"  Remaining:     {remaining_hours:.1f} hours ({remaining_hours/24:.1f} days)")
        print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python track_progress.py <log_file>")
        sys.exit(1)
    track_progress(sys.argv[1])
EOF

chmod +x scripts/track_progress.py

# Run progress tracker
python scripts/track_progress.py logs/training_*.log

# Expected output:
# ============================================================
# TRAINING PROGRESS
# ============================================================
#
# Current Status:
#   Iteration:     5,000 / 50,000 (10.0%)
#   Loss:          5.2345
#   Throughput:    12,340 tokens/sec
#   Tokens:        21.0B processed
#
# Time Estimate:
#   Remaining:     156.8 hours (6.5 days)
# ============================================================
```

### 6.4: Weights & Biases Dashboard

```bash
# If using --wandb flag, access your dashboard at:
# https://wandb.ai/<your-username>/gpt-oss-8b-moe

# Key metrics to monitor:
# • train/loss - Should decrease steadily
# • train/perplexity - Should decrease (exp(loss))
# • eval/val_loss - Should track train loss closely
# • train/learning_rate - Should follow warmup → decay schedule
# • train/grad_norm - Should be < 1.0 (due to clipping)
# • train/router_aux_loss - Should be < 0.001
# • system/tokens_per_sec - Should be stable
# • system/gpu_memory_gb - Monitor for OOM issues
```

### 6.5: Check Training Metrics

```bash
# Parse recent loss values
tail -100 logs/training_*.log | grep -oP 'loss \K[\d.]+' | tail -10

# Check for NaN/Inf issues
tail -1000 logs/training_*.log | grep -i "nan\|inf\|error"

# Count evaluation runs
grep -c "\[eval\]" logs/training_*.log

# Check router aux loss trends
grep "router_aux" logs/training_*.log | tail -20
```

---

## 7. Important Runtime Commands

### 7.1: Checkpoint Management

```bash
# List checkpoints
ls -lh out/8b_moe_run1/ckpt_*.pt

# Expected (for 8 GPUs):
# ckpt_rank00000.pt  ~5-6GB
# ckpt_rank00001.pt  ~5-6GB
# ...
# ckpt_rank00007.pt  ~5-6GB

# Check checkpoint metadata
python << EOF
import torch

ckpt = torch.load("out/8b_moe_run1/ckpt_rank00000.pt", map_location="cpu", weights_only=False)

print(f"Iteration: {ckpt['iter_num']}")
print(f"Best val loss: {ckpt['best_val_loss']:.4f}")
print(f"Tokenizer: {ckpt['tokenizer']}")
print(f"Model config: {ckpt['model_config_dict']['num_hidden_layers']} layers")
EOF
```

### 7.2: Backup Checkpoints

```bash
# Backup specific checkpoint
ITER=10000
mkdir -p backups/iter_${ITER}
cp out/8b_moe_run1/ckpt_rank*.pt backups/iter_${ITER}/

# Verify backup
ls -lh backups/iter_${ITER}/

# Backup with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p backups/${TIMESTAMP}
cp out/8b_moe_run1/ckpt_rank*.pt backups/${TIMESTAMP}/
echo "Backed up to: backups/${TIMESTAMP}"
```

### 7.3: Clean Old Checkpoints

```bash
# Training automatically keeps only last N checkpoints (--keep_last_n 5)
# To manually clean:

# List checkpoint sizes
du -sh out/8b_moe_run1/ckpt_*.pt | sort -h

# Delete specific old checkpoint (example: from old iteration)
# Be careful! Only delete if you have backups
# rm out/8b_moe_run1/ckpt_rank*.pt.old
```

### 7.4: Emergency Checkpoint Save

```bash
# If training is running, press Ctrl+C to trigger graceful shutdown
# The signal handler will save a checkpoint before exiting

# Monitor for checkpoint save:
tail -f logs/training_*.log

# You'll see:
# [train] Interrupted! Saving checkpoint...
# [ckpt] Saved checkpoint at iter 12345 (5.2 MB) to out/8b_moe_run1/ckpt_rank*.pt
```

### 7.5: Disk Space Monitoring

```bash
# Check available disk space
df -h .

# Check training directory size
du -sh out/8b_moe_run1

# Check data directory size
du -sh data/

# Monitor disk space continuously
watch -n 60 'df -h . | grep -v "Filesystem"'
```

### 7.6: Process Management

```bash
# Find training processes
ps aux | grep "train.py"

# Check GPU processes
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv

# Kill training gracefully (use Ctrl+C in tmux instead when possible)
pkill -SIGINT -f "train.py"

# Force kill (only if graceful shutdown fails)
pkill -9 -f "train.py"
```

---

## 8. Post-Training

### 8.1: Export Checkpoint to SafeTensors

```bash
# Export final checkpoint to HuggingFace format
torchrun \
  --nproc_per_node=8 \
  scripts/export_to_safetensors.py \
  --in_dir out/8b_moe_run1 \
  --ckpt_prefix ckpt \
  --max_shard_size 5GB \
  --release_dir ./release/moe-8b-final

# This creates:
# release/moe-8b-final/
#   ├── model-00001-of-00004.safetensors
#   ├── model-00002-of-00004.safetensors
#   ├── model-00003-of-00004.safetensors
#   ├── model-00004-of-00004.safetensors
#   ├── model.safetensors.index.json
#   └── config.json
```

### 8.2: Test Inference

```bash
# Create inference test script
cat > scripts/test_inference.py << 'EOF'
#!/usr/bin/env python3
import argparse
import torch
import tiktoken
from model.model import Transformer

@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=200, temperature=0.8, top_k=200):
    model.eval()
    device = next(model.parameters()).device

    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor([tokens], dtype=torch.long, device=device)

    # Generate
    for _ in range(max_tokens):
        logits, _ = model(tokens)
        next_logits = logits[:, -1, :] / temperature

        # Top-k
        if top_k > 0:
            v, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < v[:, [-1]]] = -float('inf')

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokenizer.decode(tokens[0].tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_tokens", type=int, default=200)
    args = parser.parse_args()

    print("[inference] Loading model...")

    # Load checkpoint (single rank for inference)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Build model
    from model.model import ModelConfig
    cfg = ModelConfig(**ckpt["model_config_dict"])
    model = Transformer(cfg)

    # Load weights (this assumes full checkpoint, not sharded)
    model.load_state_dict(ckpt["model_state_dict"])

    # Move to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = tiktoken.get_encoding(ckpt.get("tokenizer", "o200k_base"))

    print(f"[inference] Model: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")
    print(f"[inference] Prompt: '{args.prompt}'")
    print(f"[inference] Generating...\n")

    output = generate(model, tokenizer, args.prompt, args.max_tokens)
    print(output)
EOF

chmod +x scripts/test_inference.py

# Note: This requires merging sharded checkpoints first
# Use export_to_safetensors.py to create a full checkpoint
```

### 8.3: Evaluate on Benchmarks

```bash
# Evaluate on HellaSwag (if enabled during training)
# Or run standalone evaluation:

python scripts/evaluate_hellaswag.py \
  --checkpoint out/8b_moe_run1/ckpt_rank00000.pt \
  --num_examples 1000 \
  --batch_size 8

# Expected output:
# [hellaswag] Evaluating on 1,000 examples...
# [hellaswag] Accuracy: 0.4523 (452/1000)
```

### 8.4: Upload to HuggingFace Hub

```bash
# Install huggingface_hub
pip install huggingface-hub

# Login
huggingface-cli login

# Upload model
python << EOF
from huggingface_hub import HfApi

api = HfApi()

api.create_repo(
    repo_id="your-username/gpt-oss-moe-8b",
    exist_ok=True,
    private=False
)

api.upload_folder(
    folder_path="release/moe-8b-final",
    repo_id="your-username/gpt-oss-moe-8b",
    repo_type="model"
)

print("✓ Model uploaded to HuggingFace!")
print("https://huggingface.co/your-username/gpt-oss-moe-8b")
EOF
```

---

## 9. Troubleshooting Commands

### 9.1: Out of Memory (OOM)

```bash
# Check current GPU memory usage
nvidia-smi

# Check PyTorch memory allocation
python << EOF
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
EOF

# Solutions:
# 1. Reduce batch size
torchrun ... --batch_size 2  # Down from 4

# 2. Reduce context length
torchrun ... --block_size 2048  # Down from 4096

# 3. Increase gradient accumulation (maintains effective batch size)
torchrun ... --grad_accum_steps 64  # Up from 32

# 4. Use float16 instead of bfloat16 (saves memory, may be less stable)
torchrun ... --dtype float16
```

### 9.2: Training Divergence (NaN Loss)

```bash
# Check for NaN/Inf in logs
grep -i "nan\|inf" logs/training_*.log

# Solutions:
# 1. Reduce learning rate
torchrun ... --lr 1e-4 --min_lr 1e-5

# 2. Increase warmup
torchrun ... --warmup_iters 5000

# 3. Reduce gradient clipping threshold
torchrun ... --grad_clip 0.5

# 4. Check router aux loss coefficient (should be ~0.005 for 8 experts)
# Edit model/model.py if needed

# 5. Verify data quality
python << EOF
import numpy as np
tokens = np.load("data/tokenized/tokens_train.npy")
print(f"Token range: {tokens.min()} - {tokens.max()}")
print(f"Expected vocab size: 201088")
if tokens.max() >= 201088:
    print("⚠️  WARNING: Tokens exceed vocab size!")
EOF
```

### 9.3: Slow Training

```bash
# 1. Check GPU utilization (should be >90%)
nvidia-smi dmon -s u

# 2. Check I/O wait time
iostat -x 1 10

# 3. Profile training
python -m torch.utils.bottleneck model/train.py ... [args]

# 4. Verify data is on fast SSD (not HDD or network mount)
df -Th data/tokenized

# Solutions:
# 1. Increase batch size if memory allows
torchrun ... --batch_size 8

# 2. Move data to local SSD
cp -r /network/data/ /local/ssd/data/

# 3. Reduce evaluation frequency
torchrun ... --eval_interval 1000  # Up from 500

# 4. Enable torch.compile (experimental, requires PyTorch 2.0+)
torchrun ... --compile
```

### 9.4: Distributed Training Hangs

```bash
# 1. Enable NCCL debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 2. Increase timeout
# Edit model/train.py line ~440:
# timeout = datetime.timedelta(minutes=120)  # Up from 60

# 3. Verify all GPUs are visible
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# 4. Test NCCL communication
python << EOF
import torch
import torch.distributed as dist
import os

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

dist.init_process_group(backend='nccl')
print("✓ NCCL initialized successfully!")
dist.destroy_process_group()
EOF

# 5. Try gloo backend (slower but more stable for debugging)
# Edit model/train.py line ~441:
# dist.init_process_group(backend="gloo", timeout=timeout)
```

### 9.5: Router Load Imbalance

```bash
# Check router statistics in logs
grep "router_aux\|load_balance" logs/training_*.log | tail -20

# High router_aux_loss (>0.01) indicates poor load balancing

# Solutions:
# 1. Increase router aux loss coefficient
# Edit model/model.py, gpt_oss_moe_8b_config():
# router_aux_loss_coef = 0.01  # Up from 0.005

# 2. Train longer (load balancing improves over time)

# 3. Check expert utilization in evaluation
# Should see roughly equal usage across experts in W&B dashboard
```

### 9.6: Checkpoint Corruption

```bash
# Verify checkpoint integrity
python << EOF
import torch

try:
    ckpt = torch.load("out/8b_moe_run1/ckpt_rank00000.pt", map_location="cpu", weights_only=False)
    print(f"✓ Checkpoint loads successfully")
    print(f"  Iteration: {ckpt['iter_num']}")
    print(f"  Keys: {list(ckpt.keys())}")
except Exception as e:
    print(f"✗ Checkpoint corrupted: {e}")
EOF

# If corrupted, restore from backup
cp backups/iter_10000/ckpt_rank*.pt out/8b_moe_run1/

# Resume training from backup
torchrun ... --out_dir out/8b_moe_run1 ...
```

---

## 10. Complete CLI Reference

### Training Script Arguments

```bash
# Full argument list for model/train.py

# ============================================================
# DATA & OUTPUT
# ============================================================
--data_dir <path>              # Directory with tokenized data (REQUIRED)
--out_dir <path>               # Output directory (default: out/8b_moe_run1)
--model_config <str>           # Model config: 8b (default: 8b)

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
--batch_size <int>             # Micro batch size per GPU (default: 4)
--block_size <int>             # Context length (default: 4096, max: 131072)
--max_iters <int>              # Max training iterations (default: 50000)
--grad_accum_steps <int>       # Gradient accumulation steps (default: 32)

# Effective batch size = batch_size × block_size × grad_accum_steps × num_gpus

# ============================================================
# OPTIMIZATION
# ============================================================
--lr <float>                   # Peak learning rate (default: 3e-4)
--min_lr <float>               # Minimum learning rate (default: 3e-5)
--weight_decay <float>         # L2 regularization (default: 0.1)
--beta1 <float>                # Adam beta1 (default: 0.9)
--beta2 <float>                # Adam beta2 (default: 0.95)
--grad_clip <float>            # Gradient clipping norm (default: 1.0)

# ============================================================
# LEARNING RATE SCHEDULE
# ============================================================
--warmup_iters <int>           # Linear warmup iterations (default: 2000)
--lr_decay_iters <int>         # Cosine decay iterations (default: 50000)

# ============================================================
# MIXED PRECISION
# ============================================================
--dtype <str>                  # Training dtype: bfloat16|float16|float32
                               # (default: bfloat16)
                               # bfloat16: Best for A100/H100
                               # float16: Faster but less stable
                               # float32: Slowest but most stable

# ============================================================
# CHECKPOINTING
# ============================================================
--save_every <int>             # Save checkpoint every N iters (default: 1000)
--keep_last_n <int>            # Keep only last N checkpoints (default: 5)

# ============================================================
# EVALUATION
# ============================================================
--eval_interval <int>          # Evaluate every N iters (default: 500)
                               # Set to -1 to disable
--eval_iters <int>             # Number of eval iterations (default: 100)
--eval_hellaswag               # Enable HellaSwag evaluation (flag)
--eval_hellaswag_interval <int> # HellaSwag eval interval (default: 5000)
--eval_hellaswag_num_examples <int> # Number of examples (default: 1000)

# ============================================================
# TEXT SAMPLING
# ============================================================
--sample_every <int>           # Generate samples every N iters (default: 500)
--sample_tokens <int>          # Tokens to generate per sample (default: 200)
--temperature <float>          # Sampling temperature (default: 0.8)
--top_k <int>                  # Top-k sampling (default: 200)

# ============================================================
# LOGGING
# ============================================================
--log_interval <int>           # Log every N iters (default: 10)
--log_router_stats             # Log router aux loss (flag, default: true)

# ============================================================
# WEIGHTS & BIASES
# ============================================================
--wandb                        # Enable W&B logging (flag)
--wandb_project <str>          # W&B project name (default: gpt-oss-8b-moe)
--wandb_run_name <str>         # W&B run name (auto-generated if not set)
--wandb_entity <str>           # W&B entity/team name
--wandb_notes <str>            # Notes for this run
--wandb_tags <str> [<str> ...] # Tags for this run

# ============================================================
# SYSTEM
# ============================================================
--seed <int>                   # Random seed (default: 1337)
--compile                      # Use torch.compile (flag, experimental)
```

### Common Configuration Templates

#### Configuration 1: Quick Test (Single GPU)
```bash
python model/train.py \
  --data_dir data/tokenized \
  --out_dir out/test \
  --batch_size 1 \
  --block_size 512 \
  --max_iters 10 \
  --grad_accum_steps 1 \
  --eval_interval -1 \
  --sample_every -1 \
  --save_every -1 \
  --log_interval 1
```

#### Configuration 2: Production (8x A100 80GB)
```bash
torchrun --standalone --nproc_per_node=8 model/train.py \
  --data_dir data/tokenized \
  --out_dir out/8b_prod \
  --batch_size 4 \
  --block_size 4096 \
  --max_iters 50000 \
  --grad_accum_steps 32 \
  --dtype bfloat16 \
  --save_every 1000 \
  --keep_last_n 5 \
  --eval_interval 500 \
  --sample_every 500 \
  --log_router_stats \
  --wandb
```

#### Configuration 3: H100 Optimized (8x H100 NVL)
```bash
torchrun --standalone --nproc_per_node=8 model/train.py \
  --data_dir data/tokenized \
  --out_dir out/8b_h100 \
  --batch_size 12 \
  --block_size 4096 \
  --max_iters 50000 \
  --grad_accum_steps 12 \
  --dtype bfloat16 \
  --save_every 1000 \
  --eval_interval 500 \
  --wandb \
  --wandb_project gpt-oss-h100
```

#### Configuration 4: Long Context (8K)
```bash
torchrun --standalone --nproc_per_node=8 model/train.py \
  --data_dir data/tokenized \
  --out_dir out/8b_longctx \
  --batch_size 2 \
  --block_size 8192 \
  --grad_accum_steps 64 \
  --max_iters 50000 \
  --dtype bfloat16
```

#### Configuration 5: Memory-Constrained (4x RTX 4090)
```bash
torchrun --standalone --nproc_per_node=4 model/train.py \
  --data_dir data/tokenized \
  --out_dir out/8b_budget \
  --batch_size 1 \
  --block_size 2048 \
  --grad_accum_steps 128 \
  --dtype bfloat16 \
  --save_every 2000
```

---

## 11. Advanced Topics

### 11.1: Multi-Node Training

```bash
# For training across multiple machines with 8 GPUs each

# On master node (192.168.1.100):
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=192.168.1.100 \
  --master_port=29500 \
  model/train.py \
  --data_dir data/tokenized \
  --out_dir out/8b_multinode \
  ... [other args]

# On worker node (192.168.1.101):
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=1 \
  --master_addr=192.168.1.100 \
  --master_port=29500 \
  model/train.py \
  --data_dir data/tokenized \
  --out_dir out/8b_multinode \
  ... [other args]

# Total: 16 GPUs across 2 nodes
# Effective batch = batch_size × block_size × grad_accum × 16
```

### 11.2: Custom Learning Rate Schedules

```python
# Edit model/train.py get_lr() function (lines 196-210)
# Example: Add restarts to cosine schedule

def get_lr_with_restarts(it: int, args, num_restarts=3) -> float:
    """Cosine with warm restarts"""
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters

    # Divide training into cycles
    cycle_length = (args.lr_decay_iters - args.warmup_iters) // num_restarts
    cycle = (it - args.warmup_iters) // cycle_length
    cycle_it = (it - args.warmup_iters) % cycle_length

    # Cosine within cycle
    decay_ratio = cycle_it / cycle_length
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)
```

### 11.3: Gradient Checkpointing (Save Memory)

```python
# Edit model/model.py TransformerBlock class
# Add gradient checkpointing to reduce memory

from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x, freqs_cis, mask=None):
        # Use gradient checkpointing
        attn_out = checkpoint(
            self.attention,
            self.ln1(x),
            freqs_cis,
            mask,
            use_reentrant=False
        )
        x = x + attn_out

        moe_out, aux_loss = checkpoint(
            self.feed_forward,
            self.ln2(x),
            use_reentrant=False
        )
        x = x + moe_out

        return x, aux_loss
```

### 11.4: Custom Data Loaders

```python
# Create custom data loader for your data format
# Example: data/custom_loader.py

import torch
import numpy as np

class CustomDataLoader:
    def __init__(self, data_path, batch_size, block_size, rank, world_size):
        self.B = batch_size
        self.T = block_size
        self.rank = rank
        self.world_size = world_size

        # Load your custom data format
        self.data = self.load_data(data_path)
        self.pos = rank * batch_size * block_size

    def load_data(self, path):
        # Implement your data loading logic
        # Must return uint32 numpy array of token IDs
        pass

    def get_batch(self):
        buf = self.data[self.pos : self.pos + self.B * self.T + 1]
        x = torch.from_numpy(buf[:-1]).view(self.B, self.T)
        y = torch.from_numpy(buf[1:]).view(self.B, self.T)

        self.pos += self.B * self.T * self.world_size
        if self.pos + self.B * self.T + 1 > len(self.data):
            self.pos = self.rank * self.B * self.T

        return x, y
```

### 11.5: Model Architecture Modifications

```python
# Edit model/model.py gpt_oss_moe_8b_config()
# Example: Change number of experts

def gpt_oss_moe_8b_config() -> ModelConfig:
    return ModelConfig(
        # ... other params

        # Increase experts to 16 (from 8)
        num_local_experts=16,
        experts_per_token=2,

        # Adjust router aux loss for 16 experts
        # Rule: ~0.001 * sqrt(num_experts)
        router_aux_loss_coef=0.004,  # ~0.001 * sqrt(16)
    )
```

### 11.6: Hyperparameter Sweeps with W&B

```bash
# Create sweep configuration
cat > configs/sweep.yaml << 'EOF'
program: model/train.py
method: bayes
metric:
  name: eval/val_loss
  goal: minimize
parameters:
  lr:
    distribution: log_uniform_values
    min: 1e-4
    max: 5e-4
  weight_decay:
    values: [0.05, 0.1, 0.2]
  grad_accum_steps:
    values: [16, 32, 64]
  warmup_iters:
    values: [1000, 2000, 4000]
EOF

# Initialize sweep
wandb sweep configs/sweep.yaml

# Run sweep agents (each will try different hyperparameters)
wandb agent <sweep_id>
```

---

## Quick Reference Card

```
┌──────────────────────────────────────────────────────────────┐
│                    QUICK COMMAND REFERENCE                    │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│ Setup:                                                         │
│   pip install -r requirements.txt                             │
│   python scripts/validate_setup.py --data_dir <path>         │
│                                                                │
│ Training:                                                      │
│   Single GPU:   python model/train.py ...                    │
│   Multi GPU:    torchrun --nproc_per_node=8 model/train.py   │
│                                                                │
│ Monitoring:                                                    │
│   Logs:         tail -f logs/training_*.log                   │
│   GPUs:         watch -n 1 nvidia-smi                         │
│   Progress:     python scripts/track_progress.py <log>       │
│                                                                │
│ Checkpoints:                                                   │
│   List:         ls -lh out/*/ckpt_*.pt                        │
│   Backup:       cp out/*/ckpt_*.pt backups/                   │
│   Resume:       [automatic when out_dir has checkpoints]     │
│                                                                │
│ Export:                                                        │
│   SafeTensors:  torchrun --nproc_per_node=8 \                │
│                   scripts/export_to_safetensors.py ...        │
│                                                                │
│ Troubleshooting:                                               │
│   OOM:          Reduce --batch_size or --block_size          │
│   NaN loss:     Reduce --lr or increase --warmup_iters       │
│   Slow:         Check GPU util, I/O, increase --batch_size   │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

---

## Support & Resources

- **Documentation**: See `docs/` directory for detailed guides
- **Issues**: Report bugs on GitHub Issues
- **Discord**: Join community discussion (if available)
- **W&B Workspace**: Share training runs and compare results

---

## Conclusion

This guide covers **everything needed** to train GPT-OSS-MoE-8B from scratch. Key points:

✅ **Hardware**: 8x A100/H100 recommended, 1x A100 minimum
✅ **Data**: 10B+ tokens tokenized with o200k_base
✅ **Training**: 3-14 days depending on hardware
✅ **Cost**: $100-2,000 for full pretraining run
✅ **Result**: Production-ready 8B MoE model

**Next Steps**:
1. Run validation: `python scripts/validate_setup.py`
2. Start small: Test with single GPU first
3. Scale up: Launch multi-GPU production training
4. Monitor: Watch logs and W&B dashboard
5. Iterate: Adjust hyperparameters based on results

Happy training! 🚀


# Training Command Reference

## Single GPU Training (Testing/Development)

```bash
cd /workspace/gpt-oss-MoE-8B
source .venv/bin/activate

python model/train.py \
  --data_dir data/edu_fineweb_10BT \
  --out_dir out/8b_moe_run1 \
  --batch_size 2 \
  --block_size 2048 \
  --max_iters 1000 \
  --grad_accum_steps 32 \
  --eval_interval 100 \
  --save_every 500 \
  --log_interval 1
```

## Multi-GPU Training (Production - 8 GPUs)

```bash
cd /workspace/gpt-oss-MoE-8B
source .venv/bin/activate

torchrun --standalone --nproc_per_node=8 model/train.py \
  --data_dir data/edu_fineweb_10BT \
  --out_dir out/8b_moe_run1 \
  --batch_size 4 \
  --block_size 4096 \
  --max_iters 20000 \
  --grad_accum_steps 32 \
  --lr 3e-4 \
  --min_lr 3e-5 \
  --warmup_iters 2000 \
  --lr_decay_iters 20000 \
  --weight_decay 0.1 \
  --grad_clip 1.0 \
  --dtype bfloat16 \
  --eval_interval 500 \
  --eval_iters 100 \
  --save_every 1000 \
  --keep_last_n 5 \
  --sample_every 500 \
  --sample_tokens 200 \
  --log_interval 10 \
  --seed 1337
```

## With Weights & Biases Logging

```bash
cd /workspace/gpt-oss-MoE-8B
source .venv/bin/activate

torchrun --standalone --nproc_per_node=8 model/train.py \
  --data_dir data/edu_fineweb_10BT \
  --out_dir out/8b_moe_run1 \
  --batch_size 4 \
  --block_size 4096 \
  --max_iters 20000 \
  --grad_accum_steps 32 \
  --lr 3e-4 \
  --min_lr 3e-5 \
  --warmup_iters 2000 \
  --lr_decay_iters 20000 \
  --weight_decay 0.1 \
  --grad_clip 1.0 \
  --dtype bfloat16 \
  --eval_interval 500 \
  --eval_iters 100 \
  --save_every 1000 \
  --keep_last_n 5 \
  --sample_every 500 \
  --log_interval 10 \
  --wandb \
  --wandb_project gpt-oss-8b-moe \
  --wandb_run_name moe-8b-10BT-$(date +%Y%m%d-%H%M%S) \
  --seed 1337
```

## Quick Test Run (Verify Setup)

```bash
cd /workspace/gpt-oss-MoE-8B
source .venv/bin/activate

python model/train.py \
  --data_dir data/edu_fineweb_10BT \
  --out_dir out/test_run \
  --batch_size 2 \
  --block_size 1024 \
  --max_iters 10 \
  --grad_accum_steps 1 \
  --eval_interval -1 \
  --save_every -1 \
  --log_interval 1
```

## Parameter Explanations

### Required
- `--data_dir`: Directory with tokenized data (`data/edu_fineweb_10BT`)

### Training Configuration
- `--batch_size`: Micro batch size per GPU (default: 4)
- `--block_size`: Context length (default: 4096)
- `--max_iters`: Maximum training iterations
- `--grad_accum_steps`: Gradient accumulation steps (default: 32)

### Optimization
- `--lr`: Peak learning rate (default: 3e-4)
- `--min_lr`: Minimum learning rate (default: 3e-5)
- `--warmup_iters`: Warmup steps (default: 2000)
- `--lr_decay_iters`: LR decay iterations (default: 50000)
- `--weight_decay`: Weight decay (default: 0.1)
- `--grad_clip`: Gradient clipping (default: 1.0)

### System
- `--dtype`: Training dtype - `bfloat16`|`float16`|`float32` (default: bfloat16)
- `--seed`: Random seed (default: 1337)

### Checkpointing
- `--save_every`: Save checkpoint every N iters (default: 1000)
- `--keep_last_n`: Keep only last N checkpoints (default: 5)

### Evaluation & Logging
- `--eval_interval`: Evaluate every N iters (default: 500)
- `--eval_iters`: Number of eval iterations (default: 100)
- `--log_interval`: Log every N iters (default: 10)

### Monitoring
- `--wandb`: Enable Weights & Biases logging
- `--wandb_project`: W&B project name
- `--wandb_run_name`: W&B run name (auto-generated if not specified)

## For 8.3B Tokens (edu_fineweb_10BT)

With configuration:
- Batch size: 4 per GPU
- Block size: 4096
- Gradient accumulation: 32 steps
- 8 GPUs
- Effective batch: 4,194,304 tokens/iteration

**Recommended max_iters: ~2,000-2,500** iterations to process all tokens

