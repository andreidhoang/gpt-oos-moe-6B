# GPT-OSS-6B: Open-Source MoE Training Guide

Production-ready FSDP training setup for 8B MoE model with proper aux loss handling.

## Quick Start

### 1. Validate Setup

```bash
python scripts/validate_setup.py --data_dir ../gpt-oss-pretrain/build-nanogpt/edu_fineweb10B_o200k
```

This checks:

- ✅ Data format and tokenizer compatibility (o200k)
- ✅ Model can build and run forward/backward
- ✅ CUDA availability and memory
- ✅ Router aux loss configuration

### 2. Single GPU Test (Quick Check)

```bash
python train.py \
  --data_dir ../gpt-oss-pretrain/build-nanogpt/edu_fineweb10B_o200k \
  --out_dir out/test_run \
  --max_iters 10 \
  --log_interval 1 \
  --batch_size 2 \
  --block_size 512
```

### 3. Multi-GPU Training (Production)

```bash
# 8 GPUs
torchrun --standalone --nproc_per_node=8 train.py \
  --data_dir ../gpt-oss-pretrain/build-nanogpt/edu_fineweb10B_o200k \
  --out_dir out/8b_moe_run1 \
  --batch_size 4 \
  --block_size 4096 \
  --grad_accum_steps 32 \
  --max_iters 50000 \
  --dtype bfloat16
```

---

## Training Configuration

### Model Architecture

- **Parameters**: ~8B total, ~2.2B active per token
- **Experts**: 8 experts, top-2 routing
- **Attention**: GQA (64 query heads, 8 KV heads)
- **Context**: Up to 131K tokens (training at 4K-8K recommended)
- **Router Aux Loss**: Auto-adjusted to 0.005 (optimized for 8 experts)

### Default Training Settings

```python
# Batch configuration
batch_size = 4              # Per-GPU micro batch
block_size = 4096           # Context length
grad_accum_steps = 32       # Gradient accumulation
# Total batch = 4 × 4096 × 32 × N_GPUs tokens

# Optimization
lr = 3e-4                   # Peak learning rate
min_lr = 3e-5               # Min learning rate
weight_decay = 0.1          # Weight decay
grad_clip = 1.0             # Gradient clipping
warmup_iters = 2000         # Warmup steps
lr_decay_iters = 50000      # LR decay steps

# System
dtype = "bfloat16"          # Mixed precision (bf16 recommended)
```

### Memory Requirements

| GPUs         | Batch Size | Block Size | Memory/GPU | Throughput |
| ------------ | ---------- | ---------- | ---------- | ---------- |
| 1x A100 40GB | 2          | 2048       | ~35 GB     | ~1K tok/s  |
| 1x A100 80GB | 4          | 4096       | ~70 GB     | ~2K tok/s  |
| 8x A100 40GB | 4          | 4096       | ~32 GB     | ~16K tok/s |
| 8x A100 80GB | 8          | 4096       | ~60 GB     | ~32K tok/s |

---

## Command Line Arguments

### Data & Output

- `--data_dir`: Directory with tokenized data (required)
- `--out_dir`: Output directory for checkpoints (default: `out/8b_moe_run1`)

### Training

- `--batch_size`: Micro batch size per GPU (default: 4)
- `--block_size`: Context length (default: 4096)
- `--max_iters`: Max training iterations (default: 50000)
- `--grad_accum_steps`: Gradient accumulation (default: 32)

### Optimization

- `--lr`: Peak learning rate (default: 3e-4)
- `--min_lr`: Minimum learning rate (default: 3e-5)
- `--weight_decay`: Weight decay (default: 0.1)
- `--grad_clip`: Gradient clipping (default: 1.0)
- `--warmup_iters`: Warmup steps (default: 2000)

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
- `--log_router_stats`: Log router aux loss (default: true)

### Sampling

- `--sample_every`: Sample text every N iters (default: 500)
- `--sample_tokens`: Tokens to generate (default: 200)
- `--temperature`: Sampling temperature (default: 0.8)
- `--top_k`: Top-k sampling (default: 200)

---

## Data Preparation

### Option 1: FineWeb-Edu with o200k tokenizer (Recommended)

```bash
cd ../gpt-oss-pretrain/build-nanogpt

# Modify fineweb.py to use o200k tokenizer:
# 1. Line 30: enc = tiktoken.get_encoding("o200k_base")
# 2. Lines 37-39: Change to uint32
# 3. Line 49: Change to dtype=np.uint32

python fineweb_o200k.py  # Takes 2-4 hours for 10B tokens
```

### Option 2: Custom data

Create data loader compatible format:

- **Sharded .npy files**: `*_train_*.npy`, `*_val_*.npy`
- **OR Memory-mapped .bin**: `train.bin`, `val.bin`
- **Plus meta.json** with:
  ```json
  {
    "tokenizer": "o200k_base",
    "vocab_size": 200000,
    "dataset": "your-dataset-name"
  }
  ```

---

## Monitoring Training

### Key Metrics to Watch

1. **Training Loss**: Should decrease smoothly

   - Initial: ~10-11 (random init)
   - After 1K iters: ~6-8
   - After 10K iters: ~4-6
   - Converged: ~2-3

2. **Router Aux Loss**: Should be very small

   - ✅ Good: <0.001
   - ⚠️ Warning: 0.001-0.01
   - ❌ Bad: >0.01 (indicates router instability)

3. **Gradient Norm**: Should be stable

   - ✅ Good: 0.5-1.5 (with grad_clip=1.0)
   - ⚠️ Warning: >2.0 frequently
   - ❌ Bad: NaN or Inf

4. **Throughput**: Tokens/second
   - Check GPU utilization (`nvidia-smi`)
   - Should be near-constant after warmup
   - Drops indicate I/O bottleneck

### Warning Signs

- **Loss spikes**: Router aux loss too high, reduce coefficient
- **NaN/Inf**: Learning rate too high, gradient clipping insufficient
- **Slow training**: I/O bottleneck, check data loading
- **OOM errors**: Reduce batch_size or block_size

---

## Checkpointing

### Checkpoint Format

- **Sharded**: `ckpt_rank00000.pt`, `ckpt_rank00001.pt`, ...
- **Per-rank**: Each GPU saves its shard
- **Resume**: Automatically resumes from latest checkpoint in out_dir

### Manual Resume

```bash
# Training automatically resumes if checkpoint exists
python train.py --data_dir ... --out_dir out/existing_run
```

### Checkpoint Contents

- Model state dict (sharded)
- Optimizer state dict (sharded)
- Iteration number
- Best validation loss
- Training arguments

---

## Troubleshooting

### OOM (Out of Memory)

**Solutions**:

1. Reduce `--batch_size` (4 → 2 → 1)
2. Reduce `--block_size` (4096 → 2048 → 1024)
3. Use `--dtype float16` instead of bfloat16 (saves memory)
4. Increase `--grad_accum_steps` to maintain total batch size

### Training Too Slow

**Solutions**:

1. Check GPU utilization: `nvidia-smi dmon`
2. Move data to fast SSD (not HDD)
3. Increase `--batch_size` if memory allows
4. Use `--dtype bfloat16` (faster than float32)

### Loss Not Decreasing

**Checklist**:

1. ✅ Data tokenized correctly (check with validate_setup.py)
2. ✅ Learning rate appropriate (3e-4 is good start)
3. ✅ Router aux loss not too high (should be <0.001)
4. ✅ Gradient clipping enabled (default: 1.0)

### Router Instability

**Symptoms**: Router aux loss >0.01, expert usage imbalanced

**Fix**: Already handled! Training script auto-adjusts aux loss coef to 0.005

---

## Expected Training Time

### 10B Tokens (sample-10BT)

- **1x A100**: ~48-72 hours
- **8x A100**: ~6-10 hours
- **Cost**: ~$50-150 (depending on cloud provider)

### 100B Tokens (sample-100BT)

- **8x A100**: ~60-100 hours
- **Cost**: ~$500-1500

---

## Advanced Usage

### Custom Learning Rate Schedule

```bash
python train.py \
  --data_dir ... \
  --lr 5e-4 \
  --min_lr 5e-5 \
  --warmup_iters 1000 \
  --lr_decay_iters 30000
```

### Longer Context Training

```bash
# Requires more memory!
python train.py \
  --data_dir ... \
  --block_size 8192 \
  --batch_size 2 \
  --grad_accum_steps 64
```

### Mixed Precision Options

```bash
# bfloat16 (recommended, A100+)
python train.py --dtype bfloat16

# float16 (older GPUs)
python train.py --dtype float16

# float32 (debugging, slow)
python train.py --dtype float32
```

---

## File Structure

```
gpt-oss-MoE-8B/
├── src/
│   └── model/
│       └── model.py          # 8B MoE model definition
├── train.py                  # Main FSDP training script
├── data_loader.py            # Flexible data loading (.npy or .bin)
├── scripts/
│   └── validate_setup.py     # Pre-training validation
├── configs/
│   └── (future YAML configs)
└── out/
    └── 8b_moe_run1/         # Training outputs
        ├── ckpt_rank*.pt    # Sharded checkpoints
        └── logs.txt         # Training logs
```

---

## Citation

If you use this training setup, please cite:

```bibtex
@software{gpt_oss_8b_moe,
  title={GPT-OSS-8B: Open-Source 8B MoE Language Model},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/gpt-oss}
}
```

---

## Support

For issues or questions:

1. Check this README first
2. Run `python scripts/validate_setup.py` to diagnose issues
3. Review training logs in `out/*/logs.txt`
4. Open GitHub issue with full error message and setup details

Happy training! 🚀

# gpt-oos-moe-6B
