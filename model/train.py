#!/usr/bin/env python3
"""
FSDP Training Script for GPT-OSS-8B MoE Model

Usage:
    # Single GPU
    python train.py --data_dir data/edu_fineweb10B_o200k

    # Multi-GPU (8 GPUs)
    torchrun --standalone --nproc_per_node=8 train.py --data_dir data/edu_fineweb10B_o200k
"""

import argparse
import dataclasses
import datetime
import json
import math
import os
import signal
import sys
import time
from contextlib import nullcontext
from functools import partial
from typing import Optional

# Add project root to Python path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

# Weights & Biases (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType,
    MixedPrecision,
    ShardedStateDictConfig,
    ShardedOptimStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# Import your model
from model.model import (
    Transformer,
    ModelConfig,
    TransformerBlock,
    gpt_oss_moe_8b_config,
)

# Import data loader
from data.data_loader import get_data_loader


# ------------------------------------------------------------------------------------
# Argument Parser
# ------------------------------------------------------------------------------------

def get_args():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--data_dir", type=str, required=True, help="Directory with tokenized data")
    ap.add_argument("--out_dir", type=str, default="out/8b_moe_run1", help="Output directory")

    # Model
    ap.add_argument("--model_config", type=str, default="8b", choices=["8b"], help="Model configuration")

    # Training
    ap.add_argument("--batch_size", type=int, default=4, help="Micro batch size per GPU")
    ap.add_argument("--block_size", type=int, default=4096, help="Context length")
    ap.add_argument("--max_iters", type=int, default=50000, help="Max training iterations")
    ap.add_argument("--grad_accum_steps", type=int, default=32, help="Gradient accumulation steps")

    # Optimization
    ap.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    ap.add_argument("--min_lr", type=float, default=3e-5, help="Minimum learning rate")
    ap.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    ap.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    ap.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")

    # Learning rate schedule
    ap.add_argument("--warmup_iters", type=int, default=2000, help="Warmup iterations")
    ap.add_argument("--lr_decay_iters", type=int, default=50000, help="LR decay iterations")

    # Mixed precision
    ap.add_argument("--dtype", type=str, choices=["float32", "bfloat16", "float16"],
                    default="bfloat16", help="Training dtype")

    # Checkpointing
    ap.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N iters")
    ap.add_argument("--keep_last_n", type=int, default=5, help="Keep only last N checkpoints")

    # Evaluation
    ap.add_argument("--eval_interval", type=int, default=500, help="Evaluate every N iters")
    ap.add_argument("--eval_iters", type=int, default=100, help="Number of eval iterations")
    ap.add_argument("--eval_hellaswag", action="store_true", default=False,
                    help="Run HellaSwag evaluation (requires datasets library)")
    ap.add_argument("--eval_hellaswag_interval", type=int, default=5000,
                    help="Run HellaSwag evaluation every N iters (only if --eval_hellaswag)")
    ap.add_argument("--eval_hellaswag_num_examples", type=int, default=1000,
                    help="Number of HellaSwag examples to evaluate")
    
    # MMLU evaluation
    ap.add_argument("--eval_mmlu", action="store_true", default=False,
                    help="Run MMLU evaluation (requires datasets library)")
    ap.add_argument("--eval_mmlu_interval", type=int, default=5000,
                    help="Run MMLU evaluation every N iters (only if --eval_mmlu)")
    ap.add_argument("--eval_mmlu_subject", type=str, default="high_school_mathematics",
                    help="MMLU subject to evaluate")
    ap.add_argument("--eval_mmlu_num_examples", type=int, default=100,
                    help="Number of MMLU examples to evaluate")

    # Sampling
    ap.add_argument("--sample_every", type=int, default=500, help="Sample every N iters")
    ap.add_argument("--sample_tokens", type=int, default=200, help="Tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    ap.add_argument("--top_k", type=int, default=200, help="Top-k sampling")

    # Logging
    ap.add_argument("--log_interval", type=int, default=10, help="Log every N iters")
    ap.add_argument("--log_router_stats", action="store_true", default=True,
                    help="Log router aux loss")

    # Weights & Biases
    ap.add_argument("--wandb", action="store_true", default=False,
                    help="Enable Weights & Biases logging")
    ap.add_argument("--wandb_project", type=str, default="gpt-oss-8b-moe",
                    help="W&B project name")
    ap.add_argument("--wandb_run_name", type=str, default=None,
                    help="W&B run name (auto-generated if not specified)")
    ap.add_argument("--wandb_entity", type=str, default=None,
                    help="W&B entity (team name)")
    ap.add_argument("--wandb_notes", type=str, default=None,
                    help="Notes for this run")
    ap.add_argument("--wandb_tags", type=str, nargs="*", default=None,
                    help="Tags for this run")

    # System
    ap.add_argument("--seed", type=int, default=1337, help="Random seed")
    ap.add_argument("--compile", action="store_true", default=False,
                    help="Use torch.compile (experimental)")
    ap.add_argument("--local-rank", type=int, default=None,
                    help="Local rank (for torch.distributed.launch compatibility, ignored if using env vars)")

    return ap.parse_args()


def validate_args(args):
    """Validate training arguments"""
    assert args.batch_size > 0, f"batch_size must be positive, got {args.batch_size}"
    assert args.block_size > 0, f"block_size must be positive, got {args.block_size}"
    assert args.grad_accum_steps > 0, f"grad_accum_steps must be positive, got {args.grad_accum_steps}"
    assert args.max_iters > 0, f"max_iters must be positive, got {args.max_iters}"
    assert args.lr > 0, f"lr must be positive, got {args.lr}"
    assert args.min_lr > 0, f"min_lr must be positive, got {args.min_lr}"
    assert args.lr > args.min_lr, f"lr ({args.lr}) must be > min_lr ({args.min_lr})"
    assert args.warmup_iters < args.lr_decay_iters, \
        f"warmup_iters ({args.warmup_iters}) should be < lr_decay_iters ({args.lr_decay_iters})"
    assert args.eval_interval == -1 or args.eval_interval > 0, \
        f"eval_interval must be positive or -1 (disable), got {args.eval_interval}"
    assert args.eval_iters > 0, f"eval_iters must be positive, got {args.eval_iters}"
    
    # Additional validations
    if args.eval_hellaswag and args.eval_hellaswag_interval <= 0:
        raise ValueError(f"eval_hellaswag_interval must be > 0 when eval_hellaswag=True, got {args.eval_hellaswag_interval}")
    if args.eval_mmlu and args.eval_mmlu_interval <= 0:
        raise ValueError(f"eval_mmlu_interval must be > 0 when eval_mmlu=True, got {args.eval_mmlu_interval}")
    if args.sample_every > 0 and args.sample_tokens <= 0:
        raise ValueError(f"sample_tokens must be > 0 when sample_every > 0, got {args.sample_tokens}")
    if args.temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {args.temperature}")
    if args.top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {args.top_k}")
    
    # Validate paths
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")
    
    # Check if output directory is writable
    try:
        os.makedirs(args.out_dir, exist_ok=True)
        test_file = os.path.join(args.out_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        raise PermissionError(f"Output directory is not writable: {args.out_dir} - {e}")


# ------------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------------

def is_dist() -> bool:
    """Check if running in distributed mode"""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def rank0_print(*args, **kwargs):
    """Print only on rank 0"""
    if not is_dist() or dist.get_rank() == 0:
        print(*args, **kwargs)


def get_lr(it: int, args) -> float:
    """Learning rate schedule with warmup and cosine decay"""
    # Warmup
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    # Decay phase
    if it >= args.lr_decay_iters:
        return args.min_lr
    # Cosine decay
    decay_range = args.lr_decay_iters - args.warmup_iters
    if decay_range <= 0:
        return args.min_lr  # No decay phase, return min_lr
    decay_ratio = (it - args.warmup_iters) / decay_range
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)


def load_tokenizer(data_dir: str):
    """Load tokenizer from meta.json"""
    import tiktoken

    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}. Run data preparation first.")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    tok_name = meta.get("tokenizer", "o200k_base")
    try:
        enc = tiktoken.get_encoding(tok_name)
    except Exception:
        enc = tiktoken.get_encoding("o200k_base")
        tok_name = "o200k_base"
        rank0_print(f"[train] WARNING: tokenizer '{meta.get('tokenizer')}' not available. Using 'o200k_base'.")

    vocab_size = int(meta.get("vocab_size", getattr(enc, "n_vocab", 201_088)))
    return enc, tok_name, vocab_size


# ------------------------------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------------------------------

def save_checkpoint(
    model, optimizer, iter_num, best_val_loss, args, rank, world_size, tok_name
):
    """Save sharded checkpoint"""
    # Cleanup old checkpoints FIRST (before saving new one)
    if rank == 0 and args.keep_last_n > 0:
        import glob
        ckpts = sorted(glob.glob(os.path.join(args.out_dir, "ckpt_*.pt")))
        # Keep exactly keep_last_n * world_size checkpoints (newest)
        if len(ckpts) >= args.keep_last_n * world_size:
            # Delete oldest checkpoints, keep newest keep_last_n sets
            for old_ckpt in ckpts[:-args.keep_last_n * world_size]:
                try:
                    os.remove(old_ckpt)
                    if rank == 0:
                        rank0_print(f"[ckpt] Removed old checkpoint: {os.path.basename(old_ckpt)}")
                except OSError as e:
                    if rank == 0:
                        rank0_print(f"[ckpt] Warning: Failed to remove {old_ckpt}: {e}")
    
    # Get model config
    cfg = model.module.config if hasattr(model, "module") else model.config
    
    with FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
    ):
        model_sd = model.state_dict()
        optim_sd = FSDP.optim_state_dict(model, optimizer)

    payload = {
        "model_state_dict": model_sd,
        "optimizer_state_dict": optim_sd,
        "model_config_dict": dataclasses.asdict(cfg),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "tokenizer": tok_name,
        "args": vars(args),
    }

    # Save per-rank checkpoint
    ckpt_path = os.path.join(args.out_dir, f"ckpt_rank{rank:05d}.pt")
    try:
        torch.save(payload, ckpt_path)
        if rank == 0:
            file_size = os.path.getsize(ckpt_path) / (1024**2)  # MB
            rank0_print(f"[ckpt] Saved checkpoint at iter {iter_num} ({file_size:.1f} MB) to {args.out_dir}/ckpt_rank*.pt")
    except Exception as e:
        rank0_print(f"[ckpt] ERROR: Failed to save checkpoint: {e}")
        raise  # Re-raise to stop training

    if is_dist():
        dist.barrier()


def load_checkpoint(model, optimizer, args, rank):
    """Load sharded checkpoint if exists"""
    ckpt_path = os.path.join(args.out_dir, f"ckpt_rank{rank:05d}.pt")

    if not os.path.exists(ckpt_path):
        return 0, float("inf")  # No checkpoint found

    rank0_print(f"[train] Resuming from {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        model.load_state_dict(payload["model_state_dict"])

    shard_optim = FSDP.optim_state_dict_to_load(payload["optimizer_state_dict"], model, optimizer)
    optimizer.load_state_dict(shard_optim)

    iter_num = int(payload.get("iter_num", 0))
    best_val = float(payload.get("best_val_loss", float("inf")))

    if is_dist():
        dist.barrier()

    rank0_print(f"[train] Resumed at iter {iter_num} (best val {best_val:.4f})")
    return iter_num, best_val


# ------------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, args, device, amp_dtype):
    """Evaluate on validation set with enhanced metrics"""
    model.eval()
    losses = []
    expert_utilizations = []
    router_entropies = []

    ctx = nullcontext() if "cpu" in str(device) else torch.autocast("cuda", dtype=amp_dtype)

    for _ in range(args.eval_iters):
        x, y = val_loader.get_batch()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with ctx:
            _, out = model(x, labels=y)
        loss_val = out["loss"].item()
        if not math.isfinite(loss_val):
            rank0_print(f"[eval] WARNING: Non-finite loss detected: {loss_val}")
            loss_val = float("inf")
        losses.append(loss_val)
        
        # Collect MoE metrics if available
        if "expert_utilization" in out:
            expert_utilizations.append(out["expert_utilization"])
        if "router_entropy" in out:
            router_entropies.append(out["router_entropy"])

    model.train()

    # Aggregate metrics
    val_loss = torch.tensor([sum(losses) / max(1, len(losses))], device=device)
    
    if is_dist():
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    
    val_perplexity = math.exp(val_loss.item())  # Compute once after all_reduce

    metrics = {
        "loss": float(val_loss.item()),
        "perplexity": val_perplexity,
    }
    
    if expert_utilizations:
        # Average utilization across batches
        avg_util = torch.stack(expert_utilizations).mean(dim=0)  # (E,)
        metrics["expert_utilization"] = avg_util.detach().cpu()
        metrics["load_balance_score"] = float(avg_util.std().item() / (avg_util.mean().item() + 1e-10))
    
    if router_entropies:
        metrics["router_entropy"] = float(torch.stack(router_entropies).mean().item())

    return metrics


# ------------------------------------------------------------------------------------
# Sampling
# ------------------------------------------------------------------------------------

@torch.no_grad()
def sample_text(model, enc, device, args, amp_dtype) -> str:
    """Generate text sample (FSDP-safe collective operation)"""
    rank = dist.get_rank() if is_dist() else 0

    model_was_training = model.training
    model.eval()

    # Start with newline token
    start_tok = enc.encode("\n")[0] if hasattr(enc, 'encode') else 0
    tokens = torch.tensor([[start_tok]], device=device, dtype=torch.long)

    ctx = nullcontext() if "cpu" in str(device) else torch.autocast("cuda", dtype=amp_dtype)

    with torch.inference_mode():
        for _ in range(args.sample_tokens):
            inp = tokens[:, -args.block_size:]
            with ctx:
                logits, _ = model(inp, labels=None)
                nxt_logits = logits[:, -1, :]

            # Temperature
            if args.temperature != 1.0:
                nxt_logits = nxt_logits / max(1e-6, args.temperature)

            # Top-k sampling
            if args.top_k and args.top_k > 0:
                v, _ = torch.topk(nxt_logits, args.top_k)
                nxt_logits[nxt_logits < v[:, [-1]]] = -float("inf")

            probs = torch.softmax(nxt_logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, nxt], dim=1)

    if model_was_training:
        model.train()

    txt = enc.decode(tokens[0].tolist()) if rank == 0 else ""
    return txt


# ------------------------------------------------------------------------------------
# Main Training Loop
# ------------------------------------------------------------------------------------

def main():
    # Set environment variable for better CUDA memory management
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = get_args()
    validate_args(args)  # Validate arguments before proceeding

    # Distributed setup
    if is_dist():
        timeout = datetime.timedelta(minutes=60)
        dist.init_process_group(backend="nccl", timeout=timeout)
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rank = 0
        world_size = 1

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Set random seeds
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Initialize variables for signal handler (will be set later)
    model_ref = None
    optimizer_ref = None
    iter_num_ref = [0]
    best_val_loss_ref = [float("inf")]
    tok_name_ref = [None]

    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        rank0_print("\n[train] Interrupted! Saving checkpoint...")
        try:
            # Save checkpoint if model exists
            if model_ref is not None and optimizer_ref is not None:
                save_checkpoint(model_ref, optimizer_ref, iter_num_ref[0], best_val_loss_ref[0], args, rank, world_size, tok_name_ref[0])
        except Exception as e:
            rank0_print(f"[train] Failed to save checkpoint on interrupt: {e}")
        if args.wandb and rank == 0:
            try:
                wandb.finish()
            except:
                pass
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize Weights & Biases (only on rank 0)
    if args.wandb and rank == 0:
        if not WANDB_AVAILABLE:
            rank0_print("⚠️  wandb requested but not installed. Install with: pip install wandb")
            rank0_print("   Continuing without wandb logging...")
            args.wandb = False
        else:
            # Auto-generate run name if not specified
            if args.wandb_run_name is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                args.wandb_run_name = f"8b-moe_{timestamp}"

            # Initialize wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                notes=args.wandb_notes,
                tags=args.wandb_tags,
                config={
                    # Training args
                    "batch_size": args.batch_size,
                    "block_size": args.block_size,
                    "max_iters": args.max_iters,
                    "grad_accum_steps": args.grad_accum_steps,
                    "learning_rate": args.lr,
                    "min_lr": args.min_lr,
                    "weight_decay": args.weight_decay,
                    "beta1": args.beta1,
                    "beta2": args.beta2,
                    "grad_clip": args.grad_clip,
                    "warmup_iters": args.warmup_iters,
                    "lr_decay_iters": args.lr_decay_iters,
                    "dtype": args.dtype,
                    "seed": args.seed,
                    # System
                    "world_size": world_size,
                    "num_gpus": world_size,
                    "data_dir": args.data_dir,
                    "out_dir": args.out_dir,
                },
                resume="allow"  # Allow resuming from checkpoint
            )
            rank0_print(f"[wandb] Initialized: project={args.wandb_project}, run={args.wandb_run_name}")

    # Load tokenizer and get vocab size
    enc, tok_name, vocab_size = load_tokenizer(args.data_dir)
    tok_name_ref[0] = tok_name  # Update for signal handler

    # Build model config
    if args.model_config == "8b":
        cfg = gpt_oss_moe_8b_config()
        cfg.vocab_size = vocab_size

        # Fix aux loss coefficient if using default
        if cfg.router_aux_loss_coef == 0.02:
            rank0_print(f"[train] Adjusting router_aux_loss_coef from 0.02 to 0.005 for 8 experts")
            cfg.router_aux_loss_coef = 0.005
    else:
        raise ValueError(f"Unknown model config: {args.model_config}")

    # Adjust block size if exceeds model max
    if args.block_size > cfg.max_position_embeddings:
        rank0_print(f"[train] Reducing block_size from {args.block_size} to {cfg.max_position_embeddings}")
        args.block_size = cfg.max_position_embeddings

    # Build model on meta device (zero memory!)
    rank0_print("[train] Building model on meta device...")
    if hasattr(torch, "set_default_device"):
        torch.set_default_device("meta")
    base_model = Transformer(cfg)
    if hasattr(torch, "set_default_device"):
        torch.set_default_device("cpu")

    # Print parameter count
    if rank == 0:
        total_params = sum(p.numel() for p in base_model.parameters())
        rank0_print(f"[params] Total: {total_params/1e9:.2f}B parameters")

    # FSDP wrapping policy
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # Mixed precision
    mp_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    mixed_precision = MixedPrecision(
        param_dtype=mp_dtype,
        reduce_dtype=mp_dtype,
        buffer_dtype=mp_dtype,
    )

    # Parameter initialization function for FSDP
    def param_init_fn(module):
        # Materialize on target device
        module.to_empty(device=torch.device(device))
        # Initialize parameters
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    # Wrap with FSDP (for distributed mode or single GPU with FSDP)
    # FSDP helps with optimizer state memory even for single GPU
    if is_dist():
        rank0_print("[train] Wrapping model with FSDP...")
        model = FSDP(
            base_model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=None,  # Let FSDP manage via to_empty
            mixed_precision=mixed_precision,
            use_orig_params=True,
            limit_all_gathers=True,
            param_init_fn=param_init_fn,
        )
    else:
        # Single GPU: Use FSDP anyway to reduce optimizer memory
        # FSDP shards optimizer states which helps with large models
        rank0_print("[train] Wrapping model with FSDP (single GPU mode)...")
        # Initialize minimal process group for FSDP
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="tcp://localhost:29503",
                world_size=1,
                rank=0
            )
        model = FSDP(
            base_model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=0,
            mixed_precision=mixed_precision,
            use_orig_params=True,
            limit_all_gathers=True,
            param_init_fn=param_init_fn,
        )
    model_ref = model  # Update for signal handler

    # Print shard size
    shard_params = sum(p.numel() for p in model.parameters())
    rank0_print("=" * 70)
    rank0_print(f"Model: {args.model_config}   Device: {device}   World size: {world_size}")
    rank0_print(f"Tokenizer: {tok_name}   Vocab: {vocab_size}")
    rank0_print(f"[params] Per-rank shard: ~{shard_params/1e9:.2f}B")
    rank0_print(f"Context: block_size={args.block_size} (model max={cfg.max_position_embeddings})")
    rank0_print(f"Router aux loss coef: {cfg.router_aux_loss_coef}")

    # Print FlashAttention status
    if hasattr(model.module if hasattr(model, "module") else model, "get_flash_attn_info"):
        flash_info = (model.module if hasattr(model, "module") else model).get_flash_attn_info()
        rank0_print(f"{flash_info}")

    rank0_print("=" * 70)

    # Update W&B config with model architecture details
    if args.wandb and rank == 0:
        wandb.config.update({
            "model/config": args.model_config,
            "model/vocab_size": vocab_size,
            "model/hidden_size": cfg.hidden_size,
            "model/num_layers": cfg.num_hidden_layers,
            "model/num_attention_heads": cfg.num_attention_heads,
            "model/num_key_value_heads": cfg.num_key_value_heads,
            "model/intermediate_size": cfg.intermediate_size,
            "model/num_experts": cfg.num_local_experts,
            "model/experts_per_token": cfg.experts_per_token,
            "model/router_aux_loss_coef": cfg.router_aux_loss_coef,
            "model/max_position_embeddings": cfg.max_position_embeddings,
            "model/parameters_per_rank": shard_params,
            "model/tokenizer": tok_name,
        })

    # Data loaders
    rank0_print(f"[train] Loading data from {args.data_dir}")
    train_loader = get_data_loader(
        args.data_dir, "train", args.block_size, args.batch_size,
        process_rank=rank, num_processes=world_size, seed=args.seed + rank
    )
    val_loader = get_data_loader(
        args.data_dir, "val", args.block_size, args.batch_size,
        process_rank=rank, num_processes=world_size, seed=args.seed + 1234 + rank
    )

    # Autocast context
    amp_ctx = nullcontext() if "cpu" in device else torch.autocast("cuda", dtype=mp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=("cuda" in device and args.dtype == "float16"))

    # Optimizer - must be created AFTER FSDP wrapping
    # With FSDP use_orig_params=True, optimizer sees only sharded params
    rank0_print("[train] Creating optimizer...")
    # Ensure we're accessing parameters correctly with FSDP
    # model.parameters() should return only sharded params
    optimizer = torch.optim.AdamW(
        model.parameters(),  # This should only see sharded params with FSDP
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    optimizer_ref = optimizer  # Update for signal handler
    
    # Debug: Check parameter count per GPU (verify FSDP sharding works)
    if rank == 0:
        total_params_in_optimizer = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
        rank0_print(f"[train] Parameters in optimizer: {total_params_in_optimizer/1e9:.2f}B")
        if is_dist():
            expected_params = 6.57 / world_size
            rank0_print(f"[train] Expected per GPU: ~{expected_params:.2f}B")
            # Check if optimizer sees roughly the right number of parameters (allow 20% tolerance)
            if total_params_in_optimizer > expected_params * 1.2:
                rank0_print(f"[train] ⚠️  WARNING: Optimizer sees {total_params_in_optimizer/1e9:.2f}B params (expected ~{expected_params:.2f}B)")
                rank0_print(f"[train]   This might indicate FSDP sharding issue")
            elif total_params_in_optimizer < expected_params * 0.8:
                rank0_print(f"[train] ⚠️  WARNING: Optimizer sees {total_params_in_optimizer/1e9:.2f}B params (expected ~{expected_params:.2f}B)")
                rank0_print(f"[train]   This might indicate FSDP sharding issue")
            else:
                rank0_print(f"[train] ✅ Optimizer sees correct number of sharded parameters ({total_params_in_optimizer/1e9:.2f}B)")

    # Load checkpoint if exists
    iter_num, best_val_loss = load_checkpoint(model, optimizer, args, rank)
    iter_num_ref[0] = iter_num  # Update for signal handler
    best_val_loss_ref[0] = best_val_loss  # Update for signal handler

    # Calculate total batch size
    total_batch_size = args.batch_size * args.block_size * args.grad_accum_steps * world_size
    rank0_print(f"[train] Total batch size: {total_batch_size:,} tokens")
    rank0_print(f"[train] Gradient accumulation steps: {args.grad_accum_steps}")
    rank0_print("")

    # Training loop
    rank0_print("[train] Starting training...")
    t0 = time.time()

    while iter_num < args.max_iters:
        # Learning rate schedule
        lr = get_lr(iter_num, args)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        router_loss_accum = 0.0

        # Gradient accumulation loop
        for micro_step in range(args.grad_accum_steps):
            x, y = train_loader.get_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with amp_ctx:
                _, outputs = model(x, labels=y)
                loss = outputs["loss"] / args.grad_accum_steps

            loss_accum += loss.detach().item()
            if "router_aux_loss" in outputs:
                router_loss_accum += outputs["router_aux_loss"].item() / args.grad_accum_steps

            # Backward
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # Check for NaN/Inf loss before optimizer step
        if not math.isfinite(loss_accum):
            rank0_print(f"[train] ERROR: Non-finite loss detected at iter {iter_num}")
            rank0_print(f"[train] Loss value: {loss_accum}")
            rank0_print(f"[train] Router aux loss: {router_loss_accum}")
            # Save checkpoint for debugging
            save_checkpoint(model, optimizer, iter_num, best_val_loss, args, rank, world_size, tok_name)
            raise ValueError(f"Training diverged: loss = {loss_accum}")

        # Gradient clipping
        if args.grad_clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            norm = 0.0

        # Optimizer step
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Synchronize for accurate timing
        if "cuda" in device:
            torch.cuda.synchronize()

        # Logging
        if iter_num % args.log_interval == 0:
            # Average loss across GPUs
            loss_t = torch.tensor([loss_accum], device=device)
            if is_dist():
                dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)

            dt = time.time() - t0
            t0 = time.time()

            tokens_per_sec = total_batch_size / dt if dt > 0 else 0

            log_str = f"iter {iter_num:06d} | loss {loss_t.item():.4f} | lr {lr:.6e} | "
            log_str += f"norm {norm:.4f} | dt {dt*1000:.1f}ms | tok/s {tokens_per_sec:.0f}"

            if args.log_router_stats and router_loss_accum > 0:
                log_str += f" | router_aux {router_loss_accum:.6f}"
                if router_loss_accum > 0.01:
                    log_str += " ⚠️"

            rank0_print(log_str)

            # Log to Weights & Biases
            if args.wandb and rank == 0:
                wandb_metrics = {
                    "train/loss": loss_t.item(),
                    "train/perplexity": math.exp(loss_t.item()),  # Add perplexity
                    "train/learning_rate": lr,
                    "train/grad_norm": norm,
                    "system/tokens_per_sec": tokens_per_sec,
                    "system/dt_ms": dt * 1000,
                    "iteration": iter_num,
                }
                if args.log_router_stats and router_loss_accum > 0:
                    wandb_metrics["train/router_aux_loss"] = router_loss_accum
                
                # Add GPU memory if available
                if "cuda" in device:
                    gpu_memory_gb = torch.cuda.max_memory_allocated() / 1e9
                    wandb_metrics["system/gpu_memory_gb"] = gpu_memory_gb
                    torch.cuda.reset_peak_memory_stats()
                
                wandb.log(wandb_metrics, step=iter_num)

        # Evaluation
        if args.eval_interval > 0 and iter_num > 0 and iter_num % args.eval_interval == 0:
            eval_metrics = evaluate(model, val_loader, args, device, mp_dtype)
            val_loss = eval_metrics["loss"]
            val_perplexity = eval_metrics.get("perplexity", math.exp(val_loss))
            
            rank0_print(f"[eval] iter {iter_num:06d} | val_loss {val_loss:.4f} | val_ppl {val_perplexity:.2f}")
            
            if "router_entropy" in eval_metrics:
                rank0_print(f"[eval] router_entropy {eval_metrics['router_entropy']:.4f} | load_balance {eval_metrics.get('load_balance_score', 0):.4f}")

            # Log validation to W&B
            if args.wandb and rank == 0:
                wandb_metrics = {
                    "eval/val_loss": val_loss,
                    "eval/val_perplexity": val_perplexity,
                    "eval/best_val_loss": best_val_loss,
                }
                
                # Add MoE metrics if available
                if "router_entropy" in eval_metrics:
                    wandb_metrics["eval/router_entropy"] = eval_metrics["router_entropy"]
                if "load_balance_score" in eval_metrics:
                    wandb_metrics["eval/load_balance_score"] = eval_metrics["load_balance_score"]
                if "expert_utilization" in eval_metrics:
                    # Log per-expert utilization as histogram
                    wandb_metrics["eval/expert_utilization"] = wandb.Histogram(eval_metrics["expert_utilization"].numpy())
                
                wandb.log(wandb_metrics, step=iter_num)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_ref[0] = best_val_loss  # Update for signal handler
                rank0_print(f"[eval] New best validation loss: {best_val_loss:.4f} (perplexity: {val_perplexity:.2f})")
                save_checkpoint(model, optimizer, iter_num, best_val_loss, args, rank, world_size, tok_name)

                # Log best checkpoint to W&B
                if args.wandb and rank == 0:
                    wandb.log({"eval/best_val_loss": best_val_loss, "eval/best_perplexity": val_perplexity}, step=iter_num)

        # HellaSwag Evaluation (optional, on rank 0 only)
        if args.eval_hellaswag and iter_num > 0 and iter_num % args.eval_hellaswag_interval == 0:
            if rank == 0:
                try:
                    import sys
                    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
                    from scripts.evaluate_hellaswag import evaluate_hellaswag_simple
                    enc, _, _ = load_tokenizer(args.data_dir)
                    
                    # Get model without FSDP wrapper for evaluation
                    eval_model = model.module if hasattr(model, "module") else model
                    
                    hellaswag_results = evaluate_hellaswag_simple(
                        eval_model,
                        enc,
                        device=device,
                        num_examples=args.eval_hellaswag_num_examples,
                        max_length=args.block_size,
                    )
                    
                    rank0_print(f"[hellaswag] iter {iter_num:06d} | accuracy {hellaswag_results['accuracy']:.4f} ({hellaswag_results['correct']}/{hellaswag_results['total']})")
                    
                    if args.wandb:
                        wandb.log({
                            "eval/hellaswag_accuracy": hellaswag_results['accuracy'],
                            "eval/hellaswag_correct": hellaswag_results['correct'],
                            "eval/hellaswag_total": hellaswag_results['total'],
                        }, step=iter_num)
                except ImportError:
                    rank0_print("[hellaswag] datasets library not installed. Install with: pip install datasets")
                except RuntimeError as e:
                    rank0_print(f"[hellaswag] Runtime error: {e}")
                except Exception as e:
                    rank0_print(f"[hellaswag] Unexpected error: {e}")
                    import traceback
                    traceback.print_exc()  # Print full traceback for debugging
            if is_dist():
                dist.barrier()  # Sync all ranks

        # MMLU Evaluation (optional, on rank 0 only)
        if args.eval_mmlu and iter_num > 0 and iter_num % args.eval_mmlu_interval == 0:
            if rank == 0:
                try:
                    import sys
                    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
                    from scripts.evaluate_mmlu import evaluate_mmlu_simple
                    enc, _, _ = load_tokenizer(args.data_dir)
                    
                    # Get model without FSDP wrapper for evaluation
                    eval_model = model.module if hasattr(model, "module") else model
                    
                    mmlu_results = evaluate_mmlu_simple(
                        eval_model,
                        enc,
                        device=device,
                        subject=args.eval_mmlu_subject,
                        num_examples=args.eval_mmlu_num_examples,
                        max_length=args.block_size,
                    )
                    
                    rank0_print(f"[mmlu] iter {iter_num:06d} | subject {args.eval_mmlu_subject} | accuracy {mmlu_results['accuracy']:.4f} ({mmlu_results['correct']}/{mmlu_results['total']})")
                    
                    if args.wandb:
                        wandb.log({
                            f"eval/mmlu_{args.eval_mmlu_subject}_accuracy": mmlu_results['accuracy'],
                            f"eval/mmlu_{args.eval_mmlu_subject}_correct": mmlu_results['correct'],
                            f"eval/mmlu_{args.eval_mmlu_subject}_total": mmlu_results['total'],
                        }, step=iter_num)
                except ImportError:
                    rank0_print("[mmlu] datasets library not installed. Install with: pip install datasets")
                except RuntimeError as e:
                    rank0_print(f"[mmlu] Runtime error: {e}")
                except Exception as e:
                    rank0_print(f"[mmlu] Unexpected error: {e}")
                    import traceback
                    traceback.print_exc()  # Print full traceback for debugging
            if is_dist():
                dist.barrier()  # Sync all ranks

        # Sampling (all ranks must participate for FSDP)
        if args.sample_every > 0 and iter_num > 0 and iter_num % args.sample_every == 0:
            try:
                txt = sample_text(model, enc, device, args, mp_dtype)
                if rank == 0:
                    print("\n--- SAMPLE ---")
                    print(txt)
                    print("--------------\n")

                    # Log sample to W&B
                    if args.wandb:
                        wandb.log({
                            "samples/generated_text": wandb.Html(f"<pre>{txt}</pre>"),
                        }, step=iter_num)
            except Exception as e:
                rank0_print(f"[sample] Error: {e}")

        # Save checkpoint
        if args.save_every > 0 and iter_num > 0 and iter_num % args.save_every == 0:
            save_checkpoint(model, optimizer, iter_num, best_val_loss, args, rank, world_size, tok_name)
        
        # Update references for signal handler
        iter_num_ref[0] = iter_num
        best_val_loss_ref[0] = best_val_loss

        iter_num += 1

    # Final checkpoint
    rank0_print("[train] Training complete!")
    save_checkpoint(model, optimizer, iter_num, best_val_loss, args, rank, world_size, tok_name)

    # Finish W&B run
    if args.wandb and rank == 0:
        wandb.finish()
        rank0_print("[wandb] Run finished")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
