# Training Cost & Time Calculator for 100B Tokens
# Based on train.py configuration and realistic throughput estimates

# =============================================================================
# TRAINING CONFIGURATION (from train.py defaults)
# =============================================================================
batch_size_per_gpu = 4
block_size = 4096
grad_accum_steps = 32

# Target tokens for training
target_tokens = 100_000_000_000  # 100B tokens

# =============================================================================
# GPU CONFIGURATIONS & REALISTIC ESTIMATES
# =============================================================================
# Based on COMPLETE_SETUP_GUIDE.md estimates which account for:
# - Actual training throughput (accounting for overhead)
# - Data loading overhead
# - Checkpoint saving overhead
# - Network/communication overhead

gpu_configs = {
    "8x A100 80GB": {
        "num_gpus": 8,
        "cost_per_gpu_hour": 2.00,
        "estimated_hours": 100,  # From setup guide
        "estimated_throughput": 100_000_000_000 / 100 / 3600,  # ~278K tokens/sec effective
        "memory_per_gpu": "60 GB",
        "notes": "Most widely available, reliable"
    },
    "8x H100 NVL": {
        "num_gpus": 8,
        "cost_per_gpu_hour": 2.49,
        "estimated_hours": 48,  # From setup guide
        "estimated_throughput": 100_000_000_000 / 48 / 3600,  # ~579K tokens/sec effective
        "memory_per_gpu": "94 GB",
        "notes": "Fastest option, best for production"
    },
    "4x A100 80GB": {
        "num_gpus": 4,
        "cost_per_gpu_hour": 2.00,
        "estimated_hours": 200,  # Estimate: 2x slower than 8x
        "estimated_throughput": 100_000_000_000 / 200 / 3600,  # ~139K tokens/sec effective
        "memory_per_gpu": "60 GB",
        "notes": "Budget option, slower but cheaper"
    },
    "1x A100 80GB": {
        "num_gpus": 1,
        "cost_per_gpu_hour": 2.00,
        "estimated_hours": 800,  # Estimate: 8x slower than 8x
        "estimated_throughput": 100_000_000_000 / 800 / 3600,  # ~35K tokens/sec effective
        "memory_per_gpu": "70 GB",
        "notes": "Testing only, not recommended for production"
    }
}

# =============================================================================
# CALCULATIONS
# =============================================================================

print("=" * 80)
print("TRAINING COST & TIME ESTIMATE FOR 100B TOKENS")
print("=" * 80)
print(f"\nTraining Configuration (from train.py):")
print(f"  Batch size per GPU: {batch_size_per_gpu}")
print(f"  Block size: {block_size}")
print(f"  Gradient accumulation steps: {grad_accum_steps}")
print(f"  Target tokens: {target_tokens:,} ({target_tokens/1e9:.1f}B)")

# Calculate effective batch size per configuration
for config_name, config in gpu_configs.items():
    num_gpus = config["num_gpus"]
    effective_batch_size = batch_size_per_gpu * block_size * grad_accum_steps * num_gpus
    
    # Calculate iterations needed
    iterations_needed = target_tokens / effective_batch_size
    
    # Use realistic time estimates from guide
    total_hours = config["estimated_hours"]
    total_days = total_hours / 24
    
    # Calculate cost
    total_cost = total_hours * config["cost_per_gpu_hour"] * num_gpus
    cost_per_1B_tokens = total_cost / (target_tokens / 1e9)
    
    # Calculate effective throughput
    effective_throughput = config["estimated_throughput"]
    
    print("\n" + "-" * 80)
    print(f"Configuration: {config_name}")
    print("-" * 80)
    print(f"  GPUs: {num_gpus}")
    print(f"  Memory per GPU: {config['memory_per_gpu']}")
    print(f"  Cost per GPU-hour: ${config['cost_per_gpu_hour']:.2f}")
    print(f"  Effective throughput: {effective_throughput:,.0f} tokens/sec ({effective_throughput/1000:.1f}K tok/s)")
    print(f"\n  Effective batch size: {effective_batch_size:,} tokens/iteration")
    print(f"  Iterations needed: {iterations_needed:,.0f}")
    print(f"\n  ⏱️  Training Time:")
    print(f"     {total_hours:,.1f} hours ({total_days:.1f} days)")
    print(f"\n  💰 Estimated Cost:")
    print(f"     ${total_cost:,.2f} total")
    print(f"     ${cost_per_1B_tokens:.2f} per 1B tokens")
    print(f"\n  📝 Notes: {config['notes']}")

print("\n" + "=" * 80)
print("BATCH SIZE BREAKDOWN")
print("=" * 80)
print(f"\nFor 8 GPUs configuration:")
effective_batch = batch_size_per_gpu * block_size * grad_accum_steps * 8
print(f"  Per-GPU batch: {batch_size_per_gpu} samples")
print(f"  Sequence length: {block_size} tokens")
print(f"  Gradient accumulation: {grad_accum_steps} steps")
print(f"  Number of GPUs: 8")
print(f"  Total: {effective_batch:,} tokens per update")
print(f"\n  This means each iteration processes {effective_batch/1e6:.2f}M tokens")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print("\nFor 100B tokens training:")
print("  🚀 Best: 8x H100 NVL")
print("     • Time: ~48 hours (~2 days)")
print("     • Cost: ~$960")
print("     • Fastest and most cost-effective for production")
print("\n  ✅ Good: 8x A100 80GB")
print("     • Time: ~100 hours (~4 days)")
print("     • Cost: ~$1,600")
print("     • Most widely available, reliable")
print("\n  💰 Budget: 4x A100 80GB")
print("     • Time: ~200 hours (~8 days)")
print("     • Cost: ~$1,600")
print("     • Same cost, slower but good for testing")
print("\n  ⚠️  Not Recommended: 1x A100")
print("     • Time: ~800 hours (~33 days)")
print("     • Cost: ~$1,600")
print("     • Only for testing/debugging")

print("\n" + "=" * 80)
print("IMPORTANT NOTES")
print("=" * 80)
print("""
1. These estimates assume:
   - Stable training (no crashes/restarts)
   - Good network bandwidth for data loading
   - Efficient checkpointing (every 1000 iters)
   - Minimal evaluation overhead

2. Actual times may vary ±20% based on:
   - Data loading speed (SSD vs HDD)
   - Network topology (InfiniBand vs Ethernet)
   - System stability and restarts
   - Evaluation frequency

3. Cost considerations:
   - Spot instances can reduce cost by 50-70%
   - Long-term reservations: 20-30% discount
   - Include storage costs (~$0.10/GB/month)
   - Checkpoint storage: ~50GB per checkpoint

4. To monitor actual progress:
   - Watch tokens_per_sec in training logs
   - Adjust max_iters based on actual throughput
   - Use W&B to track training progress
""")

print("\n" + "=" * 80)
print("CONFIGURATION COMMAND")
print("=" * 80)
print("\nFor 100B tokens, you'll need to set max_iters appropriately:")
print(f"\n  # Calculate: {target_tokens:,} tokens / {effective_batch:,} tokens/iter = ~{iterations_needed:,.0f} iterations")
print("\n  # Recommended: Set max_iters to ~25,000-30,000")
print("\n  # Example command:")
print("  torchrun --standalone --nproc_per_node=8 train.py \\")
print("    --data_dir data/edu_fineweb_100BT \\")
print("    --max_iters 25000 \\")
print("    --batch_size 4 \\")
print("    --block_size 4096 \\")
print("    --grad_accum_steps 32")

