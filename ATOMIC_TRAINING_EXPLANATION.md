# Atomic-Level Training Loop Explanation: FSDP, GPU Communication, and Learning Rate

## Table of Contents
1. [System Initialization](#1-system-initialization)
2. [Training Loop: Atomic Step-by-Step](#2-training-loop-atomic-step-by-step)
3. [GPU Communication: FSDP All-Gather & All-Reduce](#3-gpu-communication-fsdp-all-gather--all-reduce)
4. [Learning Rate: Warmup + Cosine Decay](#4-learning-rate-warmup--cosine-decay)
5. [Complete Iteration Timeline](#5-complete-iteration-timeline)

---

## 1. System Initialization

### 1.1 Process Spawning

```python
# Command: python -m torch.distributed.launch --nproc_per_node=8 model/train.py

Step 1: Main Process Creates Child Processes
──────────────────────────────────────────────────────────────
Main Process (PID: 31987)
  │
  ├─→ Fork Process 0 → GPU 0 (PID: 32052, LOCAL_RANK=0)
  ├─→ Fork Process 1 → GPU 1 (PID: 32053, LOCAL_RANK=1)
  ├─→ Fork Process 2 → GPU 2 (PID: 32054, LOCAL_RANK=2)
  ├─→ Fork Process 3 → GPU 3 (PID: 32055, LOCAL_RANK=3)
  ├─→ Fork Process 4 → GPU 4 (PID: 32056, LOCAL_RANK=4)
  ├─→ Fork Process 5 → GPU 5 (PID: 32057, LOCAL_RANK=5)
  ├─→ Fork Process 6 → GPU 6 (PID: 32058, LOCAL_RANK=6)
  └─→ Fork Process 7 → GPU 7 (PID: 32059, LOCAL_RANK=7)
```

**Environment Variables Set:**
- `LOCAL_RANK`: 0-7 (which GPU this process owns)
- `RANK`: 0-7 (global rank)
- `WORLD_SIZE`: 8 (total GPUs)
- `MASTER_ADDR`: localhost
- `MASTER_PORT`: 29500

### 1.2 NCCL Process Group Initialization

```python
# Each process executes:
dist.init_process_group(
    backend="nccl",  # NVIDIA Collective Communications Library
    timeout=timedelta(minutes=60)
)

ATOMIC STEPS:
──────────────────────────────────────────────────────────────
Process 0:                    Process 1:                    Process 7:
──────────                    ──────────                    ──────────
1. Create NCCL communicator   1. Create NCCL communicator   1. Create NCCL communicator
2. Discover GPU topology      2. Discover GPU topology      2. Discover GPU topology
3. Establish NVLink paths     3. Establish NVLink paths     3. Establish NVLink paths
4. Register with rank 0       4. Register with rank 0       4. Register with rank 0
5. Wait for all ranks         5. Wait for all ranks         5. Wait for all ranks
6. Barrier sync              6. Barrier sync              6. Barrier sync ✅
```

**NCCL Topology Discovery:**
```
GPU 0 ←─ NVLink ─→ GPU 1 ←─ NVLink ─→ GPU 2 ←─ NVLink ─→ GPU 3
  ↕                  ↕                  ↕                  ↕
NVSwitch          NVSwitch          NVSwitch          NVSwitch
  ↕                  ↕                  ↕                  ↕
GPU 4 ←─ NVLink ─→ GPU 5 ←─ NVLink ─→ GPU 6 ←─ NVLink ─→ GPU 7
```

**Bandwidth:**
- NVLink: 600 GB/s per link
- NVSwitch: Full bisection bandwidth
- Communication cost: ~2ms for all-reduce of 2GB gradients

### 1.3 Model Construction on Meta Device

```python
# ALL processes execute (synchronously):
torch.set_default_device("meta")  # Zero-memory device
base_model = Transformer(cfg)     # Creates tensor METADATA only

ATOMIC BREAKDOWN:
──────────────────────────────────────────────────────────────
For each parameter tensor:
  - Shape: torch.Size([2880, 2880])
  - Dtype: torch.bfloat16
  - Device: meta
  - Requires_grad: True
  - Memory: 0 bytes (no actual data!)

Total: 6.57B parameters × 0 bytes = 0 GB allocated
```

### 1.4 FSDP Wrapping & Parameter Sharding

```python
# FSDP wraps model with auto_wrap_policy
model = FSDP(
    base_model,
    auto_wrap_policy=transformer_auto_wrap_policy,  # Wrap at TransformerBlock
    param_init_fn=param_init_fn,  # Materialize on target GPU
)

ATOMIC SHARDING PROCESS:
──────────────────────────────────────────────────────────────
For each TransformerBlock (24 blocks total):

Block 0:
  GPU 0: param_init_fn(Block0.Shard0) → Materialize on cuda:0
  GPU 1: param_init_fn(Block0.Shard1) → Materialize on cuda:1
  GPU 2: param_init_fn(Block0.Shard2) → Materialize on cuda:2
  ...
  GPU 7: param_init_fn(Block0.Shard7) → Materialize on cuda:7

Shard Size Calculation:
──────────────────────────────────────────────────────────────
Block 0 total params: ~274M parameters
Shard size per GPU: 274M / 8 = 34.25M parameters
Memory per shard: 34.25M × 2 bytes (bf16) = 68.5 MB

Each GPU stores:
  - 1/8 of Block 0 params: 68.5 MB
  - 1/8 of Block 1 params: 68.5 MB
  - ...
  - 1/8 of Block 23 params: 68.5 MB
  - Full Embedding (small): ~11 MB
  - Full LM Head (small): ~1 MB
  
Total per GPU: ~1.64 GB
```

### 1.5 Optimizer Creation

```python
# Optimizer sees only sharded parameters
optimizer = torch.optim.AdamW(
    model.parameters(),  # Returns only GPU's shard!
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1
)

ATOMIC OPTIMIZER STATE INITIALIZATION:
──────────────────────────────────────────────────────────────
For each parameter tensor in shard:

GPU 0 (Shard 0):
  param = tensor([...], dtype=bf16, device=cuda:0)  # 68.5 MB

  optimizer.state[param] = {
    'step': 0,
    'exp_avg': torch.zeros_like(param, dtype=torch.float32),  # 137 MB
    'exp_avg_sq': torch.zeros_like(param, dtype=torch.float32), # 137 MB
  }
  
Total optimizer state per GPU: 0.82B params × 4 bytes × 2 = 6.57 GB
```

---

## 2. Training Loop: Atomic Step-by-Step

### 2.1 Iteration Start: Learning Rate Update

```python
# Line 738-741: Learning rate schedule
lr = get_lr(iter_num, args)
for param_group in optimizer.param_groups:
    param_group["lr"] = lr

ATOMIC OPERATION:
──────────────────────────────────────────────────────────────
Each GPU:
  1. Call get_lr(iter_num, args)
  2. Compute lr value (e.g., 0.0003)
  3. Access optimizer.param_groups[0]
  4. Set param_group["lr"] = 0.0003
  5. This lr is stored in optimizer state, used during optimizer.step()
```

**Memory Access Pattern:**
- Read: `iter_num` (CPU register)
- Write: `optimizer.param_groups[0]["lr"]` (CPU memory)
- No GPU memory access (optimizer metadata is on CPU)

### 2.2 Zero Gradients

```python
# Line 744: Zero gradients
optimizer.zero_grad(set_to_none=True)

ATOMIC OPERATION PER PARAMETER:
──────────────────────────────────────────────────────────────
For each parameter tensor in shard:

GPU 0:
  param = tensor([...], device=cuda:0, requires_grad=True)
  
  Before:
    param.grad = tensor([...])  # Existing gradients
  
  After:
    param.grad = None  # Free memory, set to None
  
Memory freed: ~1.64 GB (gradient buffers released)
```

**Why `set_to_none=True`?**
- Frees GPU memory immediately instead of zeroing
- Faster: No memory write operation
- Memory efficient: Allows CUDA allocator to reuse freed memory

### 2.3 Gradient Accumulation Loop (32 Micro-Steps)

```python
# Line 749-765: Gradient accumulation
for micro_step in range(args.grad_accum_steps):  # 32 iterations
    x, y = train_loader.get_batch()
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    
    with amp_ctx:
        _, outputs = model(x, labels=y)
        loss = outputs["loss"] / args.grad_accum_steps
    
    loss.backward()

#### MICRO-STEP 0: Forward Pass ####
──────────────────────────────────────────────────────────────
Step 1: Data Loading
  GPU 0: Load batch[0:8192] from shard 0
  GPU 1: Load batch[8192:16384] from shard 1
  ...
  GPU 7: Load batch[57344:65536] from shard 7
  
  x shape: [batch_size=2, block_size=1024]
  Memory: 2 × 1024 × 2 bytes (uint16) = 4 KB

Step 2: Data Transfer (Non-Blocking)
  CPU → GPU transfer (PCIe/NVLink)
  x.to(device, non_blocking=True)  # Returns immediately, transfer happens async
  
Step 3: Forward Pass Through Each Layer

Layer 0: TransformerBlock 0
──────────────────────────────────────────────────────────────
3a. FSDP All-Gather Block 0 Parameters
──────────────────────────────────────────────────────────────
BEFORE:
  GPU 0: Block0.Shard0 [34.25M params, 68.5 MB]
  GPU 1: Block0.Shard1 [34.25M params, 68.5 MB]
  ...
  GPU 7: Block0.Shard7 [34.25M params, 68.5 MB]

NCCL ALL-GATHER OPERATION:
──────────────────────────────────────────────────────────────
Ring Algorithm (7 communication rounds):

Round 1:
  GPU 0 → GPU 1: Send Shard0
  GPU 1 → GPU 2: Send Shard1
  ...
  GPU 7 → GPU 0: Send Shard7
  
  After Round 1:
    GPU 0: [Shard0, Shard7]
    GPU 1: [Shard1, Shard0]
    GPU 2: [Shard2, Shard1]
    ...

Round 2-7:
  Continue rotating shards
  
After 7 rounds:
  ALL GPUs have: [Shard0, Shard1, Shard2, ..., Shard7]
  = Full Block 0 parameters (274M params, 548 MB)

Memory State:
  GPU 0: 68.5 MB (shard) + 548 MB (temporary gathered) = 616.5 MB
  This is temporary! Will be freed after computation.

3b. Compute Layer 0 Forward
──────────────────────────────────────────────────────────────
x = [batch=2, seq=1024, hidden=2880]  # 2.3 MB

# Attention computation
Q = x @ W_q  # [2, 1024, 2880] @ [2880, 2880] = [2, 1024, 2880]
K = x @ W_k  # [2, 1024, 2880] @ [1440, 2880] = [2, 1024, 1440]  (GQA)
V = x @ W_v  # [2, 1024, 2880] @ [1440, 2880] = [2, 1024, 1440]

# Attention scores (NO FlashAttention available!)
attn = Q @ K.transpose(-2, -1)  # [2, 64, 1024, 1024] = 0.27 GB per layer!
attn = attn / sqrt(head_dim)
attn = attn + causal_mask
attn = F.softmax(attn, dim=-1)   # [2, 64, 1024, 1024] = 0.27 GB

# Attention output
out = attn @ V  # [2, 64, 1024, 64] = 0.034 GB

# Store activations for backward pass
x_0 = x.clone()  # Store input (for residual connection)
attn_out = out.reshape([2, 1024, 2880])  # 2.3 MB

# MoE computation
router_logits = x @ W_router  # [2, 1024, 8]
router_probs = F.softmax(router_logits, dim=-1)
top_k_indices = torch.topk(router_probs, k=2)[1]

# Select top 2 experts per token
expert_outputs = []
for expert_id in range(8):
    mask = (top_k_indices == expert_id)
    if mask.any():
        expert_input = x[mask]
        expert_out = expert_mlp[expert_id](expert_input)
        expert_outputs.append(expert_out)
        
# Combine expert outputs
moe_out = combine_expert_outputs(expert_outputs, router_probs, top_k_indices)

# Residual connection
x = x_0 + attn_out + moe_out  # [2, 1024, 2880] = 2.3 MB

3c. FSDP Release Gathered Parameters
──────────────────────────────────────────────────────────────
After Layer 0 computation:
  FSDP immediately frees gathered Block 0 parameters
  
  GPU 0: Releases 548 MB temporary buffer
  Memory freed: 548 MB
  
  GPU 0 now only has: 68.5 MB (its shard)

Repeat for Layers 1-23:
  Each layer: All-gather → Compute → Release

Forward Pass Memory Peak:
──────────────────────────────────────────────────────────────
Base memory (sharded params): 1.64 GB
+ Temporary gathered (1-2 blocks): ~1.1 GB
+ Activations (all layers): ~7.57 GB
= Peak: ~10.3 GB per GPU during forward pass

#### MICRO-STEP 0: Backward Pass ####
──────────────────────────────────────────────────────────────
loss.backward()  # Line 765

ATOMIC BACKWARD PASS:
──────────────────────────────────────────────────────────────
Step 1: Loss Gradient Initialization
  loss = scalar tensor (requires_grad=True)
  loss.grad = None
  
  Compute: dloss/dloss = 1.0
  loss.backward() sets loss.grad = 1.0

Step 2: Backward Through Layers (Reverse Order: 23 → 0)

Layer 23 Backward:
──────────────────────────────────────────────────────────────
2a. All-Gather Block 23 Parameters
  Same as forward: Gather full Block 23 params temporarily

2b. Compute Gradients
  # Given: grad_output = gradient from loss
  grad_input = grad_output @ W.transpose()  # Backward through linear layer
  
  # Gradient w.r.t. parameters
  grad_W = grad_output.transpose() @ input  # Stored in param.grad
  
  # Gradient w.r.t. input (for next layer)
  grad_input = grad_output @ W.transpose()
  
  # Store gradients
  param.grad += grad_W / 32  # Divide by grad_accum_steps!
  # Note: += because we accumulate across micro-steps

2c. Release Gathered Parameters
  Free temporary buffers

Step 3: Gradient Accumulation
──────────────────────────────────────────────────────────────
For each parameter:
  param.grad = (param.grad + new_grad) / grad_accum_steps
  
  Actually: param.grad += new_grad / grad_accum_steps
  
  After micro-step 0:
    param.grad = 0 + (new_grad / 32)
  
  After micro-step 1:
    param.grad = (new_grad_0 / 32) + (new_grad_1 / 32)
  
  After micro-step 31:
    param.grad = sum(new_grad_i / 32) for i in [0..31]
                = average of all 32 micro-step gradients

Repeat for Micro-Steps 1-31:
  Each micro-step: Forward → Backward → Accumulate

Gradient Accumulation Complete:
──────────────────────────────────────────────────────────────
All 32 micro-steps finished:
  Each GPU has accumulated gradients for its parameter shard
  
  GPU 0: grad_0 = average of 32 micro-step gradients
  GPU 1: grad_1 = average of 32 micro-step gradients
  ...
  GPU 7: grad_7 = average of 32 micro-step gradients

Memory state:
  Sharded gradients: ~1.64 GB per GPU
```

### 2.4 Gradient Clipping

```python
# Line 777-782: Gradient clipping
if args.grad_clip > 0:
    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

ATOMIC GRADIENT CLIPPING:
──────────────────────────────────────────────────────────────
Step 1: Unscale Gradients (if using FP16)
  Not applicable (using bf16, scaler disabled)

Step 2: Compute Global Norm
──────────────────────────────────────────────────────────────
Local norm on each GPU:
  GPU 0: local_norm² = sum(grad_0²) = ||grad_0||²
  GPU 1: local_norm² = sum(grad_1²) = ||grad_1||²
  ...
  GPU 7: local_norm² = sum(grad_7²) = ||grad_7||²

All-Reduce (SUM) to get global norm:
──────────────────────────────────────────────────────────────
global_norm² = ||grad_0||² + ||grad_1||² + ... + ||grad_7||²
global_norm = sqrt(global_norm²)

NCCL All-Reduce Operation:
──────────────────────────────────────────────────────────────
Ring Algorithm (7 rounds):

Round 1:
  GPU 0 → GPU 1: Send ||grad_0||²
  GPU 1: Receives ||grad_0||², computes ||grad_0||² + ||grad_1||²
  GPU 1 → GPU 2: Send ||grad_0||² + ||grad_1||²
  ...

After 7 rounds:
  ALL GPUs have: global_norm² = sum of all local norms²

Step 3: Clip Gradients
──────────────────────────────────────────────────────────────
if global_norm > grad_clip (1.0):
    clip_coef = grad_clip / global_norm
  
    GPU 0: grad_0 *= clip_coef
    GPU 1: grad_1 *= clip_coef
    ...
    GPU 7: grad_7 *= clip_coef

Example:
  global_norm = 2.5
  clip_coef = 1.0 / 2.5 = 0.4
  
  All gradients scaled by 0.4
```

### 2.5 Optimizer Step: AdamW Update

```python
# Line 784-789: Optimizer step
optimizer.step()

ATOMIC ADAMW UPDATE:
──────────────────────────────────────────────────────────────
For each parameter tensor in shard:

Step 1: FSDP All-Reduce Gradients
──────────────────────────────────────────────────────────────
BEFORE:
  GPU 0: grad_0 (shard 0 gradients)
  GPU 1: grad_1 (shard 1 gradients)
  ...
  GPU 7: grad_7 (shard 7 gradients)

NCCL All-Reduce (SUM):
──────────────────────────────────────────────────────────────
Ring Algorithm (7 rounds):

Round 1:
  GPU 0 → GPU 1: Send grad_0
  GPU 1: Receives grad_0, computes grad_0 + grad_1
  GPU 1 → GPU 2: Send grad_0 + grad_1
  ...

After 7 rounds:
  ALL GPUs have: sum_grad = grad_0 + grad_1 + ... + grad_7

Average:
  avg_grad = sum_grad / 8  # Divide by world_size

AFTER:
  GPU 0: avg_grad (for its shard)
  GPU 1: avg_grad (for its shard)
  ...
  GPU 7: avg_grad (for its shard)

Step 2: AdamW Update (Per GPU)
──────────────────────────────────────────────────────────────
For each parameter W in GPU's shard:

  # Get optimizer state
  state = optimizer.state[W]
  step = state['step'] + 1
  exp_avg = state['exp_avg']      # Momentum buffer (fp32)
  exp_avg_sq = state['exp_avg_sq'] # Variance buffer (fp32)
  
  # Get gradient
  grad = W.grad  # Already averaged across GPUs (bf16)
  grad_fp32 = grad.float()  # Convert to fp32 for AdamW
  
  # Update biased first moment estimate (momentum)
  exp_avg = beta1 * exp_avg + (1 - beta1) * grad_fp32
  # exp_avg = 0.9 * exp_avg + 0.1 * grad_fp32
  
  # Update biased second raw moment estimate (variance)
  exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad_fp32 * grad_fp32
  # exp_avg_sq = 0.95 * exp_avg_sq + 0.05 * grad_fp32²
  
  # Bias correction
  bias_correction1 = 1 - beta1 ** step
  bias_correction2 = 1 - beta2 ** step
  
  # Compute update
  step_size = lr / bias_correction1
  
  # Weight decay
  if weight_decay > 0:
      W_fp32 = W.float()
      W_fp32 = W_fp32 * (1 - lr * weight_decay)
  
  # AdamW update
  denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)) + eps
  update = exp_avg / denom
  W_fp32 = W_fp32 - step_size * update
  
  # Convert back to bf16
  W.data = W_fp32.to(torch.bfloat16)
  
  # Update optimizer state
  state['step'] = step
  state['exp_avg'] = exp_avg
  state['exp_avg_sq'] = exp_avg_sq

ATOMIC BREAKDOWN OF ONE PARAMETER UPDATE:
──────────────────────────────────────────────────────────────
Example: W shape [2880, 2880] = 8.3M parameters

GPU 0 (has shard 0 of this weight):
  W_shard = W[0:360, :]  # 1/8 of rows
  
  Before update:
    W_shard = tensor([...], dtype=bf16, device=cuda:0)  # 69 MB
    grad = tensor([...], dtype=bf16, device=cuda:0)      # 69 MB
    exp_avg = tensor([...], dtype=fp32, device=cuda:0)   # 138 MB
    exp_avg_sq = tensor([...], dtype=fp32, device=cuda:0) # 138 MB
  
  After All-Reduce:
    grad = averaged_grad (from all 8 GPUs)
  
  After AdamW update:
    W_shard = updated_values (only GPU 0's shard updated)
    exp_avg = updated_momentum
    exp_avg_sq = updated_variance

Memory Operations:
  - Read: W_shard, grad, exp_avg, exp_avg_sq
  - Compute: exp_avg, exp_avg_sq, update
  - Write: W_shard, exp_avg, exp_avg_sq
  - No inter-GPU communication needed (already averaged gradients)

Step 3: Synchronize
──────────────────────────────────────────────────────────────
torch.cuda.synchronize()  # Line 793

ATOMIC OPERATION:
──────────────────────────────────────────────────────────────
Each GPU:
  1. Wait for all CUDA kernels to complete
  2. Ensure all memory writes are finished
  3. Return control to CPU
  
Purpose:
  - Accurate timing measurements
  - Ensure all GPUs complete before logging
```

---

## 3. GPU Communication: FSDP All-Gather & All-Reduce

### 3.1 NCCL Ring Algorithm: First Principles

**Problem:** How to efficiently sum/average data across 8 GPUs?

**Naive Approach (Poor):**
```
GPU 0 sends to GPU 1 → GPU 1 sums → sends to GPU 2 → ...
After 7 rounds: GPU 7 has sum
GPU 7 broadcasts to all others: 7 more rounds
Total: 14 communication rounds
Bandwidth: Only 1 GPU active at a time
```

**Ring Algorithm (Optimal):**
```
All GPUs communicate simultaneously in a ring topology
Total: 7 rounds (minimal!)
Bandwidth: All GPUs active simultaneously
```

### 3.2 All-Reduce: Detailed Ring Algorithm

```
RING TOPOLOGY:
──────────────────────────────────────────────────────────────
GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 4 → GPU 5 → GPU 6 → GPU 7
  ↑                                                              │
  └──────────────────────────────────────────────────────────────┘

INITIAL STATE:
──────────────────────────────────────────────────────────────
GPU 0: [g₀]
GPU 1: [g₁]
GPU 2: [g₂]
GPU 3: [g₃]
GPU 4: [g₄]
GPU 5: [g₅]
GPU 6: [g₆]
GPU 7: [g₇]

PHASE 1: REDUCE-SCATTER (7 rounds)
──────────────────────────────────────────────────────────────
Goal: Each GPU gets partial sum

Round 1:
  GPU 0 → GPU 1: Send g₀, GPU 1 computes [g₁ + g₀]
  GPU 1 → GPU 2: Send g₁, GPU 2 computes [g₂ + g₁]
  GPU 2 → GPU 3: Send g₂, GPU 3 computes [g₃ + g₂]
  ...
  GPU 7 → GPU 0: Send g₇, GPU 0 computes [g₀ + g₇]

Round 2:
  GPU 0 → GPU 1: Send [g₀ + g₇], GPU 1 computes [g₁ + g₀ + g₀ + g₇]
  GPU 1 → GPU 2: Send [g₁ + g₀], GPU 2 computes [g₂ + g₁ + g₁ + g₀]
  ...

After 7 rounds:
  Each GPU has sum of all gradients assigned to its "segment"
  
  GPU 0: sum(g₀, g₇, g₆, g₅, g₄, g₃, g₂, g₁) = total_sum
  GPU 1: total_sum (same)
  GPU 2: total_sum (same)
  ...
  GPU 7: total_sum (same)

PHASE 2: ALL-GATHER (7 rounds)
──────────────────────────────────────────────────────────────
Goal: Distribute total_sum to all GPUs

Actually, after reduce-scatter, all GPUs already have total_sum!
So all-gather is just ensuring synchronization.

Final Result:
  ALL GPUs: total_sum / 8 = average_gradient
```

**Time Complexity:**
- Communication rounds: 7 (for 8 GPUs)
- Data transferred per round: 1/8 of total gradient size
- Total data transferred: 2 × (7/8) × gradient_size ≈ 1.75 × gradient_size
- Time: ~2ms for 2GB gradients on NVLink

### 3.3 All-Gather: Detailed Ring Algorithm

```
GOAL: Each GPU needs full block parameters from all GPUs

INITIAL STATE:
──────────────────────────────────────────────────────────────
GPU 0: [Shard0]  (68.5 MB)
GPU 1: [Shard1]  (68.5 MB)
GPU 2: [Shard2]  (68.5 MB)
...
GPU 7: [Shard7]  (68.5 MB)

TARGET STATE:
──────────────────────────────────────────────────────────────
GPU 0: [Shard0, Shard1, Shard2, ..., Shard7]  (548 MB)
GPU 1: [Shard0, Shard1, Shard2, ..., Shard7]  (548 MB)
...
GPU 7: [Shard0, Shard1, Shard2, ..., Shard7]  (548 MB)

RING ALGORITHM (7 rounds):
──────────────────────────────────────────────────────────────
Round 1:
  GPU 0 sends Shard0 → GPU 1
  GPU 1 sends Shard1 → GPU 2
  GPU 2 sends Shard2 → GPU 3
  ...
  GPU 7 sends Shard7 → GPU 0
  
  After Round 1:
    GPU 0: [Shard0, Shard7]
    GPU 1: [Shard1, Shard0]
    GPU 2: [Shard2, Shard1]
    ...

Round 2:
  GPU 0 sends Shard7 → GPU 1 (GPU 1 already has Shard0, now gets Shard7)
  GPU 1 sends Shard0 → GPU 2 (GPU 2 already has Shard1, now gets Shard0)
  ...
  
  After Round 2:
    GPU 0: [Shard0, Shard7, Shard6]
    GPU 1: [Shard1, Shard0, Shard7]
    GPU 2: [Shard2, Shard1, Shard0]
    ...

Round 7:
  All GPUs have all shards!

FINAL STATE:
──────────────────────────────────────────────────────────────
GPU 0: [Shard0, Shard1, Shard2, Shard3, Shard4, Shard5, Shard6, Shard7]
GPU 1: [Shard0, Shard1, Shard2, Shard3, Shard4, Shard5, Shard6, Shard7]
...
GPU 7: [Shard0, Shard1, Shard2, Shard3, Shard4, Shard5, Shard6, Shard7]

Time: ~1ms for 548 MB block on NVLink
```

### 3.4 FSDP Communication During Forward Pass

```
FORWARD PASS THROUGH TRANSFORMERBLOCK 0:
──────────────────────────────────────────────────────────────
Timeline:

t=0ms:   FSDP calls all_gather(Block0.params)
t=0ms:   NCCL starts ring all-gather
t=1ms:   All-gather complete, all GPUs have full Block0
t=1ms:   Forward computation starts
t=5ms:   Forward computation complete
t=5ms:   FSDP releases gathered params (frees memory)
t=5ms:   Next layer starts

Memory Timeline:
──────────────────────────────────────────────────────────────
t=0ms:   Base: 1.64 GB (sharded params)
t=1ms:   Peak: 1.64 + 0.55 = 2.19 GB (gathered)
t=5ms:   Base: 1.64 GB (released)

Total forward pass (24 layers):
  Communication time: 24 × 1ms = 24ms
  Computation time: 24 × 4ms = 96ms
  Total: 120ms
```

---

## 4. Learning Rate: Warmup + Cosine Decay

### 4.1 Learning Rate Schedule Function

```python
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
        return args.min_lr
    decay_ratio = (it - args.warmup_iters) / decay_range
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)
```

### 4.2 Warmup Phase (iterations 0-1999)

```
ATOMIC COMPUTATION:
──────────────────────────────────────────────────────────────
it = 0:
  lr = 3e-4 * (0 + 1) / 2000 = 3e-4 * 1/2000 = 1.5e-7

it = 1:
  lr = 3e-4 * (1 + 1) / 2000 = 3e-4 * 2/2000 = 3.0e-7

it = 999:
  lr = 3e-4 * (999 + 1) / 2000 = 3e-4 * 1000/2000 = 1.5e-4

it = 1999:
  lr = 3e-4 * (1999 + 1) / 2000 = 3e-4 * 1.0 = 3e-4 ✅

Formula: lr(it) = lr_max * (it + 1) / warmup_iters
         Linear ramp from 0 to lr_max
```

**Why Warmup?**
- **Stability**: Large gradients at initialization can cause instability
- **Adaptive optimizers**: AdamW's momentum/variance estimates need time to stabilize
- **Gradient scale**: Prevents early training from diverging

### 4.3 Cosine Decay Phase (iterations 2000-19999)

```
ATOMIC COMPUTATION:
──────────────────────────────────────────────────────────────
it = 2000 (start of decay):
  decay_range = 20000 - 2000 = 18000
  decay_ratio = (2000 - 2000) / 18000 = 0.0
  coeff = 0.5 * (1.0 + cos(π * 0.0))
        = 0.5 * (1.0 + cos(0))
        = 0.5 * (1.0 + 1.0)
        = 1.0
  lr = 3e-5 + 1.0 * (3e-4 - 3e-5)
    = 3e-5 + 2.7e-4
    = 3e-4 ✅ (matches warmup end)

it = 11000 (middle of decay):
  decay_range = 18000
  decay_ratio = (11000 - 2000) / 18000 = 9000 / 18000 = 0.5
  coeff = 0.5 * (1.0 + cos(π * 0.5))
        = 0.5 * (1.0 + cos(π/2))
        = 0.5 * (1.0 + 0.0)
        = 0.5
  lr = 3e-5 + 0.5 * (3e-4 - 3e-5)
    = 3e-5 + 1.35e-4
    = 1.65e-4

it = 20000 (end of decay):
  decay_range = 18000
  decay_ratio = (20000 - 2000) / 18000 = 1.0
  coeff = 0.5 * (1.0 + cos(π * 1.0))
        = 0.5 * (1.0 + cos(π))
        = 0.5 * (1.0 + (-1.0))
        = 0.5 * 0.0
        = 0.0
  lr = 3e-5 + 0.0 * (3e-4 - 3e-5)
    = 3e-5 ✅ (matches min_lr)

Formula: lr(it) = lr_min + coeff * (lr_max - lr_min)
         where coeff = 0.5 * (1 + cos(π * decay_ratio))
         Smooth cosine curve from lr_max to lr_min
```

**Cosine Decay Mathematics:**
```
Cosine function: cos(θ) ranges from 1 to -1 as θ goes from 0 to π

Our transformation:
  decay_ratio ∈ [0, 1]  →  maps to θ ∈ [0, π]
  cos(π * decay_ratio) ∈ [1, -1]
  
  coeff = 0.5 * (1 + cos(π * decay_ratio))
         ∈ [0.5 * (1 + 1), 0.5 * (1 + (-1))]
         ∈ [1.0, 0.0]

  lr = lr_min + coeff * (lr_max - lr_min)
      ∈ [lr_min + 1.0 * (lr_max - lr_min), lr_min + 0.0 * (lr_max - lr_min)]
      ∈ [lr_max, lr_min]

Smooth decay curve:
  - Starts at lr_max (gradual decrease)
  - Middle: lr_max + lr_min / 2 (smooth transition)
  - Ends at lr_min (slow final approach)
```

### 4.4 Learning Rate in Optimizer Step

```
ATOMIC USAGE IN ADAMW:
──────────────────────────────────────────────────────────────
optimizer.step() internally uses:
  step_size = lr / bias_correction1
  
  bias_correction1 = 1 - beta1^step
  For step = 1: bias_correction1 = 1 - 0.9¹ = 0.1
  For step = 1000: bias_correction1 = 1 - 0.9¹⁰⁰⁰ ≈ 1.0
  
  Effective step_size = lr / bias_correction1
  
  Early training: step_size ≈ lr / 0.1 = 10 × lr (larger steps!)
  Later training: step_size ≈ lr / 1.0 = lr (normal steps)

Parameter Update:
  W = W - step_size * (exp_avg / sqrt(exp_avg_sq))
  
  With warmup:
    Early: lr small → step_size small → conservative updates
    Later: lr large → step_size large → aggressive updates
  
  With decay:
    Early decay: lr = 3e-4 → large updates
    Middle decay: lr = 1.65e-4 → medium updates
    Late decay: lr = 3e-5 → small updates (fine-tuning)
```

---

## 5. Complete Iteration Timeline

```
ITERATION 1000: COMPLETE ATOMIC TIMELINE
═══════════════════════════════════════════════════════════════

t=0ms:   ITERATION START
──────────────────────────────────────────────────────────────
         Learning Rate Update:
           - Call get_lr(1000, args)
           - Compute: lr = 3e-4 * 1001/2000 = 1.5e-4
           - Set optimizer.param_groups[0]["lr"] = 1.5e-4
           - Time: ~0.001ms

t=0.001ms: Zero Gradients
──────────────────────────────────────────────────────────────
         - Set all param.grad = None
         - Free gradient memory buffers
         - Time: ~0.1ms

t=0.1ms:  GRADIENT ACCUMULATION LOOP START (32 micro-steps)
──────────────────────────────────────────────────────────────

Micro-Step 0:
t=0.1ms:  - Load batch: x, y = train_loader.get_batch()
          - Transfer to GPU: x.to(device, non_blocking=True)
          - Time: ~0.5ms

t=0.6ms:  Forward Pass Start
          ──────────────────────────────────────────────────
          Layer 0:
          t=0.6ms:  All-gather Block0 params (1ms)
          t=1.6ms:  Forward computation (4ms)
          t=5.6ms:  Release gathered params
          
          Layer 1:
          t=5.6ms:  All-gather Block1 params (1ms)
          t=6.6ms:  Forward computation (4ms)
          t=10.6ms: Release gathered params
          
          ... (repeat for 24 layers)
          
          t=120ms: Forward complete

t=120ms:  Backward Pass Start
          ──────────────────────────────────────────────────
          Layer 23:
          t=120ms:  All-gather Block23 params (1ms)
          t=121ms:  Backward computation (4ms)
          t=125ms:  Release gathered params
          
          Layer 22:
          t=125ms:  All-gather Block22 params (1ms)
          t=126ms:  Backward computation (4ms)
          t=130ms:  Release gathered params
          
          ... (repeat for 24 layers)
          
          t=240ms: Backward complete

t=240ms:  Accumulate gradients
          ──────────────────────────────────────────────────
          For each param:
            param.grad += (new_grad / 32)
          Time: ~1ms

t=241ms:  Micro-Step 0 Complete

Repeat Micro-Steps 1-31:
  Each micro-step: ~241ms
  Total accumulation time: 32 × 241ms = 7,712ms (7.7 seconds)

t=7712ms: GRADIENT ACCUMULATION COMPLETE
──────────────────────────────────────────────────────────────
         All 32 micro-steps finished
         Each GPU has accumulated gradients

t=7712ms: Gradient Clipping
──────────────────────────────────────────────────────────────
         - Compute local norm on each GPU: ~0.5ms
         - All-reduce global norm: ~2ms
         - Clip if needed: ~0.1ms
         Total: ~2.6ms

t=7714.6ms: Optimizer Step
──────────────────────────────────────────────────────────────
         Step 1: All-reduce gradients (2ms)
         Step 2: AdamW update per GPU (10ms)
         Step 3: Synchronize (0.1ms)
         Total: ~12ms

t=7726.6ms: Synchronize & Logging
──────────────────────────────────────────────────────────────
         - torch.cuda.synchronize(): ~0.1ms
         - All-reduce loss for logging: ~0.1ms
         - Print/log: ~1ms
         Total: ~1.2ms

t=7727.8ms: ITERATION COMPLETE
──────────────────────────────────────────────────────────────
         iter_num += 1
         Next iteration starts

TOTAL ITERATION TIME: ~7.73 seconds
──────────────────────────────────────────────────────────────
Breakdown:
  - Gradient accumulation: 7.7s (99.6%)
  - Optimizer step: 0.012s (0.15%)
  - Logging: 0.001s (0.01%)
  - Other: 0.017s (0.24%)

Throughput:
  Total tokens: 2 × 1024 × 32 × 8 = 524,288 tokens
  Time: 7.73 seconds
  Throughput: 524,288 / 7.73 = 67,825 tokens/second
```

---

## Summary: Key Atomic Operations

### **Memory Operations:**
1. **All-Gather**: GPU 0 receives 7 shards, temporarily uses 548 MB
2. **Computation**: Matrix multiplications on GPU, ~4ms per layer
3. **All-Reduce**: 7 communication rounds, ~2ms total
4. **Gradient Accumulation**: += operation, divides by 32

### **Communication Operations:**
1. **Ring Topology**: Physical NVLink connections
2. **Ring Algorithm**: 7 rounds for 8 GPUs (optimal)
3. **Bandwidth**: 600 GB/s per NVLink
4. **Latency**: ~1-2ms per all-reduce/all-gather

### **Learning Rate:**
1. **Warmup**: Linear ramp from 0 to 3e-4 over 2000 iterations
2. **Decay**: Cosine curve from 3e-4 to 3e-5 over 18000 iterations
3. **Usage**: Applied in optimizer.step() as step_size

### **Training Efficiency:**
- **99.6%** time spent in gradient accumulation (forward/backward)
- **0.15%** time spent in optimizer step
- **Communication overhead**: ~3% (all-gather + all-reduce)
- **GPU utilization**: 100% (always computing)

This atomic-level breakdown shows exactly how FSDP, NCCL, and learning rate scheduling work together to train large models efficiently across multiple GPUs.

