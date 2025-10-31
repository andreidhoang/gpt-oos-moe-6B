# model.py — GPT-OSS-8B-style Transformer with MoE, GQA, RoPE(+stretch), sink-bias,
# optional FlashAttention, and FSDP-friendly reset_parameters() on all modules.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Literal

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Optional Flash-Attention-2 / 3 -----------------------------------------
# FlashAttention3 is API-compatible with FlashAttention2
# FA3 provides significant speedups on Hopper GPUs (H100+) and improved memory efficiency
# Installation: pip install flash-attn --no-build-isolation
# For FA3: Requires CUDA 12.3+, Hopper architecture (compute capability 9.0)
# FA2 works on Ampere (A100, compute capability 8.0) and newer

_flash_available = False
_flash_version = None
try:
    # Try flash_attn >= 2.x / 3.x API (same interface)
    from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func
    _flash_available = True

    # Detect version for informational purposes
    try:
        import flash_attn
        _flash_version = getattr(flash_attn, '__version__', 'unknown')
    except:
        _flash_version = 'unknown'

except Exception:
    _flash_attn_func = None
    _flash_available = False
    _flash_version = None


# ------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------

@dataclass
class RopeScalingConfig:
    # Simple scalar stretch for YaRN-like extension (pragmatic approximation)
    factor: float = 32.0

@dataclass
class ModelConfig:
    # Core dims
    vocab_size: int = 201_088
    hidden_size: int = 2880
    num_hidden_layers: int = 24
    head_dim: int = 64

    # Attention (GQA)
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    attention_bias: bool = True
    attention_dropout: float = 0.0
    dropout: float = 0.0

    # Patterns / positions
    max_position_embeddings: int = 131_072
    sliding_window: int = 128
    layer_types: Optional[List[Literal["sliding_attention", "full_attention"]]] = None

    # MoE
    num_local_experts: int = 8
    experts_per_token: int = 2
    router_aux_loss_coef: float = 0.01  # conservative; 0.01–0.1 common

    # MLP inside each expert (SwiGLU uses 2*FF)
    intermediate_size: int = 2880
    swiglu_clip: float = 7.0

    # RoPE / YaRN
    rope_theta: float = 150_000.0
    rope_scaling: RopeScalingConfig = field(default_factory=RopeScalingConfig)

    # Sink (null-attention bias)
    enable_sink_logit: bool = True
    sink_logit_init: float = 4.0  # positive → allows "attend to nothing"

    # Norms / init / tying
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    eos_token_id: Optional[int] = None

    def __post_init__(self):
        if self.layer_types is None:
            # Default: alternate full <-> sliding attention
            self.layer_types = [
                "full_attention" if i % 2 == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        assert len(self.layer_types) == self.num_hidden_layers
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.head_dim > 0
        # Note: GPT OSS MoE 8B uses hidden_size=2880 with num_attention_heads=64 and head_dim=64
        # This doesn't strictly equal 64*64=4096, but the architecture works correctly
        # Removed assertion: head_dim * num_attention_heads == hidden_size
        assert self.num_key_value_heads <= self.num_attention_heads, \
            f"num_key_value_heads ({self.num_key_value_heads}) > num_attention_heads ({self.num_attention_heads})"
        assert self.experts_per_token <= self.num_local_experts, \
            f"experts_per_token ({self.experts_per_token}) > num_local_experts ({self.num_local_experts})"
        assert self.max_position_embeddings > 0, \
            f"max_position_embeddings must be positive, got {self.max_position_embeddings}"
        assert self.intermediate_size > 0, \
            f"intermediate_size must be positive, got {self.intermediate_size}"
        assert self.rope_scaling.factor > 0, \
            f"rope_scaling.factor must be positive, got {self.rope_scaling.factor}"
        assert self.router_aux_loss_coef >= 0, \
            f"router_aux_loss_coef must be non-negative, got {self.router_aux_loss_coef}"

    @property
    def group_size(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm is ~15% faster than LayerNorm and empirically works just as well.
    It normalizes using only the RMS (no mean centering), reducing computation.

    Reference: https://arxiv.org/abs/1910.07467

    Args:
        dim: Normalization dimension
        eps: Epsilon for numerical stability (default: 1e-5)

    Shape:
        - Input: (*, dim)
        - Output: (*, dim) same shape as input
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm.

        Formula: x * rsqrt(mean(x^2) + eps) * weight

        Note: We cast to float32 for numerical stability during variance computation,
        then cast back to input dtype.
        """
        # Compute variance in float32 for stability
        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # Normalize
        x = x * torch.rsqrt(var + self.eps)
        # Scale by learnable weight
        return self.weight * x

    def reset_parameters(self):
        """
        Reset parameters to initial values.
        Called by FSDP for meta device materialization.
        """
        with torch.no_grad():
            self.weight.fill_(1.0)


def swiglu(x: torch.Tensor, clip: Optional[float] = None) -> torch.Tensor:
    """
    SwiGLU (Swish-Gated Linear Unit) activation function.

    SwiGLU splits the input into two halves: gate and value.
    It applies Swish (SiLU) to the gate and multiplies with the value.

    Formula: SwiGLU(x) = Swish(gate) * up
             where gate, up = split(x, dim=-1)
             and Swish(x) = x * sigmoid(x)

    Reference: https://arxiv.org/abs/2002.05202 (GLU Variants Improve Transformer)

    Args:
        x: Input tensor with last dimension = 2 * desired_output_dim
        clip: Optional clipping value for numerical stability (typically 7.0)

    Returns:
        Tensor with last dimension = input_last_dim / 2

    """
    # Split into two halves
    up, gate = x.chunk(2, dim=-1)
    # Optional clipping for training stability
    # Prevents extreme values that can cause NaN losses
    if clip is not None:
        up = up.clamp(-clip, clip)
        gate = gate.clamp(-clip, clip)
    # Apply SwiGLU: silu(gate) * up
    # F.silu is Swish activation: x * sigmoid(x)
    return F.silu(gate) * up

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings with YaRN-style scaling.

    References:
    - RoPE: https://arxiv.org/abs/2104.09864
    - YaRN: https://arxiv.org/abs/2309.00071

    Applies rotational transformations to queries and keys based on their positions,
    providing relative position information. Supports context extension through YaRN scaling.

    Args:
        head_dim: Dimension of each attention head
        rope_theta: Base frequency parameter (higher = better long context)
        scale_cfg: YaRN scaling configuration

    Shape:
        - Input: (B*H, T, head_dim) where B=batch, H=heads, T=sequence length
        - Positions: (B*H, T) position indices
        - Output: (B*H, T, head_dim)
    """

    def __init__(self, head_dim: int, rope_theta: float, scale_cfg: RopeScalingConfig):
        super().__init__()
        self.head_dim = int(head_dim)
        self.theta = float(rope_theta)
        self.factor = float(scale_cfg.factor)

        # Placeholders; real buffers (CPU) in reset_parameters()
        self.register_buffer("inv_freq_base", torch.empty(0), persistent=False)
        self._seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

        # Immediately build once so rank0 (non-meta) has valid buffers
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize or reset RoPE frequency buffers.

        Called during:
        1. Initial construction (rank 0)
        2. FSDP meta materialization (non-rank-0)
        3. Checkpoint loading
        """
        device = torch.device("cpu")  # Build on CPU, FSDP will move as needed

        # Compute base inverse frequencies
        # inv_freq[i] = 1 / (theta ^ (2i / head_dim)) for i in [0, head_dim/2)
        inv_freq = 1.0 / (
            self.theta ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
                / self.head_dim
            )
        )

        self.register_buffer("inv_freq_base", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self.register_buffer(
            "cos_cached",
            torch.empty(0, dtype=torch.float32, device=device),
            persistent=False
        )
        self.register_buffer(
            "sin_cached",
            torch.empty(0, dtype=torch.float32, device=device),
            persistent=False
        )

    def _update_cache(self, seqlen: int, device: torch.device, dtype: torch.dtype):
        """
        Update cos/sin cache if needed.

        Args:
            seqlen: Maximum sequence length needed
            device: Target device
            dtype: Target dtype
        """
        # check if cache is valid
        if (
            seqlen <= self._seq_len_cached and
            self.cos_cached.device == device and
            self.cos_cached.dtype == dtype and
            self.cos_cached.shape[0] >= seqlen  # Ensure cache is large enough
        ): return

        # compute scaled positions
        # YaRN: divide positions by scaling factor
        pos = torch.arange(seqlen, device=device, dtype=torch.float32) / self.factor # ie: [0, 1, 2, 3] / 32 = [0, 0.03125, 0.0625, 0.09375]

        # compute frequencies: outer product of positions and inv_freq
        # freqs[i, j] = pos[i] * inv_freq[j]
        freqs = torch.einsum("s,f->sf", pos, self.inv_freq_base.to(device=device)) # shape [seqlen, head_dim/2]

        # compute cos and sin
        cos = torch.cos(freqs).to(dtype) # shape [seqlen, head_dim/2]
        sin = torch.sin(freqs).to(dtype) 

        # interleave for application to x
        # this creates the pattern needed for complex rotation
        cos = torch.stack([cos, cos], dim=-1).reshape(seqlen, -1) # shape [seqlen, head_dim]
        sin = torch.stack([sin, sin], dim=-1).reshape(seqlen, -1) 

        # cache
        self.cos_cached = cos # shape [seqlen, head_dim]
        self.sin_cached = sin # shape [seqlen, head_dim]
        self._seq_len_cached = seqlen # ie: 4

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensor.

        Args:
            x: Input tensor (B*H, T, head_dim)
            positions: Position indices (B*H, T)

        Returns:
            Rotated tensor (B*H, T, head_dim)
        """
        
        BxH, T, Dh = x.shape # ie: (32, 4, 8)

        # Ensure cache is up to date
        max_pos = int(positions.max().item()) + 1
        self._update_cache(max_pos, x.device, x.dtype)

        # Validate positions are within cache bounds
        if positions.max().item() >= self.cos_cached.shape[0]:
            raise ValueError(
                f"Position {positions.max().item()} exceeds cache size {self.cos_cached.shape[0]}. "
                f"This suggests a bug in position computation or max_position_embeddings is too small."
            )

        # Index cos/sin by positions
        cos = self.cos_cached[positions] # shape [B*H, T, head_dim]
        sin = self.sin_cached[positions] # shape [B*H, T, head_dim]

        # Split into even and odd indices
        x1, x2 = x[..., ::2], x[..., 1::2] # shape [B*H, T, head_dim/2], shape [B*H, T, head_dim/2]

        xr1 = x1 * cos[..., ::2] - x2 * sin[..., ::2]
        xr2 = x1 * sin[..., ::2] + x2 * cos[..., ::2]

        # Interleave back
        out = torch.empty_like(x)
        out[..., ::2] = xr1
        out[..., 1::2] = xr2

        return out

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with optional Flash Attention.

    GQA reduces KV cache size by grouping multiple query heads per KV head.
    For example, with 64 query heads and 8 KV heads, we have 8 queries per KV head,
    resulting in 8x reduction in KV cache size.

    Features:
    - Grouped query attention for efficiency
    - RoPE for relative position encoding
    - Flash Attention 2/3 for speed (when available)
    - Sliding window attention (alternating layers)
    - Attention sinks for long-context stability

    Args:
        cfg: Model configuration

    Shape:
        - Input: (B, T, H) where B=batch, T=sequence, H=hidden_dim
        - Positions: (B, T) position indices for RoPE
        - Causal mask: (T, T) boolean mask (True=keep)
        - Output: (B, T, H)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        H = cfg.hidden_size
        self.n_head = int(cfg.num_attention_heads)
        self.n_kv = int(cfg.num_key_value_heads)
        self.dh = int(cfg.head_dim)
        self.group_size = int(cfg.group_size)
        self.scale = 1.0 / math.sqrt(self.dh)

        # Store flash attention availability for logging
        self._flash_available = _flash_available
        self._flash_version = _flash_version

        # Dropouts
        self.drop_attn = nn.Dropout(cfg.attention_dropout)
        self.drop_resid = nn.Dropout(cfg.dropout)

        # RoPE for positional encoding
        self.rope = RotaryEmbedding(self.dh, cfg.rope_theta, cfg.rope_scaling)

        # Query, Key, Value projections
        # Q: (H) -> (n_head * head_dim)
        # K, V: (H) -> (n_kv * head_dim)  [smaller due to grouping]
        self.q = nn.Linear(H, self.n_head * self.dh, bias=cfg.attention_bias)
        self.k = nn.Linear(H, self.n_kv * self.dh, bias=cfg.attention_bias)
        self.v = nn.Linear(H, self.n_kv * self.dh, bias=cfg.attention_bias)

        # Output projection
        self.o = nn.Linear(self.n_head * self.dh, H, bias=True)

        # Attention sinks (learnable "attend to nothing" logit per head)
        self.use_sink = bool(cfg.enable_sink_logit)
        if self.use_sink:
            self.sink_logit = nn.Parameter(
                torch.full((self.n_head,), float(cfg.sink_logit_init))
            )
        else:
            # Dummy buffer for FSDP compatibility
            self.register_buffer("sink_logit", torch.empty(0), persistent=False)

        # Store init settings for reset_parameters
        self.init_std = float(cfg.initializer_range)
        self.sink_init = float(cfg.sink_logit_init)

        # Initialize once for rank 0
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize or reset parameters.

        Called during:
        1. Initial construction (rank 0)
        2. FSDP meta materialization (non-rank-0)
        3. Checkpoint loading
        """
        init_std = getattr(self, "init_std", 0.02)

        with torch.no_grad():
            # Initialize projection weights
            for m in (self.q, self.k, self.v, self.o):
                nn.init.normal_(m.weight, mean=0.0, std=init_std)
                if getattr(m, "bias", None) is not None:
                    m.bias.zero_()

            # Initialize sink logit if used
            if (
                getattr(self, "use_sink", False)
                and hasattr(self, "sink_logit")
                and self.sink_logit.numel() > 0
            ):
                self.sink_logit.fill_(getattr(self, "sink_init", 4.0))

        # Reset RoPE buffers
        if hasattr(self, "rope") and hasattr(self.rope, "reset_parameters"):
            self.rope.reset_parameters()

    def _kv_expand(self, kv: torch.Tensor) -> torch.Tensor:
        """
        Expand KV heads to match query heads for grouped query attention.

        Args:
            kv: (B, T, n_kv * head_dim)

        Returns:
            (B, n_head, T, head_dim)
        """
        B, T, _ = kv.shape
        # Reshape to (B, T, n_kv, head_dim)
        kv = kv.view(B, T, self.n_kv, self.dh)
        # Expand each KV head to group_size query heads
        # (B, T, n_kv, head_dim) -> (B, T, n_kv, group_size, head_dim)
        kv = kv.unsqueeze(3).expand(B, T, self.n_kv, self.group_size, self.dh) # shape [B, T, 8, 64] -> [B, T, 8, 8, 64]
        # Reshape to (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        kv = kv.reshape(B, T, self.n_head, self.dh).transpose(1, 2).contiguous()
        return kv

    @staticmethod
    def _build_local_mask(T: int, device: torch.device, win: int) -> torch.Tensor:
        """
        Build sliding window mask for local attention.

        Args:
            T: Sequence length
            device: Target device
            win: Window size

        Returns:
            (T, T) boolean mask (True=keep)
        """

        i = torch.arange(T, device=device)
        j = torch.arange(T, device=device)
        dist = i[:, None] - j[None, :]  # Distance matrix
        return (dist >= 0) & (dist < win)  # Keep only recent tokens

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        causal_mask: torch.Tensor,
        is_sliding_layer: bool,
        sliding_window: int,
    ) -> torch.Tensor:
        """
        Forward pass of grouped query attention.

        Args:
            x: Input tensor (B, T, H)
            positions: Position indices (B, T) for RoPE
            causal_mask: Causal attention mask (T, T) boolean
            is_sliding_layer: Whether to use sliding window
            sliding_window: Window size for sliding attention

        Returns:
            Output tensor (B, T, H)
        """
        B, T, H = x.shape

        # Q: (B, T, n_head * head_dim)
        # K, V: (B, T, n_kv * head_dim)
        q = self.q(x).view(B, T, self.n_head, self.dh).transpose(1, 2).contiguous() # shape [B, T, n_head, head_dim] -> [B, n_head, T, head_dim] -> [B, 64, T, 64]

        k = self._kv_expand(self.k(x)) # shape [B, n_head, T, head_dim] -> [B, 64, T, 64]
        v = self._kv_expand(self.v(x)) # shape [B, n_head, T, head_dim] -> [B, 64, T, 64]
        # Apply RoPE to queries and keys
        # Need to reshape for RoPE: (B, n_head, T, head_dim) -> (B*n_head, T, head_dim)
        pos_rep = positions.repeat_interleave(self.n_head, 0) # shape [B*n_head, T]

        q = self.rope.apply(
            q.view(B * self.n_head, T, self.dh),
            pos_rep
        ).view(B, self.n_head, T, self.dh)

        k = self.rope.apply(
            k.view(B * self.n_head, T, self.dh),
            pos_rep
        ).view(B, self.n_head, T, self.dh)

        # Fast path: Flash Attention
        # Use FlashAttention on full attention layers (sliding layers need manual attention for sinks)
        # Note: Attention sinks are only needed for sliding window layers,
        # so FlashAttention can be used on full attention layers even if sinks are enabled globally
        use_flash = _flash_available and (not is_sliding_layer)

        if use_flash:
            # Flash Attention expects (B, T, H, D) layout
            qf = q.transpose(1, 2)  # (B, T, n_head, head_dim)
            kf = k.transpose(1, 2)
            vf = v.transpose(1, 2)

            # Flash attention with causal masking
            out = _flash_attn_func(qf, kf, vf, causal=True)  # (B, T, n_head, head_dim)

            # Reshape and project
            out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.dh)
            out = self.o(out)
            return self.drop_resid(out)
        # Manual attention path (supports sliding window + sinks)
        # Compute attention scores: Q @ K^T
        att = torch.einsum("bhtd,bhsd->bhts", q, k) * self.scale  # (B, n_head, T, T)

        # Apply causal mask
        mask = causal_mask  # (T, T) True=keep

        # Apply sliding window if enabled
        if is_sliding_layer:
            local = self._build_local_mask(T, x.device, sliding_window)
            mask = mask & local  # (T, T)

        # Mask out invalid positions
        att = att.masked_fill(~mask.view(1, 1, T, T), float("-inf"))

        # Append attention sink column if enabled
        if (
            getattr(self, "use_sink", False)
            and hasattr(self, "sink_logit")
            and self.sink_logit.numel() > 0
        ):
            # Sink column: learnable "attend to nothing" logit
            sink_col = self.sink_logit.view(1, self.n_head, 1, 1).expand(B, -1, T, 1)
            att = torch.cat([att, sink_col], dim=-1)  # (B, n_head, T, T+1)

        # Softmax to get attention probabilities
        p = F.softmax(att, dim=-1)

        # Drop sink probability (it represents "attend to nothing")
        if (
            getattr(self, "use_sink", False)
            and hasattr(self, "sink_logit")
            and self.sink_logit.numel() > 0
        ):
            p = p[..., :-1]  # (B, n_head, T, T)

        # Apply attention dropout
        p = self.drop_attn(p)

        # Apply attention to values
        y = torch.einsum("bhts,bhsd->bhtd", p, v).contiguous()

        # Reshape and project
        y = y.transpose(1, 2).reshape(B, T, self.n_head * self.dh)
        y = self.o(y)

        return self.drop_resid(y)
        

class MoE(nn.Module):
    """
    A fast, vectorized MoE layer using einsum.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        H = cfg.hidden_size
        E = cfg.num_local_experts
        FF = cfg.intermediate_size
        self.E = int(E)
        self.K = int(cfg.experts_per_token)
        self.clip = float(cfg.swiglu_clip)
        self.router_aux_loss_coef = float(cfg.router_aux_loss_coef)

        # Expert parameters: W_in (H -> 2*FF), W_out (FF -> H)
        self.W_in = nn.Parameter(torch.empty(E, H, 2 * FF))
        self.b_in = nn.Parameter(torch.zeros(E, 2 * FF))
        self.W_out = nn.Parameter(torch.empty(E, FF, H))
        self.b_out = nn.Parameter(torch.zeros(E, H))

        # Router
        self.router = nn.Linear(H, E, bias=True)

        # store init std for reset
        self.init_std = float(cfg.initializer_range)

        # init once for rank0
        self.reset_parameters()

    def reset_parameters(self):
        init_std = getattr(self, "init_std", 0.02)
        with torch.no_grad():
            nn.init.normal_(self.W_in,  mean=0.0, std=init_std)
            nn.init.normal_(self.W_out, mean=0.0, std=init_std)
            self.b_in.zero_(); self.b_out.zero_()
            nn.init.normal_(self.router.weight, mean=0.0, std=init_std)
            if self.router.bias is not None:
                self.router.bias.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x: (B,T,H) -> (S,H) where S=B*T
        B, T, H = x.shape
        S = B * T
        x_flat = x.view(S, H)

        # Route tokens to experts
        logits = self.router(x_flat)  # (S, E)

        # Top-K routing
        topk_weights, topk_indices = torch.topk(logits, self.K, dim=-1)  # (S, K), (S, K)
        topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32).to(x.dtype)

        # Create a one-hot mask for selected experts for each token and top-k choice
        # (S, K) -> (S, K, E)
        expert_mask = F.one_hot(topk_indices, num_classes=self.E)

        # Combine the mask with the weights
        # (S, K, E) * (S, K, 1) -> (S, K, E)
        gating_weights = expert_mask * topk_weights.unsqueeze(-1)

        # Sum weights over K choices to get the final weight for each expert for each token
        # (S, K, E) -> (S, E)
        final_expert_weights = gating_weights.sum(dim=1)

        # Aux (Switch-style) loss for load balancing
        # IMPORTANT: Use final_expert_weights to account for ALL K experts and their weights
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        importance = probs.mean(dim=0)          # (E,) - average probability over tokens
        load = final_expert_weights.mean(dim=0)  # (E,) - average routing weight per expert (accounts for all K)
        aux_loss = self.router_aux_loss_coef * self.E * (importance * load).sum()

        # --- Vectorized Expert Computation ---
        # 1. Apply all experts' W_in to all tokens
        #    'sh,ehd->sed': (S,H) @ (E,H,2FF) -> (S,E,2FF)
        expert_inputs = torch.einsum('sh,ehd->sed', x_flat, self.W_in) + self.b_in
        
        # 2. Apply SwiGLU activation
        #    swiglu halves the last dimension
        expert_outputs = swiglu(expert_inputs, clip=self.clip) # (S,E,FF)

        # 3. Apply all experts' W_out
        #    'sef,efh->seh': (S,E,FF) @ (E,FF,H) -> (S,E,H)
        expert_outputs = torch.einsum('sef,efh->seh', expert_outputs, self.W_out) + self.b_out

        # 4. Weight the expert outputs by the router weights and sum
        #    'seh,se->sh': (S,E,H) * (S,E) -> (S,H)
        weighted_output = torch.einsum('seh,se->sh', expert_outputs, final_expert_weights)

        # Reshape back to (B, T, H)
        out = weighted_output.view(B, T, H)

        # Compute additional metrics for monitoring
        aux_dict = {"router_aux_loss": aux_loss}
        
        # Expert utilization: fraction of tokens each expert handles
        # Count how many tokens each expert is selected for (weighted by routing weights)
        expert_utilization = final_expert_weights.mean(dim=0)  # (E,) average routing weight per expert
        
        # Router entropy: measure of routing diversity (higher = more diverse)
        router_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()  # scalar
        
        # Note: Load balance score is computed in Transformer.forward() after aggregating across layers
        # Removing per-layer computation to avoid redundancy
        
        aux_dict["expert_utilization"] = expert_utilization.detach()  # (E,) per-expert usage
        aux_dict["router_entropy"] = router_entropy.detach()  # scalar

        return out, aux_dict


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.moe = MoE(cfg)

    def reset_parameters(self):
        # Submodules handle their own init; keep for FSDP completeness
        if hasattr(self.norm1, "reset_parameters"): self.norm1.reset_parameters()
        if hasattr(self.attn,  "reset_parameters"): self.attn.reset_parameters()
        if hasattr(self.norm2, "reset_parameters"): self.norm2.reset_parameters()
        if hasattr(self.moe,   "reset_parameters"): self.moe.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        causal_mask: torch.Tensor,
        is_sliding_layer: bool,
        sliding_window: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        a = self.attn(self.norm1(x), positions, causal_mask, is_sliding_layer, sliding_window)
        x = x + a
        m, aux = self.moe(self.norm2(x))
        x = x + m
        return x, aux

class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        H = cfg.hidden_size
        self.embed = nn.Embedding(cfg.vocab_size, H)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm_f = RMSNorm(H, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(H, cfg.vocab_size, bias=False)

        # store init std for reset
        self.init_std = float(cfg.initializer_range)

        # optional tying
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

        # Store FlashAttention status
        self._flash_available = _flash_available
        self._flash_version = _flash_version

        # init once for rank0
        self.reset_parameters()

    def get_flash_attn_info(self) -> str:
        """
        Get FlashAttention status information.

        Returns:
            String describing FlashAttention availability and version
        """
        if not _flash_available:
            return "FlashAttention: Not available (will use manual attention)"

        version_str = f"v{_flash_version}" if _flash_version and _flash_version != 'unknown' else "unknown version"

        # Check which layers can use FlashAttention
        full_attn_layers = sum(1 for lt in self.config.layer_types if lt == "full_attention")
        total_layers = len(self.config.layer_types)

        # FlashAttention works on full attention layers even with sinks enabled
        # (sinks are only needed for sliding window layers)
        return f"FlashAttention: {version_str} - active on {full_attn_layers}/{total_layers} full attention layers"

    @staticmethod
    def build_causal_mask(T: int, device, dtype=torch.bool) -> torch.Tensor:
        i = torch.arange(T, device=device)
        j = torch.arange(T, device=device)
        return (j[None, :] <= i[:, None]).to(dtype)

    def reset_parameters(self):
        init_std = getattr(self, "init_std", 0.02)
        with torch.no_grad():
            nn.init.normal_(self.embed.weight, mean=0.0, std=init_std)
            if self.lm_head.weight is not self.embed.weight:
                nn.init.normal_(self.lm_head.weight, mean=0.0, std=init_std)
        # Cascade to blocks + final norm
        for blk in getattr(self, "layers", []):
            if hasattr(blk, "reset_parameters"):
                blk.reset_parameters()
        if hasattr(self.norm_f, "reset_parameters"):
            self.norm_f.reset_parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T = input_ids.shape
        device = input_ids.device
        x = self.embed(input_ids)
        x = self.drop(x)
        positions = torch.arange(T, device=device).view(1, T).expand(B, T) # shape [1, T] -> [B, T]
        causal_mask = self.build_causal_mask(T, device) # (T,T) bool # shape [T, T] -> [B, T, T]

        aux_losses: List[torch.Tensor] = []
        expert_utilizations: List[torch.Tensor] = []
        router_entropies: List[torch.Tensor] = []
        
        for i, layer in enumerate(self.layers):
            is_sliding = (self.config.layer_types[i] == "sliding_attention")
            x, aux = layer(x, positions, causal_mask, is_sliding, self.config.sliding_window)
            if aux:
                if "router_aux_loss" in aux:
                    aux_losses.append(aux["router_aux_loss"])
                if "expert_utilization" in aux:
                    expert_utilizations.append(aux["expert_utilization"])
                if "router_entropy" in aux:
                    router_entropies.append(aux["router_entropy"])

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        aux_out: Dict[str, torch.Tensor] = {}
        if labels is not None:
            # next-token loss
            logits_flat = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            targets = labels[:, 1:].contiguous().view(-1)
            nll = F.cross_entropy(logits_flat, targets, ignore_index=-100)
            if aux_losses:
                aux_total = torch.stack(aux_losses).mean()
                nll = nll + aux_total
                aux_out["router_aux_loss"] = aux_total.detach()
            
            # Aggregate MoE metrics across layers
            if expert_utilizations:
                # Average utilization across all layers and tokens
                avg_expert_util = torch.stack(expert_utilizations).mean(dim=0)  # (E,)
                aux_out["expert_utilization"] = avg_expert_util.detach()
                
                # Load balance: std/mean of utilization
                load_balance = avg_expert_util.std() / (avg_expert_util.mean() + 1e-10)
                aux_out["load_balance_score"] = load_balance.detach()
            
            if router_entropies:
                # Average router entropy across layers
                avg_entropy = torch.stack(router_entropies).mean()
                aux_out["router_entropy"] = avg_entropy.detach()
            
            loss = nll
        return logits, {"loss": loss, **aux_out}


# ------------------------------------------------------------------------------------
# Quick param sanity check for the 20B config
# ------------------------------------------------------------------------------------

def gpt_oss_moe_6b_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=201_088,
        hidden_size=2880,
        num_hidden_layers=24,
        head_dim=64,
        num_attention_heads=64,
        num_key_value_heads=8,
        attention_bias=True,
        attention_dropout=0.0,
        dropout=0.0,
        max_position_embeddings=131_072,
        sliding_window=128,
        num_local_experts=8,
        experts_per_token=2,
        router_aux_loss_coef=0.005,  # Adjusted for 8 experts (was 0.02, training script uses 0.005)
        intermediate_size=2880,
        swiglu_clip=7.0,
        rope_theta=150_000.0,
        enable_sink_logit=True,   # sink-bias enabled (flash kept on full-attn layers)
        sink_logit_init=4.0,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        tie_word_embeddings=False,
        eos_token_id=None,
    )


if __name__ == "__main__":
    cfg = gpt_oss_moe_6b_config()
    model = Transformer(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params/1e9:.3f} B")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable:        {trainable/1e9:.3f} B")