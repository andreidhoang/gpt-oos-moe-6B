#!/usr/bin/env python3
"""
Export FSDP-sharded training checkpoints (.pt with optimizer) -> HuggingFace-style weights-only .safetensors

Adapted for GPT-OSS MoE-8B with:
- MoE architecture (32 experts, top-4 routing)
- Grouped Query Attention (GQA)
- YaRN RoPE scaling
- NO transformers dependency (we implement simple sharding here)
- Loads metadata only on rank 0 to avoid ShardedTensor local-rank mismatch
- Writes to a NEW folder (default under ./release)

Run (same world size as training, e.g. 8):
  torchrun --nproc_per_node=8 scripts/export_to_safetensors.py \
    --in_dir out/8b_moe_run1 \
    --ckpt_prefix ckpt \
    --max_shard_size 5GB \
    --release_dir ./release/moe-8b-iter50000

Requires:
  pip install safetensors
"""

import os, json, argparse, datetime, sys
import torch
import torch.distributed as dist
from functools import partial

from safetensors.torch import save_file

# Add parent directory to path to import model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import your model bits
from model.model import Transformer, ModelConfig, TransformerBlock, RopeScalingConfig

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy


# --------------------------- args ---------------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="Folder with sharded .pt files")
    ap.add_argument("--ckpt_prefix", type=str, default="ckpt", help="Prefix of sharded training ckpts")
    ap.add_argument("--release_dir", type=str, default="", help="Absolute output dir (default: auto under ./release)")
    ap.add_argument("--wrap_policy", type=str, choices=["transformer","size"], default="transformer")
    ap.add_argument("--min_params", type=int, default=2_000_000)
    ap.add_argument("--dtype", type=str, choices=["float32","bfloat16","float16"], default="bfloat16")
    ap.add_argument("--max_shard_size", type=str, default="5GB", help='e.g. "2GB", "5GB", "10GB"')
    return ap.parse_args()


# --------------------- misc helpers ---------------------
def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE","1")) > 1

def rank0_print(*a, **k):
    if (not is_dist()) or dist.get_rank() == 0:
        print(*a, **k)

def sharded_path(in_dir, prefix, rank):
    return os.path.join(in_dir, f"{prefix}_rank{rank:05d}.pt")

_UNITS = {"B":1, "KB":1024, "MB":1024**2, "GB":1024**3, "TB":1024**4}
def parse_size(s: str) -> int:
    if isinstance(s, (int, float)): return int(s)
    s = s.strip().upper().replace(" ", "")
    for u in ["TB","GB","MB","KB","B"]:
        if s.endswith(u):
            return int(float(s[:-len(u)]) * _UNITS[u])
    return int(s)  # raw bytes

# allowlist ShardedTensor in case safe loader is used elsewhere
try:
    from torch.distributed._shard.sharded_tensor.api import ShardedTensor
    import torch.serialization as _ser
    _ser.add_safe_globals([ShardedTensor])
except Exception:
    pass


# ---------------------- config normalization ----------------------
def _massage_cfg_for_model(cfg_dict: dict) -> dict:
    """Turn nested dicts into expected types (e.g., rope_scaling dict -> RopeScalingConfig)."""
    if cfg_dict is None:
        return {}
    cfg = dict(cfg_dict)
    rs = cfg.get("rope_scaling", None)
    if isinstance(rs, dict):
        # support alias 'scale' -> 'factor'
        if "factor" not in rs and "scale" in rs:
            rs["factor"] = rs.pop("scale")
        cfg["rope_scaling"] = RopeScalingConfig(**rs)
    return cfg


# ---------------------- build FSDP model ----------------------
def build_model_from_cfg(cfg_dict: dict, device: str, args) -> FSDP:
    cfg = _massage_cfg_for_model(cfg_dict)
    model_cfg = ModelConfig(**cfg)

    if args.wrap_policy == "transformer":
        auto_wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
    else:
        auto_wrap = partial(size_based_auto_wrap_policy, min_num_params=max(1_000_000, args.min_params))

    dmap = {"float32":torch.float32,"bfloat16":torch.bfloat16,"float16":torch.float16}
    mp = MixedPrecision(param_dtype=dmap[args.dtype], reduce_dtype=dmap[args.dtype], buffer_dtype=dmap[args.dtype])

    # meta build -> to_empty on correct device
    if hasattr(torch, "set_default_device"): torch.set_default_device("meta")
    base = Transformer(model_cfg)
    if hasattr(torch, "set_default_device"): torch.set_default_device("cpu")

    def _param_init_fn(m: torch.nn.Module):
        m.to_empty(device=torch.device(device))  # weights will be overwritten by ckpt load

    model = FSDP(
        base,
        auto_wrap_policy=auto_wrap,
        device_id=None,
        mixed_precision=mp,
        use_orig_params=True,
        limit_all_gathers=True,
        param_init_fn=_param_init_fn,
    )
    return model


# --------------------- sharding (no transformers) ---------------------
def shard_state_dict_for_hf(state: dict, max_shard_bytes: int, base_name="model"):
    """
    Split a full weights dict into shards <= max_shard_bytes.
    Returns:
      shards: { filename -> {param_name: tensor, ...} }
      index:  {"metadata":{"total_size":bytes},"weight_map":{param_name:filename}}
    """
    shards_list = []   # list[dict[param_name->tensor]]
    shard_sizes = []   # bytes per shard
    weight_map = {}
    total = 0

    current = {}
    cur_size = 0

    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue
        t_bytes = tensor.element_size() * tensor.numel()
        if current and cur_size + t_bytes > max_shard_bytes:
            shards_list.append(current)
            shard_sizes.append(cur_size)
            current, cur_size = {}, 0
        current[name] = tensor
        cur_size += t_bytes
        total += t_bytes
    if current:
        shards_list.append(current)
        shard_sizes.append(cur_size)

    n = len(shards_list)
    shards = {}
    for i, shard in enumerate(shards_list, start=1):
        fname = f"{base_name}-{i:05d}-of-{n:05d}.safetensors"
        shards[fname] = shard
        for k in shard.keys():
            weight_map[k] = fname

    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    return shards, index


# --------------------------- main ---------------------------
def main():
    args = get_args()

    # init dist
    if is_dist():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        rank = dist.get_rank(); world = dist.get_world_size()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rank, world = 0, 1

    # sanity: shards exist
    for r in range(world):
        p = sharded_path(args.in_dir, args.ckpt_prefix, r)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing shard for rank {r}: {p}")

    # ---------- load metadata ONLY on rank 0, then broadcast ----------
    if (not is_dist()) or rank == 0:
        meta = torch.load(sharded_path(args.in_dir, args.ckpt_prefix, 0),
                          map_location="cpu", weights_only=False)  # trusted file you created
        cfg_dict = meta.get("model_config_dict", {})
        iter_num = int(meta.get("iter_num", 0))
        tok_name = meta.get("tokenizer", "o200k_base")
        best_val_loss = float(meta.get("best_val_loss", float("inf")))
    else:
        cfg_dict, iter_num, tok_name, best_val_loss = None, 0, "", float("inf")

    if is_dist():
        obj = [cfg_dict, iter_num, tok_name, best_val_loss]
        dist.broadcast_object_list(obj, src=0)
        cfg_dict, iter_num, tok_name, best_val_loss = obj

    # decide output dir (outside training folder by default)
    if args.release_dir:
        release_dir = args.release_dir
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        release_dir = f"./release/moe-8b-iter{iter_num:06d}-{stamp}"
    if (not is_dist()) or rank == 0:
        os.makedirs(release_dir, exist_ok=True)
        rank0_print(f"[export] writing HF weights to: {release_dir}")
    if is_dist(): dist.barrier()

    # ---------- build model & load THIS rank's shard ----------
    model = build_model_from_cfg(cfg_dict, device, args)
    my_path = sharded_path(args.in_dir, args.ckpt_prefix, rank)
    payload = torch.load(my_path, map_location="cpu", weights_only=False)  # trusted file

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        model.load_state_dict(payload["model_state_dict"])

    if is_dist(): dist.barrier()

    # ---------- gather full weights to rank 0 (CPU) ----------
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        full_state = model.state_dict()

    # ---------- rank 0: save shards + index + config ----------
    if (not is_dist()) or rank == 0:
        max_bytes = parse_size(args.max_shard_size)
        shards, index = shard_state_dict_for_hf(full_state, max_bytes, base_name="model")

        # write shards
        for fname, shard in shards.items():
            save_file(shard, os.path.join(release_dir, fname))

        # write index
        with open(os.path.join(release_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)

        # minimal config.json (adjust fields if your config differs)
        cfg = _massage_cfg_for_model(cfg_dict)

        # Handle rope_scaling config
        rope_scaling_dict = None
        if isinstance(cfg.get("rope_scaling"), RopeScalingConfig):
            rope_scaling_dict = {"type": "yarn", "factor": cfg["rope_scaling"].factor}
        elif isinstance(cfg.get("rope_scaling"), dict):
            rope_scaling_dict = cfg["rope_scaling"]

        hf_cfg = dict(
            # Model type
            model_type="gpt_oss_moe",
            architectures=["GPTOSSMoEForCausalLM"],

            # Core dimensions
            vocab_size=int(cfg["vocab_size"]),
            hidden_size=int(cfg["hidden_size"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]),

            # Attention (GQA)
            num_attention_heads=int(cfg["num_attention_heads"]),
            num_key_value_heads=int(cfg.get("num_key_value_heads", cfg["num_attention_heads"])),
            head_dim=int(cfg.get("head_dim", 64)),
            attention_bias=bool(cfg.get("attention_bias", True)),
            attention_dropout=float(cfg.get("attention_dropout", 0.0)),

            # MoE configuration
            num_local_experts=int(cfg.get("num_local_experts", 8)),
            num_experts_per_tok=int(cfg.get("experts_per_token", 2)),
            router_aux_loss_coef=float(cfg.get("router_aux_loss_coef", 0.01)),

            # MLP / FFN
            intermediate_size=int(cfg.get("intermediate_size", cfg["hidden_size"])),
            hidden_act="swiglu",

            # Position embeddings
            max_position_embeddings=int(cfg["max_position_embeddings"]),
            rope_theta=float(cfg.get("rope_theta", 10000.0)),
            rope_scaling=rope_scaling_dict,
            sliding_window=int(cfg.get("sliding_window", 0)),

            # Normalization
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-5)),

            # Regularization
            dropout=float(cfg.get("dropout", 0.0)),

            # Model structure
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            eos_token_id=cfg.get("eos_token_id", None),
            bos_token_id=cfg.get("bos_token_id", None),
            pad_token_id=cfg.get("pad_token_id", None),

            # Training metadata
            torch_dtype=args.dtype,
            tokenizer_name=tok_name,
            training_iteration=iter_num,
            best_val_loss=best_val_loss if best_val_loss != float("inf") else None,
        )

        with open(os.path.join(release_dir, "config.json"), "w") as f:
            json.dump(hf_cfg, f, indent=2)

        # Model card / README
        with open(os.path.join(release_dir, "README.md"), "w") as f:
            f.write(
                f"# GPT-OSS MoE-8B Checkpoint\n\n"
                f"Exported weights from GPT-OSS MoE-8B training.\n\n"
                f"## Model Details\n\n"
                f"- **Architecture**: Mixture of Experts (MoE) Transformer\n"
                f"- **Total Parameters**: ~{cfg['num_local_experts'] * cfg['hidden_size'] * cfg['intermediate_size'] * 3 / 1e9:.1f}B (all experts)\n"
                f"- **Active Parameters**: ~{cfg['num_experts_per_tok'] * cfg['hidden_size'] * cfg['intermediate_size'] * 3 / 1e9:.1f}B per forward pass\n"
                f"- **Experts**: {cfg['num_local_experts']} total, top-{cfg['experts_per_token']} routing\n"
                f"- **Context Length**: {cfg['max_position_embeddings']:,} tokens\n"
                f"- **Attention**: Grouped Query Attention ({cfg['num_attention_heads']} query heads, {cfg.get('num_key_value_heads', 8)} KV heads)\n\n"
                f"## Training Details\n\n"
                f"- **Source Checkpoint**: `{args.in_dir}/{args.ckpt_prefix}_rank*.pt`\n"
                f"- **Training Iteration**: {iter_num:,}\n"
                f"- **Best Validation Loss**: {best_val_loss:.4f}\n"
                f"- **Tokenizer**: {tok_name}\n"
                f"- **Export Date**: {datetime.datetime.now().isoformat()}\n"
                f"- **Export dtype**: {args.dtype}\n"
                f"- **Max Shard Size**: {args.max_shard_size}\n\n"
                f"## Usage\n\n"
                f"This model checkpoint is in HuggingFace-compatible safetensors format.\n"
                f"To use with transformers (requires custom modeling code):\n\n"
                f"```python\n"
                f"# Note: Requires custom modeling implementation for GPT-OSS MoE architecture\n"
                f"# See the original training repository for model code\n"
                f"```\n\n"
                f"## Files\n\n"
                f"- `config.json`: Model architecture configuration\n"
                f"- `model.safetensors.index.json`: Weight shard mapping\n"
                f"- `model-*.safetensors`: Model weight shards\n\n"
                f"## License\n\n"
                f"[Specify license - recommend Apache 2.0 for open source]\n"
            )

        rank0_print(f"[export] ✓ Done! Files written to: {release_dir}")
        rank0_print(f"[export]   - {len(shards)} safetensors shards")
        rank0_print(f"[export]   - Total size: {index['metadata']['total_size'] / 1e9:.2f} GB")

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
