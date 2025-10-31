import argparse
import os
import json
import torch
from torch.nn import functional as F
import tiktoken

# Assuming the script is run from the root of the gpt-oos-moe-6B directory
# and the model definition is in model.model
try:
    from model.model import Transformer, gpt_oss_moe_6b_config
except ImportError:
    # If running from scripts/ directory, adjust path
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.model import Transformer, gpt_oss_moe_8b_config

def print_check(message, status):
    print(f"- {message}: {'✅' if status else '❌'}")
    if not status:
        exit(1)

def validate_data(data_dir):
    print("\n--- Validating Data ---")
    meta_path = os.path.join(data_dir, 'meta.json')
    status = os.path.exists(meta_path)
    print_check("meta.json exists", status)

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    tokenizer_name = meta.get('tokenizer', '')
    status = tokenizer_name == 'o200k_base'
    print_check(f"Tokenizer compatibility ({tokenizer_name})", status)

    try:
        tiktoken.get_encoding("o200k_base")
        print_check("tiktoken o200k_base encoding is available", True)
    except Exception as e:
        print_check(f"tiktoken o200k_base encoding is available ({e})", False)

    # Check for at least one training data file
    train_files = [f for f in os.listdir(data_dir) if f.endswith('.bin') or f.endswith('.npy')]
    status = len(train_files) > 0
    print_check("Training data files found", status)

def validate_model():
    print("\n--- Validating Model ---")
    try:
        cfg = gpt_oss_moe_8b_config()
        model = Transformer(cfg)
        print_check("Model can be built", True)
    except Exception as e:
        print_check(f"Model can be built ({e})", False)
        return None, None

    # Check for router aux loss configuration
    has_aux_loss = hasattr(model.layers[0].feed_forward, 'aux_loss_weight')
    print_check("Router aux loss configuration found", has_aux_loss)
    return model, cfg

def validate_forward_backward(model, cfg):
    print("\n--- Validating Forward/Backward Pass ---")
    if model is None or cfg is None:
        print_check("Skipping due to model build failure", False)
        return

    try:
        # Create dummy data
        batch_size = 2
        block_size = 64
        x = torch.randint(0, cfg.vocab_size, (batch_size, block_size))
        y = torch.randint(0, cfg.vocab_size, (batch_size, block_size))

        # Move to GPU if available
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
            model = model.cuda()
        
        model.train()
        logits, loss = model(x, y)
        print_check("Forward pass successful", True)

        loss.backward()
        print_check("Backward pass successful", True)

    except Exception as e:
        print(f"  Error during forward/backward pass: {e}")
        print_check("Forward/Backward pass", False)

def validate_cuda():
    print("\n--- Validating System ---")
    cuda_available = torch.cuda.is_available()
    print_check("CUDA availability", cuda_available)

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"  Found {gpu_count} GPU(s).")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {gpu_name} ({total_mem:.2f} GB)")
        
        try:
            # Check memory
            tensor = torch.randn(1024, 1024, 1024, dtype=torch.float16, device='cuda')
            del tensor
            torch.cuda.empty_cache()
            print_check("CUDA memory seems sufficient for a basic check", True)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print_check("CUDA memory seems sufficient for a basic check", False)
            else:
                raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation script for GPT-OSS MoE setup.")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the tokenized data.')
    args = parser.parse_args()

    print("Starting validation...")
    
    validate_data(args.data_dir)
    model, cfg = validate_model()
    validate_cuda()
    validate_forward_backward(model, cfg)

    print("\n✅ Validation complete. Setup looks good!")
