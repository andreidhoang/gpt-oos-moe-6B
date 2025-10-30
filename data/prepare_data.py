#!/usr/bin/env python3
"""
Data Preparation Helper for GPT-OSS-8B MoE Training

Utilities for:
1. Converting FineWeb/custom data to o200k tokenized format
2. Validating existing tokenized data
3. Creating/updating meta.json
4. Data statistics and diagnostics

Usage:
    # Convert FineWeb dataset
    python scripts/prepare_data.py convert-fineweb \
        --source ../gpt-oss-pretrain/build-nanogpt/sample-10BT \
        --output data/fineweb_o200k \
        --format npy

    # Validate existing data
    python scripts/prepare_data.py validate \
        --data_dir ../gpt-oss-pretrain/build-nanogpt/edu_fineweb10B_o200k

    # Show data statistics
    python scripts/prepare_data.py stats \
        --data_dir ../gpt-oss-pretrain/build-nanogpt/edu_fineweb10B_o200k
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np

try:
    import tiktoken
except ImportError:
    print("❌ tiktoken not installed. Install with: pip install tiktoken")
    sys.exit(1)


def create_meta_json(
    output_dir: str,
    tokenizer: str = "o200k_base",
    vocab_size: int = 200000,
    dataset: str = "custom",
    **extra_fields
):
    """
    Create meta.json file for tokenized data

    Args:
        output_dir: Directory to save meta.json
        tokenizer: Tokenizer name
        vocab_size: Vocabulary size
        dataset: Dataset name
        **extra_fields: Additional metadata fields
    """
    meta_path = os.path.join(output_dir, "meta.json")

    meta = {
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "dataset": dataset,
        **extra_fields
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Created {meta_path}")
    print(f"   Tokenizer: {tokenizer}")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Dataset: {dataset}")


def convert_fineweb_to_o200k(
    source_dir: str,
    output_dir: str,
    output_format: Literal["npy", "bin"] = "npy",
    dataset_name: str = "fineweb_edu",
):
    """
    Convert FineWeb dataset from GPT-2 tokenization to o200k

    Args:
        source_dir: Directory with original .npy files (GPT-2 tokenized)
        output_dir: Directory to save o200k tokenized files
        output_format: Output format ("npy" or "bin")
        dataset_name: Dataset name for meta.json
    """
    print("=" * 70)
    print("CONVERTING FINEWEB TO O200K")
    print("=" * 70)

    # Load tokenizers
    print("\nLoading tokenizers...")
    gpt2_enc = tiktoken.get_encoding("gpt2")
    o200k_enc = tiktoken.get_encoding("o200k_base")
    print(f"✅ GPT-2 vocab: {gpt2_enc.n_vocab}")
    print(f"✅ o200k vocab: {o200k_enc.n_vocab}")

    # Find source files
    source_files = sorted(glob.glob(os.path.join(source_dir, "*.npy")))
    if not source_files:
        print(f"❌ No .npy files found in {source_dir}")
        return

    print(f"\n✅ Found {len(source_files)} source files")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each file
    total_tokens_processed = 0

    for i, source_file in enumerate(source_files):
        basename = os.path.basename(source_file)
        print(f"\n[{i+1}/{len(source_files)}] Processing {basename}...")

        # Load GPT-2 tokenized data
        try:
            data_gpt2 = np.load(source_file)
            print(f"   Loaded {len(data_gpt2):,} GPT-2 tokens")
        except Exception as e:
            print(f"   ❌ Failed to load {basename}: {e}")
            continue

        # Decode to text
        try:
            # Convert to list of ints (numpy arrays may cause issues)
            tokens_list = data_gpt2.astype(np.int64).tolist()
            text = gpt2_enc.decode(tokens_list)
            print(f"   Decoded to {len(text):,} characters")
        except Exception as e:
            print(f"   ❌ Failed to decode {basename}: {e}")
            continue

        # Encode with o200k
        try:
            tokens_o200k = o200k_enc.encode(text)
            print(f"   Encoded to {len(tokens_o200k):,} o200k tokens")
        except Exception as e:
            print(f"   ❌ Failed to encode {basename}: {e}")
            continue

        # Save output
        output_file = os.path.join(output_dir, basename)
        try:
            # o200k tokens can exceed 65535, so use uint32
            tokens_array = np.array(tokens_o200k, dtype=np.uint32)
            np.save(output_file, tokens_array)
            print(f"   ✅ Saved {output_file}")
            total_tokens_processed += len(tokens_o200k)
        except Exception as e:
            print(f"   ❌ Failed to save {output_file}: {e}")
            continue

    # Create meta.json
    create_meta_json(
        output_dir,
        tokenizer="o200k_base",
        vocab_size=o200k_enc.n_vocab,
        dataset=dataset_name,
        total_tokens=total_tokens_processed,
        num_files=len(source_files),
    )

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"Total tokens processed: {total_tokens_processed:,}")
    print(f"Output directory: {output_dir}")


def validate_data(data_dir: str) -> bool:
    """
    Validate tokenized data directory

    Args:
        data_dir: Directory containing tokenized data

    Returns:
        True if validation passes
    """
    print("=" * 70)
    print("VALIDATING DATA")
    print("=" * 70)

    # Check meta.json
    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(meta_path):
        print(f"❌ Missing {meta_path}")
        return False

    try:
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"✅ Loaded meta.json")
        print(f"   Tokenizer: {meta.get('tokenizer', 'unknown')}")
        print(f"   Vocab size: {meta.get('vocab_size', 'unknown')}")
        print(f"   Dataset: {meta.get('dataset', 'unknown')}")
    except Exception as e:
        print(f"❌ Failed to load meta.json: {e}")
        return False

    # Check tokenizer compatibility
    tokenizer = meta.get("tokenizer", "").lower()
    if "o200k" not in tokenizer:
        print(f"⚠️  WARNING: Tokenizer is {tokenizer}, expected o200k-based")
        print(f"   Model expects vocab size ~201088")

    # Check data files
    npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
    bin_files = glob.glob(os.path.join(data_dir, "*.bin"))

    if npy_files:
        print(f"\n✅ Found {len(npy_files)} .npy files")

        # Validate sample file
        sample = npy_files[0]
        try:
            data = np.load(sample)
            print(f"\nSample validation ({os.path.basename(sample)}):")
            print(f"   Shape: {data.shape}")
            print(f"   Dtype: {data.dtype}")
            print(f"   Token range: [{data.min()}, {data.max()}]")

            # Check dtype
            if data.dtype not in [np.uint32, np.int32, np.uint16, np.int16, np.int64]:
                print(f"   ⚠️  WARNING: Unexpected dtype {data.dtype}")

            # Check token range
            vocab_size = meta.get("vocab_size", 0)
            if data.max() >= vocab_size:
                print(f"   ❌ ERROR: Token {data.max()} exceeds vocab {vocab_size}")
                return False

            if data.min() < 0:
                print(f"   ❌ ERROR: Negative token ID {data.min()}")
                return False

            print(f"   ✅ Token range valid")

        except Exception as e:
            print(f"   ❌ Failed to load sample: {e}")
            return False

    elif bin_files:
        print(f"\n✅ Found {len(bin_files)} .bin files")
        for bf in bin_files:
            size_mb = os.path.getsize(bf) / (1024 * 1024)
            print(f"   {os.path.basename(bf)}: {size_mb:.1f} MB")

    else:
        print(f"❌ No data files found in {data_dir}")
        return False

    print("\n" + "=" * 70)
    print("✅ VALIDATION PASSED")
    print("=" * 70)
    return True


def show_data_stats(data_dir: str):
    """
    Display statistics about tokenized data

    Args:
        data_dir: Directory containing tokenized data
    """
    print("=" * 70)
    print("DATA STATISTICS")
    print("=" * 70)

    # Load meta.json
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print("\nMetadata:")
        for key, value in meta.items():
            print(f"  {key}: {value}")

    # Analyze .npy files
    npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    if npy_files:
        print(f"\n.npy Shards: {len(npy_files)}")

        train_shards = [f for f in npy_files if "train" in os.path.basename(f).lower()]
        val_shards = [f for f in npy_files if "val" in os.path.basename(f).lower()]

        print(f"  Training shards: {len(train_shards)}")
        print(f"  Validation shards: {len(val_shards)}")

        # Calculate total tokens
        total_tokens = 0
        for npy_file in npy_files:
            try:
                data = np.load(npy_file)
                total_tokens += len(data)
            except Exception as e:
                print(f"  ⚠️  Could not load {os.path.basename(npy_file)}: {e}")

        print(f"\nTotal tokens: {total_tokens:,}")
        print(f"Total tokens (billions): {total_tokens / 1e9:.2f}B")

        # Size on disk
        total_size = sum(os.path.getsize(f) for f in npy_files)
        print(f"Total size on disk: {total_size / (1024**3):.2f} GB")

    # Analyze .bin files
    bin_files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
    if bin_files:
        print(f"\n.bin Files: {len(bin_files)}")

        for bf in bin_files:
            basename = os.path.basename(bf)
            size_mb = os.path.getsize(bf) / (1024 * 1024)

            # Try to infer token count (assume uint32)
            try:
                data = np.memmap(bf, dtype=np.uint32, mode="r")
                num_tokens = len(data)
                print(f"  {basename}:")
                print(f"    Size: {size_mb:.1f} MB")
                print(f"    Tokens: {num_tokens:,} ({num_tokens/1e9:.2f}B)")
            except Exception as e:
                print(f"  {basename}: {size_mb:.1f} MB (could not read: {e})")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Data preparation utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Convert FineWeb command
    convert_parser = subparsers.add_parser(
        "convert-fineweb",
        help="Convert FineWeb dataset from GPT-2 to o200k tokenization"
    )
    convert_parser.add_argument(
        "--source", type=str, required=True,
        help="Source directory with GPT-2 tokenized .npy files"
    )
    convert_parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for o200k tokenized files"
    )
    convert_parser.add_argument(
        "--format", type=str, default="npy", choices=["npy", "bin"],
        help="Output format (default: npy)"
    )
    convert_parser.add_argument(
        "--dataset", type=str, default="fineweb_edu",
        help="Dataset name for meta.json (default: fineweb_edu)"
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate tokenized data directory"
    )
    validate_parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing tokenized data"
    )

    # Stats command
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show data statistics"
    )
    stats_parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing tokenized data"
    )

    # Create meta command
    meta_parser = subparsers.add_parser(
        "create-meta",
        help="Create meta.json file"
    )
    meta_parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save meta.json"
    )
    meta_parser.add_argument(
        "--tokenizer", type=str, default="o200k_base",
        help="Tokenizer name (default: o200k_base)"
    )
    meta_parser.add_argument(
        "--vocab_size", type=int, default=200000,
        help="Vocabulary size (default: 200000)"
    )
    meta_parser.add_argument(
        "--dataset", type=str, default="custom",
        help="Dataset name (default: custom)"
    )

    args = parser.parse_args()

    if args.command == "convert-fineweb":
        convert_fineweb_to_o200k(
            args.source,
            args.output,
            args.format,
            args.dataset
        )
    elif args.command == "validate":
        success = validate_data(args.data_dir)
        sys.exit(0 if success else 1)
    elif args.command == "stats":
        show_data_stats(args.data_dir)
    elif args.command == "create-meta":
        create_meta_json(
            args.output_dir,
            args.tokenizer,
            args.vocab_size,
            args.dataset
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
