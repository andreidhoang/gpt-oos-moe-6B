"""
Sequential Data Loader for GPT-OSS-8B Training

Based on DataLoaderLite from Andrej Karpathy's train_gpt2.py
Adapted for 8B MoE model with improvements:
- Sequential reading (100% data coverage)
- Staggered starting positions (reduced I/O spikes)
- Compatible with existing train.py interface
"""

import glob
import os
import threading
from queue import Queue
from typing import Literal, Tuple

import numpy as np
import torch


def load_tokens(filename):
    """Load .npy shard and convert to torch tensor"""
    npt = np.load(filename)
    npt = npt.astype(np.int64)  # Use int64 for o200k tokenizer
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    """
    Sequential data loader (DataLoaderLite pattern)

    Features:
    - Sequential reading for 100% data coverage
    - Position tracking per GPU
    - Automatic shard advancement
    - Clear epoch boundaries
    """

    def __init__(self, data_dir, split, B, T, process_rank, num_processes):
        self.B = B  # batch_size
        self.T = T  # block_size (context length)
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # Get the shard filenames
        shards = glob.glob(os.path.join(data_dir, f"*{split}*.npy"))
        shards = sorted(shards)
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split} in {data_dir}"

        if process_rank == 0:
            print(f"[DataLoader] Found {len(shards)} shards for split '{split}'")

        # Stagger starting shard across ranks to reduce I/O spikes
        if len(self.shards) >= num_processes:
            shard_offset = (len(self.shards) // num_processes) * process_rank
            self.current_shard = shard_offset % len(self.shards)
        else:
            self.current_shard = process_rank % len(self.shards)

        # Load initial shard
        self.tokens = load_tokens(self.shards[self.current_shard])

        # Each GPU starts at different position within shard
        self.current_position = self.B * self.T * self.process_rank

        # Tracking
        self.shards_completed = 0
        self.tokens_processed = 0
        
        # Prefetch buffer for next shard (background loading)
        self.prefetch_queue = Queue(maxsize=1)
        self.prefetch_thread = None
        self._prefetch_next_shard()

        if process_rank == 0:
            print(
                f"[DataLoader] Starting at shard {self.current_shard}/{len(self.shards)}, "
                f"position {self.current_position:,}"
            )

    def reset(self):
        """Reset to beginning (useful for validation)"""
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
        if self.process_rank == 0:
            print(f"[DataLoader] Reset to shard 0")
        self._prefetch_next_shard()
    
    def _prefetch_next_shard(self):
        """Prefetch next shard in background thread"""
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            return  # Already prefetching
        
        def _load_next():
            next_shard_idx = (self.current_shard + 1) % len(self.shards)
            try:
                next_tokens = load_tokens(self.shards[next_shard_idx])
                self.prefetch_queue.put((next_shard_idx, next_tokens), block=False)
            except:
                pass  # Queue full or error, skip prefetch
        
        self.prefetch_thread = threading.Thread(target=_load_next, daemon=True)
        self.prefetch_thread.start()
    
    def _get_prefetched_shard(self):
        """Get prefetched shard if available"""
        try:
            next_shard_idx, next_tokens = self.prefetch_queue.get_nowait()
            return next_shard_idx, next_tokens
        except:
            return None, None

    def next_batch(self):
        """Get next batch (original DataLoaderLite interface)"""
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]

        # Check if we got enough tokens
        if len(buf) < B*T + 1:
            # Not enough tokens, advance to next shard
            # Try to use prefetched shard first
            next_shard_idx, next_tokens = self._get_prefetched_shard()
            if next_tokens is not None:
                self.current_shard = next_shard_idx
                self.tokens = next_tokens
            else:
                # Fallback to loading synchronously
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = load_tokens(self.shards[self.current_shard])
            
            self.current_position = B * T * self.process_rank
            self.shards_completed += 1
            
            # Prefetch next shard
            self._prefetch_next_shard()

            # Log shard transitions
            if self.shards_completed % 10 == 0 and self.process_rank == 0:
                print(
                    f"[DataLoader] Advanced to shard {self.current_shard} "
                    f"({self.shards_completed} shards completed)"
                )

            # Recursive call with new shard
            return self.next_batch()

        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)   # targets

        # Advance the position in the tensor
        self.current_position += B * T * self.num_processes
        self.tokens_processed += B * T

        # If loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            # Try to use prefetched shard first
            next_shard_idx, next_tokens = self._get_prefetched_shard()
            if next_tokens is not None:
                self.current_shard = next_shard_idx
                self.tokens = next_tokens
            else:
                # Fallback to loading synchronously
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = load_tokens(self.shards[self.current_shard])
            
            self.current_position = B * T * self.process_rank
            self.shards_completed += 1
            
            # Prefetch next shard
            self._prefetch_next_shard()

            # Log shard transitions
            if self.shards_completed % 10 == 0 and self.process_rank == 0:
                print(
                    f"[DataLoader] Advanced to shard {self.current_shard} "
                    f"({self.shards_completed} shards completed)"
                )

        return x, y

    def get_batch(self):
        """Alias for train.py compatibility"""
        return self.next_batch()


def get_data_loader(
    data_dir: str,
    split: Literal["train", "val"],
    block_size: int,
    batch_size: int,
    process_rank: int = 0,
    num_processes: int = 1,
    seed: int = 42,
):
    """
    Create data loader compatible with train.py

    Uses sequential reading (DataLoaderLite pattern) for:
    - 100% data coverage
    - Deterministic training
    - Clear epoch boundaries

    Args:
        data_dir: Directory containing tokenized .npy shards
        split: "train" or "val"
        block_size: Context length (T)
        batch_size: Batch size per process (B)
        process_rank: DDP rank
        num_processes: Number of DDP processes
        seed: Random seed (not used in sequential mode)

    Returns:
        DataLoaderLite instance with get_batch() method
    """
    # Check for .npy shards
    npy_pattern = os.path.join(data_dir, f"*{split}*.npy")
    npy_shards = glob.glob(npy_pattern)

    if not npy_shards:
        raise FileNotFoundError(
            f"No .npy shards found in {data_dir}\n"
            f"Expected files matching pattern: {npy_pattern}\n"
            f"\nRun data preparation first!"
        )

    if process_rank == 0:
        print(f"[DataLoader] Using SEQUENTIAL reading (DataLoaderLite pattern)")
        print(f"[DataLoader] Found {len(npy_shards)} .npy shards")

    return DataLoaderLite(
        data_dir=data_dir,
        split=split,
        B=batch_size,
        T=block_size,
        process_rank=process_rank,
        num_processes=num_processes
    )
