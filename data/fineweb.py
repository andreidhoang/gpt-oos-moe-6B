"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import json
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
# Configuration options:
# - "sample-10BT":  10 billion tokens (small, fast)
# - "sample-100BT": 100 billion tokens (medium, recommended)
# - "sample-350BT": 350 billion tokens (large, production)
remote_name = "sample-10BT"  # Change this to use different dataset sizes

# Auto-set local directory based on remote_name
local_dir = f"edu_fineweb_{remote_name.replace('sample-', '').replace('-', '_')}"
shard_size = int(1e8) # 100M tokens per shard

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# init the tokenizer
enc = tiktoken.get_encoding("o200k_base")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of int64 tokens
    # Using int64 because o200k vocab size (~200k) exceeds uint16 range (65536)
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.int64)
    return tokens_np

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard (int64 for o200k tokenizer)
    all_tokens_np = np.empty((shard_size,), dtype=np.int64)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}.npy")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}.npy")
        write_datafile(filename, all_tokens_np[:token_count])

# Create meta.json file for train.py compatibility
meta_path = os.path.join(DATA_CACHE_DIR, "meta.json")
vocab_size = getattr(enc, "n_vocab", len(enc._mergeable_ranks))
meta = {
    "tokenizer": "o200k_base",
    "vocab_size": vocab_size,
    "dataset": "fineweb-edu"
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"\n✅ Created {meta_path}")
print(f"   Tokenizer: {meta['tokenizer']}")
print(f"   Vocab size: {meta['vocab_size']}")
print(f"   Dataset: {meta['dataset']}")
