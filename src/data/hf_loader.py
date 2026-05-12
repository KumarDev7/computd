"""
HuggingFace dataset loader for Ultra-FineWeb with LFM2.5 tokenizer.

Streams data from https://huggingface.co/datasets/openbmb/Ultra-FineWeb
Tokenizer from https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

_N_PARQUET_FILES = 1
_CACHE_TOKEN_LIMIT = 50_000_000


def load_tokenizer(tokenizer_name: str = "LiquidAI/LFM2.5-1.2B-Thinking"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _build_cache(cache_path: str, tokenizer, hf_split: str = "en"):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    import re

    print(f'[data] cache not found — downloading {_N_PARQUET_FILES} parquet shards from Ultra-FineWeb...')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    prefix = "data/ultrafineweb_en/ultrafineweb-en-part-"
    all_ids = []
    for i in range(1, _N_PARQUET_FILES + 1):
        shard = f"{prefix}{i:04d}-of-2048.parquet"
        print(f'[data] downloading {shard}...')
        local = hf_hub_download("openbmb/Ultra-FineWeb", shard, repo_type="dataset")
        df = pd.read_parquet(local, columns=["content"])
        texts = df["content"].dropna().tolist()
        print(f'[data]   shard {i}: {len(texts)} documents, tokenizing...')
        batch = tokenizer(texts, add_special_tokens=False)
        for ids in batch["input_ids"]:
            all_ids.extend(ids)
            if len(all_ids) >= _CACHE_TOKEN_LIMIT:
                break
        print(f'[data]   total tokens so far: {len(all_ids):,}')
        if len(all_ids) >= _CACHE_TOKEN_LIMIT:
            break

    arr = np.array(all_ids[:_CACHE_TOKEN_LIMIT], dtype=np.int32)
    np.save(cache_path, arr)
    print(f'[data] built cache: {len(arr):,} tokens saved to {cache_path}')
    return arr


class CachedDataLoader:
    def __init__(self, cache_path, tokenizer_name="LiquidAI/LFM2.5-1.2B-Thinking",
                 seq_len=1024, batch_size=4, seed=42):
        self.tokenizer = load_tokenizer(tokenizer_name)
        self.vocab_size = self.tokenizer.vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed
        if not os.path.exists(cache_path):
            self._data = _build_cache(cache_path, self.tokenizer, hf_split="en").astype(np.int32)
        else:
            self._data = np.load(cache_path).astype(np.int32)
            print(f'[data] loaded cache: {len(self._data):,} tokens from {cache_path}')
        self._rng = np.random.default_rng(seed)

    def __iter__(self):
        while True:
            starts = self._rng.integers(0, len(self._data) - self.seq_len - 1,
                                         size=self.batch_size)
            batch = np.stack([self._data[s:s + self.seq_len + 1] for s in starts])
            yield jnp.array(batch)