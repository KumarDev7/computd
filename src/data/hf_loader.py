"""
HuggingFace dataset loader for Ultra-FineWeb with LFM2.5 tokenizer.

Streams data from https://huggingface.co/datasets/openbmb/Ultra-FineWeb
Tokenizer from https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking
"""
import os
import numpy as np
import jax
import jax.numpy as jnp


def load_tokenizer(tokenizer_name: str = "LiquidAI/LFM2.5-1.2B-Thinking"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


class CachedDataLoader:
    def __init__(self, cache_path, tokenizer_name="LiquidAI/LFM2.5-1.2B-Thinking",
                 seq_len=1024, batch_size=4, seed=42):
        self.tokenizer = load_tokenizer(tokenizer_name)
        self.vocab_size = self.tokenizer.vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed
        self._data = np.load(cache_path).astype(np.int32)
        self._rng = np.random.default_rng(seed)
        print(f'[data] loaded cache: {len(self._data):,} tokens from {cache_path}')

    def __iter__(self):
        while True:
            starts = self._rng.integers(0, len(self._data) - self.seq_len - 1,
                                         size=self.batch_size)
            batch = np.stack([self._data[s:s + self.seq_len + 1] for s in starts])
            yield jnp.array(batch)