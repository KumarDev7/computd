"""
Character-level text data loader for Tiny Shakespeare.
"""
import numpy as np
import jax.numpy as jnp


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.ch2id = {c: i for i, c in enumerate(chars)}
        self.id2ch = {i: c for i, c in enumerate(chars)}

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.ch2id[c] for c in text], dtype=np.int32)

    def decode(self, ids) -> str:
        return "".join(self.id2ch[int(i)] for i in ids)


def shakespeare_loader(path: str, batch_size: int, seq_len: int,
                       split: str = "train", seed: int = 0):
    """
    Infinite generator of (batch, seq_len+1) token batches from the text file.
    split='train' uses first 90%, split='val' uses last 10%.
    """
    text = open(path).read()
    tok  = CharTokenizer(text)
    data = tok.encode(text)

    n     = len(data)
    split_at = int(0.9 * n)
    chunk = data[:split_at] if split == "train" else data[split_at:]

    rng = np.random.default_rng(seed)
    while True:
        starts = rng.integers(0, len(chunk) - seq_len - 1, size=batch_size)
        batch  = np.stack([chunk[s : s + seq_len + 1] for s in starts])
        yield jnp.array(batch), tok
