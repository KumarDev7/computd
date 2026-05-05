import numpy as np
import jax
import jax.numpy as jnp


def random_token_batches(vocab_size: int, batch_size: int, seq_len: int, seed: int = 0):
    """Infinite generator of random token batches for smoke-testing."""
    rng = np.random.default_rng(seed)
    while True:
        batch = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
        yield jnp.array(batch)


def synthetic_copy_task(vocab_size: int, batch_size: int, seq_len: int, seed: int = 0):
    """
    Copy task: predict x[t] from x[t-1]. Slightly harder than random.
    Each sequence is a repeated pattern of length pattern_len.
    """
    rng = np.random.default_rng(seed)
    pattern_len = max(2, seq_len // 4)
    while True:
        patterns = rng.integers(1, vocab_size, size=(batch_size, pattern_len), dtype=np.int32)
        repeats = (seq_len // pattern_len) + 2
        seqs = np.tile(patterns, (1, repeats))[:, :seq_len]
        yield jnp.array(seqs)


def make_bigram_table(vocab_size: int, seed: int = 42) -> np.ndarray:
    """Create a deterministic transition table for the bigram task.

    Each token i maps to a unique successor next_table[i], so the model
    must learn input-dependent predictions. Using a fixed seed ensures
    the same table is shared between train and test generators.
    """
    rng = np.random.default_rng(seed)
    return rng.permutation(vocab_size).astype(np.int32)


def bigram_task(vocab_size: int, batch_size: int, seq_len: int, seed: int = 0,
                transition_table: np.ndarray | None = None):
    """
    Deterministic bigram task: a fixed transition table maps token A → token B.
    Each token deterministically follows the previous one.
    This tests whether the model can learn input-dependent predictions.
    Perfect for DWA: different tokens should activate different pool vectors.

    Args:
        transition_table: Pre-built table from make_bigram_table(). If None,
            creates one from the seed (old behavior — different tables per seed).
    """
    rng = np.random.default_rng(seed)
    if transition_table is None:
        transition_table = make_bigram_table(vocab_size, seed=seed)
    next_table = transition_table

    while True:
        first = rng.integers(0, vocab_size, size=(batch_size,), dtype=np.int32)
        seqs = np.zeros((batch_size, seq_len), dtype=np.int32)
        seqs[:, 0] = first
        for t in range(1, seq_len):
            seqs[:, t] = next_table[seqs[:, t - 1]]
        yield jnp.array(seqs)
