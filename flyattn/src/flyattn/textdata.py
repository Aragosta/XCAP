"""Real-text corpora with an honest out-of-distribution split.

Tokenisation is byte-level (vocab 256). That is deliberate: a learned subword
vocabulary is fitted on the training domain, so every OOD number it produces is
partly a tokeniser artefact. Bytes make the OOD comparison clean.

Splits
------
train / val   Project Gutenberg (NLTK `gutenberg`) minus three held-out books.
ood_book      the three held-out Gutenberg books - unseen author, same register.
ood_brown     the Brown corpus - different register, different era, edited prose.
ood_reuters   Reuters newswire - different register, heavy on numerals and names.

`hard_mask` marks the positions a byte-bigram model fitted on the training split
finds hardest. In-distribution loss is dominated by easy continuation bytes, so
the hard subset is where architectural differences have room to show up.
"""
from __future__ import annotations

import os
import re
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "text")
HELD_OUT_BOOKS = ("carroll-alice.txt", "melville-moby_dick.txt", "milton-paradise.txt")


def _read_dir(path, limit_bytes=None, skip=()):
    chunks, total = [], 0
    for root, _, files in os.walk(path):
        for f in sorted(files):
            if f.startswith(".") or f in ("README", "CONTENTS", "cats.txt"):
                continue
            if f in skip:
                continue
            p = os.path.join(root, f)
            try:
                b = open(p, "rb").read()
            except OSError:
                continue
            chunks.append(b)
            total += len(b)
            if limit_bytes and total >= limit_bytes:
                return b"\n".join(chunks)[:limit_bytes]
    return b"\n".join(chunks)


def load_corpora(val_frac=0.05, ood_limit=2_000_000, data_dir=DATA):
    g = _read_dir(os.path.join(data_dir, "gutenberg"), skip=HELD_OUT_BOOKS)
    held = b"\n".join(open(os.path.join(data_dir, "gutenberg", b), "rb").read()
                      for b in HELD_OUT_BOOKS)
    # Brown ships POS-tagged as word/TAG; strip the tags back to plain prose so
    # the shift measured is register, not annotation format.
    brown = re.sub(rb"/[^\s]+", b"", _read_dir(os.path.join(data_dir, "brown"),
                                               limit_bytes=3 * ood_limit))
    reuters = _read_dir(os.path.join(data_dir, "reuters", "training"),
                        limit_bytes=ood_limit)
    arr = np.frombuffer(g, dtype=np.uint8)
    n_val = int(len(arr) * val_frac)
    return dict(
        train=arr[:-n_val].copy(),
        val=arr[-n_val:].copy(),
        ood_book=np.frombuffer(held, dtype=np.uint8)[:ood_limit].copy(),
        ood_brown=np.frombuffer(brown, dtype=np.uint8)[:ood_limit].copy(),
        ood_reuters=np.frombuffer(reuters, dtype=np.uint8)[:ood_limit].copy(),
    )


def bigram_logprobs(train: np.ndarray, alpha=0.5) -> np.ndarray:
    """Add-alpha byte bigram model; used to define the 'hard position' subset."""
    counts = np.zeros((256, 256), np.float64) + alpha
    np.add.at(counts, (train[:-1], train[1:]), 1.0)
    return np.log(counts / counts.sum(1, keepdims=True))


def batches(data: np.ndarray, batch_size: int, seq_len: int, rng, n_batches=None,
            deterministic=False):
    """Yield (inputs, targets) int64 arrays of shape (B, T)."""
    hi = len(data) - seq_len - 1
    i = 0
    while n_batches is None or i < n_batches:
        if deterministic:
            starts = (np.arange(batch_size) * (hi // batch_size)
                      + i * seq_len) % hi
        else:
            starts = rng.integers(0, hi, batch_size)
        idx = starts[:, None] + np.arange(seq_len + 1)[None, :]
        chunk = data[idx].astype(np.int64)
        yield chunk[:, :-1], chunk[:, 1:]
        i += 1
