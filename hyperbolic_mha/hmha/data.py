"""WikiText-2: download, cache, and byte-level tokenisation.

Real text, not synthetic. WikiText-2 is a standard LM benchmark of verified
Wikipedia articles, so results here sit on the same corpus the literature uses.

Tokenisation is **byte-level** (vocab 256). At ~6M parameters a word-level
vocabulary of ~33k would put most of the parameter budget in the embedding
table, which would make the attention comparison a comparison of embeddings
instead. Bytes also make the headline metric **bits-per-byte**, which is
tokeniser-independent. The cost is that perplexity here is per *byte* and so is
not comparable to published per-token WikiText-2 numbers -- stated as a
limitation rather than glossed over.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch

# HuggingFace is not reachable from this environment; this mirror is.
_BASE = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2"
_SPLITS = {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"}

VOCAB_SIZE = 256


@dataclass
class Corpus:
    train: torch.Tensor
    valid: torch.Tensor
    test: torch.Tensor
    source: str

    def summary(self) -> dict:
        return {
            "source": self.source,
            "vocab_size": VOCAB_SIZE,
            "tokenizer": "byte-level (utf-8)",
            "train_bytes": int(self.train.numel()),
            "valid_bytes": int(self.valid.numel()),
            "test_bytes": int(self.test.numel()),
        }


def default_cache_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "results" / "data"


def download(cache_dir: Path | None = None) -> dict[str, Path]:
    """Fetch the three splits into ``cache_dir``, skipping any already present."""
    cache_dir = Path(cache_dir or default_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for split, filename in _SPLITS.items():
        dest = cache_dir / f"wikitext2.{split}.txt"
        if not dest.exists() or dest.stat().st_size == 0:
            urllib.request.urlretrieve(f"{_BASE}/{filename}", dest)
        paths[split] = dest
    return paths


def encode(text: str) -> torch.Tensor:
    """UTF-8 bytes as uint8 token ids."""
    return torch.frombuffer(bytearray(text.encode("utf-8")), dtype=torch.uint8).clone()


def decode(tokens: torch.Tensor) -> str:
    """Inverse of :func:`encode`; malformed sequences are replaced, not raised.

    A partially-trained model emits arbitrary byte sequences, so generation
    samples routinely contain invalid UTF-8. Replacing keeps sample dumps
    readable instead of crashing the report.
    """
    return bytes(tokens.to(torch.uint8).tolist()).decode("utf-8", errors="replace")


def load_corpus(cache_dir: Path | None = None, max_train_bytes: int | None = None) -> Corpus:
    """Load WikiText-2 as byte tensors, optionally truncating the training split."""
    paths = download(cache_dir)
    splits = {}
    for split, path in paths.items():
        tokens = encode(path.read_text(encoding="utf-8"))
        if split == "train" and max_train_bytes is not None:
            tokens = tokens[:max_train_bytes]
        splits[split] = tokens
    return Corpus(source=f"WikiText-2 ({_BASE})", **splits)


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of ``(inputs, next-token targets)`` at random offsets."""
    if data.numel() < seq_len + 1:
        raise ValueError(f"split has {data.numel()} bytes, need > {seq_len}")
    idx = torch.randint(
        0, data.numel() - seq_len - 1, (batch_size,), generator=generator
    )
    x = torch.stack([data[i : i + seq_len] for i in idx]).long()
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in idx]).long()
    return x.to(device), y.to(device)


def iter_eval_batches(
    data: torch.Tensor, batch_size: int, seq_len: int, max_batches: int | None = None
):
    """Deterministic non-overlapping batches for evaluation.

    Evaluation must not be random: both arms have to see byte-for-byte the same
    held-out windows or the comparison is noise.
    """
    stride = seq_len
    n_windows = (data.numel() - 1) // stride
    windows = [
        (i * stride, i * stride + seq_len)
        for i in range(n_windows)
        if i * stride + seq_len + 1 <= data.numel()
    ]
    if max_batches is not None:
        windows = windows[: max_batches * batch_size]

    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        if not chunk:
            continue
        x = torch.stack([data[a:b] for a, b in chunk]).long()
        y = torch.stack([data[a + 1 : b + 1] for a, b in chunk]).long()
        yield x, y
