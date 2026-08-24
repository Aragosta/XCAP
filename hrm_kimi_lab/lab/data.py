"""Char-level LM data from a real text corpus (tiny Shakespeare, 1.1 MB)."""
from pathlib import Path
import torch

DATA = Path(__file__).resolve().parent.parent / "data" / "tinyshakespeare.txt"


class CharData:
    def __init__(self, path: Path = DATA, val_frac: float = 0.1):
        text = path.read_text(encoding="utf-8")
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n_val = int(len(ids) * val_frac)
        self.train, self.val = ids[:-n_val], ids[-n_val:]

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def batch(self, split: str, batch_size: int, seq_len: int, generator: torch.Generator):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([data[i : i + seq_len] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + seq_len] for i in ix])
        return x, y

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)
