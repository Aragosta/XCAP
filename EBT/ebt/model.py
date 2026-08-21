"""A minimal encoder-style transformer whose only moving part is the attention."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn
from torch import Tensor

from .attention import Attention, AttentionConfig, attention_flops


@dataclass
class ModelConfig:
    vocab_size: int
    n_classes: int
    seq_len: int
    attn: AttentionConfig
    n_layers: int = 2
    mlp_ratio: int = 4
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.attn.d_model
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(replace(cfg.attn, dropout=cfg.dropout))
        self.mlp = nn.Sequential(
            nn.Linear(d, cfg.mlp_ratio * d), nn.GELU(),
            nn.Linear(cfg.mlp_ratio * d, d), nn.Dropout(cfg.dropout),
        )

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask)
        return x + self.mlp(self.ln2(x))


class Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.attn.d_model
        self.tok = nn.Embedding(cfg.vocab_size, d)
        self.pos = nn.Embedding(cfg.seq_len, d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.n_classes)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        n = x.size(1)
        h = self.tok(x) + self.pos(torch.arange(n, device=x.device))[None]
        for blk in self.blocks:
            h = blk(h, key_padding_mask)
        return self.head(self.ln_f(h))

    # ------------------------------------------------------------------ stats
    def attention_stats(self) -> dict[str, float]:
        """Mean of the per-layer diagnostics from the most recent forward pass."""
        per_layer = [b.attn.last_stats for b in self.blocks if b.attn.last_stats]
        if not per_layer:
            return {}
        keys = per_layer[0]
        return {k: sum(d[k] for d in per_layer) / len(per_layer) for k in keys}

    def flops_per_sequence(self, effective_k: float | None = None) -> float:
        d = self.cfg.attn.d_model
        attn = self.cfg.n_layers * attention_flops(self.cfg.attn, self.cfg.seq_len, effective_k)
        mlp = self.cfg.n_layers * 2 * self.cfg.seq_len * d * d * self.cfg.mlp_ratio
        head = self.cfg.seq_len * d * self.cfg.n_classes
        return float(attn + mlp + head)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(task, attn_cfg: AttentionConfig, n_layers: int = 2, dropout: float = 0.0) -> Model:
    return Model(ModelConfig(
        vocab_size=task.vocab_size, n_classes=task.n_classes, seq_len=task.seq_len,
        attn=attn_cfg, n_layers=n_layers, dropout=dropout,
    ))
