"""Attention under test: softmax rows vs. sparsemax rows.

One knob, two settings:

    normaliser "softmax"    the usual dense attention row: every weight is
                            strictly positive, however irrelevant the key
               "sparsemax"  the Euclidean projection of the scores onto the
                            probability simplex, which sets weights below a
                            data-dependent threshold to *exactly* zero

Everything else -- projections, head layout, the learnable per-head temperature,
masking -- is identical between the two, so a difference in a benchmark number
is a difference in the normaliser and nothing else.

The temperature matters and is deliberately learnable: sparsemax is not scale
invariant, so a hardcoded 1/sqrt(d) would silently decide how sparse the model
is allowed to be. Softmax gets the same parameter so neither side is favoured.

Every module records diagnostics for the last forward pass in ``last_stats``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .sparsemax import get_normaliser

NORMALISERS = ("softmax", "sparsemax")


@dataclass
class AttentionConfig:
    d_model: int = 128
    n_heads: int = 4
    normaliser: str = "softmax"
    causal: bool = False
    learn_temp: bool = True
    dropout: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        if self.normaliser not in NORMALISERS:
            raise ValueError(f"normaliser must be one of {NORMALISERS}, got {self.normaliser!r}")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not self.name:
            self.name = self.normaliser


class Attention(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.normalise = get_normaliser(cfg.normaliser)
        self.log_temp = nn.Parameter(torch.zeros(cfg.n_heads), requires_grad=cfg.learn_temp)
        self.last_stats: dict[str, float] = {}

    # ---------------------------------------------------------------- helpers
    def _split(self, t: Tensor) -> Tensor:
        b, n, _ = t.shape
        return t.view(b, n, self.cfg.n_heads, self.d_head).transpose(1, 2)  # [B,H,N,dh]

    def _scores(self, q: Tensor, k: Tensor) -> Tensor:
        temp = self.log_temp.exp().view(1, -1, 1, 1)
        return (q @ k.transpose(-1, -2)) * (temp / math.sqrt(self.d_head))

    def _allowed(self, n: int, valid: Tensor) -> Tensor:
        """[B,1,Nq,Nk] boolean mask of attendable pairs."""
        allowed = valid[:, None, None, :].expand(-1, 1, n, -1)
        if self.cfg.causal:
            pos = torch.arange(n, device=valid.device)
            allowed = allowed & (pos[:, None] >= pos[None, :])
        return allowed

    # ----------------------------------------------------------------- forward
    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """x: [B,N,d]; key_padding_mask: [B,N] with True marking a real token."""
        b, n, _ = x.shape
        q, k, v = (self._split(t) for t in self.qkv(x).chunk(3, dim=-1))
        valid = (
            torch.ones(b, n, device=x.device, dtype=torch.bool)
            if key_padding_mask is None
            else key_padding_mask
        )

        allowed = self._allowed(n, valid)
        attn = self.normalise(self._scores(q, k), dim=-1, mask=allowed)
        attn = attn * valid[:, None, :, None]            # zero out padded query rows
        self.last_stats = self._attn_stats(attn, allowed.expand_as(attn))

        out = (self.drop(attn) @ v).transpose(1, 2).reshape(b, n, self.cfg.d_model)
        return self.out_proj(out)

    # ------------------------------------------------------------------ stats
    @torch.no_grad()
    def _attn_stats(self, attn: Tensor, allowed: Tensor) -> dict[str, float]:
        live_rows = (attn.sum(-1) > 0).to(attn.dtype)      # padded rows don't count
        n_live = live_rows.sum().clamp(min=1)
        allowed_f = allowed.to(attn.dtype) * live_rows[..., None]
        n_allowed = allowed_f.sum().clamp(min=1)
        nonzero = ((attn > 0) & allowed).to(attn.dtype) * live_rows[..., None]
        rows = allowed_f.sum(-1).clamp(min=1)
        ent = -(attn.clamp(min=1e-12).log() * attn * allowed_f).sum(-1)
        return {
            "attn_zero_frac": float(1.0 - nonzero.sum() / n_allowed),
            "attn_support": float((nonzero.sum(-1) * live_rows).sum() / n_live),
            "attn_support_frac": float(((nonzero.sum(-1) / rows) * live_rows).sum() / n_live),
            "attn_entropy": float((ent * live_rows).sum() / n_live),
            "attn_max": float(attn.max()),
        }


def attention_flops(cfg: AttentionConfig, n: int) -> float:
    """Analytic multiply-accumulate count for one sequence, all heads.

    Q/K/V/out projections plus the two score/context matmuls.  Identical for both
    normalisers: sparsemax changes which weights are non-zero, not how many are
    computed, so its real cost shows up in wall clock (the threshold search is a
    sort) rather than here.
    """
    d = cfg.d_model
    proj = 4 * n * d * d
    attn = 2 * cfg.n_heads * (n ** 2) * (d / cfg.n_heads)
    return float(proj + attn)
