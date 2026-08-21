"""Attention as an energy landscape.

Two independent axes.  The **score** decides what "match" means; the **gate**
decides how scores become weights.

score
    "dot"      standard scaled dot product,  s_ij = q_i . k_j / sqrt(d)
    "energy"   negative squared distance,    E_ij = ||q_i - k_j||^2,  s = -E/T
    "transe"   TransE relational energy,     E_ij = ||g_r(q_i) - k_j||^2
               with g_r(z) = z + r_i, r_i drawn from a small learned relation
               codebook by a per-query selector
    "rotate"   RotatE relational energy, same but g_r(z) = z o r_i, a rotation
               by codebook angles in 2D subspaces

gate
    "softmax"    competitive: weights sum to 1, every memory gets some mass
    "sparsemax"  competitive but with exact zeros
    "sigmoid"    *non-competitive*: a_j = sigma((tau_j - E_ij)/T), each memory
                 judged on its own, so zero, one or many can be active at once

An identity worth knowing before reading any result
---------------------------------------------------
    -||q - k||^2 = 2 q.k - ||q||^2 - ||k||^2

so energy attention is dot-product attention plus a per-key bias -||k||^2 (the
per-query term is constant inside a softmax row).  Under a *competitive* gate
the two differ only by that key-norm penalty: energy attention prefers keys of
small norm, all else equal.  The difference is therefore expected to be small,
and `tests/test_energy.py` pins the identity down.  The genuinely new object is
the **sigmoid gate**, which drops the sum-to-one constraint entirely, and the
relational scores, which put a learned bottleneck between query and memory.

Similarly, if the relation vector were a free linear function of the query
token, r_i = W_r x_i, then q_i + r_i = (W_q + W_r) x_i and TransE attention
would collapse exactly back to energy attention.  That is why the relation
comes from a **codebook** of `n_relations` vectors chosen by a selector: the
bottleneck is the whole point, and it is what makes "which relation is this
query asking about?" an inspectable quantity.

Every module records diagnostics for the last forward pass in ``last_stats``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .sparsemax import get_normaliser, masked_softmax, sparsemax

SCORES = ("dot", "energy", "transe", "rotate")
GATES = ("softmax", "sparsemax", "sigmoid")
RELATIONAL = ("transe", "rotate")


@dataclass
class AttentionConfig:
    d_model: int = 128
    n_heads: int = 4
    score: str = "dot"
    gate: str = "softmax"
    n_relations: int = 4          # codebook size for the relational scores
    causal: bool = False
    learn_temp: bool = True
    dropout: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        if self.score not in SCORES:
            raise ValueError(f"score must be one of {SCORES}, got {self.score!r}")
        if self.gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}, got {self.gate!r}")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.score in RELATIONAL and self.n_relations < 2:
            raise ValueError("relational scores need at least 2 relation slots")
        if self.score == "rotate" and (self.d_model // self.n_heads) % 2:
            raise ValueError("rotate needs an even head dimension (it rotates 2D subspaces)")
        if not self.name:
            self.name = f"{self.score}-{self.gate}"


class Attention(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.cfg = cfg
        h, dh = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.d_head = dh
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.log_temp = nn.Parameter(torch.zeros(h), requires_grad=cfg.learn_temp)

        if cfg.gate == "sigmoid":
            # tau_j = per-head offset + a data-dependent term read off the key token
            self.tau_bias = nn.Parameter(torch.zeros(h))
            self.tau_proj = nn.Linear(cfg.d_model, h, bias=False)
        if cfg.score in RELATIONAL:
            self.rel_select = nn.Linear(cfg.d_model, h * cfg.n_relations, bias=False)
            slots = dh // 2 if cfg.score == "rotate" else dh
            self.relations = nn.Parameter(torch.randn(h, cfg.n_relations, slots) * 0.02)

        self.last_stats: dict[str, float] = {}
        self.last_attn: Tensor | None = None       # [B,H,N,N], kept for introspection
        self.last_relation_probs: Tensor | None = None   # [B,H,N,R]

    # ---------------------------------------------------------------- helpers
    def _split(self, t: Tensor) -> Tensor:
        b, n, _ = t.shape
        return t.view(b, n, self.cfg.n_heads, self.d_head).transpose(1, 2)  # [B,H,N,dh]

    def _temp(self) -> Tensor:
        return self.log_temp.exp().view(1, -1, 1, 1)

    def _relation(self, x: Tensor, q: Tensor) -> Tensor:
        """Apply g_r to the query, with r chosen per query from the codebook."""
        b, n, _ = x.shape
        h, r = self.cfg.n_heads, self.cfg.n_relations
        logits = self.rel_select(x).view(b, n, h, r).permute(0, 2, 1, 3)     # [B,H,N,R]
        probs = torch.softmax(logits, dim=-1)
        self.last_relation_probs = probs.detach()
        mix = probs @ self.relations                                          # [B,H,N,slots]
        if self.cfg.score == "transe":
            return q + mix                                                    # g_r(z) = z + r
        # rotate: g_r(z) = z o r, a rotation of each 2D subspace by angle mix
        cos, sin = torch.cos(mix), torch.sin(mix)
        a, bb = q[..., 0::2], q[..., 1::2]
        return torch.stack([a * cos - bb * sin, a * sin + bb * cos], dim=-1).flatten(-2)

    def _energy(self, q: Tensor, k: Tensor) -> Tensor:
        """E_ij = ||q_i - k_j||^2, computed without materialising the difference."""
        return (q.pow(2).sum(-1)[..., :, None]
                - 2.0 * (q @ k.transpose(-1, -2))
                + k.pow(2).sum(-1)[..., None, :]).clamp(min=0)

    def _scores(self, x: Tensor, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor | None]:
        """Return (scores, energy).  ``energy`` is None for the dot product."""
        scale = self._temp() / math.sqrt(self.d_head)
        if self.cfg.score == "dot":
            return (q @ k.transpose(-1, -2)) * scale, None
        if self.cfg.score in RELATIONAL:
            q = self._relation(x, q)
        energy = self._energy(q, k)
        return -energy * scale, energy

    def _allowed(self, n: int, valid: Tensor) -> Tensor:
        allowed = valid[:, None, None, :].expand(-1, 1, n, -1)
        if self.cfg.causal:
            pos = torch.arange(n, device=valid.device)
            allowed = allowed & (pos[:, None] >= pos[None, :])
        return allowed

    def _gate(self, scores: Tensor, allowed: Tensor, x: Tensor) -> Tensor:
        if self.cfg.gate == "softmax":
            return masked_softmax(scores, dim=-1, mask=allowed)
        if self.cfg.gate == "sparsemax":
            return sparsemax(scores, dim=-1, mask=allowed)
        # sigmoid: judge every memory independently against its own threshold.
        # `scores` is already -E/T, so this is sigma((tau_j - E_ij)/T).
        tau = (self.tau_proj(x).transpose(1, 2)[:, :, None, :]
               + self.tau_bias.view(1, -1, 1, 1)) * (self._temp() / math.sqrt(self.d_head))
        return torch.sigmoid(scores + tau) * allowed

    # ----------------------------------------------------------------- forward
    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        b, n, _ = x.shape
        q, k, v = (self._split(t) for t in self.qkv(x).chunk(3, dim=-1))
        valid = (
            torch.ones(b, n, device=x.device, dtype=torch.bool)
            if key_padding_mask is None
            else key_padding_mask
        )
        allowed = self._allowed(n, valid)
        scores, energy = self._scores(x, q, k)
        attn = self._gate(scores, allowed, x) * valid[:, None, :, None]

        ctx = self.drop(attn) @ v
        if self.cfg.gate == "sigmoid":
            # a non-competitive gate does not normalise itself.  Dividing by
            # 1 + sum(a) keeps the scale bounded *and* keeps the semantics: a
            # query that matches no memory at all contributes nothing.
            ctx = ctx / (1.0 + attn.sum(-1, keepdim=True))

        self.last_attn = attn.detach()
        self.last_stats = self._attn_stats(attn, allowed.expand_as(attn), energy)
        return self.out_proj(ctx.transpose(1, 2).reshape(b, n, self.cfg.d_model))

    # ------------------------------------------------------------------ stats
    @torch.no_grad()
    def _attn_stats(self, attn: Tensor, allowed: Tensor, energy: Tensor | None) -> dict[str, float]:
        live = (attn.sum(-1) > 0).to(attn.dtype)
        n_live = live.sum().clamp(min=1)
        allowed_f = allowed.to(attn.dtype) * live[..., None]
        n_allowed = allowed_f.sum().clamp(min=1)
        nonzero = ((attn > 0) & allowed).to(attn.dtype) * live[..., None]
        p = attn / attn.sum(-1, keepdim=True).clamp(min=1e-9)      # entropy of the shape
        ent = -(p.clamp(min=1e-12).log() * p * allowed_f).sum(-1)
        stats = {
            "attn_zero_frac": float(1.0 - nonzero.sum() / n_allowed),
            "attn_support": float((nonzero.sum(-1) * live).sum() / n_live),
            "attn_entropy": float((ent * live).sum() / n_live),
            "attn_max": float(attn.max()),
            "attn_row_mass": float((attn.sum(-1) * live).sum() / n_live),
            "attn_active": float(((attn > 0.5).to(attn.dtype).sum(-1) * live).sum() / n_live),
        }
        if energy is not None:
            e = energy[allowed.expand_as(energy)]
            stats.update(energy_mean=float(e.mean()), energy_min=float(energy.min()),
                         energy_std=float(e.std()))
        if self.last_relation_probs is not None:
            rp = self.last_relation_probs
            stats["relation_entropy"] = float(-(rp.clamp(min=1e-12).log() * rp).sum(-1).mean())
            stats["relation_slots_used"] = float(
                (rp.mean(dim=(0, 2)) > 0.05).to(rp.dtype).sum(-1).mean())
        return stats


def attention_flops(cfg: AttentionConfig, n: int) -> float:
    """Multiply-accumulates for one sequence, all heads.

    The energy expands to the same q.k matmul plus two O(N d) norm terms, so the
    quadratic cost is identical to the dot product; the relational scores add a
    selector and a codebook mix, both linear in N.
    """
    d, h = cfg.d_model, cfg.n_heads
    proj = 4 * n * d * d
    attn = 2 * h * (n ** 2) * (d / h)
    extra = 2 * n * d if cfg.score != "dot" else 0
    if cfg.score in RELATIONAL:
        extra += n * d * h * cfg.n_relations + n * d * cfg.n_relations
    if cfg.gate == "sigmoid":
        extra += n * d * h
    return float(proj + attn + extra)
