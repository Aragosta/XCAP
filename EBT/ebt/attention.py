"""Attention variants under test.

All variants share the same projections, the same head layout and the same
learnable per-head temperature, so the only thing that differs between runs is
the mechanism itself:

    routing    "none"      every head sees every token (standard attention)
               "topk"      MoSA expert-choice routing: each head hard-selects
                           its own top-k tokens and runs a k x k attention
               "sparsemax" differentiable routing: sparsemax over the router
                           scores; tokens with exact-zero probability are
                           dropped, the rest are gathered and gated by their
                           routing weight.  The number of kept tokens is
                           learned per (sequence, head) instead of hardcoded.

    normaliser "softmax"   dense attention rows, no exact zeros
               "sparsemax" projection onto the simplex, exact zeros inside the
                           selected block

Every module records diagnostics for the last forward pass in ``last_stats``.

Causality note
--------------
Expert-choice routing scores the whole sequence, so *which* tokens a head picks
depends on future tokens.  That is inherent to expert choice (it is equally
true of expert-choice MoE) and is why the benchmark tasks in this repo are
bidirectional/encoder-style.  The ``causal`` flag masks attention *within* the
selected block by original position and is exercised by the tests, but it does
not make the routing decision causal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from .sparsemax import get_normaliser, sparsemax

ROUTINGS = ("none", "topk", "sparsemax")


@dataclass
class AttentionConfig:
    d_model: int = 128
    n_heads: int = 4
    routing: str = "none"
    normaliser: str = "softmax"
    capacity_ratio: float = 0.25     # k = ceil(capacity_ratio * N) for topk routing
    min_capacity: int = 4
    causal: bool = False
    learn_temp: bool = True
    router_gate: str = "mean"        # sparsemax router: "mean" (p * |S|) or "raw" (p)
    dropout: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        if self.routing not in ROUTINGS:
            raise ValueError(f"routing must be one of {ROUTINGS}, got {self.routing!r}")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 < self.capacity_ratio <= 1.0:
            raise ValueError("capacity_ratio must be in (0, 1]")
        if self.router_gate not in ("mean", "raw"):
            raise ValueError("router_gate must be 'mean' or 'raw'")
        if not self.name:
            self.name = f"{self.routing}-{self.normaliser}"


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
        if cfg.routing != "none":
            self.router = nn.Linear(cfg.d_model, cfg.n_heads, bias=False)
        else:
            self.router = None
        self.last_stats: dict[str, float] = {}
        # set True to keep .grad on the router logits (see metrics.router_grad_coverage)
        self.retain_router_grad = False
        self.last_router_scores: Tensor | None = None

    # ---------------------------------------------------------------- helpers
    def _split(self, t: Tensor) -> Tensor:
        b, n, _ = t.shape
        return t.view(b, n, self.cfg.n_heads, self.d_head).transpose(1, 2)  # [B,H,N,dh]

    def _scores(self, q: Tensor, k: Tensor) -> Tensor:
        temp = self.log_temp.exp().view(1, -1, 1, 1)
        return (q @ k.transpose(-1, -2)) * (temp / math.sqrt(self.d_head))

    def capacity(self, n: int) -> int:
        return max(self.cfg.min_capacity, min(n, math.ceil(self.cfg.capacity_ratio * n)))

    def _allowed(self, q_pos: Tensor, k_pos: Tensor, k_alive: Tensor) -> Tensor:
        """[B,H,Nq,Nk] boolean mask of attendable pairs."""
        allowed = k_alive[:, :, None, :].expand(-1, -1, q_pos.size(-1), -1)
        if self.cfg.causal:
            allowed = allowed & (q_pos[:, :, :, None] >= k_pos[:, :, None, :])
        return allowed

    # ------------------------------------------------------------------ router
    def _route(self, x: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
        """Return (idx [B,H,C], gate [B,H,C], alive [B,H,C], stats).

        ``valid`` is [B,N]; padded tokens must never consume routing capacity.
        """
        b, n, _ = x.shape
        scores = self.router(x).transpose(1, 2)                     # [B,H,N]
        route_mask = valid[:, None, :].expand(b, self.cfg.n_heads, n)
        if self.retain_router_grad and scores.requires_grad:
            scores.retain_grad()
            self.last_router_scores = scores

        if self.cfg.routing == "topk":
            cap = min(self.capacity(n), int(valid.sum(-1).max()))
            masked = scores.masked_fill(~route_mask, float("-inf"))
            idx = masked.topk(cap, dim=-1).indices.sort(dim=-1).values
            gate = torch.sigmoid(scores.gather(-1, idx))
            alive = route_mask.gather(-1, idx)
            gate = gate * alive
            support = alive.sum(-1).to(x.dtype)
        else:  # sparsemax router: support size is learned
            p = sparsemax(scores, dim=-1, mask=route_mask)          # [B,H,N]
            support_n = (p > 0).sum(-1)                             # [B,H]
            support = support_n.to(p.dtype)
            cap = int(support_n.max().clamp(min=1).item())
            idx = p.topk(cap, dim=-1).indices.sort(dim=-1).values
            p_sel = p.gather(-1, idx)
            alive = p_sel > 0
            gate = p_sel * support[..., None] if self.cfg.router_gate == "mean" else p_sel
            gate = gate * alive

        hits = torch.zeros(b, n, device=x.device, dtype=x.dtype)
        hits.scatter_add_(1, idx.reshape(b, -1), alive.to(x.dtype).reshape(b, -1))
        covered = (hits > 0) & valid
        stats = {
            "route_capacity": float(idx.size(-1)),
            "route_support": float(support.mean()),
            "route_support_std": float(support.std()) if support.numel() > 1 else 0.0,
            "token_coverage": float(covered.sum() / valid.sum().clamp(min=1)),
        }
        return idx, gate, alive, stats

    # ----------------------------------------------------------------- forward
    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """x: [B,N,d]; key_padding_mask: [B,N] True = real token."""
        b, n, _ = x.shape
        q, k, v = (self._split(t) for t in self.qkv(x).chunk(3, dim=-1))
        pos = torch.arange(n, device=x.device).view(1, 1, n).expand(b, self.cfg.n_heads, n)
        valid = (
            torch.ones(b, n, device=x.device, dtype=torch.bool)
            if key_padding_mask is None
            else key_padding_mask
        )
        stats: dict[str, float] = {}

        if self.cfg.routing == "none":
            q_pos = k_pos = pos
            alive = valid[:, None, :].expand(b, self.cfg.n_heads, n)
            gate = None
            stats.update(route_capacity=float(n), route_support=float(n),
                         route_support_std=0.0, token_coverage=1.0)
        else:
            idx, gate, sel_alive, stats = self._route(x, valid)
            gather = idx[..., None].expand(-1, -1, -1, self.d_head)
            q, k, v = (t.gather(2, gather) for t in (q, k, v))
            q_pos = k_pos = idx
            alive = sel_alive & valid[:, None, :].expand(b, self.cfg.n_heads, n).gather(2, idx)

        scores = self._scores(q, k)
        allowed = self._allowed(q_pos, k_pos, alive)
        attn = self.normalise(scores, dim=-1, mask=allowed)
        attn = attn * alive[..., None]                      # kill dead query rows
        stats.update(self._attn_stats(attn, allowed))

        out = self.drop(attn) @ v                            # [B,H,C,dh]
        if gate is not None:
            out = out * gate[..., None]

        if self.cfg.routing == "none":
            out = out.transpose(1, 2).reshape(b, n, self.cfg.d_model)
        else:
            dense = torch.zeros(b, self.cfg.n_heads, n, self.d_head, dtype=out.dtype, device=out.device)
            dense = dense.scatter_add(2, idx[..., None].expand(-1, -1, -1, self.d_head), out)
            out = dense.transpose(1, 2).reshape(b, n, self.cfg.d_model)

        self.last_stats = stats
        return self.out_proj(out)

    # ------------------------------------------------------------------ stats
    @torch.no_grad()
    def _attn_stats(self, attn: Tensor, allowed: Tensor) -> dict[str, float]:
        live_rows = (attn.sum(-1) > 0).to(attn.dtype)          # dead slots don't count
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


def attention_flops(cfg: AttentionConfig, n: int, effective_k: float | None = None) -> float:
    """Analytic multiply-accumulate count for one sequence, all heads.

    Counts the projections (Q,K,V,out) plus the two score/context matmuls.
    ``effective_k`` overrides the block width (used for the sparsemax router,
    whose support size is data dependent).
    """
    d, h = cfg.d_model, cfg.n_heads
    proj = 4 * n * d * d
    if cfg.routing == "none":
        width = n
    elif effective_k is not None:
        width = effective_k
    else:
        width = max(cfg.min_capacity, min(n, math.ceil(cfg.capacity_ratio * n)))
    route = n * d * h if cfg.routing != "none" else 0.0
    attn = 2 * h * (width ** 2) * (d / h)
    return float(proj + route + attn)
