"""The transformer under test: MoE blocks with a cross-layer attention residual.

Both experimental arms instantiate this exact class; ``ModelConfig.attention``
selects ``"euclidean"`` or ``"hyperbolic"`` and nothing else changes. Everything
outside the attention module -- embeddings, norms, MoE, residuals, tying, init --
is shared, which is what licenses attributing any measured difference to geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import build_attention
from .moe import MoEFeedForward


@dataclass
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 704
    max_seq_len: int = 256
    dropout: float = 0.0

    attention: str = "euclidean"  # "euclidean" | "hyperbolic"
    attn_residual: bool = True

    # MoE
    n_routed_experts: int = 4
    n_shared_experts: int = 1
    top_k: int = 2
    load_balance_coef: float = 0.01
    router_z_coef: float = 1e-3

    # hyperbolic-only knobs (ignored by the Euclidean arm)
    curvature: float = 1.0
    learnable_curvature: bool = False
    score_sign: str = "corrected"
    aggregation: str = "lorentz_centroid"
    score_scale: str = "sqrt_d"

    tag: str = field(default="", compare=False)

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class Block(nn.Module):
    """Pre-norm block with an MoE FFN and an optional cross-layer attention residual.

    The attention residual carries each layer's *attention output* forward to the
    next layer through a learnable per-layer gate:

        a_l = attn_l(norm(x)) + lambda_l * a_{l-1}
        x   = x + a_l
        x   = x + moe(norm(x))

    ``lambda_l`` starts at 0, so the model begins as an ordinary pre-norm
    transformer and only opens the extra path if it helps -- the residual can
    never make initialisation worse.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = build_attention(
            cfg.attention,
            cfg.d_model,
            cfg.n_heads,
            dropout=cfg.dropout,
            curvature=cfg.curvature,
            learnable_curvature=cfg.learnable_curvature,
            score_sign=cfg.score_sign,
            aggregation=cfg.aggregation,
            score_scale=cfg.score_scale,
        )
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.moe = MoEFeedForward(
            cfg.d_model,
            cfg.d_ff,
            n_routed_experts=cfg.n_routed_experts,
            n_shared_experts=cfg.n_shared_experts,
            top_k=cfg.top_k,
            load_balance_coef=cfg.load_balance_coef,
            router_z_coef=cfg.router_z_coef,
        )
        self.attn_residual = cfg.attn_residual
        if cfg.attn_residual:
            self.residual_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self, x: torch.Tensor, prev_attn: torch.Tensor | None = None, need_stats: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        a = self.attn(self.attn_norm(x), need_stats=need_stats)
        if self.attn_residual and prev_attn is not None:
            a = a + self.residual_gate * prev_attn
        x = x + a
        x = x + self.moe(self.ffn_norm(x))
        return x, (a if self.attn_residual else None)


class MoETransformer(nn.Module):
    """Byte-level causal LM. Weight-tied embeddings; the arm is one config flag."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tied

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        need_stats: bool = False,
    ) -> dict:
        x = self.drop(self.embed(idx))

        prev_attn = None
        for block in self.blocks:
            x, prev_attn = block(x, prev_attn, need_stats=need_stats)

        logits = self.lm_head(self.final_norm(x))
        out = {"logits": logits}

        aux = torch.stack([b.moe.last_aux_loss.to(logits.device) for b in self.blocks]).sum()
        out["aux_loss"] = aux

        if targets is not None:
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1)
            )
            out["ce_loss"] = ce
            out["loss"] = ce + aux
        return out

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
    ) -> torch.Tensor:
        """Autoregressive sampling. Recomputes the full prefix each step (no KV
        cache) -- fine at these sizes, and it keeps both arms on the same path."""
        self.eval()
        for _ in range(max_new_tokens):
            window = idx[:, -self.cfg.max_seq_len :]
            logits = self(window)["logits"][:, -1] / max(temperature, 1e-5)
            if top_k is not None:
                kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            idx = torch.cat([idx, torch.multinomial(logits.softmax(dim=-1), 1)], dim=1)
        return idx

    def param_counts(self) -> dict:
        """Total, trainable, and active-per-token parameter counts.

        Sparse models are misleading if only total params are quoted: a token
        only touches the shared expert plus ``top_k`` routed experts, so
        ``active`` is the number that governs per-token compute.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        inactive = 0
        for block in self.blocks:
            n_routed = block.moe.n_routed_experts
            skipped = n_routed - block.moe.top_k
            if n_routed:
                per_expert = sum(p.numel() for p in block.moe.experts[0].parameters())
                inactive += skipped * per_expert
        return {
            "total": total,
            "trainable": trainable,
            "active_per_token": total - inactive,
            "embedding": self.embed.weight.numel(),
        }

    @torch.no_grad()
    def attention_stats(self) -> dict:
        """Diagnostics from the last ``need_stats=True`` forward pass."""
        entropies = [
            b.attn.last_attn_entropy for b in self.blocks if b.attn.last_attn_entropy is not None
        ]
        radii = [
            getattr(b.attn, "last_radius", None)
            for b in self.blocks
            if getattr(b.attn, "last_radius", None) is not None
        ]
        stats = {
            "attn_entropy_mean": sum(entropies) / len(entropies) if entropies else float("nan"),
            "expert_entropy_mean": sum(b.moe.expert_utilisation_entropy() for b in self.blocks)
            / len(self.blocks),
            "expert_entropy_max": float(
                torch.log(torch.tensor(float(self.cfg.n_routed_experts)))
            ),
        }
        if radii:
            stats["manifold_radius_mean"] = sum(radii) / len(radii)
        if self.cfg.attention == "hyperbolic":
            stats["curvature_mean"] = float(
                torch.stack([b.attn.curvature.mean() for b in self.blocks]).mean()
            )
        if self.cfg.attn_residual:
            stats["attn_residual_gate_absmean"] = float(
                torch.stack([b.residual_gate.abs().mean() for b in self.blocks]).mean()
            )
        return stats
