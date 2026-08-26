"""Sparse mixture-of-experts FFN: one always-on shared expert plus top-k routed experts.

Identical in both arms of the experiment -- only the attention module differs --
so this file exists to make the baseline a realistic modern block rather than a
dense toy, not as a variable under test.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    """SwiGLU feed-forward: ``W_down(silu(W_gate x) * W_up x)``."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoEFeedForward(nn.Module):
    """Top-k routed experts + a shared expert applied to every token.

    The shared expert absorbs the knowledge every token needs, letting the routed
    experts specialise -- the arrangement used by current sparse LLMs.

    Two auxiliary losses keep routing healthy and are returned for logging:
      * **load-balance loss** ``n * sum_i f_i * P_i`` (Switch Transformer), where
        ``f_i`` is the fraction of tokens routed to expert ``i`` and ``P_i`` the
        mean router probability. Minimised by uniform routing.
      * **router z-loss** ``mean(logsumexp(logits)^2)``, which keeps the router
        logits from drifting to large magnitudes.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        top_k: int = 2,
        load_balance_coef: float = 0.01,
        router_z_coef: float = 1e-3,
    ):
        super().__init__()
        if top_k > n_routed_experts:
            raise ValueError(f"top_k {top_k} exceeds n_routed_experts {n_routed_experts}")
        self.n_routed_experts, self.top_k = n_routed_experts, top_k
        self.load_balance_coef, self.router_z_coef = load_balance_coef, router_z_coef

        # Routed experts are narrower so that total params stay comparable to a
        # dense FFN of width d_ff while only top_k of them run per token.
        d_expert = max(1, d_ff // n_routed_experts) * 2
        self.experts = nn.ModuleList(
            [SwiGLUExpert(d_model, d_expert) for _ in range(n_routed_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [SwiGLUExpert(d_model, d_expert) for _ in range(n_shared_experts)]
        )
        self.router = nn.Linear(d_model, n_routed_experts, bias=False)

        self.last_aux_loss = torch.tensor(0.0)
        self.last_expert_fractions: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        flat = x.reshape(-1, d)

        logits = self.router(flat)
        probs = logits.softmax(dim=-1)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)
        # Renormalise over the selected experts so each token's gates sum to 1.
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            # Gather the tokens routed here; skipping empties is what makes this sparse.
            token_idx, slot = (topk_idx == i).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            gate = topk_probs[token_idx, slot].unsqueeze(-1)
            out.index_add_(0, token_idx, gate * expert(flat[token_idx]))

        for shared in self.shared_experts:
            out = out + shared(flat)

        self.last_aux_loss = self._aux_loss(probs, topk_idx, flat.shape[0], logits)
        return out.view(b, t, d)

    def _aux_loss(
        self,
        probs: torch.Tensor,
        topk_idx: torch.Tensor,
        n_tokens: int,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        counts = torch.bincount(
            topk_idx.reshape(-1), minlength=self.n_routed_experts
        ).to(probs.dtype)
        fractions = counts / (n_tokens * self.top_k)
        mean_prob = probs.mean(dim=0)

        load_balance = self.n_routed_experts * (fractions * mean_prob).sum()
        z_loss = logits.logsumexp(dim=-1).pow(2).mean()

        self.last_expert_fractions = fractions.detach()
        return self.load_balance_coef * load_balance + self.router_z_coef * z_loss

    def expert_utilisation_entropy(self) -> float:
        """Entropy in nats of the routed-token distribution.

        ``log(n_routed_experts)`` means perfectly balanced routing; near 0 means
        the router has collapsed onto a single expert.
        """
        if self.last_expert_fractions is None:
            return float("nan")
        f = self.last_expert_fractions.clamp_min(1e-12)
        return float(-(f * f.log()).sum())
