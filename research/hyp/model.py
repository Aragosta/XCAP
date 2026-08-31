"""MoE + multi-head-attention transformer, written from scratch.

Small enough to train on 4 CPU cores, but structurally a real MoE MHA stack:
pre-norm blocks, causal MHA with per-head Q/K/V, and a top-k routed mixture of
experts in place of the feed-forward. Every tensor the probes need (residual
stream per layer, router logits, expert assignments) is captured on the way
through, because the geometry questions are all about intermediate state.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict


@dataclass
class Config:
    vocab_size: int = 259
    d_model: int = 192
    n_layers: int = 6
    n_heads: int = 6
    seq_len: int = 256
    n_experts: int = 6
    top_k: int = 2
    d_ff: int = 384
    dropout: float = 0.1
    moe: bool = True
    aux_loss_coef: float = 0.01


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class MHA(nn.Module):
    """Plain multi-head attention: n_heads independent heads, no GQA sharing.

    Kept as true MHA on purpose -- the question is about what the attention
    geometry does, and key/value sharing would confound the head-wise reads.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.seq_len, cfg.seq_len)).view(1, 1, cfg.seq_len, cfg.seq_len)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class MoE(nn.Module):
    """Top-k routed mixture of experts with a Switch-style load-balancing loss.

    Routing decisions are recorded because 'what does the router partition on'
    is one of the things being measured, not just an implementation detail.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.top_k = cfg.top_k
        self.n_experts = cfg.n_experts
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(cfg.d_model, cfg.d_ff) for _ in range(cfg.n_experts))
        self.last_router_logits = None
        self.last_top_idx = None

    def forward(self, x, capture=False):
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        logits = self.router(flat)
        probs = F.softmax(logits, dim=-1)
        top_w, top_i = probs.topk(self.top_k, dim=-1)
        top_w = top_w / top_w.sum(-1, keepdim=True)

        out = torch.zeros_like(flat)
        for e in range(self.n_experts):
            hit = (top_i == e)
            if not hit.any():
                continue
            tok = hit.any(-1).nonzero(as_tuple=True)[0]
            w = (top_w * hit).sum(-1)[tok].unsqueeze(-1)
            out[tok] += w * self.experts[e](flat[tok])

        # Switch load balance: n_experts * <fraction routed> . <mean gate prob>
        frac = torch.zeros(self.n_experts, device=x.device, dtype=probs.dtype)
        frac.scatter_add_(0, top_i.reshape(-1),
                          torch.ones_like(top_i.reshape(-1), dtype=probs.dtype))
        frac = frac / frac.sum().clamp(min=1)
        aux = self.n_experts * (frac * probs.mean(0)).sum()

        if capture:
            self.last_router_logits = logits.detach().view(B, T, self.n_experts)
            self.last_top_idx = top_i.detach().view(B, T, self.top_k)
        return out.view(B, T, C), aux


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = MHA(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.moe = MoE(cfg) if cfg.moe else None
        self.ff = None if cfg.moe else Expert(cfg.d_model, cfg.d_ff * cfg.top_k)

    def forward(self, x, capture=False):
        x = x + self.attn(self.n1(x))
        if self.moe is not None:
            d, aux = self.moe(self.n2(x), capture=capture)
        else:
            d, aux = self.ff(self.n2(x)), x.new_zeros(())
        return x + self.drop(d), aux


class MoETransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.nf = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, capture=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]

        # Residual stream at every depth: this IS the trajectory the Koopman and
        # hyperbolic probes treat as a dynamical system in layer-time.
        stream = [x] if capture else None
        aux_total = x.new_zeros(())
        for b in self.blocks:
            x, aux = b(x, capture=capture)
            aux_total = aux_total + aux
            if capture:
                stream.append(x)

        logits = self.head(self.nf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            loss = loss + self.cfg.aux_loss_coef * aux_total / max(1, self.cfg.n_layers)
        return logits, loss, stream

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def n_active_params(self):
        c = self.cfg
        per_layer_attn = 4 * c.d_model * c.d_model
        if c.moe:
            per_layer_ff = c.top_k * 2 * c.d_model * c.d_ff
        else:
            per_layer_ff = 2 * c.d_model * c.d_ff * c.top_k
        return c.n_layers * (per_layer_attn + per_layer_ff)
