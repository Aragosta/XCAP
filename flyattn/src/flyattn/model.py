"""Baseline architecture for every training experiment here: a decoder-only
transformer with multi-head attention and a top-k mixture-of-experts FFN
(MoE-MHA). Everything under test is a switch on this one model.

Switches
--------
attn_kind
    "qk"        standard scaled dot-product attention.
    "keybias"   T2. softmax(-g||q-k||^2) == softmax(2g q.k - g||k||^2) because
                softmax is translation invariant in the query and the -||q||^2
                term is constant across keys. So Euclidean-distance attention is
                exactly standard attention with a per-key bias. This arm adds
                only that bias term, with g learnable per head.
    "euclid"    the literal -g||q-k||^2 form, kept to verify the identity
                numerically rather than on the whiteboard.
    "lorentz_ip" the Lorentzian inner product of *unconstrained* vectors used
                directly as the score. The indefinite signature is then just a
                fixed diagonal sign flip J, and softmax(<q,k>_L) = softmax((Jq).k),
                so W_Q absorbs it exactly: the same function class as "qk". This
                arm exists to make that concrete. (It is only exactly equivalent
                with use_rope=False; RoPE does not commute with J, which changes
                the score by an artefact of the rotation, not by geometry.)
    "hyperbolic" q and k are lifted to the hyperboloid, x -> (sqrt(1+||sx||^2), sx),
                and scored by -g * d_H(q,k)^2 with d_H = arccosh(-<q,k>_L). Here
                the signature does buy something: the lift couples ||q|| to ||k||
                multiplicatively, so unlike the Euclidean case the score does NOT
                decompose into (dot product + per-key bias). This is the only
                distance-attention variant T2's algebra does not kill.
    "hyperbolic_d" the same with -g * d_H instead of -g * d_H^2.
    "hyperbolic_ip" the Lorentzian inner product of the *lifted* vectors,
                g * <q,k>_L = -g * cosh(d_H). Monotone in hyperbolic distance and,
                like "hyperbolic", not decomposable into dot product + key bias.
                RoPE is norm-preserving, so the lifted time coordinate is
                unaffected by it and the geometry stays well defined.
qk_init_gain
    T4. multiplies the standard init std of W_Q and W_K, sweeping the
    rank-collapse / entropy-collapse transition.
attn_mask_fn
    T3/T8. callable(T) -> bool tensor (T,T) of *allowed* positions, intersected
    with the causal mask. Used to install S1/H2, connectome, or rewired masks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 256
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    seq_len: int = 256
    n_experts: int = 4
    top_k: int = 2
    d_ff: int = 256
    dropout: float = 0.0
    attn_kind: str = "qk"
    share_k: bool = False       # T9: one K (and Q) bundle read by every head
    share_q: bool = False
    qk_init_gain: float = 1.0
    aux_loss_weight: float = 0.01
    tie_embeddings: bool = True
    use_rope: bool = True
    extras: dict = field(default_factory=dict)

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


def rope_cache(seq_len, d_head, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    t = torch.arange(seq_len, device=device).float()
    f = torch.outer(t, inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None, :x.shape[2]], sin[None, None, :x.shape[2]]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # T9: 21 MBON types read the same ~2000-Kenyon-cell bundle. Sharing K
        # across heads tests whether head diversity belongs in the values rather
        # than the keys; the freed parameters are not given back, so the shared
        # arms are strictly smaller.
        self.q = nn.Linear(cfg.d_model, cfg.d_head if cfg.share_q else cfg.d_model,
                           bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_head if cfg.share_k else cfg.d_model,
                           bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        std = cfg.d_model ** -0.5
        for w in (self.q.weight, self.k.weight):
            nn.init.normal_(w, std=std * cfg.qk_init_gain)
        for w in (self.v.weight, self.o.weight):
            nn.init.normal_(w, std=std)
        if cfg.attn_kind in ("keybias", "euclid", "hyperbolic", "hyperbolic_d",
                             "hyperbolic_ip"):
            # log-gamma, one per head; init at the value that reproduces the
            # standard 1/sqrt(d) temperature
            self.log_gamma = nn.Parameter(
                torch.full((cfg.n_heads,), math.log(0.5 * cfg.d_head ** -0.5)))
        if cfg.attn_kind in ("hyperbolic", "hyperbolic_d", "hyperbolic_ip"):
            # radial scale of the lift, per head: it sets how far up the
            # hyperboloid typical activations sit, i.e. how curved the geometry
            # they actually experience is
            self.log_scale = nn.Parameter(torch.zeros(cfg.n_heads))
        self.last_entropy = None
        self.head_mask = None      # (H,) float, for pruning whole heads
        self.register_buffer("struct_mask", torch.empty(0), persistent=False)

    def forward(self, x, cos, sin, causal):
        B, T, C = x.shape
        H, D = self.cfg.n_heads, self.cfg.d_head
        hq, hk = (1 if self.cfg.share_q else H), (1 if self.cfg.share_k else H)
        q = self.q(x).view(B, T, hq, D).transpose(1, 2)
        k = self.k(x).view(B, T, hk, D).transpose(1, 2)
        if self.cfg.use_rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if hq == 1:
            q = q.expand(B, H, T, D)
        if hk == 1:
            k = k.expand(B, H, T, D)
        v = self.v(x).view(B, T, H, D).transpose(1, 2)

        if self.cfg.attn_kind == "qk":
            att = (q @ k.transpose(-2, -1)) * (D ** -0.5)
        elif self.cfg.attn_kind == "lorentz_ip":
            j = torch.ones(D, device=q.device)
            j[0] = -1.0
            att = ((q * j) @ k.transpose(-2, -1)) * (D ** -0.5)
        elif self.cfg.attn_kind in ("hyperbolic", "hyperbolic_d", "hyperbolic_ip"):
            g = self.log_gamma.exp().view(1, H, 1, 1)
            s_ = self.log_scale.exp().view(1, H, 1, 1)
            qs, ks = q * s_, k * s_
            q0 = (1.0 + (qs * qs).sum(-1)).sqrt()
            k0 = (1.0 + (ks * ks).sum(-1)).sqrt()
            # -<q,k>_L = q0 k0 - qs.ks >= 1 on the hyperboloid
            z = q0[..., None] * k0[:, :, None, :] - qs @ ks.transpose(-2, -1)
            if self.cfg.attn_kind == "hyperbolic_ip":
                att = -g * z          # g * <q,k>_L = -g * cosh(d_H)
            else:
                d = torch.acosh(z.clamp_min(1.0 + 1e-6))
                att = -g * (d if self.cfg.attn_kind == "hyperbolic_d" else d * d)
        else:
            g = self.log_gamma.exp().view(1, H, 1, 1)
            if self.cfg.attn_kind == "keybias":
                att = 2 * g * (q @ k.transpose(-2, -1)) - g * (k * k).sum(-1)[:, :, None, :]
            else:  # literal Euclidean distance kernel
                d2 = ((q * q).sum(-1)[..., None] - 2 * (q @ k.transpose(-2, -1))
                      + (k * k).sum(-1)[:, :, None, :])
                att = -g * d2

        mask = causal
        if self.struct_mask.numel():
            mask = mask & self.struct_mask[:T, :T]
        att = att.masked_fill(~mask, float("-inf"))
        p = att.softmax(-1)
        with torch.no_grad():
            self.last_entropy = (-(p.clamp_min(1e-9).log() * p).sum(-1)
                                 ).mean((0, 2)).detach()
        y = p @ v
        if self.head_mask is not None:
            y = y * self.head_mask.view(1, H, 1, 1)
        return self.o(y.transpose(1, 2).reshape(B, T, C))


class MoE(nn.Module):
    """Top-k mixture of SwiGLU experts with the standard load-balancing loss."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        E, C, F_ = cfg.n_experts, cfg.d_model, cfg.d_ff
        self.gate = nn.Linear(C, E, bias=False)
        self.w1 = nn.Parameter(torch.randn(E, C, F_) * C ** -0.5)
        self.w3 = nn.Parameter(torch.randn(E, C, F_) * C ** -0.5)
        self.w2 = nn.Parameter(torch.randn(E, F_, C) * F_ ** -0.5)
        self.aux = torch.tensor(0.0)

    def forward(self, x):
        B, T, C = x.shape
        E, k = self.cfg.n_experts, self.cfg.top_k
        flat = x.reshape(-1, C)
        logits = self.gate(flat)
        probs = logits.softmax(-1)
        topv, topi = probs.topk(k, -1)
        topv = topv / topv.sum(-1, keepdim=True)
        # load balancing (Switch Transformer): E * mean(frac_tokens) . mean(prob)
        frac = torch.zeros(E, device=x.device, dtype=probs.dtype)
        frac.scatter_add_(0, topi.reshape(-1),
                          torch.ones_like(topi.reshape(-1), dtype=probs.dtype))
        frac = frac / frac.sum()
        self.aux = E * (frac * probs.mean(0)).sum()
        out = torch.zeros_like(flat)
        for e in range(E):
            sel, slot = (topi == e).nonzero(as_tuple=True)
            if sel.numel() == 0:
                continue
            h = flat[sel]
            y = (F.silu(h @ self.w1[e]) * (h @ self.w3[e])) @ self.w2[e]
            out.index_add_(0, sel, y * topv[sel, slot][:, None])
        return out.view(B, T, C)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n1 = nn.RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.n2 = nn.RMSNorm(cfg.d_model)
        self.moe = MoE(cfg)

    def forward(self, x, cos, sin, causal):
        x = x + self.attn(self.n1(x), cos, sin, causal)
        return x + self.moe(self.n2(x))


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.emb.weight
        cos, sin = rope_cache(cfg.seq_len, cfg.d_head, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer(
            "causal", torch.tril(torch.ones(cfg.seq_len, cfg.seq_len, dtype=torch.bool)),
            persistent=False)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.emb(idx)
        causal = self.causal[:T, :T]
        for b in self.blocks:
            x = b(x, self.cos, self.sin, causal)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        aux = sum(b.moe.aux for b in self.blocks) / len(self.blocks)
        return logits, loss + self.cfg.aux_loss_weight * aux

    def set_struct_mask(self, mask: torch.Tensor | None):
        for b in self.blocks:
            b.attn.struct_mask = (torch.empty(0, dtype=torch.bool)
                                  if mask is None else mask)

    def attention_entropies(self):
        return torch.stack([b.attn.last_entropy for b in self.blocks])

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
