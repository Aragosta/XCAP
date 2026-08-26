"""Euclidean and hyperbolic multi-head attention behind one interface.

Both modules take ``(B, T, d_model)`` Euclidean activations and return the same
shape, so a model can swap one for the other with a single config flag. They
carry *identical* parameter sets -- four projections of shape ``(d_model, d_model)``
-- because the manifold's time coordinate is derived from the spatial part rather
than learned. That parity is what makes the comparison a controlled experiment.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import expmap0, klein_gyromidpoint, logmap0, lorentz_centroid

# Attention weights that are exactly zero would drag the Lorentz centroid toward
# the lightlike cone; masked positions get a large finite score instead of -inf.
_MASK_VALUE = -1e9


class RotaryEmbedding(nn.Module):
    """Standard RoPE, cached per (seq_len, device, dtype)."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: tuple | None = None

    def _cos_sin(self, seq_len: int, device, dtype):
        if self._cache is not None:
            cos, sin, n, dev, dt = self._cache
            if n >= seq_len and dev == device and dt == dtype:
                return cos[:seq_len], sin[:seq_len]
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)
        self._cache = (cos, sin, seq_len, device, dtype)
        return cos, sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply RoPE to ``(B, H, T, head_dim)``."""
        seq_len = x.shape[-2]
        cos, sin = self._cos_sin(seq_len + offset, x.device, x.dtype)
        cos = cos[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = sin[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin


class EuclideanMHA(nn.Module):
    """Ordinary causal multi-head attention: RoPE + scaled dot product."""

    kind = "euclidean"

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, rope: bool = True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim) if rope else None

        self.last_attn_entropy: float | None = None

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, need_stats: bool = False) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self._split(self.w_q(x)), self._split(self.w_k(x)), self._split(self.w_v(x))
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)

        if need_stats:
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(_causal_mask(t, x.device), _MASK_VALUE)
            attn = scores.softmax(dim=-1)
            self.last_attn_entropy = _entropy(attn)
            out = attn @ v
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )

        return self.w_o(out.transpose(1, 2).reshape(b, t, self.d_model))


class HyperbolicMHA(nn.Module):
    """Causal MHA with attention scored by Lorentz inner product.

    Pipeline, per head:
      1. Euclidean projection to Q/K/V, then RoPE, in the tangent space at the origin.
      2. ``expmap0`` lifts each head vector onto the hyperboloid of curvature ``-c``.
         Lifting after the linear map is exactly the spec's ``exp_o(W log_o(x))``,
         since ``log_o(exp_o(u)) = u`` -- one exp/log pair cancels, so we skip it.
      3. Scores from the Lorentz inner product via the signature trick (below).
      4. Softmax, then a manifold-aware weighted mean of the value points.
      5. ``logmap0`` returns to flat space for the residual stream and MLP.

    Args:
        score_sign: ``"corrected"`` uses ``+<q,k>_L``, which *decreases* with
            geodesic distance, so attention concentrates on nearby tokens.
            ``"spec"`` uses ``-<q,k>_L`` exactly as written in the reference,
            which increases with distance and therefore attends to the farthest
            tokens. Kept as a flag so the difference is measured, not argued.
        aggregation: ``"lorentz_centroid"`` (default), ``"klein"`` (the Einstein
            gyromidpoint the spec describes), or ``"tangent_mean"`` (average in
            tangent space -- an ablation isolating how much the curved
            aggregation matters versus the curved scoring).
        score_scale: ``"sqrt_d"`` divides by ``sqrt(head_dim)`` as the spec says;
            ``"learned"`` adds one scalar per head. Lorentz products span
            ``cosh(sqrt(c) d)``, a far wider dynamic range than dot products, so
            the fixed scale can saturate the softmax -- ``"learned"`` tests that.
    """

    kind = "hyperbolic"

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        rope: bool = True,
        curvature: float = 1.0,
        learnable_curvature: bool = False,
        score_sign: str = "corrected",
        aggregation: str = "lorentz_centroid",
        score_scale: str = "sqrt_d",
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        if score_sign not in ("corrected", "spec"):
            raise ValueError(f"unknown score_sign {score_sign!r}")
        if aggregation not in ("lorentz_centroid", "klein", "tangent_mean"):
            raise ValueError(f"unknown aggregation {aggregation!r}")
        if score_scale not in ("sqrt_d", "learned"):
            raise ValueError(f"unknown score_scale {score_scale!r}")

        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.score_sign_value = 1.0 if score_sign == "corrected" else -1.0
        self.score_sign, self.aggregation, self.score_scale = score_sign, aggregation, score_scale

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim) if rope else None

        # Curvature is stored through softplus so it stays strictly positive
        # whatever the optimiser does. raw = log(exp(c) - 1) inverts softplus.
        raw_c = math.log(math.expm1(curvature))
        raw = torch.full((n_heads, 1, 1), raw_c)
        self.raw_curvature = nn.Parameter(raw, requires_grad=learnable_curvature)
        self.learnable_curvature = learnable_curvature

        if score_scale == "learned":
            self.log_temperature = nn.Parameter(torch.zeros(n_heads, 1, 1))

        # Signature vector diag(-1, 1, ..., 1). Scaling one operand elementwise and
        # then matmul is equivalent to Q M K^T but avoids a second matmul.
        signature = torch.ones(self.head_dim + 1)
        signature[0] = -1.0
        self.register_buffer("signature", signature, persistent=False)

        self.last_attn_entropy: float | None = None
        self.last_radius: float | None = None

    @property
    def curvature(self) -> torch.Tensor:
        """Per-head curvature ``c > 0``, shaped ``(H, 1, 1)`` to broadcast."""
        return F.softplus(self.raw_curvature)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, need_stats: bool = False) -> torch.Tensor:
        b, t, _ = x.shape
        c = self.curvature.to(x.dtype)

        q, k, v = self._split(self.w_q(x)), self._split(self.w_k(x)), self._split(self.w_v(x))
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)

        # (B, H, T, head_dim) -> (B, H, T, head_dim + 1) on the hyperboloid.
        q_h, k_h, v_h = expmap0(q, c), expmap0(k, c), expmap0(v, c)

        # <q, k>_L for every pair, as one matmul: scale q by the signature, then
        # contract. This is the Q M K^T trick with M folded into an elementwise op.
        inner = (q_h * self.signature.to(x.dtype)) @ k_h.transpose(-1, -2)

        scores = self.score_sign_value * inner
        if self.score_scale == "learned":
            scores = scores * self.log_temperature.exp() / math.sqrt(self.head_dim)
        else:
            scores = scores / math.sqrt(self.head_dim)

        scores = scores.masked_fill(_causal_mask(t, x.device), _MASK_VALUE)
        attn = scores.softmax(dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)

        if self.aggregation == "lorentz_centroid":
            out_h = lorentz_centroid(attn, v_h, c)
            out = logmap0(out_h, c)
        elif self.aggregation == "klein":
            out_h = klein_gyromidpoint(attn, v_h, c)
            out = logmap0(out_h, c)
        else:  # tangent_mean -- curved scoring, flat aggregation
            out = attn @ v

        if need_stats:
            self.last_attn_entropy = _entropy(attn)
            self.last_radius = float(out.detach().norm(dim=-1).mean())

        return self.w_o(out.transpose(1, 2).reshape(b, t, self.d_model))


def build_attention(kind: str, d_model: int, n_heads: int, dropout: float = 0.0, **kwargs):
    """Construct an attention module by name, ignoring options the arm cannot use."""
    if kind == "euclidean":
        return EuclideanMHA(d_model, n_heads, dropout=dropout, rope=kwargs.get("rope", True))
    if kind == "hyperbolic":
        allowed = {
            "rope",
            "curvature",
            "learnable_curvature",
            "score_sign",
            "aggregation",
            "score_scale",
        }
        return HyperbolicMHA(
            d_model, n_heads, dropout=dropout, **{k: v for k, v in kwargs.items() if k in allowed}
        )
    raise ValueError(f"unknown attention kind {kind!r}")


def _causal_mask(t: int, device) -> torch.Tensor:
    """True where a query may NOT attend (strictly upper triangular)."""
    return torch.ones(t, t, dtype=torch.bool, device=device).triu(1)


def _entropy(attn: torch.Tensor) -> float:
    """Mean attention entropy in nats -- low means collapsed/one-hot attention."""
    with torch.no_grad():
        return float(-(attn.clamp_min(1e-12).log() * attn).sum(dim=-1).mean())
