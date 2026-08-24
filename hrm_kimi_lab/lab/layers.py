"""HRM-Text layers, ported for CPU.

Source: github.com/sapientinc/HRM-Text -> models/layers.py, models/transformer.py.
Only change vs. upstream: FlashAttention is replaced by
torch.nn.functional.scaled_dot_product_attention (no CUDA here). Gating,
RoPE, init scheme and SwiGLU shapes follow upstream exactly.
"""
from typing import Optional, Tuple

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

CosSin = Tuple[Tensor, Tensor]


def trunc_normal_init_(tensor: Tensor, std: float = 1.0):
    return tensor.normal_().fmod_(3.0).mul_(1.014762601732121 * std)


def find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: Tensor, cos_sin: CosSin):
    # x: [b, seq_len, num_heads, head_dim]
    cos, sin = cos_sin
    return ((x * cos.unsqueeze(-2)) + (rotate_half(x) * sin.unsqueeze(-2))).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, base):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> CosSin:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class LinearInit(nn.Module):
    """Truncated LeCun-normal linear (upstream `LinearInit`)."""

    def __init__(self, in_features, out_features, bias=False, batch_out_features=(), init_std=None):
        super().__init__()
        if init_std is None:
            init_std = 1.0 / (in_features ** 0.5)
        self.weight = nn.Parameter(
            trunc_normal_init_(
                torch.empty((math.prod(batch_out_features) * out_features, in_features)), std=init_std
            )
        )
        self.bias = nn.Parameter(torch.zeros(math.prod(batch_out_features) * out_features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)


class ScaledEmbeddingInit(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, init_std):
        super().__init__()
        self.scale = 1.0 / init_std
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * F.embedding(x, self.embedding_weight)


class Attention(nn.Module):
    """Upstream HRM-Text gated MHA (sigmoid gate on the attention output)."""

    def __init__(self, hidden_size, head_dim, num_heads, init_std_in=None, init_std_out=None):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.gqkv_proj = LinearInit(
            hidden_size, head_dim, batch_out_features=(4 * num_heads,), bias=False, init_std=init_std_in
        )
        self.o_proj = LinearInit(head_dim * num_heads, hidden_size, bias=False, init_std=init_std_out)

    def forward(self, hidden_states: Tensor, cos_sin: Optional[CosSin] = None) -> Tensor:
        b, t, _ = hidden_states.shape
        gqkv = self.gqkv_proj(hidden_states).view(b, t, 4 * self.num_heads, self.head_dim)
        gate, query, key, value = gqkv.split(self.num_heads, dim=-2)
        if cos_sin is not None:
            query = apply_rotary_pos_emb(query, cos_sin)
            key = apply_rotary_pos_emb(key, cos_sin)
        # [b, h, t, hd] for SDPA
        out = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), is_causal=True
        ).transpose(1, 2)
        out = (torch.sigmoid(gate) * out).reshape(b, t, self.num_heads * self.head_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size, init_std_in=None, init_std_out=None):
        super().__init__()
        self.gate_up_proj = LinearInit(
            hidden_size, intermediate_size, batch_out_features=(2,), bias=False, init_std=init_std_in
        )
        self.down_proj = LinearInit(intermediate_size, hidden_size, bias=False, init_std=init_std_out)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rms_norm(x: Tensor, eps: float = 1e-5) -> Tensor:
    return F.rms_norm(x, (x.shape[-1],), eps=eps)
