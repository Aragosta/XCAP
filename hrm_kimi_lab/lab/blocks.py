"""Blocks combining an HRM-Text style residual stream with swappable
token mixers (HRM gated MHA vs. Kimi Delta Attention) and
channel mixers (dense SwiGLU vs. Kimi Stable LatentMoE)."""
from dataclasses import dataclass
from typing import Literal, Optional

import math
import torch
from torch import Tensor, nn

from lab.layers import Attention, CosSin, RotaryEmbedding, SwiGLU, find_multiple, rms_norm

from src.kda import KDAConfig, KimiDeltaAttention          # vendored: pablo-reyes8/kimi-k3-pytorch
from src.stable_latent_moe import StableLatentMoE          # vendored: idem
from src.stable_latent_moe.config import StableLatentMoEConfig


@dataclass
class BlockConfig:
    hidden_size: int = 128
    num_heads: int = 4
    expansion: float = 4.0
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    mixer: Literal["mha", "kda"] = "mha"
    ffn: Literal["dense", "moe"] = "dense"

    # MoE (Kimi Stable LatentMoE), scaled down for a small model
    moe_routed_experts: int = 8
    moe_top_k: int = 2
    moe_shared_experts: int = 1
    moe_latent_ratio: float = 0.5

    # KDA
    kda_chunk_size: int = 64
    kda_conv_kernel: int = 4      # set to 1 to disable KDA's short conv (ablation)
    mha_conv_kernel: int = 0      # set to 4 to give MHA the same short conv (ablation)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def intermediate_size(self) -> int:
        # upstream HRM-Text formula, but with a smaller rounding multiple
        return find_multiple(round(self.expansion * self.hidden_size * 2 / 3), 32)

    @property
    def init_in_std(self) -> float:
        return 1.0 / math.sqrt(self.hidden_size)

    @property
    def init_out_std(self) -> float:
        return 1.0 / math.sqrt(self.intermediate_size)


class KDAMixer(nn.Module):
    """Kimi Delta Attention as a drop-in token mixer (linear attention, no RoPE)."""

    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.kda = KimiDeltaAttention(
            KDAConfig(
                d_model=cfg.hidden_size,
                num_heads=cfg.num_heads,
                key_head_dim=cfg.head_dim,
                value_head_dim=cfg.head_dim,
                short_conv_kernel_size=cfg.kda_conv_kernel,
                chunk_size=cfg.kda_chunk_size,
                secondary_tile_size=min(16, cfg.kda_chunk_size),
                init_std=1.0 / math.sqrt(cfg.hidden_size),
            )
        )

    def forward(self, x: Tensor, cos_sin: Optional[CosSin] = None) -> Tensor:
        return self.kda(x, mode="chunkwise").hidden_states


class MoEFFN(nn.Module):
    """Kimi Stable LatentMoE as a drop-in channel mixer."""

    def __init__(self, cfg: BlockConfig):
        super().__init__()
        hidden = cfg.intermediate_size
        latent = max(32, int(cfg.hidden_size * cfg.moe_latent_ratio))
        self.moe = StableLatentMoE(
            StableLatentMoEConfig(
                d_model=cfg.hidden_size,
                latent_dim=latent,
                num_shared_experts=cfg.moe_shared_experts,
                num_routed_experts=cfg.moe_routed_experts,
                routed_experts_per_token=cfg.moe_top_k,
                shared_expert_hidden_dim=hidden,
                routed_expert_hidden_dim=hidden,
                init_std=1.0 / math.sqrt(cfg.hidden_size),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        # Quantile-balancing routing-bias updates are training-only (upstream contract).
        return self.moe(x, update_routing_bias=self.training)


class Block(nn.Module):
    """Pre-norm residual block: mixer + FFN."""

    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.eps = cfg.norm_eps
        if cfg.mixer == "mha":
            self.mixer = Attention(
                cfg.hidden_size, cfg.head_dim, cfg.num_heads,
                init_std_in=cfg.init_in_std, init_std_out=cfg.init_in_std,
                conv_kernel=cfg.mha_conv_kernel,
            )
        else:
            self.mixer = KDAMixer(cfg)
        if cfg.ffn == "dense":
            self.ffn = SwiGLU(
                cfg.hidden_size, cfg.intermediate_size,
                init_std_in=cfg.init_in_std, init_std_out=cfg.init_out_std,
            )
        else:
            self.ffn = MoEFFN(cfg)

    def forward(self, x: Tensor, cos_sin: Optional[CosSin]) -> Tensor:
        x = x + self.mixer(rms_norm(x, self.eps), cos_sin)
        return x + self.ffn(rms_norm(x, self.eps))


class Stack(nn.Module):
    """A stack of blocks + its own RoPE table (upstream `Transformer`)."""

    def __init__(self, cfg: BlockConfig, n_layers: int):
        super().__init__()
        self.cfg = cfg
        self.needs_rope = cfg.mixer == "mha"
        if self.needs_rope:
            self.rotary_emb = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(n_layers)])

    def forward(self, x: Tensor) -> Tensor:
        cos_sin = self.rotary_emb(x.shape[1]) if self.needs_rope else None
        for layer in self.layers:
            x = layer(x, cos_sin)
        return rms_norm(x, self.cfg.norm_eps)
