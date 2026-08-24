"""HRM recurrent core (sapientinc/HRM-Text, models/baselines/hrm_nocarry_bp_warmup.py)
with swappable H/L block types, plus a plain (non-recurrent) baseline."""
from dataclasses import dataclass, field, replace
from typing import Literal, Optional

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lab.blocks import Block, BlockConfig, Stack
from lab.layers import ScaledEmbeddingInit, rms_norm, trunc_normal_init_


class RecurrentBlock(nn.Module):
    """HRM level: a Stack applied to (hidden_state + input_injection)."""

    def __init__(self, cfg: BlockConfig, n_layers: int):
        super().__init__()
        self.core = Stack(cfg, n_layers)

    def forward(self, hidden_states: Tensor, input_injection: Tensor) -> Tensor:
        return self.core(hidden_states + input_injection)


class HRMCore(nn.Module):
    """H/L hierarchical recurrence with 1-step-gradient + backprop warmup.

    Gradient policy is upstream's: only the last `bp_steps` level applications
    receive gradients (H prioritised, at least one for L); earlier cycles run
    under no_grad, so depth is free in memory.
    """

    def __init__(self, h_cfg: BlockConfig, l_cfg: BlockConfig,
                 H_cycles: int, L_cycles: int, h_layers: int = 1, l_layers: int = 1,
                 bp_min_steps: int = 2, bp_max_steps: int = 5, bp_warmup_ratio: float = 0.3):
        super().__init__()
        self.H_level = RecurrentBlock(h_cfg, h_layers)
        self.L_level = RecurrentBlock(l_cfg, l_layers)
        self.H_cycles, self.L_cycles = H_cycles, L_cycles
        self.bp_min_steps, self.bp_max_steps = bp_min_steps, bp_max_steps
        self.bp_warmup_ratio = bp_warmup_ratio
        self.zL_init = nn.Buffer(
            trunc_normal_init_(torch.empty(h_cfg.hidden_size), std=1.0), persistent=True
        )

    def bp_steps_for(self, step: int, total_steps: int) -> int:
        warmup = total_steps * self.bp_warmup_ratio
        progress = min(1.0, step / warmup) if warmup > 0 else 1.0
        return self.bp_min_steps + int(progress * (self.bp_max_steps - self.bp_min_steps))

    def forward(self, x: Tensor, bp_steps: int = 2) -> Tensor:
        z_H, z_L = x, self.zL_init.expand_as(x)
        H_bp = min(self.H_cycles, bp_steps - 1)
        L_bp = bp_steps - H_bp
        total_L = self.H_cycles * self.L_cycles
        for i in range(self.H_cycles):
            for k in range(i * self.L_cycles, (i + 1) * self.L_cycles):
                with torch.set_grad_enabled(torch.is_grad_enabled() and k >= total_L - L_bp):
                    z_L = self.L_level(z_L, z_H)
            with torch.set_grad_enabled(torch.is_grad_enabled() and i >= self.H_cycles - H_bp):
                z_H = self.H_level(z_H, z_L)
        return z_H


class PlainCore(nn.Module):
    """Non-recurrent transformer baseline."""

    def __init__(self, cfg: BlockConfig, n_layers: int):
        super().__init__()
        self.stack = Stack(cfg, n_layers)

    def bp_steps_for(self, step: int, total_steps: int) -> int:
        return 0

    def forward(self, x: Tensor, bp_steps: int = 0) -> Tensor:
        return self.stack(x)


class LM(nn.Module):
    def __init__(self, core: nn.Module, vocab_size: int, hidden_size: int, norm_eps: float = 1e-5):
        super().__init__()
        self.core = core
        self.norm_eps = norm_eps
        init_std = 1.0 / math.sqrt(hidden_size)
        self.embed = ScaledEmbeddingInit(vocab_size, hidden_size, init_std=init_std)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, std=init_std)

    def forward(self, tokens: Tensor, bp_steps: int = 2) -> Tensor:
        x = self.embed(tokens)
        x = self.core(x, bp_steps=bp_steps)
        return self.lm_head(rms_norm(x, self.norm_eps))

    def loss(self, tokens: Tensor, targets: Tensor, bp_steps: int = 2) -> Tensor:
        logits = self(tokens, bp_steps=bp_steps)
        return F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())


# ---------------------------------------------------------------- variants

@dataclass
class Variant:
    name: str
    kind: Literal["plain", "hrm"]
    mixer_h: str = "mha"
    ffn_h: str = "dense"
    mixer_l: str = "mha"
    ffn_l: str = "dense"
    n_layers: int = 4          # plain only
    h_layers: int = 1
    l_layers: int = 1
    H_cycles: int = 2
    L_cycles: int = 2
    bp_min_steps: int = 2
    bp_max_steps: int = 5
    note: str = ""


def build(variant: Variant, vocab_size: int, base: BlockConfig) -> LM:
    if variant.kind == "plain":
        cfg = replace(base, mixer=variant.mixer_h, ffn=variant.ffn_h)
        core = PlainCore(cfg, variant.n_layers)
    else:
        h_cfg = replace(base, mixer=variant.mixer_h, ffn=variant.ffn_h)
        l_cfg = replace(base, mixer=variant.mixer_l, ffn=variant.ffn_l)
        core = HRMCore(
            h_cfg, l_cfg, variant.H_cycles, variant.L_cycles,
            variant.h_layers, variant.l_layers,
            bp_min_steps=variant.bp_min_steps, bp_max_steps=variant.bp_max_steps,
        )
    return LM(core, vocab_size, base.hidden_size, base.norm_eps)
