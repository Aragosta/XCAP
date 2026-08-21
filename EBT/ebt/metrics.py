"""Evaluation, profiling and mechanism diagnostics."""

from __future__ import annotations

import time
from statistics import mean

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import Attention
from .tasks import Task


def masked_loss_and_acc(logits: Tensor, y: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    sel = mask.reshape(-1)
    lg = logits.reshape(-1, logits.size(-1))[sel]
    tg = y.reshape(-1)[sel]
    return F.cross_entropy(lg, tg), (lg.argmax(-1) == tg).float().mean()


@torch.no_grad()
def evaluate(model: nn.Module, task: Task, n_batches: int, batch_size: int, seed: int) -> dict[str, float]:
    model.eval()
    g = torch.Generator().manual_seed(seed)
    losses, accs, stats = [], [], []
    for _ in range(n_batches):
        x, y, m = task.batch(batch_size, g)
        loss, acc = masked_loss_and_acc(model(x), y, m)
        losses.append(float(loss)); accs.append(float(acc))
        stats.append(model.attention_stats())
    out = {"loss": mean(losses), "acc": mean(accs)}
    if stats and stats[0]:
        out.update({k: mean(s[k] for s in stats) for k in stats[0]})
    model.train()
    return out


def router_grad_coverage(model: nn.Module, task: Task, batch_size: int, seed: int) -> dict[str, float]:
    """How much of the routing signal actually receives gradient.

    The usual argument against hard top-k routing is that unselected tokens get
    exactly zero gradient, so the router cannot learn to *start* selecting them.
    This measures that directly: the fraction of router logits with a non-zero
    gradient after one backward pass.
    """
    attns = [m for m in model.modules() if isinstance(m, Attention) and m.router is not None]
    if not attns:
        return {}
    for a in attns:
        a.retain_router_grad = True
    g = torch.Generator().manual_seed(seed)
    x, y, m = task.batch(batch_size, g)
    model.zero_grad(set_to_none=True)
    loss, _ = masked_loss_and_acc(model(x), y, m)
    loss.backward()
    covered, magnitude = [], []
    for a in attns:
        gr = a.last_router_scores.grad
        if gr is None:
            continue
        covered.append(float((gr != 0).float().mean()))
        magnitude.append(float(gr.abs().mean()))
        a.retain_router_grad = False
        a.last_router_scores = None
    model.zero_grad(set_to_none=True)
    return {
        "router_grad_frac": mean(covered) if covered else 0.0,
        "router_grad_mag": mean(magnitude) if magnitude else 0.0,
    }


def benchmark_speed(model: nn.Module, task: Task, batch_size: int, iters: int = 12,
                    warmup: int = 3, seed: int = 0) -> dict[str, float]:
    g = torch.Generator().manual_seed(seed)
    x, y, m = task.batch(batch_size, g)

    def _fwd():
        with torch.no_grad():
            model(x)

    def _fwd_bwd():
        model.zero_grad(set_to_none=True)
        loss, _ = masked_loss_and_acc(model(x), y, m)
        loss.backward()

    out = {}
    for label, fn in (("fwd_ms", _fwd), ("fwd_bwd_ms", _fwd_bwd)):
        for _ in range(warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        out[label] = (time.perf_counter() - t0) / iters * 1e3   # ms per batch
    model.zero_grad(set_to_none=True)
    out["tokens_per_s_fwd"] = batch_size * task.seq_len / (out["fwd_ms"] / 1e3)
    return out


def attention_memory_bytes(model: nn.Module, seq_len: int, batch_size: int,
                           effective_width: float | None = None) -> float:
    """Bytes held by the materialised attention matrices (the term that blows up)."""
    total = 0.0
    for mod in model.modules():
        if isinstance(mod, Attention):
            if mod.cfg.routing == "none":
                w = seq_len
            elif effective_width is not None:
                w = effective_width
            else:
                w = mod.capacity(seq_len)
            total += batch_size * mod.cfg.n_heads * w * w * 4
    return total
