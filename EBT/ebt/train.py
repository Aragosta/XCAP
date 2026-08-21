"""Training loop and single-run experiment driver."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from statistics import mean, pstdev

import torch

from .attention import AttentionConfig
from .metrics import (attention_memory_bytes, benchmark_speed, evaluate,
                      masked_loss_and_acc)
from .model import build_model
from .tasks import build_task


@dataclass
class TrainConfig:
    steps: int = 1500
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup: int = 100
    grad_clip: float = 1.0
    eval_every: int = 100
    eval_batches: int = 8
    eval_batch_size: int = 64
    seed: int = 0
    n_layers: int = 2
    dropout: float = 0.0
    acc_threshold: float = 0.9      # for "steps to reach" sample efficiency


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(3.14159265 * p)).item()))


def run(task_name: str, attn_cfg: AttentionConfig, train_cfg: TrainConfig,
        seq_len: int = 128, verbose: bool = False) -> dict:
    torch.manual_seed(train_cfg.seed)
    task = build_task(task_name, seq_len)
    model = build_model(task, attn_cfg, n_layers=train_cfg.n_layers, dropout=train_cfg.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    g = torch.Generator().manual_seed(train_cfg.seed + 10_000)

    history, grad_norms = [], []
    steps_to_threshold = None
    t0 = time.perf_counter()
    for step in range(train_cfg.steps):
        for grp in opt.param_groups:
            grp["lr"] = _lr_at(step, train_cfg)
        x, y, m = task.batch(train_cfg.batch_size, g)
        loss, acc = masked_loss_and_acc(model(x), y, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        grad_norms.append(float(gn))
        opt.step()

        if (step + 1) % train_cfg.eval_every == 0 or step == train_cfg.steps - 1:
            ev = evaluate(model, task, train_cfg.eval_batches, train_cfg.eval_batch_size,
                          seed=99_999)
            ev["step"] = step + 1
            ev["train_loss"] = float(loss.detach())
            history.append(ev)
            if steps_to_threshold is None and ev["acc"] >= train_cfg.acc_threshold:
                steps_to_threshold = step + 1
            if verbose:
                print(f"  [{attn_cfg.name}/{task_name}] step {step+1:5d} "
                      f"loss {ev['loss']:.4f} acc {ev['acc']:.3f}")
    train_seconds = time.perf_counter() - t0

    final = history[-1]
    speed = benchmark_speed(model, task, train_cfg.batch_size)
    result = {
        "task": task_name,
        "variant": attn_cfg.name,
        "seed": train_cfg.seed,
        "attn_cfg": asdict(attn_cfg),
        "train_cfg": asdict(train_cfg),
        "seq_len": seq_len,
        "final": final,
        "best_acc": max(h["acc"] for h in history),
        "final_acc": final["acc"],
        "final_loss": final["loss"],
        "steps_to_acc": steps_to_threshold,
        "history": history,
        "params": model.n_params(),
        "flops_per_seq": model.flops_per_sequence(),
        "attn_matrix_bytes": attention_memory_bytes(model, seq_len, train_cfg.batch_size),
        "train_seconds": train_seconds,
        "grad_norm_mean": mean(grad_norms),
        "grad_norm_std": pstdev(grad_norms),
        "state_dict": model.state_dict(),
        **speed,
    }
    return result
