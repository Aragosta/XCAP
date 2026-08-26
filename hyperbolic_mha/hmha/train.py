"""Training loop and evaluation for one experimental arm."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import torch

from .data import Corpus, get_batch, iter_eval_batches
from .metrics import bits_per_byte, peak_rss_mb
from .model import ModelConfig, MoETransformer

LN2 = math.log(2.0)


@dataclass
class TrainConfig:
    steps: int = 1500
    batch_size: int = 16
    seq_len: int = 256
    lr: float = 3e-3
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 12
    seed: int = 0
    log_every: int = 100


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to ``lr * min_lr_ratio``."""
    warmup = max(1, int(cfg.steps * cfg.warmup_ratio))
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, cfg.steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine)


@torch.no_grad()
def evaluate(
    model: MoETransformer,
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    max_batches: int | None = None,
    need_stats: bool = False,
) -> dict:
    """Cross-entropy and bits-per-byte over deterministic held-out windows.

    Token-weighted, not batch-weighted: a short final batch must not count the
    same as a full one.
    """
    model.eval()
    total_nats, total_tokens = 0.0, 0
    for x, y in iter_eval_batches(data, batch_size, seq_len, max_batches):
        out = model(x, y, need_stats=need_stats)
        n = y.numel()
        total_nats += out["ce_loss"].item() * n
        total_tokens += n

    mean_nats = total_nats / max(1, total_tokens)
    result = {
        "ce_loss": mean_nats,
        "bits_per_byte": bits_per_byte(mean_nats),
        "perplexity_per_byte": math.exp(min(mean_nats, 20.0)),
        "eval_tokens": total_tokens,
    }
    if need_stats:
        result.update(model.attention_stats())
    model.train()
    return result


def train_arm(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    corpus: Corpus,
    verbose: bool = True,
) -> dict:
    """Train one arm end to end and return every metric the report needs."""
    torch.manual_seed(train_cfg.seed)
    generator = torch.Generator().manual_seed(train_cfg.seed + 9973)

    model = MoETransformer(model_cfg)
    counts = model.param_counts()

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # norms, gates and curvature are scalars/vectors: decaying them is noise
        (no_decay if p.dim() < 2 else decay).append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
    )

    history: list[dict] = []
    best = {"bits_per_byte": float("inf"), "step": -1}
    grad_norms: list[float] = []
    nonfinite_steps = 0

    model.train()
    start = time.perf_counter()
    train_compute_s = 0.0

    for step in range(train_cfg.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, train_cfg)

        x, y = get_batch(corpus.train, train_cfg.batch_size, train_cfg.seq_len, generator)

        step_start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = model(x, y)
        out["loss"].backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        # A non-finite grad norm means the update would poison every weight; skip
        # it and count it, so instability shows up in the report as a number.
        if torch.isfinite(gnorm):
            opt.step()
        else:
            nonfinite_steps += 1
        train_compute_s += time.perf_counter() - step_start
        grad_norms.append(float(gnorm) if torch.isfinite(gnorm) else float("nan"))

        if (step + 1) % train_cfg.eval_every == 0 or step == train_cfg.steps - 1:
            metrics = evaluate(
                model,
                corpus.valid,
                train_cfg.batch_size,
                train_cfg.seq_len,
                train_cfg.eval_batches,
                need_stats=True,
            )
            record = {
                "step": step + 1,
                "tokens": (step + 1) * train_cfg.batch_size * train_cfg.seq_len,
                "train_ce": out["ce_loss"].item(),
                "train_aux": out["aux_loss"].item(),
                "elapsed_s": time.perf_counter() - start,
                **{f"val_{k}": v for k, v in metrics.items()},
            }
            history.append(record)
            if record["val_bits_per_byte"] < best["bits_per_byte"]:
                best = {"bits_per_byte": record["val_bits_per_byte"], "step": step + 1}
            if verbose:
                print(
                    f"    step {step + 1:5d}/{train_cfg.steps}  "
                    f"train_ce {record['train_ce']:.4f}  "
                    f"val_bpb {record['val_bits_per_byte']:.4f}  "
                    f"{record['elapsed_s']:.0f}s",
                    flush=True,
                )

    wall_s = time.perf_counter() - start
    final_val = evaluate(
        model, corpus.valid, train_cfg.batch_size, train_cfg.seq_len, train_cfg.eval_batches
    )
    final_test = evaluate(
        model, corpus.test, train_cfg.batch_size, train_cfg.seq_len, train_cfg.eval_batches
    )

    tokens_seen = train_cfg.steps * train_cfg.batch_size * train_cfg.seq_len
    return {
        "model_config": model_cfg.to_dict(),
        "train_config": asdict(train_cfg),
        "params": counts,
        "history": history,
        "final_val": final_val,
        "final_test": final_test,
        "best_val_bits_per_byte": best["bits_per_byte"],
        "best_val_step": best["step"],
        "attention_stats": model.attention_stats(),
        "cost": {
            "wall_s": wall_s,
            "train_compute_s": train_compute_s,
            "ms_per_step": 1000.0 * train_compute_s / train_cfg.steps,
            "tokens_per_s": tokens_seen / train_compute_s if train_compute_s else 0.0,
            "tokens_seen": tokens_seen,
            "peak_rss_mb": peak_rss_mb(),
        },
        "stability": {
            "nonfinite_grad_steps": nonfinite_steps,
            "grad_norm_mean": float(
                torch.tensor([g for g in grad_norms if g == g]).mean()
            ),
            "grad_norm_max": float(
                torch.tensor([g for g in grad_norms if g == g]).max()
            ),
        },
        "_model": model,
    }
