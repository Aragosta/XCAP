"""Synthetic tasks: multi-query associative recall (MQAR) and its harder variants.

MQAR (Arora et al., the Zoology harness) is the task that separates
recall-capable architectures from ones that merely match perplexity: a sequence
of key-value pairs followed by queries, scored only at the answer positions.
Difficulty is the number of KV pairs, which is also the recurrent-state pressure.

`gap_recall` is the long-range variant used for the T5 curvature diagnostic: the
key-value pair sits at a controlled distance from its query, so a mask that
bottlenecks long-range information fails at large gaps specifically.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import Config, Transformer

IGNORE = -100


def mqar_batch(batch, seq_len, vocab, n_kv, rng, n_query=None):
    """Returns x (B,T), y (B,T) with IGNORE everywhere but the answer positions."""
    n_query = n_query or n_kv
    x = np.zeros((batch, seq_len), np.int64)
    y = np.full((batch, seq_len), IGNORE, np.int64)
    half = vocab // 2
    for b in range(batch):
        keys = rng.choice(np.arange(1, half), size=n_kv, replace=False)
        vals = rng.choice(np.arange(half, vocab), size=n_kv, replace=True)
        seq = np.zeros(seq_len, np.int64)
        seq[:2 * n_kv:2] = keys
        seq[1:2 * n_kv:2] = vals
        # queries occupy the remainder, each followed by its answer slot
        pos = 2 * n_kv
        qi = rng.choice(n_kv, size=n_query, replace=False)
        for t in qi:
            if pos + 1 >= seq_len:
                break
            seq[pos] = keys[t]
            seq[pos + 1] = vals[t]
            y[b, pos] = vals[t]      # predict the value at the query position
            pos += 2
        # fill the tail with distractor keys that were never bound
        if pos < seq_len:
            seq[pos:] = rng.integers(1, half, seq_len - pos)
        x[b] = seq
    return x, y


def gap_recall_batch(batch, seq_len, vocab, gap, rng):
    """One key-value pair at a controlled distance from its query."""
    x = rng.integers(1, vocab // 2, (batch, seq_len)).astype(np.int64)
    y = np.full((batch, seq_len), IGNORE, np.int64)
    qpos = seq_len - 2
    kpos = np.maximum(0, qpos - gap)
    for b in range(batch):
        key = rng.integers(1, vocab // 2)
        val = rng.integers(vocab // 2, vocab)
        x[b, kpos] = key
        x[b, kpos + 1] = val
        x[b, qpos] = key
        x[b, qpos + 1] = val
        y[b, qpos] = val
    return x, y


def train_synth(cfg: Config, task_fn, steps=1500, batch_size=32, lr=2e-3,
                seed=0, threads=1, struct_mask=None, eval_batches=8,
                warmup=100, progress=None):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = Transformer(cfg)
    if struct_mask is not None:
        model.set_struct_mask(struct_mask)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    rng = np.random.default_rng(seed + 991)
    erng = np.random.default_rng(31337)
    evb = [task_fn(batch_size, erng) for _ in range(eval_batches)]
    curve = []
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = task_fn(batch_size, rng)
        logits, _ = model(torch.from_numpy(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               torch.from_numpy(y).reshape(-1), ignore_index=IGNORE)
        aux = sum(b.moe.aux for b in model.blocks) / len(model.blocks)
        opt.zero_grad(set_to_none=True)
        (loss + cfg.aux_loss_weight * aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, step / warmup) * 0.5 * (
                1 + math.cos(math.pi * min(1.0, step / steps)))
        opt.step()
        if step % 250 == 0 or step == steps:
            acc, el = eval_synth(model, evb)
            curve.append(dict(step=step, train_loss=float(loss.item()),
                              acc=acc, eval_loss=el, elapsed=time.time() - t0))
            if progress:
                progress(curve[-1])
    return dict(curve=curve, final=curve[-1], n_params=model.n_params()), model


@torch.no_grad()
def eval_synth(model, evb):
    model.eval()
    correct = tot = 0
    lsum = 0.0
    for x, y in evb:
        logits, _ = model(torch.from_numpy(x))
        yt = torch.from_numpy(y)
        m = yt != IGNORE
        pred = logits.argmax(-1)
        correct += int((pred[m] == yt[m]).sum())
        tot += int(m.sum())
        lsum += float(F.cross_entropy(logits[m], yt[m], reduction="sum"))
    model.train()
    return correct / max(tot, 1), lsum / max(tot, 1)
