"""Shared training/evaluation harness.

House rules baked in here, per the pre-registration:
  * Never report in-distribution loss alone. Every run returns held-out loss,
    three OOD losses at increasing distance, a hard-subset loss, and the whole
    sample-efficiency curve.
  * The hard subset is the 20% of held-out positions where a byte-bigram model
    fitted on the training split is least confident. Easy continuation bytes
    dominate average loss and hide architectural differences.
  * Attention entropy per layer and head is logged throughout training, since
    entropy collapse is the diagnostic for the T4 criticality axis.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from .model import Config, Transformer
from .textdata import batches


def make_eval_batches(data, batch_size, seq_len, n_batches, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for x, y in batches(data, batch_size, seq_len, rng, n_batches=n_batches):
        out.append((torch.from_numpy(x), torch.from_numpy(y)))
    return out


def hard_mask_for(evb, bigram_lp, frac=0.2):
    """Boolean mask over eval positions marking the hardest `frac` under bigram."""
    scores = []
    for x, y in evb:
        scores.append(bigram_lp[x.numpy(), y.numpy()])
    s = np.concatenate([v.ravel() for v in scores])
    thr = np.quantile(s, frac)
    return [torch.from_numpy(bigram_lp[x.numpy(), y.numpy()] <= thr) for x, y in evb]


@torch.no_grad()
def evaluate(model, evb, hard=None):
    model.eval()
    tot, n = 0.0, 0
    htot, hn = 0.0, 0
    for i, (x, y) in enumerate(evb):
        logits, _ = model(x)
        ll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             y.reshape(-1), reduction="none")
        tot += ll.sum().item(); n += ll.numel()
        if hard is not None:
            m = hard[i].reshape(-1)
            htot += ll[m].sum().item(); hn += int(m.sum())
    model.train()
    out = {"loss": tot / n, "bpb": tot / n / math.log(2)}
    if hard is not None and hn:
        out["hard_loss"] = htot / hn
    return out


def train_run(cfg: Config, train_data, eval_sets, *, steps=3000, batch_size=32,
              lr=1e-3, warmup=100, weight_decay=0.1, seed=0, eval_every=500,
              hard=None, log_entropy=True, threads=2, data_budget=None,
              struct_mask=None, progress=None):
    """Train one model. `eval_sets` maps name -> list of (x, y) batches."""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = Transformer(cfg)
    if struct_mask is not None:
        model.set_struct_mask(struct_mask)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                            betas=(0.9, 0.95))
    rng = np.random.default_rng(seed + 777)
    data = train_data if data_budget is None else train_data[:data_budget]
    gen = batches(data, batch_size, cfg.seq_len, rng)
    curve, ent_log = [], []
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = next(gen)
        _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, step / warmup) * (
                0.5 * (1 + math.cos(math.pi * min(1.0, step / steps))))
        opt.step()
        if log_entropy and (step % 50 == 0 or step == 1):
            ent_log.append([step, model.attention_entropies().tolist(),
                            float(loss.item()), float(gn)])
        if step % eval_every == 0 or step == steps:
            row = {"step": step, "train_loss": float(loss.item()),
                   "tokens": step * batch_size * cfg.seq_len,
                   "grad_norm": float(gn), "elapsed": time.time() - t0}
            for name, evb in eval_sets.items():
                r = evaluate(model, evb, hard if name == "val" else None)
                row[name] = r["loss"]
                if "hard_loss" in r:
                    row[name + "_hard"] = r["hard_loss"]
            curve.append(row)
            if progress:
                progress(row)
    return dict(config=asdict(cfg), steps=steps, seed=seed, batch_size=batch_size,
                lr=lr, data_budget=data_budget, curve=curve,
                entropy_log=ent_log, final=curve[-1],
                n_params=model.n_params(), wall_s=time.time() - t0), model
