#!/usr/bin/env python3
"""Train every attention variant on every probe task and dump the raw results.

    python experiments/run_benchmark.py --steps 1000 --seeds 2 --workers 4

Results land in results/results.json; build the report with report.py.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ebt.train import TrainConfig, run
from ebt.variants import VARIANT_NAMES, variant

ROOT = Path(__file__).resolve().parents[1]


def _job(args) -> dict:
    task, name, seed, opts = args
    torch.set_num_threads(opts["threads"])
    cfg = TrainConfig(
        steps=opts["steps"], batch_size=opts["batch_size"], seed=seed,
        n_layers=opts["n_layers"], eval_every=opts["eval_every"],
        eval_batches=opts["eval_batches"], lr=opts["lr"],
    )
    attn = variant(name, d_model=opts["d_model"], n_heads=opts["n_heads"])
    t0 = time.perf_counter()
    res = run(task, attn, cfg, seq_len=opts["seq_len"])
    print(f"[done {time.perf_counter()-t0:6.1f}s] {task:20s} {name:22s} seed={seed} "
          f"acc={res['final_acc']:.3f} loss={res['final_loss']:.3f}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+",
                    default=["associative_recall", "needle", "majority"])
    ap.add_argument("--variants", nargs="+", default=VARIANT_NAMES)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "results" / "results.json"))
    a = ap.parse_args()

    opts = {k: getattr(a, k) for k in
            ("steps", "batch_size", "n_layers", "eval_every", "eval_batches", "lr",
             "d_model", "n_heads", "seq_len", "threads")}
    jobs = [(t, v, s, opts) for t in a.tasks for v in a.variants for s in range(a.seeds)]
    print(f"{len(jobs)} runs on {a.workers} workers "
          f"({a.steps} steps, N={a.seq_len}, d={a.d_model})", flush=True)

    t0 = time.perf_counter()
    if a.workers > 1:
        with mp.get_context("spawn").Pool(a.workers) as pool:
            results = pool.map(_job, jobs)
    else:
        results = [_job(j) for j in jobs]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(a), "results": results}, indent=1))
    print(f"wrote {out} ({len(results)} runs, {time.perf_counter()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
