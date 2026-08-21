#!/usr/bin/env python3
"""Cost side of the comparison: how each variant scales with sequence length.

Untrained models, forward and forward+backward wall clock, analytic FLOPs and
the size of the materialised attention matrices.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ebt.metrics import attention_memory_bytes, benchmark_speed
from ebt.model import build_model
from ebt.tasks import build_task
from ebt.variants import VARIANT_NAMES, variant

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", nargs="+", type=int, default=[128, 256, 512, 1024])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--capacity-ratio", type=float, default=0.25)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "scaling.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    rows = []
    for n in a.seq_lens:
        task = build_task("majority", n)
        for name in VARIANT_NAMES:
            torch.manual_seed(0)
            cfg = variant(name, d_model=a.d_model, n_heads=a.n_heads,
                          capacity_ratio=a.capacity_ratio)
            model = build_model(task, cfg, n_layers=a.n_layers)
            with torch.no_grad():
                model(task.batch(2, torch.Generator().manual_seed(0))[0])
            width = model.attention_stats().get("route_support", n)
            speed = benchmark_speed(model, task, a.batch_size, iters=a.iters, warmup=2)
            rows.append({
                "seq_len": n, "variant": name, **speed,
                "flops_per_seq": model.flops_per_sequence(
                    width if cfg.routing == "sparsemax" else None),
                "attn_bytes": attention_memory_bytes(
                    model, n, a.batch_size, width if cfg.routing == "sparsemax" else None),
                "effective_width": width,
            })
            print(f"N={n:5d} {name:22s} fwd {speed['fwd_ms']:7.1f} ms  "
                  f"fwd+bwd {speed['fwd_bwd_ms']:7.1f} ms", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(a), "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
