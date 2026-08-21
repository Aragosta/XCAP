#!/usr/bin/env python3
"""Out-of-distribution evaluation: does the score function change what transfers?

Every other experiment here evaluates on the training distribution.  This one
trains once per variant and then evaluates the *same weights* under shifts the
model never saw:

  length      trained at N, evaluated at 1.5x and 2x N
  distractors trained with F facts, evaluated with 2F and 3F facts in the same
              sequence (more memories competing for the same query)
  norm-shift  the input embeddings are scaled at eval time, which is the shift
              the theory actually speaks to: a dot product is unbounded in the
              norms, a distance is not (dot-product attention is not Lipschitz;
              L2 attention is -- Kim et al., ICML 2021)

The third one is the decisive test.  If the energy view buys robustness, it
buys it because it is a metric, and the metric property is precisely a
statement about norms.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ebt.metrics import masked_loss_and_acc
from ebt.model import build_model
from ebt.tasks import Relational, build_task
from ebt.train import TrainConfig, run
from ebt.variants import variant

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def evaluate_on(model, task, batches: int, batch_size: int, seed: int,
                embed_scale: float = 1.0) -> float:
    model.eval()
    g = torch.Generator().manual_seed(seed)
    accs = []
    old = model.tok.weight.data.clone()
    if embed_scale != 1.0:
        model.tok.weight.data.mul_(embed_scale)
    for _ in range(batches):
        x, y, m = task.batch(batch_size, g)
        accs.append(float(masked_loss_and_acc(model(x), y, m)[1]))
    model.tok.weight.data.copy_(old)
    model.train()
    return sum(accs) / len(accs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+",
                    default=["dot-softmax", "energy-softmax", "energy-softmax-tied",
                             "energy-sigmoid"])
    ap.add_argument("--task", default="relational")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "generalisation.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    rows = []
    for name in a.variants:
        for seed in range(a.seeds):
            cfg = variant(name, d_model=a.d_model, n_heads=a.n_heads)
            tcfg = TrainConfig(steps=a.steps, lr=a.lr, n_layers=a.n_layers, seed=seed,
                               eval_every=a.steps, eval_batches=4)
            res = run(a.task, cfg, tcfg, seq_len=a.seq_len)
            model = build_model(build_task(a.task, a.seq_len), cfg, n_layers=a.n_layers)
            model.load_state_dict(res["state_dict"])

            row = {"variant": name, "seed": seed, "in_dist": res["final_acc"]}
            # length shift: the learned positional table only covers seq_len, so
            # extend it by repeating the last row (a deliberately dumb extension:
            # what we measure is the attention's robustness, not the position code)
            for mult in (1.5, 2.0):
                n = int(a.seq_len * mult)
                big = build_task(a.task, n)
                m2 = build_model(big, cfg, n_layers=a.n_layers)
                sd = dict(res["state_dict"])
                pos = sd["pos.weight"]
                sd["pos.weight"] = torch.cat(
                    [pos, pos[-1:].expand(n - pos.size(0), -1)], 0)
                m2.load_state_dict(sd)
                row[f"len_x{mult}"] = evaluate_on(m2, big, 4, 64, seed=555)
            # distractor shift: same length, more competing facts
            if a.task == "relational":
                for mult in (2, 3):
                    shifted = Relational(seq_len=a.seq_len,
                                         n_subjects_shown=min(8, 3 * mult))
                    row[f"facts_x{mult}"] = evaluate_on(model, shifted, 4, 64, seed=555)
            # norm shift: scale the embeddings the attention sees
            for scale in (0.5, 2.0, 4.0):
                row[f"norm_x{scale}"] = evaluate_on(
                    model, build_task(a.task, a.seq_len), 4, 64, seed=555, embed_scale=scale)
            rows.append(row)
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in row.items()), flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(a), "rows": rows}, indent=1))

    # aggregate
    print("\n=== mean over seeds (accuracy) ===")
    keys = [k for k in rows[0] if k not in ("variant", "seed")]
    width = max(len(v) for v in a.variants)
    print("variant".ljust(width), "  ".join(k.rjust(9) for k in keys))
    for name in a.variants:
        mine = [r for r in rows if r["variant"] == name]
        print(name.ljust(width),
              "  ".join(f"{sum(r[k] for r in mine)/len(mine):9.3f}" for k in keys))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
