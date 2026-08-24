"""Extra test: test-time compute scaling of the recurrent variants.

An HRM trained with N outer (H) cycles can be *run* with a different number of
cycles. This evaluates each trained recurrent checkpoint at H_cycles in
1..8 without any retraining, which is the cheapest probe of whether the
recurrence has learned an iterative refinement or just a fixed-depth function.
"""
import argparse, json, math
from pathlib import Path

import torch

from lab.blocks import BlockConfig
from lab.data import CharData
from lab.model import build
from lab.train import evaluate
from lab.variants import VARIANTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--cycles", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6, 8])
    p.add_argument("--eval-iters", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)

    data = CharData()
    out = {}
    for ckpt_path in sorted(a.results.glob("*.pt")):
        name = ckpt_path.stem.split("__seed")[0]
        if name != ckpt_path.stem:      # only the seed-0 checkpoints
            continue
        variant = VARIANTS[name]
        if variant.kind != "hrm":
            continue
        ckpt = torch.load(ckpt_path, weights_only=False)
        base = BlockConfig(hidden_size=ckpt["hidden"], num_heads=ckpt["heads"],
                           max_seq_len=ckpt["seq_len"])
        curve = {}
        for cycles in a.cycles:
            model = build(
                type(variant)(**{**variant.__dict__, "H_cycles": cycles}),
                data.vocab_size, base,
            )
            model.load_state_dict(ckpt["state_dict"])
            loss = evaluate(model, data, a.batch_size, ckpt["seq_len"], a.eval_iters)
            curve[cycles] = {"val_loss": loss, "val_bpc": loss / math.log(2)}
            print(f"{variant.name:24s} H_cycles={cycles} val_bpc {loss/math.log(2):.4f}", flush=True)
        out[variant.name] = {"trained_H_cycles": variant.H_cycles, "curve": curve}
    (a.results / "loop_scaling.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
