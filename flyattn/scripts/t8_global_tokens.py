"""T8 - asymmetric global tokens at the M3 budget.

M3's census says ~1.1% of FlyWire neurons are strongly asymmetric (680
broadcasters, 782 integrators at a minimum-total-degree cutoff of 50), not the
~30% a naive rich-club reading suggests. That 1.1% is the global-token budget.

Arms, all on a local (window) base mask at matched density and matched total
global budget:
  none          local window only
  symmetric     BigBird-style globals: read everything and read by everything
  asymmetric    half integrators (full read, restricted write), half broadcasters
                (restricted read, full write)

The asymmetric variant is a one-line mask change and appears to be untried.
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flyattn.model import Config  # noqa: E402
from flyattn import textdata, train as T, masks as MK, synth  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
SEQ = 256


def build(kind, n_global):
    base = MK.window_mask(SEQ, 10)
    if kind == "none":
        return base
    return MK.global_token_mask(SEQ, n_global, kind=kind, base=base,
                                rng=np.random.default_rng(7))


def main(steps=1500, seeds=(0, 1), n_globals=(4, 8), threads=1):
    corp = textdata.load_corpora()
    lp = textdata.bigram_logprobs(corp["train"])
    ev = {k: T.make_eval_batches(corp[k], 16, SEQ, 12, seed=5)
          for k in ("val", "ood_book", "ood_brown", "ood_reuters")}
    hard = T.hard_mask_for(ev["val"], lp)
    runs = []
    for ng in n_globals:
        for kind in ("none", "symmetric", "asymmetric"):
            for sd in seeds:
                m = build(kind, ng)
                cfg = Config(d_model=96, n_layers=3, n_heads=4, seq_len=SEQ,
                             n_experts=4, top_k=2, d_ff=192)
                res, _ = T.train_run(cfg, corp["train"], ev, steps=steps,
                                     batch_size=16, seed=sd, hard=hard,
                                     threads=threads, eval_every=500,
                                     struct_mask=m)
                res.update(kind=kind, n_global=ng, seed=sd,
                           mask_density=MK.density(m))
                runs.append(res)
                print(f"{kind:11s} ng={ng} seed={sd} density={MK.density(m):.4f} "
                      f"val={res['final']['val']:.4f} "
                      f"hard={res['final']['val_hard']:.4f} "
                      f"brown={res['final']['ood_brown']:.4f}", flush=True)
                json.dump(dict(seq_len=SEQ, runs=runs),
                          open(os.path.join(RES, "t8_global_tokens.json"), "w"),
                          indent=2)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seeds", default="0,1")
    a = p.parse_args()
    main(steps=a.steps, threads=a.threads,
         seeds=tuple(int(s) for s in a.seeds.split(",")))
