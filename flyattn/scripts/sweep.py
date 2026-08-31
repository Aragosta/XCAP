"""T3 / T4 - the geometric-temperature x criticality grid on real text.

T3 sweeps the attention-mask structure along the S1/H2 temperature axis at fixed
N, density and degree sequence. The hot limit (configuration model) is one
endpoint of that continuum, not a separate "random" condition. FlyWire's fitted
beta from M1 is one of the cells.

T4 sweeps the QK initialisation gain, which moves the model across the
rank-collapse / entropy-collapse transition, and is crossed with structure so the
result is an interaction, not a main effect.

Moderators recorded for every cell: data scale (small vs full training budget),
task difficulty (average loss vs the hard-token subset), and distribution shift
(three OOD corpora).

Usage: sweep.py --grid t3 --shard 0 --n-shards 4
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flyattn.model import Config  # noqa: E402
from flyattn import textdata, train as T, masks as MK  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
SEQ = 256
MEAN_DEGREE = 20.0          # ~8.5% causal density at T=256
GAMMA = 3.1                 # from M1's fit to FlyWire
FLY_BETA = 1.30             # from M1

# "bX" places node i at angle 2*pi*i/T, so the S1 geometry is sequence locality.
# "bXr" keeps the same beta but assigns angles to positions at random: same
# topology, locality destroyed. That separates "geometry as a locality prior"
# from "geometry as topology", which is otherwise confounded on sequence data.
STRUCTURES = ["dense", "config", "b1.05", "b1.30", "b1.60", "b2.00", "b3.00",
              "b5.00", "b8.00", "window", "b1.30r", "b4.00r"]


def build_mask(name, rng):
    if name == "dense":
        return None
    if name == "config":
        return MK.configuration_mask(SEQ, MEAN_DEGREE, GAMMA, rng)
    if name == "window":
        return MK.window_mask(SEQ, int(MEAN_DEGREE / 2))
    if name.startswith("b"):
        rand_angles = name.endswith("r")
        beta = float(name[1:-1] if rand_angles else name[1:])
        return MK.s1_mask(SEQ, beta, MEAN_DEGREE, GAMMA, rng,
                          positions_as_angles=not rand_angles)
    raise ValueError(name)


def cells(grid):
    if grid == "t3":
        for s in STRUCTURES:
            for scale in ("small", "full"):
                for sd in (0, 1):
                    yield dict(structure=s, data_scale=scale, seed=sd, qk_gain=1.0)
    elif grid == "t4":
        for s in ("dense", "config", "b1.30", "b4.00"):
            for g in (0.25, 0.5, 1.0, 2.0, 4.0):
                for sd in (0, 1):
                    yield dict(structure=s, data_scale="full", seed=sd, qk_gain=g)
    else:
        raise ValueError(grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="t3")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()

    corp = textdata.load_corpora()
    lp = textdata.bigram_logprobs(corp["train"])
    ev = {k: T.make_eval_batches(corp[k], 16, SEQ, 12, seed=5)
          for k in ("val", "ood_book", "ood_brown", "ood_reuters")}
    hard = T.hard_mask_for(ev["val"], lp)
    budgets = {"small": 250_000, "full": None}

    todo = [c for i, c in enumerate(cells(a.grid)) if i % a.n_shards == a.shard]
    out_path = os.path.join(RES, f"sweep_{a.grid}_shard{a.shard}.json")
    done = json.load(open(out_path))["runs"] if os.path.exists(out_path) else []
    seen = {(r["structure"], r["data_scale"], r["seed"], r["qk_gain"]) for r in done}

    for c in todo:
        key = (c["structure"], c["data_scale"], c["seed"], c["qk_gain"])
        if key in seen:
            continue
        rng = np.random.default_rng(4242 + c["seed"])
        mask = build_mask(c["structure"], rng)
        cfg = Config(d_model=96, n_layers=3, n_heads=4, seq_len=SEQ,
                     n_experts=4, top_k=2, d_ff=192, qk_init_gain=c["qk_gain"])
        print(f"== {key} mask_density="
              f"{MK.density(mask) if mask is not None else 1.0:.4f}", flush=True)
        res, model = T.train_run(
            cfg, corp["train"], ev, steps=a.steps, batch_size=16, seed=c["seed"],
            hard=hard, threads=a.threads, eval_every=250,
            data_budget=budgets[c["data_scale"]],
            struct_mask=mask,
            progress=lambda r: print(f"   step {r['step']} val {r['val']:.4f} "
                                     f"hard {r.get('val_hard', 0):.4f} "
                                     f"({r['elapsed']:.0f}s)", flush=True))
        res.update(c)
        res["mask_density"] = (MK.density(mask) if mask is not None else 1.0)
        if mask is not None:
            i, j = torch.tril(mask, -1).nonzero(as_tuple=True)
            res["mask_mean_span"] = float((i - j).abs().float().mean())
            res["mask_frac_long"] = float(((i - j).abs() > SEQ // 4).float().mean())
        done.append(res)
        json.dump(dict(grid=a.grid, seq_len=SEQ, mean_degree=MEAN_DEGREE,
                       gamma=GAMMA, fly_beta=FLY_BETA, runs=done),
                  open(out_path, "w"), indent=2)


if __name__ == "__main__":
    main()
