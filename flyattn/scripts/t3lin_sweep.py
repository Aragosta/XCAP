"""T3-linear: the span sweep redone against what the theory actually predicts.

Two fixes over T3:

  * linear-distance masks. T3 used the S1 circle, where the first and last
    positions are angular neighbours, inserting a diameter-2 shortcut at every
    beta and masking the very phase transition the sweep crosses.
  * the prediction is about DEPTH, not about beta. Long-range percolation says
    the mask's graph distance is the number of layers two tokens need before
    they can interact. So the grid crosses beta with layer count: a mask whose
    graph distance exceeds the model's depth should be crippled, and should
    recover when layers are added. Masks whose distance already fits in the
    depth should gain nothing from more layers beyond the usual capacity effect.

Measured graph distances at seq 160, mean degree 20:
    beta 1.3 -> 2.10   beta 2.0 -> 2.36   beta 3.0 -> 3.01   window -> 5.79

so L = 2 sits below the window's requirement and above nothing else; L = 4 sits
above all of them. The prediction is an interaction: the window (and beta 6)
should improve far more from L=2 to L=4 than the low-beta masks do.

Run on both architectures - plain MHA and MoE-MHA at matched active width - so
the result is not a property of the mixture layer.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flyattn.model import Config  # noqa: E402
from flyattn import textdata, train as T, masks as MK  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
SEQ = int(os.environ.get("FLYATTN_SEQ", 160))
MEAN_DEGREE = 20.0
STRUCTS = ("dense", "window", "b1.30", "b2.00", "b3.00", "b6.00")


def build(spec, rng):
    if spec == "dense":
        return None
    if spec == "window":
        return MK.window_mask(SEQ, int(MEAN_DEGREE / 2))
    return MK.powerlaw_span_mask(SEQ, float(spec[1:]), MEAN_DEGREE, rng)


def cells():
    for arch in ("moe", "mha"):
        for st in STRUCTS:
            for L in (2, 4):
                for sd in (0, 1):
                    yield dict(arch=arch, structure=st, n_layers=L, seed=sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()

    corp = textdata.load_corpora()
    lp = textdata.bigram_logprobs(corp["train"])
    ev = {k: T.make_eval_batches(corp[k], 16, SEQ, 8, seed=5)
          for k in ("val", "ood_book", "ood_brown", "ood_reuters")}
    hard = T.hard_mask_for(ev["val"], lp)

    out_path = os.path.join(RES, f"t3lin_shard{a.shard}.json")
    done = json.load(open(out_path))["runs"] if os.path.exists(out_path) else []
    seen = {tuple(r["key"]) for r in done}

    todo = [c for i, c in enumerate(cells()) if i % a.n_shards == a.shard]
    for c in todo:
        k = (c["arch"], c["structure"], c["n_layers"], c["seed"])
        if k in seen:
            continue
        rng = np.random.default_rng(4242 + c["seed"])
        mask = build(c["structure"], rng)
        cfg = Config(d_model=96, n_layers=c["n_layers"], n_heads=4, seq_len=SEQ,
                     n_experts=4, top_k=2, d_ff=192, use_moe=(c["arch"] == "moe"))
        gd = MK.graph_distance(mask) if mask is not None else (1.0, 1.0)
        print(f"== {k} meanD={gd[0]:.2f} maxD={gd[1]:.0f}", flush=True)
        res, _ = T.train_run(
            cfg, corp["train"], ev, steps=a.steps, batch_size=16, seed=c["seed"],
            hard=hard, threads=a.threads, eval_every=250, struct_mask=mask,
            progress=lambda r: print(f"   step {r['step']} val {r['val']:.4f} "
                                     f"({r['elapsed']:.0f}s)", flush=True))
        res.update(c)
        res["key"] = list(k)
        res["mean_graph_distance"] = gd[0]
        res["max_graph_distance"] = gd[1]
        res["mask_density"] = MK.density(mask) if mask is not None else 1.0
        done.append(res)
        json.dump(dict(seq_len=SEQ, runs=done), open(out_path, "w"), indent=2)


if __name__ == "__main__":
    main()
