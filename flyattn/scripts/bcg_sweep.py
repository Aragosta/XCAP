"""Tests B, C and G.

B  Per-row temperature. tau_c(i) ~ sqrt(log k_i), so a single global 1/sqrt(d)
   is critical for at most one row. Crossed with how heterogeneous k_i actually
   is: dense causal attention (k_i = i+1, heterogeneous by construction),
   heavy-tailed degree sequences (gamma 2.1 and 3.1), a near-homogeneous one
   (gamma 10), and the sliding window (k_i uniform - the zero-prediction cell).
   Prediction: the correction helps in proportion to the spread of k_i, and does
   nothing for the window.

C  Hyperbolic depth x mask tail. T2c found hyperbolic attention pays only in
   deep layers. Prediction under the compounding story: the effect tracks the
   heaviness of the mask's tail and vanishes for a purely local mask.
   (Test A has already falsified the Levy version of that mechanism, so this now
   runs as a direct test of the claim itself rather than of its explanation.)

G  sigma-Reparam x structure. Zhai et al. prevent entropy collapse by
   reparameterising W_Q, W_K as gamma * W / sigma(W). If the T3 optimum in beta
   flattens once entropy collapse is prevented, the "structure" benefit was a
   training-stability benefit wearing a costume. The off arm is the existing T3
   full-data sweep, run under an identical configuration.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flyattn.model import Config  # noqa: E402
from flyattn import textdata, train as T, masks as MK  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
SEQ, MEAN_DEGREE, GAMMA = 256, 20.0, 3.1
HIP = "hyperbolic_ip"


def build_mask(spec, rng):
    """spec: 'dense' | 'window' | 'config' | 'b<beta>' | 'b<beta>g<gamma>'."""
    if spec == "dense":
        return None
    if spec == "window":
        return MK.window_mask(SEQ, int(MEAN_DEGREE / 2))
    if spec == "config":
        return MK.configuration_mask(SEQ, MEAN_DEGREE, GAMMA, rng)
    gamma = GAMMA
    if "g" in spec[1:]:
        b, g = spec[1:].split("g")
        beta, gamma = float(b), float(g)
    else:
        beta = float(spec[1:])
    return MK.s1_mask(SEQ, beta, MEAN_DEGREE, gamma, rng)


def cells(test):
    if test == "b":
        for st in ("dense", "window", "b2.00g2.1", "b2.00g3.1", "b2.00g10"):
            for rt in (False, True):
                for sd in (0, 1):
                    yield dict(structure=st, row_temp=rt, seed=sd, n_layers=3,
                               layer_kinds=None, sigma_reparam=False)
    elif test == "c":
        for st in ("window", "b2.00", "b1.05"):
            for place in ("qk", "deep2"):
                for sd in (0, 1):
                    lk = (("qk", "qk", HIP, HIP) if place == "deep2" else None)
                    yield dict(structure=st, row_temp=False, seed=sd, n_layers=4,
                               layer_kinds=lk, sigma_reparam=False, place=place)
    elif test == "g":
        for st in ("dense", "config", "b1.30", "b2.00", "b8.00", "window"):
            for sd in (0, 1):
                yield dict(structure=st, row_temp=False, seed=sd, n_layers=3,
                           layer_kinds=None, sigma_reparam=True)
    else:
        raise ValueError(test)


def key(c):
    return (c["structure"], c["row_temp"], c["sigma_reparam"],
            str(c["layer_kinds"]), c["seed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, choices=("b", "c", "g"))
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

    out_path = os.path.join(RES, f"test_{a.test}_shard{a.shard}.json")
    done = json.load(open(out_path))["runs"] if os.path.exists(out_path) else []
    seen = {tuple(r["key"]) for r in done}

    todo = [c for i, c in enumerate(cells(a.test)) if i % a.n_shards == a.shard]
    for c in todo:
        k = key(c)
        if k in seen:
            continue
        rng = np.random.default_rng(4242 + c["seed"])
        mask = build_mask(c["structure"], rng)
        cfg = Config(d_model=96, n_layers=c["n_layers"], n_heads=4, seq_len=SEQ,
                     n_experts=4, top_k=2, d_ff=192, row_temp=c["row_temp"],
                     sigma_reparam=c["sigma_reparam"],
                     layer_kinds=tuple(c["layer_kinds"]) if c["layer_kinds"] else ())
        print(f"== {k} density="
              f"{MK.density(mask) if mask is not None else 1.0:.4f}", flush=True)
        res, _ = T.train_run(
            cfg, corp["train"], ev, steps=a.steps, batch_size=16, seed=c["seed"],
            hard=hard, threads=a.threads, eval_every=500, struct_mask=mask,
            progress=lambda r: print(f"   step {r['step']} val {r['val']:.4f} "
                                     f"hard {r.get('val_hard',0):.4f} "
                                     f"({r['elapsed']:.0f}s)", flush=True))
        res.update({kk: vv for kk, vv in c.items() if kk != "layer_kinds"})
        res["layer_kinds"] = list(c["layer_kinds"]) if c["layer_kinds"] else None
        res["key"] = list(k)
        res["mask_density"] = MK.density(mask) if mask is not None else 1.0
        if mask is not None:
            kk = (mask & torch.tril(torch.ones_like(mask)).bool()).sum(-1).float()
            res["k_row_min"] = float(kk.min())
            res["k_row_max"] = float(kk.max())
            res["k_row_cv"] = float(kk.std() / kk.mean())
        done.append(res)
        json.dump(dict(test=a.test, runs=done), open(out_path, "w"), indent=2)


if __name__ == "__main__":
    main()
