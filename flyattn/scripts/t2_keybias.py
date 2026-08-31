"""T2 - the key-norm bias one-liner.

    softmax(-g||q-k||^2) = softmax(-g||q||^2 + 2g q.k - g||k||^2)
                         = softmax(2g q.k - g||k||^2)

because softmax is invariant to adding a constant across the key axis and
-g||q||^2 is exactly such a constant. So Euclidean-distance attention is
standard attention plus a per-key bias -g||k||^2 (with a learnable temperature).

Arms:
  qk       standard scaled dot product (baseline)
  keybias  the bias term only, gamma learnable per head
  euclid   the literal distance kernel, present to check the identity numerically

Gate: if `keybias` is indistinguishable from `qk` on sample efficiency, hard-token
loss and OOD loss, the "distance instead of QK" direction is dead in Euclidean
space and only the hyperbolic (indefinite-signature) version is worth trying.
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flyattn.model import Config, Transformer  # noqa: E402
from flyattn import textdata, train as T  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def identity_check():
    """Numerically verify the algebra the whole arm rests on."""
    torch.manual_seed(0)
    B, H, Tq, Tk, D = 2, 3, 7, 11, 16
    q = torch.randn(B, H, Tq, D)
    k = torch.randn(B, H, Tk, D)
    g = torch.rand(1, H, 1, 1) + 0.1
    d2 = ((q * q).sum(-1)[..., None] - 2 * (q @ k.transpose(-2, -1))
          + (k * k).sum(-1)[:, :, None, :])
    a = (-g * d2).softmax(-1)
    b = (2 * g * (q @ k.transpose(-2, -1)) - g * (k * k).sum(-1)[:, :, None, :]).softmax(-1)
    return float((a - b).abs().max())


def main(steps=3000, seeds=(0, 1), arms=("qk", "keybias", "euclid"),
         batch_size=32, threads=1, tag=""):
    err = identity_check()
    print(f"identity max |softmax difference| = {err:.3e}", flush=True)

    corp = textdata.load_corpora()
    lp = textdata.bigram_logprobs(corp["train"])
    seq = 128
    ev = {k: T.make_eval_batches(corp[k], batch_size, seq, 20, seed=5)
          for k in ("val", "ood_book", "ood_brown", "ood_reuters")}
    hard = T.hard_mask_for(ev["val"], lp)

    out = []
    for arm in arms:
        for sd in seeds:
            cfg = Config(attn_kind=arm, d_model=128, n_layers=4, n_heads=4,
                         seq_len=seq, n_experts=4, top_k=2, d_ff=256)
            print(f"== arm={arm} seed={sd}", flush=True)
            res, _ = T.train_run(cfg, corp["train"], ev, steps=steps,
                                 batch_size=batch_size, seed=sd, hard=hard,
                                 threads=threads, eval_every=250,
                                 progress=lambda r: print(
                                     f"   step {r['step']} val {r['val']:.4f} "
                                     f"hard {r.get('val_hard', float('nan')):.4f} "
                                     f"brown {r['ood_brown']:.4f} "
                                     f"({r['elapsed']:.0f}s)", flush=True))
            res["arm"] = arm
            out.append(res)
            json.dump(dict(identity_max_abs_diff=err, runs=out),
                      open(os.path.join(RES, f"t2_keybias{tag}.json"), "w"), indent=2)
    summarise(out)


def summarise(runs):
    keys = ["val", "val_hard", "ood_book", "ood_brown", "ood_reuters"]
    arms = sorted({r["arm"] for r in runs})
    print(f"\n{'arm':10s}" + "".join(f"{k:>14s}" for k in keys))
    for a in arms:
        f = [r["final"] for r in runs if r["arm"] == a]
        print(f"{a:10s}" + "".join(
            f"{np.mean([x[k] for x in f]):>9.4f}±{np.std([x[k] for x in f]):.3f}"
            for k in keys))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--arms", default="qk,keybias,euclid")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    main(steps=a.steps, arms=tuple(a.arms.split(",")),
         seeds=tuple(int(s) for s in a.seeds.split(",")),
         threads=a.threads, tag=a.tag)
