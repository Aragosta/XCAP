"""T2c - is hyperbolic geometry a *deep-layer* phenomenon?

T2b gave every head a free radial scale s controlling how far up the hyperboloid
its activations sit, i.e. how much curvature it actually experiences. The
whole-model arms converged to nearly flat space (s ~ 1.02-1.26 for the better
arm), but s rose monotonically with depth in all four runs, both arms, both
seeds. That is the one positive signal in the hyperbolic direction, and it has a
cheap direct test: put the hyperbolic attention only where the model asked for
curvature and see whether it is better than putting it everywhere, or nowhere.

Arms (4-layer model, `hip` = hyperbolic_ip = score -g*cosh(d_H)):
  qk / qk / qk / qk      baseline                        (from T2)
  hip / hip / hip / hip  hyperbolic everywhere           (from T2b)
  qk / qk / hip / hip    deep only  - the prediction
  hip / hip / qk / qk    shallow only - the control that makes "deep" mean
                         something rather than "some layers"
  qk / qk / qk / hip     last layer only

If depth is what matters, deep-only >= everywhere > shallow-only. If the ordering
is flat, the s gradient was a free parameter drifting, not a signal.
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flyattn.model import Config  # noqa: E402
from flyattn import textdata, train as T  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
H = "hyperbolic_ip"
ARMS = {
    "deep2":    ("qk", "qk", H, H),
    "shallow2": (H, H, "qk", "qk"),
    "last1":    ("qk", "qk", "qk", H),
}


def main(arms, seeds=(0, 1), steps=3000, batch_size=32, threads=1, tag=""):
    corp = textdata.load_corpora()
    lp = textdata.bigram_logprobs(corp["train"])
    seq = 128
    ev = {k: T.make_eval_batches(corp[k], batch_size, seq, 20, seed=5)
          for k in ("val", "ood_book", "ood_brown", "ood_reuters")}
    hard = T.hard_mask_for(ev["val"], lp)

    out_path = os.path.join(RES, f"t2c_depth{tag}.json")
    out = json.load(open(out_path))["runs"] if os.path.exists(out_path) else []
    seen = {(r["arm"], r["seed"]) for r in out}

    for arm in arms:
        for sd in seeds:
            if (arm, sd) in seen:
                continue
            cfg = Config(attn_kind="qk", layer_kinds=ARMS[arm], d_model=128,
                         n_layers=4, n_heads=4, seq_len=seq, n_experts=4,
                         top_k=2, d_ff=256)
            print(f"== arm={arm} {ARMS[arm]} seed={sd}", flush=True)
            res, model = T.train_run(
                cfg, corp["train"], ev, steps=steps, batch_size=batch_size,
                seed=sd, hard=hard, threads=threads, eval_every=500,
                progress=lambda r: print(
                    f"   step {r['step']} val {r['val']:.4f} "
                    f"hard {r.get('val_hard', float('nan')):.4f} "
                    f"({r['elapsed']:.0f}s)", flush=True))
            res["arm"] = arm
            res["layer_kinds"] = list(ARMS[arm])
            res["learned_scale"] = [
                (b.attn.log_scale.exp().tolist()
                 if hasattr(b.attn, "log_scale") else None)
                for b in model.blocks]
            out.append(res)
            json.dump(dict(arms={k: list(v) for k, v in ARMS.items()}, runs=out),
                      open(out_path, "w"), indent=2)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default="deep2,shallow2,last1")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    main(tuple(a.arms.split(",")), tuple(int(s) for s in a.seeds.split(",")),
         steps=a.steps, threads=a.threads, tag=a.tag)
