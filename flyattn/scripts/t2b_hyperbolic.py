"""T2b - is the Lorentzian inner product a genuinely different similarity?

T2 killed Euclidean-distance attention by showing it is standard attention plus
a per-key bias. The stated escape hatch was hyperbolic space, where the
indefinite signature is supposed to make the similarity genuinely different.
That claim splits in two, and only one half survives inspection:

  lorentz_ip     <q,k>_L on unconstrained vectors. J = diag(-1,1,...,1) is a
                 fixed invertible map, so softmax(<q,k>_L) = softmax((Jq).k) and
                 W_Q absorbs it: the same function class as standard attention.
                 The signature on its own buys nothing. Checked by construction
                 in `equivalence_check`.
  hyperbolic     q, k lifted to the hyperboloid and scored by -g*d_H^2. Here
                 -<q,k>_L = sqrt(1+||sq||^2) sqrt(1+||sk||^2) - sq.sk, in which
                 the query norm *multiplies* the key norm. There is no additive
                 decomposition, so T2's argument does not apply.
  hyperbolic_ip  the same lift scored by g*<q,k>_L = -g*cosh(d_H), the form used
                 in most hyperbolic-attention work. Also not decomposable.

So the constraint, not the signature, is what makes hyperbolic attention a
different function. Two arms test whether that difference is worth anything.

`equivalence_check` demonstrates the lorentz_ip claim by construction: flip the
sign of the first row of W_Q and the two models produce identical logits.
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flyattn.model import Config, Transformer  # noqa: E402
from flyattn import textdata, train as T  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def equivalence_check():
    """lorentz_ip with W_Q -> J W_Q reproduces qk exactly."""
    torch.manual_seed(0)
    # RoPE does not commute with J, so the algebraic claim is stated and checked
    # on the un-rotated model; with RoPE the two differ by a rotation artefact.
    cfg_a = Config(d_model=64, n_layers=2, n_heads=4, seq_len=32, attn_kind="qk",
                   use_rope=False)
    cfg_b = Config(d_model=64, n_layers=2, n_heads=4, seq_len=32,
                   attn_kind="lorentz_ip", use_rope=False)
    a, b = Transformer(cfg_a), Transformer(cfg_b)
    b.load_state_dict(a.state_dict())
    D = cfg_a.d_head
    with torch.no_grad():
        for blk in b.blocks:
            w = blk.attn.q.weight.view(cfg_a.n_heads, D, -1)
            w[:, 0, :] *= -1.0            # J acts on the first coordinate of each head
    x = torch.randint(0, 256, (4, 32))
    la, _ = a(x)
    lb, _ = b(x)
    return float((la - lb).abs().max().item())


def main(steps=3000, seeds=(0, 1), arms=("hyperbolic", "hyperbolic_ip"),
         batch_size=32, threads=1, tag=""):
    err = equivalence_check()
    print(f"lorentz_ip vs qk after W_Q -> J W_Q: max |logit difference| = "
          f"{err:.3e}", flush=True)

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
            res, model = T.train_run(
                cfg, corp["train"], ev, steps=steps, batch_size=batch_size,
                seed=sd, hard=hard, threads=threads, eval_every=250,
                progress=lambda r: print(
                    f"   step {r['step']} val {r['val']:.4f} "
                    f"hard {r.get('val_hard', float('nan')):.4f} "
                    f"brown {r['ood_brown']:.4f} ({r['elapsed']:.0f}s)", flush=True))
            res["arm"] = arm
            if arm.startswith("hyperbolic"):
                res["learned_scale"] = [b.attn.log_scale.exp().tolist()
                                        for b in model.blocks]
                res["learned_gamma"] = [b.attn.log_gamma.exp().tolist()
                                        for b in model.blocks]
            out.append(res)
            json.dump(dict(equivalence_max_abs_diff=err, runs=out),
                      open(os.path.join(RES, f"t2b_hyperbolic{tag}.json"), "w"),
                      indent=2)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--arms", default="hyperbolic,hyperbolic_ip")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    main(steps=a.steps, arms=tuple(a.arms.split(",")),
         seeds=tuple(int(s) for s in a.seeds.split(",")),
         threads=a.threads, tag=a.tag)
