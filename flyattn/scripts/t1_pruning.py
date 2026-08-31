"""T1 - the shuffling threshold: at what density does weight *position* start to matter?

Train to convergence, then prune the attention value path (W_V and W_O, the
weights a connectome mask would constrain) by global magnitude, sweeping density
from dense down to 1e-4. At each density, four arms:

  trained        the trained mask, weights left where they are
  shuffled_w     the trained mask, surviving weight values permuted among the
                 surviving positions  (the ESA ablation: does topology alone carry
                 the performance, or does the weight-to-position assignment?)
  shuffled_mask  a random mask with the same per-layer count, weights taken from
                 the trained model at those positions (the Frankle ablation)
  random_both    random mask and permuted weights (full randomisation reference)

The gate: find the density at which `trained` separates from `shuffled_w`. If
that density is far below where a transformer actually operates, weight-position
coupling is not what a connectome mask would be buying you, and the program
stops.

No retraining after pruning (post-training pruning is the sensitive regime;
pruning at init is the insensitive one), but a short fine-tune arm is included
to show how much of the gap is recoverable.
"""
import json, os, sys, copy
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flyattn.model import Config, Transformer  # noqa: E402
from flyattn import textdata, train as T  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
DENSITIES = [1.0, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
             5e-4, 2e-4, 1e-4]


def vo_params(model):
    """(name, tensor) for every W_V / W_O matrix, the connectome-maskable path."""
    out = []
    for li, b in enumerate(model.blocks):
        out.append((f"L{li}.v", b.attn.v.weight))
        out.append((f"L{li}.o", b.attn.o.weight))
    return out


def build_masks(model, density, rng):
    """Global-magnitude mask over all W_V/W_O, plus per-layer counts."""
    ps = vo_params(model)
    flat = torch.cat([p.detach().abs().reshape(-1) for _, p in ps])
    k = max(1, int(round(density * flat.numel())))
    thr = torch.topk(flat, k, largest=True).values[-1]
    masks = {n: (p.detach().abs() >= thr) for n, p in ps}
    # ensure the exact count
    return masks


def apply_arm(model, base_state, masks, arm, rng):
    """Write the pruned weights for one arm into `model` in place."""
    with torch.no_grad():
        for n, p in vo_params(model):
            w0 = base_state[n].clone()
            m = masks[n]
            if arm == "trained":
                p.copy_(w0 * m)
            elif arm == "shuffled_w":
                idx = m.nonzero(as_tuple=True)
                vals = w0[idx]
                perm = torch.from_numpy(rng.permutation(len(vals)))
                new = torch.zeros_like(w0)
                new[idx] = vals[perm]
                p.copy_(new)
            elif arm in ("shuffled_mask", "random_both"):
                cnt = int(m.sum())
                flat = torch.zeros(w0.numel(), dtype=torch.bool)
                if cnt:
                    pos = torch.from_numpy(
                        rng.choice(w0.numel(), size=cnt, replace=False))
                    flat[pos] = True
                m2 = flat.view_as(w0)
                if arm == "shuffled_mask":
                    p.copy_(w0 * m2)
                else:
                    vals = w0[m.nonzero(as_tuple=True)]
                    perm = torch.from_numpy(rng.permutation(len(vals)))
                    new = torch.zeros_like(w0)
                    new[m2] = vals[perm]
                    p.copy_(new)
            else:
                raise ValueError(arm)


def main(steps=6000, batch_size=32, threads=2, n_shuffle=5, seed=0):
    torch.set_num_threads(threads)
    corp = textdata.load_corpora()
    lp = textdata.bigram_logprobs(corp["train"])
    seq = 128
    ev = {k: T.make_eval_batches(corp[k], batch_size, seq, 20, seed=5)
          for k in ("val", "ood_book", "ood_brown", "ood_reuters")}
    hard = T.hard_mask_for(ev["val"], lp)

    ckpt = os.path.join(RES, "t1_base_model.pt")
    cfg = Config(d_model=128, n_layers=4, n_heads=4, seq_len=seq,
                 n_experts=4, top_k=2, d_ff=256)
    if os.path.exists(ckpt):
        model = Transformer(cfg)
        model.load_state_dict(torch.load(ckpt))
        base_run = json.load(open(os.path.join(RES, "t1_base_run.json")))
    else:
        print("training base model...", flush=True)
        base_run, model = T.train_run(
            cfg, corp["train"], ev, steps=steps, batch_size=batch_size, seed=seed,
            hard=hard, threads=threads, eval_every=500,
            progress=lambda r: print(f"   step {r['step']} val {r['val']:.4f} "
                                     f"({r['elapsed']:.0f}s)", flush=True))
        torch.save(model.state_dict(), ckpt)
        json.dump(base_run, open(os.path.join(RES, "t1_base_run.json"), "w"))
    base_state = {n: p.detach().clone() for n, p in vo_params(model)}
    n_vo = sum(p.numel() for _, p in vo_params(model))
    print(f"W_V/W_O parameter count: {n_vo}", flush=True)

    dense = T.evaluate(model, ev["val"], hard)
    print(f"dense val loss {dense['loss']:.4f} hard {dense['hard_loss']:.4f}", flush=True)

    rows = []
    for d in DENSITIES:
        rng = np.random.default_rng(12345)
        masks = build_masks(model, d, rng)
        kept = int(sum(int(m.sum()) for m in masks.values()))
        rec = {"density": d, "kept_weights": kept}
        for arm in ("trained", "shuffled_w", "shuffled_mask", "random_both"):
            reps = 1 if arm == "trained" else n_shuffle
            vals = {k: [] for k in ("val", "val_hard", "ood_brown", "ood_reuters")}
            for r in range(reps):
                apply_arm(model, base_state, masks, arm,
                          np.random.default_rng(1000 * r + 7))
                m1 = T.evaluate(model, ev["val"], hard)
                m2 = T.evaluate(model, ev["ood_brown"])
                m3 = T.evaluate(model, ev["ood_reuters"])
                vals["val"].append(m1["loss"]); vals["val_hard"].append(m1["hard_loss"])
                vals["ood_brown"].append(m2["loss"]); vals["ood_reuters"].append(m3["loss"])
            rec[arm] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in vals.items()}
        # restore
        with torch.no_grad():
            for n, p in vo_params(model):
                p.copy_(base_state[n])
        rows.append(rec)
        print(f"density {d:<8g} kept {kept:>7d}  "
              + "  ".join(f"{a}={rec[a]['val'][0]:.4f}" for a in
                          ("trained", "shuffled_w", "shuffled_mask", "random_both")),
              flush=True)
        json.dump(dict(dense=dense, n_vo_params=n_vo, rows=rows,
                       base_final=base_run["final"]),
                  open(os.path.join(RES, "t1_pruning.json"), "w"), indent=2)

    # where does position start to matter?
    thr = None
    for r in rows:
        gap = r["shuffled_w"]["val"][0] - r["trained"]["val"][0]
        sd = r["shuffled_w"]["val"][1] + 1e-9
        if gap > 3 * sd and gap > 0.01:
            thr = r["density"]
    print(f"\nlowest density at which trained beats shuffled_w by >3sd: {thr}")


if __name__ == "__main__":
    main()
