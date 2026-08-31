"""T1b - the same shuffling question, but with fine-tuning after pruning.

T1 prunes and evaluates without retraining, which is the sensitive regime but
also the one where the network is simply destroyed below ~5% density: every arm
sits at the unigram floor, so "shuffling makes no difference" is true but vacuous.
Fine-tuning under a fixed mask is what lets the question be asked at all at low
density - the mask stays where it was placed, only the surviving weights move.

Arms are the same three: the trained mask with its weights, the trained mask with
weights permuted inside it, and a mask shuffled within the layer.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flyattn.model import Config, Transformer  # noqa: E402
from flyattn import textdata, train as T  # noqa: E402
from t1_pruning import vo_params, build_masks, apply_arm  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
DENSITIES = [0.3, 0.1, 0.05, 0.02, 0.01]


def finetune(model, masks, ev, hard, steps, train_data, lr=3e-4, batch_size=32,
             seq=128, seed=0):
    """Fine-tune with the mask enforced after every optimiser step."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0,
                            betas=(0.9, 0.95))
    rng = np.random.default_rng(seed + 5)
    gen = textdata.batches(train_data, batch_size, seq, rng)
    for step in range(steps):
        x, y = next(gen)
        _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for n, p in vo_params(model):
                p.mul_(masks[n])
    m1 = T.evaluate(model, ev["val"], hard)
    m2 = T.evaluate(model, ev["ood_brown"])
    return dict(val=m1["loss"], val_hard=m1["hard_loss"], ood_brown=m2["loss"])


def main(steps=400, threads=1, reps=2, shard=0, n_shards=1):
    torch.set_num_threads(threads)
    corp = textdata.load_corpora()
    lp = textdata.bigram_logprobs(corp["train"])
    seq = 128
    ev = {k: T.make_eval_batches(corp[k], 32, seq, 20, seed=5)
          for k in ("val", "ood_brown")}
    hard = T.hard_mask_for(ev["val"], lp)
    cfg = Config(d_model=128, n_layers=4, n_heads=4, seq_len=seq,
                 n_experts=4, top_k=2, d_ff=256)

    out_path = os.path.join(RES, f"t1b_finetune_shard{shard}.json")
    rows = json.load(open(out_path))["rows"] if os.path.exists(out_path) else []
    seen = {(r["density"], r["arm"], r["rep"]) for r in rows}

    jobs = [(d, a, r) for d in DENSITIES
            for a in ("trained", "shuffled_w", "shuffled_mask")
            for r in range(1 if a == "trained" else reps)]
    jobs = [j for i, j in enumerate(jobs) if i % n_shards == shard]

    for d, arm, rep in jobs:
        if (d, arm, rep) in seen:
            continue
        model = Transformer(cfg)
        model.load_state_dict(torch.load(os.path.join(RES, "t1_base_model.pt")))
        base_state = {n: p.detach().clone() for n, p in vo_params(model)}
        masks = build_masks(model, d, np.random.default_rng(12345))
        apply_arm(model, base_state, masks, arm, np.random.default_rng(1000 * rep + 7))
        if arm == "shuffled_mask":
            # the mask that must now be enforced is the shuffled one
            masks = {n: (p.detach() != 0) for n, p in vo_params(model)}
        pre = T.evaluate(model, ev["val"], hard)
        post = finetune(model, masks, ev, hard, steps, corp["train"], seed=rep)
        row = dict(density=d, arm=arm, rep=rep, pre_val=pre["loss"],
                   pre_hard=pre["hard_loss"], **post)
        rows.append(row)
        print(f"d={d:<6g} {arm:14s} rep{rep}  pre {pre['loss']:.4f} -> "
              f"post {post['val']:.4f} (hard {post['val_hard']:.4f}, "
              f"brown {post['ood_brown']:.4f})", flush=True)
        json.dump(dict(steps=steps, rows=rows), open(out_path, "w"), indent=2)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    a = p.parse_args()
    main(steps=a.steps, threads=a.threads, shard=a.shard, n_shards=a.n_shards)
