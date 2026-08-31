"""T5 - curvature diagnostic, then surgery.

M2 measured curvature on the connectome. Here the same measurement is applied to
the *attention masks* from the T3 sweep, and used to predict where long-range
information fails: `gap_recall` places a key-value pair a controlled distance
from its query, so a mask whose negatively curved edges bottleneck long paths
should fail at large gaps specifically while doing fine at short ones.

Then SDRF (stochastic discrete Ricci flow) is applied to the worst mask and the
task is re-run, to see whether the predicted failure is actually recovered.

Caveat carried from the literature: SDRF's own gains are modest and concentrated
on low-homophily data, and it trades over-squashing against over-smoothing.
"""
import json, os, sys
import numpy as np
import torch
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flyattn.model import Config  # noqa: E402
from flyattn import masks as MK, synth, curvature as CV  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
SEQ, VOCAB, MEAN_DEG, GAMMA = 256, 64, 20.0, 3.1


def mask_stats(m: torch.Tensor):
    a = sp.csr_matrix(m.numpy().astype(np.int8))
    a.setdiag(0); a.eliminate_zeros()
    tri = sp.triu(a, 1).tocoo()
    ed = np.stack([tri.row, tri.col], 1)
    c = CV.bfc_edges(a, ed)
    i, j = tri.row, tri.col
    return dict(n_edges=int(len(ed)), bfc_mean=float(c.mean()),
                bfc_min=float(c.min()), bfc_frac_neg=float((c < 0).mean()),
                bfc_q05=float(np.quantile(c, 0.05)),
                mean_span=float(np.abs(i - j).mean()),
                frac_long=float((np.abs(i - j) > SEQ // 4).mean())), a, ed, c


def build(name, rng):
    if name == "dense":
        return None
    if name == "config":
        return MK.configuration_mask(SEQ, MEAN_DEG, GAMMA, rng)
    if name == "window":
        return MK.window_mask(SEQ, int(MEAN_DEG / 2))
    return MK.s1_mask(SEQ, float(name[1:]), MEAN_DEG, GAMMA, rng)


def main(names=("dense", "config", "b1.30", "b4.00", "window"),
         gaps=(16, 200), seeds=(0, 1), steps=1200, threads=1, do_sdrf=True):
    out = {"masks": {}, "runs": []}
    built = {}
    for nm in names:
        m = build(nm, np.random.default_rng(4242))
        built[nm] = m
        if m is not None:
            st, _, _, _ = mask_stats(m)
            out["masks"][nm] = st
            print(f"{nm:8s} {st}", flush=True)

    if do_sdrf:
        worst = min((n for n in names if built[n] is not None),
                    key=lambda n: out["masks"][n]["bfc_q05"])
        print(f"SDRF target (most negatively curved 5th pct): {worst}", flush=True)
        a = sp.csr_matrix(built[worst].numpy().astype(np.int8))
        a.setdiag(0); a.eliminate_zeros()
        a2, log = CV.sdrf(a, n_iter=40, rng=np.random.default_rng(0))
        m2 = torch.from_numpy(np.asarray(a2.todense()) > 0)
        m2.fill_diagonal_(True)
        built[f"{worst}+sdrf"] = m2
        st, _, _, _ = mask_stats(m2)
        out["masks"][f"{worst}+sdrf"] = st
        out["sdrf_log"] = log
        print(f"{worst}+sdrf {st}", flush=True)
        names = tuple(names) + (f"{worst}+sdrf",)

    for nm in names:
        for gap in gaps:
            for sd in seeds:
                cfg = Config(vocab_size=VOCAB, d_model=96, n_layers=3, n_heads=4,
                             seq_len=SEQ, n_experts=4, top_k=2, d_ff=192)
                fn = (lambda b, r, g=gap: synth.gap_recall_batch(b, SEQ, VOCAB, g, r))
                res, _ = synth.train_synth(cfg, fn, steps=steps, seed=sd,
                                           threads=threads, struct_mask=built[nm])
                row = dict(mask=nm, gap=gap, seed=sd, **res["final"])
                out["runs"].append(row)
                print(f"  {nm:12s} gap={gap:<4d} seed={sd} acc={row['acc']:.3f} "
                      f"loss={row['eval_loss']:.4f}", flush=True)
                json.dump(out, open(os.path.join(RES, "t5_curvature.json"), "w"),
                          indent=2)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--names", default="dense,config,b1.30,b4.00,window")
    p.add_argument("--gaps", default="16,200")
    p.add_argument("--seeds", default="0,1")
    a = p.parse_args()
    main(names=tuple(a.names.split(",")), gaps=tuple(int(g) for g in a.gaps.split(",")),
         seeds=tuple(int(s) for s in a.seeds.split(",")), steps=a.steps,
         threads=a.threads)
