"""M4 - what span distribution does *dense* attention learn on its own?

T3 found that the useful thing about the S1 masks was their span distribution
P(s), with an optimum near the tail exponent beta = 2D = 2. That raises an
obvious question the trained baseline can answer for free: if a dense model is
left to distribute its attention however it likes, does it produce a power-law
span distribution, and with what exponent?

For every head we accumulate the attention mass at each causal span s = i - j
over held-out text, then fit the tail exponent by MLE on the mass-weighted
distribution. Per layer, so the depth trend is visible.
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flyattn.model import Config, Transformer  # noqa: E402
from flyattn import textdata, train as T  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def span_profiles(model, batches, seq):
    """Attention mass per causal span, per layer and head."""
    L, H = model.cfg.n_layers, model.cfg.n_heads
    acc = np.zeros((L, H, seq))
    store = {}

    def hook(li):
        def f(mod, inp, out):
            pass
        return f

    # re-run the attention maths with the same code path, capturing p
    for x, _ in batches:
        B, Tq = x.shape
        cos, sin = model.cos, model.sin
        h = model.emb(x)
        causal = model.causal[:Tq, :Tq]
        for li, blk in enumerate(model.blocks):
            a = blk.attn
            z = blk.n1(h)
            D = a.cfg.d_head
            q = a.q(z).view(B, Tq, H, D).transpose(1, 2)
            k = a.k(z).view(B, Tq, H, D).transpose(1, 2)
            if a.cfg.use_rope:
                from flyattn.model import apply_rope
                q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            att = (q @ k.transpose(-2, -1)) * (D ** -0.5)
            att = att.masked_fill(~causal, float("-inf"))
            p = att.softmax(-1)                      # (B, H, Tq, Tq)
            idx = (torch.arange(Tq)[:, None] - torch.arange(Tq)[None, :])
            for s in range(Tq):
                m = idx == s
                if m.any():
                    acc[li, :, s] += p[:, :, m].sum((0, 2)).detach().numpy()
            h = blk(h, cos, sin, causal)
    return acc


def fit_tail(mass, s_min=4):
    """MLE power-law exponent for the mass-weighted span distribution."""
    s = np.arange(len(mass))
    m = (s >= s_min) & (mass > 0)
    if m.sum() < 8:
        return np.nan
    w = mass[m] / mass[m].sum()
    # weighted Hill estimator: beta = 1 + 1 / E_w[log(s / s_min)]
    return 1.0 + 1.0 / np.sum(w * np.log(s[m] / (s_min - 0.5)))


def main():
    torch.set_num_threads(4)
    corp = textdata.load_corpora()
    seq = 128
    ev = T.make_eval_batches(corp["val"], 16, seq, 8, seed=5)
    cfg = Config(d_model=128, n_layers=4, n_heads=4, seq_len=seq,
                 n_experts=4, top_k=2, d_ff=256)
    model = Transformer(cfg)
    model.load_state_dict(torch.load(os.path.join(RES, "t1_base_model.pt")))
    model.eval()
    with torch.no_grad():
        acc = span_profiles(model, ev, seq)
    acc = acc / acc.sum(-1, keepdims=True)

    out = {"per_head_beta": [], "per_layer": []}
    print(f"{'layer':6s}{'head':5s}{'beta_tail':>10s}{'mass s=0':>10s}"
          f"{'mass s<=4':>11s}{'mass s>32':>11s}{'mean span':>11s}")
    for li in range(cfg.n_layers):
        betas = []
        for hh in range(cfg.n_heads):
            m = acc[li, hh]
            b = fit_tail(m)
            betas.append(b)
            s = np.arange(seq)
            print(f"{li:<6d}{hh:<5d}{b:10.3f}{m[0]:10.3f}{m[:5].sum():11.3f}"
                  f"{m[32:].sum():11.3f}{(m*s).sum():11.2f}")
        out["per_head_beta"].append([float(b) for b in betas])
        out["per_layer"].append(dict(
            layer=li, beta_median=float(np.nanmedian(betas)),
            mass_le4=float(acc[li, :, :5].sum(-1).mean()),
            mass_gt32=float(acc[li, :, 32:].sum(-1).mean()),
            mean_span=float((acc[li] * np.arange(seq)).sum(-1).mean())))
    out["span_mass"] = acc.tolist()
    json.dump(out, open(os.path.join(RES, "m4_learned_span.json"), "w"))
    print("\nper-layer median tail exponent:",
          [round(r["beta_median"], 3) for r in out["per_layer"]])
    print("per-layer mean span:          ",
          [round(r["mean_span"], 2) for r in out["per_layer"]])


if __name__ == "__main__":
    main()
