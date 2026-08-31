"""Which geometry is softmax attention actually closer to?

Softmax attention scores exp(q.k) are an inner product in an infinite-dimensional
feature space -- a Koopman/kernel lift. Linear attention replaces that lift with
an explicit finite feature map (same mechanism, truncated). Hyperbolic attention
replaces the inner product with a negative-curvature distance (a different
mechanism entirely).

This swaps each surrogate into a *trained* softmax model and measures the damage
to validation loss. No retraining, so all surrogates degrade; the question is
which one degrades less, i.e. which geometry the learned attention was already
closest to. Each surrogate gets a temperature fitted on held-out data so it is
compared at its best.
"""
import argparse, json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import Vocab, load_gsm8k, encode_corpus
from model import Config, MoETransformer, MHA

HERE = os.path.dirname(os.path.abspath(__file__))


def attn_softmax(q, k, v, mask, **kw):
    a = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    a = a.masked_fill(mask == 0, float("-inf"))
    return F.softmax(a, -1) @ v


def attn_hyperbolic(q, k, v, mask, beta=1.0, scale=0.7, **kw):
    """Score by negative geodesic distance on the Poincare ball (c = 1).

    q and k are mapped into the ball with expmap0 after a global rescale; the
    curvature radius the vectors span is what `scale` controls.
    """
    def to_ball(x):
        n = x.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        r = (scale * n / n.amax(dim=-2, keepdim=True).clamp(min=1e-9))
        return torch.tanh(r) * x / n

    qb, kb = to_ball(q), to_ball(k)
    sq = torch.cdist(qb.flatten(0, 1), kb.flatten(0, 1)).view(*q.shape[:-1], k.size(-2)) ** 2
    nq = (qb * qb).sum(-1, keepdim=True)
    nk = (kb * kb).sum(-1).unsqueeze(-2)
    den = ((1 - nq) * (1 - nk)).clamp(min=1e-6)
    d = torch.acosh((1 + 2 * sq / den).clamp(min=1 + 1e-7))
    a = (-beta * d).masked_fill(mask == 0, float("-inf"))
    return F.softmax(a, -1) @ v


def attn_dist_kernel(q, k, v, mask, beta=1.0, squared=False, **kw):
    """Softmax over negative FLAT distance -- the control for the hyperbolic kernel.

    Without this, a hyperbolic-kernel win is unattributable: it keeps the softmax
    normalisation that linear attention throws away, so the comparison would be
    measuring softmax-vs-no-softmax rather than curved-vs-flat.
    """
    d = torch.cdist(q.flatten(0, 1), k.flatten(0, 1)).view(*q.shape[:-1], k.size(-2))
    s = d ** 2 if squared else d
    a = (-beta * s).masked_fill(mask == 0, float("-inf"))
    return F.softmax(a, -1) @ v


def attn_softmax_temp(q, k, v, mask, beta=1.0, **kw):
    """Baseline with a free temperature, so tuning alone explains nothing."""
    a = beta * (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    a = a.masked_fill(mask == 0, float("-inf"))
    return F.softmax(a, -1) @ v


def _phi_performer(x, W, scale):
    """Positive random features: E[phi(q).phi(k)] = exp(q.k)  (Performer)."""
    x = x * scale
    p = x @ W.T
    return torch.exp(p - (x * x).sum(-1, keepdim=True) / 2
                     - p.amax(dim=-1, keepdim=True)) / math.sqrt(W.size(0))


def attn_linear_rff(q, k, v, mask, W=None, scale=1.0, **kw):
    """Linear attention via an explicit feature map -- the Koopman truncation."""
    qf, kf = _phi_performer(q, W, scale), _phi_performer(k, W, scale)
    kv = torch.einsum("bhtf,bhtd->bhtfd", kf, v).cumsum(2)
    z = kf.cumsum(2)
    num = torch.einsum("bhtf,bhtfd->bhtd", qf, kv)
    den = torch.einsum("bhtf,bhtf->bht", qf, z).unsqueeze(-1)
    return num / den.clamp(min=1e-6)


def attn_linear_elu(q, k, v, mask, scale=1.0, **kw):
    qf, kf = F.elu(q * scale) + 1, F.elu(k * scale) + 1
    kv = torch.einsum("bhtf,bhtd->bhtfd", kf, v).cumsum(2)
    z = kf.cumsum(2)
    num = torch.einsum("bhtf,bhtfd->bhtd", qf, kv)
    den = torch.einsum("bhtf,bhtf->bht", qf, z).unsqueeze(-1)
    return num / den.clamp(min=1e-6)


def patch(model, fn, **kw):
    """Replace the attention computation, keeping all trained weights."""
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        sh = lambda t: t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        y = fn(sh(q), sh(k), sh(v), self.mask[:, :, :T, :T], **kw)
        return self.drop(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))
    MHA.forward = forward


def evaluate(model, batches):
    tot = 0.0
    with torch.no_grad():
        for x, y in batches:
            tot += model(x, y)[1].item()
    return tot / len(batches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/moe/ckpt_best.pt")
    ap.add_argument("--batches", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(2)

    ck = torch.load(os.path.join(HERE, args.ckpt), map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    model = MoETransformer(cfg); model.load_state_dict(ck["model"]); model.eval()
    vocab = Vocab.load(os.path.join(HERE, "runs", "vocab.json"))
    va = encode_corpus(load_gsm8k(os.path.join(HERE, "data/gsm8k_test.jsonl")), vocab)

    rng = np.random.default_rng(0)
    bs = []
    for _ in range(args.batches):
        ix = rng.integers(0, len(va) - cfg.seq_len - 1, size=8)
        bs.append((torch.from_numpy(np.stack([va[i:i + cfg.seq_len] for i in ix]).astype(np.int64)),
                   torch.from_numpy(np.stack([va[i + 1:i + 1 + cfg.seq_len] for i in ix]).astype(np.int64))))
    # split so each surrogate's free scalar is tuned on one half, scored on the other
    tune, test = bs[:len(bs) // 2], bs[len(bs) // 2:]

    orig = MHA.forward
    patch(model, attn_softmax)
    base = evaluate(model, test)
    print(f"softmax MHA (trained)              val loss {base:.4f}  ppl {math.exp(base):6.2f}")

    res = {"softmax": base}
    dh = cfg.d_model // cfg.n_heads
    trials = {
        "softmax + temperature": (attn_softmax_temp,
                                  [dict(beta=b) for b in (0.5, 0.8, 1.0, 1.25, 2.0)]),
        "euclid dist kernel (flat)": (attn_dist_kernel,
                                      [dict(beta=b, squared=sq) for b in (0.25, 0.5, 1, 2, 4, 8)
                                       for sq in (False, True)]),
        "hyperbolic (curving)": (attn_hyperbolic,
                                 [dict(beta=b, scale=s) for b in (0.5, 1, 2, 4, 8)
                                  for s in (0.3, 0.6, 0.9)]),
        "linear elu (lifting)": (attn_linear_elu,
                                 [dict(scale=s) for s in (0.25, 0.5, 1.0, 2.0)]),
    }
    for D in (64, 256):
        W = torch.randn(D, dh)
        trials[f"linear RFF D={D} (lifting)"] = (
            attn_linear_rff, [dict(W=W, scale=s) for s in (0.25, 0.5, 1.0)])

    for name, (fn, grid) in trials.items():
        best, bk = float("inf"), None
        for kw in grid:
            patch(model, fn, **kw)
            v = evaluate(model, tune)
            if v < best and np.isfinite(v):
                best, bk = v, kw
        patch(model, fn, **bk)
        v = evaluate(model, test)
        res[name] = v
        pk = {k: ("W" if torch.is_tensor(x) else x) for k, x in bk.items()}
        print(f"{name:34s} val loss {v:.4f}  ppl {math.exp(min(v,20)):6.2f}  "
              f"delta {v-base:+.4f}   {pk}")

    MHA.forward = orig
    json.dump(res, open(os.path.join(HERE, "results", "attention_kernel.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
