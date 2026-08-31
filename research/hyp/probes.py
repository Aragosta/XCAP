"""Run the geometric / Koopman probes on a trained MoE MHA model.

Each probe corresponds to one claim:
  P1  delta-hyperbolicity   is the residual stream tree-like at all?
  P2  radial hypothesis     does reasoning depth live in the radius?
  P3  Koopman-in-depth      how linear is the layer-to-layer map?
  P4  chart comparison      does curving beat lifting?  (the central test)
  P5  energy                does a hyperbolic energy track model confidence?
  P6  MoE routing           does the router partition angle, not radius?
  P7  sibling distance      is cross-branch distance ~ 2r?
"""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geometry as g
from distortion import hyp_advantage
from data import Vocab, load_gsm8k, encode_probe_set, PAD
from model import Config, MoETransformer

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- activations
def collect(ckpt_path, n_examples=400, max_tokens=6000, seed=0, shuffle_text=False):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    model = MoETransformer(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    torch.set_num_threads(2)

    vocab = Vocab.load(os.path.join(HERE, "runs", "vocab.json"))
    texts = load_gsm8k(os.path.join(HERE, "data/gsm8k_test.jsonl"))
    X, D, A, E = encode_probe_set(texts, vocab, cfg.seq_len, limit=n_examples)

    if shuffle_text:
        # Destroys word order but preserves the unigram distribution: any
        # structure that survives this is not structure in the reasoning.
        rng = np.random.default_rng(seed)
        for i in range(len(X)):
            m = X[i] != PAD
            v = X[i][m]; rng.shuffle(v); X[i][m] = v

    streams, experts, logps = [], [], []
    with torch.no_grad():
        for s in range(0, len(X), 32):
            xb = torch.from_numpy(X[s:s + 32])
            logits, _, stream = model(xb, capture=True)
            streams.append([h.numpy() for h in stream])
            lp = torch.log_softmax(logits[:, :-1], -1)
            tgt = xb[:, 1:]
            logps.append(torch.gather(lp, 2, tgt[..., None])[..., 0].numpy())
            if cfg.moe:
                experts.append(np.stack([b.moe.last_top_idx[..., 0].numpy()
                                         for b in model.blocks], 1))
    L = len(streams[0])
    H = [np.concatenate([s[l] for s in streams], 0) for l in range(L)]   # (N,T,d)
    lp = np.concatenate(logps, 0)
    ex = np.concatenate([experts[i] for i in range(len(experts))], 0) if experts else None

    # Keep answer tokens only: the question is not a reasoning trace.
    mask = (A == 1) & (D >= 1)
    mask[:, -1] = False
    idx = np.argwhere(mask)
    rng = np.random.default_rng(seed)
    if len(idx) > max_tokens:
        idx = idx[rng.choice(len(idx), max_tokens, replace=False)]
    b, t = idx[:, 0], idx[:, 1]

    meta = dict(depth=D[b, t].astype(float), pos=t.astype(float), ex=E[b],
                logp=lp[b, t], tok=X[b, t])
    Hs = np.stack([H[l][b, t] for l in range(L)], 0).astype(np.float64)   # (L+1,n,d)
    exs = ex[b, :, t] if ex is not None else None                          # (n,n_layers)
    return Hs, meta, exs, cfg, ck.get("val")


def split(n, frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    k = int(frac * n)
    return p[:k], p[k:]


# ------------------------------------------------------------------ P1  delta
def p1_delta(Hs, n_pts=700, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for l in range(Hs.shape[0]):
        H = Hs[l]
        s = rng.choice(len(H), min(n_pts, len(H)), replace=False)
        A = H[s]
        D = np.linalg.norm(A[:, None] - A[None], axis=-1)
        act = g.delta_hyperbolicity(D, 120000, seed)
        actm = g.delta_max(D)
        G = g.matched_gaussian(A, seed)
        DG = np.linalg.norm(G[:, None] - G[None], axis=-1)
        nul = g.delta_hyperbolicity(DG, 120000, seed)
        nulm = g.delta_max(DG)
        out.append(dict(layer=l, act_mean=act["delta_rel_mean"], act_p95=act["delta_rel_p95"],
                        act_max=actm["delta_rel"], null_max=nulm["delta_rel"],
                        null_mean=nul["delta_rel_mean"], null_p95=nul["delta_rel_p95"],
                        ratio_max=actm["delta_rel"] / max(nulm["delta_rel"], 1e-9),
                        ratio=act["delta_rel_mean"] / max(nul["delta_rel_mean"], 1e-9)))
    return out


def p1b_curvature_advantage(Hs, n_pts=300, k=8, seed=0):
    """Primary structural test: hyperbolic advantage relative to a matched null.

    Calibrated in validate_instrument.py -- a tree-metric cloud scores ~3.9, a
    Gaussian carrying the same covariance scores ~1.0. Only the ratio is
    meaningful; the raw advantage favours any concentrated cloud.
    """
    rng = np.random.default_rng(seed)
    out = []
    for l in range(Hs.shape[0]):
        A = Hs[l][rng.choice(len(Hs[l]), min(n_pts, len(Hs[l])), replace=False)]
        D = np.linalg.norm(A[:, None] - A[None], axis=-1)
        a = hyp_advantage(D, k=k, seed=seed)
        N = g.matched_gaussian(A, seed)
        DN = np.linalg.norm(N[:, None] - N[None], axis=-1)
        b = hyp_advantage(DN, k=k, seed=seed)
        out.append(dict(layer=l, adv_data=a["advantage"], adv_null=b["advantage"],
                        ratio=a["advantage"] / max(b["advantage"], 1e-9),
                        hyp_stress=a["hyp_stress"], euc_stress=a["euc_stress"]))
    return out


# ----------------------------------------------------------------- P2  radial
def partial_corr(a, b, ctrl):
    """corr(a,b) with `ctrl` linearly regressed out of both."""
    C = np.stack([ctrl, np.ones_like(ctrl)], 1)
    ra = a - C @ np.linalg.lstsq(C, a, rcond=None)[0]
    rb = b - C @ np.linalg.lstsq(C, b, rcond=None)[0]
    return float(np.corrcoef(ra, rb)[0, 1])


def p2_radial(Hs, meta):
    out = []
    for l in range(Hs.shape[0]):
        H = Hs[l]
        r = np.linalg.norm(H - H.mean(0), axis=-1)
        out.append(dict(layer=l, mean_norm=float(r.mean()),
                        r_vs_depth=float(np.corrcoef(r, meta["depth"])[0, 1]),
                        r_vs_pos=float(np.corrcoef(r, meta["pos"])[0, 1]),
                        # position and depth are correlated by construction, so
                        # the depth claim only survives if it does here:
                        r_vs_depth_partial=partial_corr(r, meta["depth"], meta["pos"])))
    return out


# ---------------------------------------------------------------- P3  Koopman
def p3_koopman(Hs):
    tr, te = split(Hs.shape[1])
    out, d = [], Hs.shape[2]
    for l in range(Hs.shape[0] - 1):
        X, Y = Hs[l], Hs[l + 1]
        W, P = g.fit_linear(X[tr], Y[tr], X[te], Y[te])
        ev = g.koopman_spectrum(W, d)
        # Residual-stream layers are x + f(x), so A sits near the identity;
        # what matters is the spectrum's spread and how much of the update is
        # linearly predictable at all.
        out.append(dict(layer=l, r2=g.r2(Y[te], P),
                        r2_update=g.r2(Y[te] - X[te], P - X[te]),
                        spec_radius=float(np.abs(ev).max()),
                        spec_mean=float(np.abs(ev).mean()),
                        n_expanding=int((np.abs(ev) > 1.0).sum()),
                        n_dims=int(d)))
    # one global operator for the whole stack (a true Koopman operator)
    Xg = np.concatenate([Hs[l] for l in range(Hs.shape[0] - 1)], 0)
    Yg = np.concatenate([Hs[l + 1] for l in range(Hs.shape[0] - 1)], 0)
    tr2, te2 = split(len(Xg))
    Wg, Pg = g.fit_linear(Xg[tr2], Yg[tr2], Xg[te2], Yg[te2])
    return out, dict(global_r2=g.r2(Yg[te2], Pg),
                     global_r2_update=g.r2(Yg[te2] - Xg[te2], Pg - Xg[te2]),
                     global_spec_radius=float(np.abs(g.koopman_spectrum(Wg, d)).max()))


# ------------------------------------------------------- P4  chart comparison
def p4_charts(Hs, radii, rff_dims=(192, 1024),
              gammas=(0.05, 0.2, 0.5, 1.0, 2.0), seed=0):
    """Curving vs lifting, scored identically: held-out R^2 in ambient space.

    Curvature is NOT the free parameter. Rescaling a fixed cloud to fill the
    ball of curvature c makes the chart identical for every c -- curvature and
    fill fraction are the same knob. The quantity that actually varies the
    geometry is how many curvature radii the cloud spans, so the sweep is over
    R_max = 2*artanh(fill) at c = 1. R_max -> 0 recovers the Euclidean chart
    exactly, so this is a properly nested comparison.

    The RFF (Koopman) route gets its bandwidth tuned on held-out data while the
    hyperbolic chart gets a single fixed form, which biases the comparison
    against the hyperbolic hypothesis on purpose.
    """
    tr, te = split(Hs.shape[1], seed=seed)
    rows = []
    for l in range(Hs.shape[0] - 1):
        X, Y = Hs[l], Hs[l + 1]
        mu = X[tr].mean(0)
        rmax = max(np.linalg.norm(X - mu, axis=-1).max(),
                   np.linalg.norm(Y - mu, axis=-1).max())
        rec = dict(layer=l)

        # A residual block is h + f(h) with ||f|| << ||h||, so R^2 against
        # h_{l+1} is dominated by the identity and every chart scores ~0.99.
        # Scoring the *update* removes that free baseline and is the only
        # version of the question with any discriminating power.
        def score_chart(P):
            return g.r2(Y[te], P), g.r2(Y[te] - X[te], P - X[te])

        _, P0 = g.fit_linear(X[tr], Y[tr], X[te], Y[te])
        rec["euclid"], rec["euclid_d"] = score_chart(P0)

        best = (rec["euclid_d"], 0.0)
        for R in radii:
            fill = np.tanh(R / 2.0)                 # c = 1: d(0,x) = 2 artanh|x|
            s = fill / max(rmax, 1e-12)
            Xb, Yb = (X - mu) * s, (Y - mu) * s
            Zx, Zy = g.logmap0(Xb, 1.0), g.logmap0(Yb, 1.0)
            _, Pz = g.fit_linear(Zx[tr], Zy[tr], Zx[te], Zy[te])
            Pamb = g.expmap0(Pz, 1.0) / s + mu       # back to ambient to score
            v, vd = score_chart(Pamb)
            rec[f"hyp_R={R:g}"] = v
            rec[f"hypd_R={R:g}"] = vd
            if vd > best[0]:
                best = (vd, R)
        rec["hyp_best_d"], rec["hyp_best_R"] = best

        # Koopman route: lift to random Fourier features, stay linear there.
        Xc = (X - mu) / np.linalg.norm(X[tr] - mu, axis=-1).mean()
        for D in rff_dims:
            bv = -np.inf
            for gam in gammas:
                F = g.rff_lift(Xc, D, gam, seed)
                _, Pf = g.fit_linear(F[tr], Y[tr], F[te], Y[te], ridge=1e-4)
                bv = max(bv, score_chart(Pf)[1])
            rec[f"rff{D}_d"] = bv

        # Control: an elementwise warp with no hyperbolic structure.
        sc = np.linalg.norm(X[tr] - mu, axis=-1).mean()
        for a in (1.0, 3.0):
            Wx, Wy = np.tanh(a * Xc) / a, np.tanh(a * (Y - mu) / sc) / a
            _, Pw = g.fit_linear(Wx[tr], Wy[tr], Wx[te], Wy[te])
            Pamb = np.arctanh((a * Pw).clip(-0.999, 0.999)) / a * sc + mu
            rec[f"tanh{a:g}_d"] = score_chart(Pamb)[1]

        rec["hyp_gain_d"] = rec["hyp_best_d"] - rec["euclid_d"]
        rows.append(rec)
    return rows


def p4_multiseed(Hs, radii, seeds=(0, 1, 2)):
    """Repeat the chart comparison over train/test splits.

    The hyperbolic gain is small enough that a single split cannot distinguish
    it from noise, so report the spread across splits alongside the mean.
    """
    runs = [p4_charts(Hs, radii, seed=s) for s in seeds]
    out = []
    for l in range(len(runs[0])):
        keys = [k for k in runs[0][l] if k != "layer"]
        rec = dict(layer=l)
        for k in keys:
            v = np.array([r[l][k] for r in runs], dtype=float)
            rec[k] = float(v.mean())
            rec[k + "_sd"] = float(v.std())
        rec["hyp_gain_sd"] = rec.get("hyp_gain_d_sd", np.nan)
        out.append(rec)
    return out


# ----------------------------------------------------------------- P5  energy
def p5_energy(Hs, meta, c=1.0):
    """Wrapped-normal energy on the ball, including the sinh volume term."""
    out = []
    for l in range(Hs.shape[0]):
        H = Hs[l]
        mu = H.mean(0)
        Xb, _ = g.to_ball(H, mu, c)
        v = g.logmap0(Xb, c)
        var = v.var(0).clip(1e-12)
        quad = 0.5 * ((v ** 2) / var).sum(-1)
        nv = np.linalg.norm(v, axis=-1).clip(1e-9)
        n = H.shape[1]
        # (n-1) log(sinh|v| / |v|) is the hyperbolic volume/entropy term: it is
        # what makes the capacity at radius r grow like e^{(n-1)r}.
        vol = (n - 1) * (np.log(np.sinh(nv.clip(max=30)) / nv))
        E_hyp = quad + vol
        E_euc = quad
        out.append(dict(layer=l,
                        E_hyp_vs_logp=float(np.corrcoef(-E_hyp, meta["logp"])[0, 1]),
                        E_euc_vs_logp=float(np.corrcoef(-E_euc, meta["logp"])[0, 1]),
                        vol_share=float(np.abs(vol).mean() / (np.abs(quad).mean() + 1e-9)),
                        E_vs_depth=float(np.corrcoef(E_hyp, meta["depth"])[0, 1])))
    return out


# ------------------------------------------------------------- P6  MoE router
def logistic_probe(F, y, n_class, steps=400, seed=0):
    """Held-out accuracy of a linear probe -- no sklearn in this environment."""
    torch.manual_seed(seed)
    tr, te = split(len(F), seed=seed)
    Xtr = torch.tensor(F[tr], dtype=torch.float32)
    Xte = torch.tensor(F[te], dtype=torch.float32)
    m, s = Xtr.mean(0), Xtr.std(0).clamp(min=1e-6)
    Xtr, Xte = (Xtr - m) / s, (Xte - m) / s
    ytr = torch.tensor(y[tr]); yte = torch.tensor(y[te])
    W = torch.nn.Linear(F.shape[1], n_class)
    opt = torch.optim.Adam(W.parameters(), lr=0.05, weight_decay=1e-4)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(W(Xtr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (W(Xte).argmax(-1) == yte).float().mean().item()
    base = float(np.bincount(y, minlength=n_class).max() / len(y))
    return acc, base


def p6_routing(Hs, experts, cfg):
    """Router input at layer l is the residual stream entering that block."""
    out = []
    for l in range(cfg.n_layers):
        H = Hs[l]
        y = experts[:, l].astype(np.int64)
        z = H - H.mean(0)
        r = np.linalg.norm(z, axis=-1, keepdims=True)
        u = z / r.clip(1e-9)                       # pure direction
        # Radial basis expansion of r: the probe can now fit ANY function of
        # the radius, with the same input width as the angular probe. Whatever
        # it still cannot predict is genuinely not in the radius.
        ctr = np.quantile(r, np.linspace(0.01, 0.99, 128))[None, :]
        w = np.diff(np.quantile(r, [0.05, 0.95]))[0] / 8 + 1e-9
        rf = np.hstack([np.exp(-((r - ctr) / w) ** 2), r, r ** 2,
                        np.log(r.clip(1e-9))])
        a_ang, base = logistic_probe(u, y, cfg.n_experts)
        a_rad, _ = logistic_probe(rf, y, cfg.n_experts)
        a_full, _ = logistic_probe(z, y, cfg.n_experts)
        out.append(dict(layer=l, majority=base, angle_only=a_ang,
                        radius_only=a_rad, full=a_full,
                        angle_dims=int(u.shape[1]), radius_dims=int(rf.shape[1]),
                        n_used=int(len(np.unique(y)))))
    return out


# ------------------------------------------------------ P7  sibling distances
def p7_siblings(Hs, meta, max_depth=7, seed=0):
    """Distance between same-depth tokens from *different* traces vs depth.

    The hyperbolic picture predicts d ~ 2r: two deep branches must be reached
    through the root, so cross-branch distance grows linearly with depth and
    is roughly twice the radius. A flat space has no such law.
    """
    rng = np.random.default_rng(seed)
    out = []
    for l in range(Hs.shape[0]):
        H = Hs[l]
        mu = H.mean(0)
        rows = []
        for d in range(1, max_depth + 1):
            sel = np.where(meta["depth"] == d)[0]
            if len(sel) < 40:
                continue
            i = rng.choice(sel, 3000)
            j = rng.choice(sel, 3000)
            ok = meta["ex"][i] != meta["ex"][j]      # different traces only
            i, j = i[ok], j[ok]
            r = np.linalg.norm(H[sel] - mu, axis=-1).mean()
            dist = np.linalg.norm(H[i] - H[j], axis=-1).mean()
            rows.append(dict(depth=d, radius=float(r), cross_dist=float(dist),
                             ratio=float(dist / max(r, 1e-9)), n=int(len(i))))
        if rows:
            dd = np.array([x["depth"] for x in rows], float)
            rr = np.array([x["radius"] for x in rows])
            cc = np.array([x["cross_dist"] for x in rows])
            slope_r = float(np.polyfit(dd, rr, 1)[0]) if len(dd) > 1 else np.nan
            slope_d = float(np.polyfit(dd, cc, 1)[0]) if len(dd) > 1 else np.nan
            out.append(dict(layer=l, rows=rows, radius_slope=slope_r,
                            dist_slope=slope_d,
                            slope_ratio=float(slope_d / slope_r) if slope_r else np.nan))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/moe/ckpt_best.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--shuffle-text", action="store_true")
    ap.add_argument("--examples", type=int, default=400)
    ap.add_argument("--tokens", type=int, default=6000)
    args = ap.parse_args()

    Hs, meta, experts, cfg, val = collect(os.path.join(HERE, args.ckpt),
                                          args.examples, args.tokens,
                                          shuffle_text=args.shuffle_text)
    print(f"tokens {Hs.shape[1]}  layers {Hs.shape[0]}  d {Hs.shape[2]}  "
          f"moe={cfg.moe}  val={val}", flush=True)

    res = dict(ckpt=args.ckpt, shuffled=args.shuffle_text, val=val,
               n_tokens=int(Hs.shape[1]), moe=bool(cfg.moe),
               depth_max=float(meta["depth"].max()))
    res["p1_delta"] = p1_delta(Hs);                      print("P1 done", flush=True)
    res["p1b_curvature"] = p1b_curvature_advantage(Hs);  print("P1b done", flush=True)
    res["p2_radial"] = p2_radial(Hs, meta);              print("P2 done", flush=True)
    res["p3_koopman"], res["p3_global"] = p3_koopman(Hs); print("P3 done", flush=True)
    res["p4_charts"] = p4_multiseed(Hs, [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
    print("P4 done", flush=True)
    res["p5_energy"] = p5_energy(Hs, meta);              print("P5 done", flush=True)
    if experts is not None:
        res["p6_routing"] = p6_routing(Hs, experts, cfg); print("P6 done", flush=True)
    res["p7_siblings"] = p7_siblings(Hs, meta);          print("P7 done", flush=True)

    path = args.out or (args.ckpt.replace("/", "_").replace(".pt", "") +
                        ("_shuf" if args.shuffle_text else "") + ".json")
    path = os.path.join(HERE, "results", path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(res, open(path, "w"), indent=1, default=float)
    print("wrote", path)


if __name__ == "__main__":
    main()
