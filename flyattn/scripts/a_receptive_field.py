"""Test A - the effective receptive field, and the Levy prediction. No training.

Claim: composing L attention layers whose per-hop span distribution has tail
P(s) ~ s^-beta gives an L-fold convolution that is one-sided alpha-stable with
alpha = beta - 1, so the receptive field grows as L^(1/alpha), and beta = 2D = 2
(alpha = 1, Cauchy) is the marginal case where the L-fold convolution is the
same distribution rescaled - a depth-invariant receptive field.

Three things are measured, in this order, because each depends on the last:

1. beta_step - the *empirical* tail exponent of the mask's own one-hop span
   distribution, fitted by log-log CCDF regression away from both the knee and
   the T-truncation. The nominal beta of the generator is not used as the
   prediction input: at T = 256 the realised P(s) is truncated and its true
   exponent differs from the nominal one.
2. alpha_L - the tail exponent of the sum of L iid hops drawn from that same
   P(s), on an unbounded line (no boundary, so no saturation artefact).
   Prediction: alpha_L = beta_step - 1, constant in L.
3. width growth and self-similarity - q90 of the L-fold sum against L, and the
   collapse of CCDFs rescaled by their own median. Prediction: slope
   1/(beta_step - 1), and the collapse residual minimised at alpha = 1.

Separately, `layers_to_cover` walks the *actual* truncated mask and reports how
many hops are needed to reach position 0 from the last position - the practically
useful number, and the one the unbounded analysis deliberately excludes.
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flyattn import masks as MK  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
SEQ, MEAN_DEG, GAMMA = 256, 20.0, 3.1
DEPTHS = (1, 2, 4, 8, 16)
N_MC = 200_000


def step_distribution(mask):
    """P(s) over one causal hop, uniform among allowed keys, averaged over rows."""
    m = (mask & torch.tril(torch.ones_like(mask)).bool()).numpy()
    T = m.shape[0]
    acc = np.zeros(T)
    for i in range(T // 2, T):          # rows with a full history behind them
        allowed = np.flatnonzero(m[i, :i + 1])
        if len(allowed):
            acc[i - allowed] += 1.0 / len(allowed)
    return acc / acc.sum()


def ccdf_exponent(samples, lo_q=0.60, hi_q=0.995, min_pts=8):
    """Tail exponent by least squares on log CCDF vs log s.

    For P(S > s) ~ s^-alpha the slope of log CCDF against log s is -alpha.
    Fitted between two quantiles so neither the body nor the extreme tail
    (where the empirical CCDF is a step function over a handful of points)
    drives the fit.
    """
    x = np.sort(samples[samples > 0].astype(float))
    if len(x) < 200:
        return np.nan
    lo, hi = np.quantile(x, lo_q), np.quantile(x, hi_q)
    if not (hi > lo > 0):
        return np.nan
    grid = np.unique(np.geomspace(lo, hi, 40))
    if len(grid) < min_pts:
        return np.nan
    n = len(x)
    ccdf = np.array([(x > g).sum() / n for g in grid])
    ok = ccdf > 0
    if ok.sum() < min_pts:
        return np.nan
    slope = np.polyfit(np.log(grid[ok]), np.log(ccdf[ok]), 1)[0]
    return float(-slope)


def sample_sum(p, L, n, rng):
    """Sum of L iid hops drawn from pmf p, on an unbounded line."""
    s = np.arange(len(p))
    draws = rng.choice(s, size=(n, L), p=p)
    return draws.sum(1)


def collapse_residual(sets):
    """Mean |CCDF - reference CCDF| after rescaling each depth by its median."""
    grid = np.geomspace(0.05, 6.0, 120)
    ref, resid = None, []
    for v in sets:
        v = v[v > 0].astype(float)
        med = np.median(v)
        if med <= 0:
            continue
        c = np.array([(v / med > g).mean() for g in grid])
        if ref is None:
            ref = c
        else:
            resid.append(float(np.mean(np.abs(c - ref))))
    return float(np.mean(resid)) if resid else np.nan


def layers_to_cover(mask, n_walk=4000, max_hop=64, rng=None):
    """Hops needed to walk from the last position back to position 0."""
    rng = rng or np.random.default_rng(0)
    m = (mask & torch.tril(torch.ones_like(mask)).bool()).numpy()
    T = m.shape[0]
    nb = [np.flatnonzero(m[i, :i + 1]) for i in range(T)]
    out = []
    for _ in range(n_walk):
        pos, h = T - 1, 0
        while pos > 0 and h < max_hop:
            cand = nb[pos]
            cand = cand[cand < pos]
            if not len(cand):
                break
            pos = int(cand[rng.integers(len(cand))])
            h += 1
        out.append(h)
    return float(np.mean(out)), float(np.median(out))


def main():
    out = {}
    names = ("b1.05", "b1.30", "b2.00", "b3.00", "b5.00", "b8.00", "window")
    for name in names:
        rng = np.random.default_rng(4242)
        mask = (MK.window_mask(SEQ, int(MEAN_DEG / 2)) if name == "window"
                else MK.s1_mask(SEQ, float(name[1:]), MEAN_DEG, GAMMA, rng))
        p = step_distribution(mask)
        one_hop = sample_sum(p, 1, N_MC, np.random.default_rng(1))
        beta_step = ccdf_exponent(one_hop, 0.50, 0.98)
        rec = dict(beta_nominal=(None if name == "window" else float(name[1:])),
                   beta_step=beta_step,
                   alpha_predicted=(None if not np.isfinite(beta_step)
                                    else float(beta_step - 1)),
                   depths=list(DEPTHS), alpha=[], q90=[], mean=[])
        sets = []
        for L in DEPTHS:
            v = sample_sum(p, L, N_MC, np.random.default_rng(100 + L))
            sets.append(v)
            rec["alpha"].append(ccdf_exponent(v))
            rec["q90"].append(float(np.quantile(v, 0.90)))
            rec["mean"].append(float(v.mean()))
        Ls = np.log(np.array(DEPTHS, float))
        rec["growth_exponent"] = float(np.polyfit(Ls, np.log(rec["q90"]), 1)[0])
        rec["growth_predicted"] = (None if not rec["alpha_predicted"]
                                   or rec["alpha_predicted"] <= 0
                                   else float(1.0 / rec["alpha_predicted"]))
        rec["collapse_residual"] = collapse_residual(sets)
        rec["layers_to_cover_mean"], rec["layers_to_cover_median"] = \
            layers_to_cover(mask, rng=np.random.default_rng(7))
        out[name] = rec
        print(f"{name}: beta_step={beta_step:.3f} alpha_pred={rec['alpha_predicted']:.3f} "
              f"alpha(L)={[round(a,3) for a in rec['alpha']]}", flush=True)

    json.dump(out, open(os.path.join(RES, "a_receptive_field.json"), "w"), indent=2)
    print(f"\n{'mask':8s}{'beta_nom':>9s}{'beta_step':>10s}{'a_pred':>8s}"
          f"{'a_meas':>8s}{'growth':>8s}{'g_pred':>8s}{'collapse':>10s}{'cover_L':>9s}")
    for k, r in out.items():
        am = np.nanmean(r["alpha"][1:])
        bn = r["beta_nominal"]
        gp = r["growth_predicted"]
        print(f"{k:8s}{'-' if bn is None else f'{bn:9.2f}'}"
              f"{r['beta_step']:10.3f}{r['alpha_predicted']:8.3f}{am:8.3f}"
              f"{r['growth_exponent']:8.3f}"
              f"{'-' if gp is None else f'{gp:8.3f}'}"
              f"{r['collapse_residual']:10.4f}{r['layers_to_cover_median']:9.1f}")


if __name__ == "__main__":
    main()
