"""A2 - the diameter phase transition of long-range percolation.

This is the mathematics the span idea actually rests on. For long-range
percolation on Z^d, where P(i ~ j) ~ |i-j|^-s, the graph distance between two
nodes at Euclidean distance r has three regimes separated by s = d and s = 2d
(Benjamini-Berger; Coppersmith-Gamarnik-Sviridenko; Biskup):

    s < d          bounded diameter
    s = d          Theta(log N / log log N)
    d < s < 2d     polylogarithmic: (log r)^Delta,  Delta = 1/log2(2d/s)
    s = 2d         N^theta for some theta in (0,1)   (sublinear polynomial)
    s > 2d         Theta(N)                          (linear)

For a causal attention mask on a sequence, d = 1 and s is exactly the span
exponent beta. The graph distance is the number of *layers* needed for two
tokens to be able to influence one another. So the theory says:

    beta < 2   depth requirement grows polylogarithmically with context length
    beta = 2   sublinear polynomial
    beta > 2   depth must grow LINEARLY with context length

which makes beta = 2D the boundary beyond which a fixed-depth transformer
cannot connect its own context, however long you make it.

This script builds the graphs at n = 2^7 .. 2^14 at fixed mean degree and fits
the observed scaling, which is what distinguishes the three regimes. Distances
are linear (no wrap-around), unlike the S1 masks used in T3 - the circle's
wrap-around bridge is precisely what would destroy the phase transition.
"""
import json, os, sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, shortest_path

RES = os.path.join(os.path.dirname(__file__), "..", "results")
MEAN_DEGREE = 20.0
SIZES = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)


def solve_c(beta, n, kbar):
    """Scale so that mean degree = kbar under p(s) = min(1, c s^-beta)."""
    s = np.arange(1, n)
    w = (n - s) / n                      # fraction of nodes having a partner at s
    lo, hi = 1e-9, 1e9
    for _ in range(200):
        c = np.sqrt(lo * hi)
        k = 2.0 * np.sum(w * np.minimum(1.0, c * s ** (-beta)))
        if k < kbar:
            lo = c
        else:
            hi = c
    return np.sqrt(lo * hi)


def build(beta, n, rng, kbar=MEAN_DEGREE):
    """Long-range percolation on a path of n nodes, plus the nearest-neighbour
    bonds that keep the graph connected (the local band every mask has)."""
    c = solve_c(beta, n, kbar)
    rows, cols = [int_ := np.arange(n - 1)], [int_ + 1]   # local band
    for s in range(2, n):
        p = min(1.0, c * s ** (-beta))
        m = n - s
        cnt = rng.binomial(m, p)
        if cnt:
            i = rng.choice(m, size=cnt, replace=False)
            rows.append(i)
            cols.append(i + s)
    r = np.concatenate(rows)
    cl = np.concatenate(cols)
    a = sp.csr_matrix((np.ones(len(r), np.int8), (r, cl)), shape=(n, n))
    a = a + a.T
    a.data[:] = 1
    return a.tocsr()


def distances(a, n_src=12, rng=None):
    """Mean and max graph distance from a few sources (BFS)."""
    rng = rng or np.random.default_rng(0)
    n = a.shape[0]
    src = rng.choice(n, size=min(n_src, n), replace=False)
    d = shortest_path(a, method="D", unweighted=True, indices=src)
    d = d[np.isfinite(d)]
    return float(d.mean()), float(np.percentile(d, 99)), float(d.max())


def main():
    out = {}
    betas = (1.0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 6.0)
    print(f"{'beta':>5s}" + "".join(f"{n:>8d}" for n in SIZES) + "   fit")
    for beta in betas:
        means, maxes = [], []
        for n in SIZES:
            a = build(beta, n, np.random.default_rng(1234))
            mu, p99, mx = distances(a, rng=np.random.default_rng(7))
            means.append(mu); maxes.append(mx)
        L = np.array(SIZES, float)
        mu = np.array(means)
        # candidate laws: mean distance ~ n^theta   vs   ~ (log n)^Delta
        theta = np.polyfit(np.log(L), np.log(mu), 1)[0]
        delta = np.polyfit(np.log(np.log2(L)), np.log(mu), 1)[0]
        r_pow = np.corrcoef(np.log(L), np.log(mu))[0, 1] ** 2
        r_log = np.corrcoef(np.log(np.log2(L)), np.log(mu))[0, 1] ** 2
        delta_pred = (1.0 / np.log2(2.0 / beta)) if 1 < beta < 2 else np.nan
        out[str(beta)] = dict(sizes=list(SIZES), mean_dist=means, max_dist=maxes,
                              theta=float(theta), delta=float(delta),
                              r2_power=float(r_pow), r2_polylog=float(r_log),
                              delta_predicted=(None if not np.isfinite(delta_pred)
                                               else float(delta_pred)))
        best = "n^theta" if r_pow > r_log else "(log n)^Delta"
        print(f"{beta:5.2f}" + "".join(f"{m:8.2f}" for m in means)
              + f"   theta={theta:.3f}(R2={r_pow:.3f})  Delta={delta:.2f}"
              f"(R2={r_log:.3f})  best={best}"
              + (f"  Delta_pred={delta_pred:.2f}" if np.isfinite(delta_pred) else ""))
    json.dump(out, open(os.path.join(RES, "a2_diameter_scaling.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
