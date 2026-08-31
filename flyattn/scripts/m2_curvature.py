"""M2 - Balanced Forman curvature of FlyWire against degree-matched rewirings.

Prediction under test: the connectome is more negatively curved than a
degree-preserving rewiring of itself. If it is, connectome-derived attention
masks inherit the over-squashing bottlenecks that negative curvature causes in
any message-passing architecture, which is a reason to expect them to
underperform before a single GPU hour is spent.

Curvature is computed exactly on a random subset of edges (the per-edge cost is
O(d_i d_j), and tens of thousands of edges pin the distribution down far more
tightly than the difference we are looking for).
"""
import json, os, sys, time
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flyattn.connectome import load_connectome, giant_component  # noqa: E402
from flyattn.curvature import bfc_edges, sample_edges  # noqa: E402
from flyattn.nulls import rewire_undirected  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "m2_curvature.json")


def summarise(c):
    return dict(mean=float(c.mean()), sd=float(c.std()), median=float(np.median(c)),
                frac_negative=float((c < 0).mean()),
                q01=float(np.quantile(c, 0.01)), q05=float(np.quantile(c, 0.05)),
                q25=float(np.quantile(c, 0.25)), q75=float(np.quantile(c, 0.75)),
                min=float(c.min()), max=float(c.max()), n=int(len(c)))


def main(n_edges=20000, n_null=3):
    t0 = time.time()
    c = load_connectome()
    a = c.undirected_simple()
    gc = giant_component(a)
    a = a[gc][:, gc].tocsr()
    a.setdiag(0); a.eliminate_zeros()
    print(f"[{time.time()-t0:.0f}s] N={a.shape[0]} E={a.nnz//2}", flush=True)

    rng = np.random.default_rng(0)
    ed = sample_edges(a, n_edges, rng)
    t = time.time()
    curv = bfc_edges(a, ed)
    print(f"[{time.time()-t0:.0f}s] empirical BFC on {len(ed)} edges "
          f"({time.time()-t:.0f}s): mean={curv.mean():.4f} "
          f"frac_neg={(curv<0).mean():.4f}", flush=True)

    triu = sp.triu(a, 1).tocoo()
    nulls = []
    for r in range(n_null):
        t = time.time()
        u, v = rewire_undirected(triu.row, triu.col, a.shape[0], swaps_per_edge=5.0,
                                 rng=np.random.default_rng(10 + r))
        an = sp.csr_matrix((np.ones(len(u), np.int8), (u, v)), shape=a.shape)
        an = (an + an.T).tocsr(); an.data[:] = 1
        an.setdiag(0); an.eliminate_zeros()
        edn = sample_edges(an, n_edges, np.random.default_rng(100 + r))
        cn = bfc_edges(an, edn)
        nulls.append(cn)
        print(f"[{time.time()-t0:.0f}s] null {r+1}/{n_null} ({time.time()-t:.0f}s): "
              f"mean={cn.mean():.4f} frac_neg={(cn<0).mean():.4f}", flush=True)

    nullcat = np.concatenate(nulls)
    nm = np.array([n.mean() for n in nulls])
    z = (curv.mean() - nm.mean()) / (nm.std(ddof=1) + 1e-12)
    res = dict(n_nodes=int(a.shape[0]), n_edges_total=int(a.nnz // 2),
               n_edges_sampled=int(n_edges), n_null=n_null,
               empirical=summarise(curv), null_pooled=summarise(nullcat),
               null_means=nm.tolist(), z_of_mean=float(z),
               delta_mean=float(curv.mean() - nullcat.mean()),
               delta_frac_negative=float((curv < 0).mean() - (nullcat < 0).mean()),
               empirical_hist=np.histogram(curv, bins=60, range=(-2, 1))[0].tolist(),
               null_hist=np.histogram(nullcat, bins=60, range=(-2, 1))[0].tolist(),
               hist_edges=np.histogram_bin_edges(curv, bins=60, range=(-2, 1)).tolist())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if "hist" not in k}, indent=2))


if __name__ == "__main__":
    main(n_edges=int(sys.argv[1]) if len(sys.argv) > 1 else 20000,
         n_null=int(sys.argv[2]) if len(sys.argv) > 2 else 3)
