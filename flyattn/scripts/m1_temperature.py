"""M1 - fit FlyWire's geometric temperature beta (and the degree exponent gamma).

Procedure
---------
1. Build the >=5-synapse undirected connectome, take its giant component.
2. Draw random node-induced subsamples. The S1/H2 ensemble is closed under
   random node sampling (hidden degrees rescale by the sampling fraction, beta
   is unchanged), so beta fitted on a subsample estimates beta of the whole
   graph. Fitting at several scales is the empirical check on that assumption -
   if the estimates drift with N, the geometric description is not holding.
3. For each subsample: calibrate hidden degrees to the observed degree sequence
   at the candidate beta, sample an S1 graph, and bisect beta until the ensemble
   mean local clustering matches the observed one.
4. Locate the fitted beta relative to beta_c = 1 (below it clustering vanishes in
   the thermodynamic limit) and beta = 2D = 2 (above it system-spanning
   long-range links disappear).

The clustering statistic is the mean local clustering coefficient over nodes of
degree >= 2, computed identically on data and model.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flyattn.connectome import load_connectome, giant_component  # noqa: E402
from flyattn import s1h2  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "m1_temperature.json")


def induced(a, nodes):
    sub = a[nodes][:, nodes].tocsr()
    sub.setdiag(0); sub.eliminate_zeros()
    return sub


def main(scales=(8000, 16000, 32000), reps=2, calib_iters=8):
    t0 = time.time()
    c = load_connectome()
    a = c.undirected_simple().astype(np.float32)
    gc = giant_component(a)
    a = induced(a, gc)
    deg_full = np.asarray(a.sum(1)).ravel()
    c_full = s1h2.mean_clustering(a)
    gam_full = s1h2.fit_gamma(deg_full)
    print(f"[{time.time()-t0:.0f}s] giant component N={a.shape[0]} "
          f"<k>={deg_full.mean():.2f} mean_clustering={c_full:.4f} "
          f"gamma={gam_full['gamma']:.3f} (k_min={gam_full['k_min']})", flush=True)

    fits = []
    for ns in scales:
        for r in range(reps):
            rng = np.random.default_rng(1000 * r + ns)
            nodes = rng.choice(a.shape[0], size=ns, replace=False)
            sub = induced(a, np.sort(nodes))
            k = np.asarray(sub.sum(1)).ravel()
            keep = np.flatnonzero(k > 0)
            sub = induced(sub, keep)
            k = np.asarray(sub.sum(1)).ravel()
            c_sub = s1h2.mean_clustering(sub)
            gam = s1h2.fit_gamma(k)
            print(f"[{time.time()-t0:.0f}s] scale={ns} rep={r}: n={len(k)} "
                  f"<k>={k.mean():.2f} c={c_sub:.4f} gamma={gam['gamma']:.3f}",
                  flush=True)
            beta, trace, status = s1h2.fit_beta(
                k, c_sub, rng, lo=1.05, hi=8.0,
                tol=max(0.002, 0.02 * c_sub), max_iter=10)
            print(f"[{time.time()-t0:.0f}s]   -> beta={beta:.3f} ({status})", flush=True)
            fits.append(dict(scale=ns, rep=r, n=int(len(k)), mean_degree=float(k.mean()),
                             clustering=float(c_sub), gamma=gam, beta=float(beta),
                             status=status,
                             trace=[[float(x), float(y)] for x, y in trace]))
            json.dump(dict(full=dict(n=int(a.shape[0]), mean_degree=float(deg_full.mean()),
                                     clustering=float(c_full), gamma=gam_full),
                           fits=fits, beta_c=1.0, beta_2D=2.0),
                      open(OUT, "w"), indent=2)
    b = np.array([f["beta"] for f in fits])
    print(f"\nbeta estimates: {np.round(b,3).tolist()}  mean={b.mean():.3f} sd={b.std():.3f}")
    print(f"phase: {'hot/non-geometric' if b.mean()<1 else ('quasi-geometric (1<beta<2)' if b.mean()<2 else 'cold / over-clustered (beta>2D)')}")


if __name__ == "__main__":
    main()
