"""M3 - rich-club sizing and the broadcaster/integrator budget.

Question asked: how many neurons are strongly in/out asymmetric ("global tokens"
with a restricted read or write), versus how many sit in the balanced rich club?
The claim under test is ~676 broadcasters and ~638 integrators against ~37k
balanced rich-club neurons, i.e. ~1% of the brain asymmetric rather than ~30%.

Every count here is reported as a function of the minimum-degree cutoff, because
the 5x ratio rule is trivially satisfied by low-degree neurons (a 1-in/5-out cell
is a "broadcaster" under the raw rule) and the headline number is entirely a
function of where that cutoff sits.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flyattn.connectome import load_connectome  # noqa: E402
from flyattn.nulls import rewire_directed  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "m3_richclub.json")
RATIO = 5.0


def rich_club_phi(u, v, deg, ks):
    """Undirected rich-club coefficient phi(k) for each k in ks."""
    order = np.argsort(-deg, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(deg))
    du, dv = deg[u], deg[v]
    emin = np.minimum(du, dv)
    out = []
    for k in ks:
        nk = int((deg > k).sum())
        ek = int((emin > k).sum())
        out.append(ek / (nk * (nk - 1) / 2) if nk > 1 else np.nan)
    return np.array(out)


def main(n_null=10):
    c = load_connectome()
    kin, kout = c.in_out_degree()
    ktot = kin + kout
    a = c.undirected_simple()
    deg = np.asarray(a.sum(1)).ravel().astype(np.int64)
    uu, vv = sp_triu(a)

    # --- asymmetry census as a function of the degree cutoff -----------------
    rows = []
    for cut in [0, 5, 10, 20, 30, 50, 100]:
        m = ktot >= cut
        b = m & (kout >= RATIO * np.maximum(kin, 1e-9)) & (kin < kout)
        i = m & (kin >= RATIO * np.maximum(kout, 1e-9)) & (kout < kin)
        # strict version: the 5x rule with a floor on the *large* side too
        rows.append(dict(min_total_degree=cut, n_neurons=int(m.sum()),
                         broadcasters=int(b.sum()), integrators=int(i.sum()),
                         asym_fraction_of_brain=float((b.sum() + i.sum()) / c.n)))

    # --- rich club -----------------------------------------------------------
    ks = np.unique(np.percentile(deg, np.arange(50, 100, 2)).astype(int))
    ks = np.unique(np.concatenate([ks, np.array([50, 75, 100, 150, 200, 300, 500])]))
    phi = rich_club_phi(uu, vv, deg, ks)
    rng = np.random.default_rng(0)
    null = np.zeros((n_null, len(ks)))
    for r in range(n_null):
        t = time.time()
        p, q = rewire_directed(c.pre, c.post, c.n, swaps_per_edge=5.0,
                               rng=np.random.default_rng(100 + r))
        au, av, adeg = undirected_from_directed(p, q, c.n)
        null[r] = rich_club_phi(au, av, adeg, ks)
        print(f"  null {r+1}/{n_null} ({time.time()-t:.0f}s)", flush=True)
    phin = phi / null.mean(0)

    # rich-club membership: neurons above the smallest k where the normalised
    # coefficient is significantly > 1 and stays there
    z = (phi - null.mean(0)) / (null.std(0) + 1e-12)
    sig = (phin > 1.0) & (z > 2)
    k_star = int(ks[np.argmax(sig)]) if sig.any() else None
    rc = int((deg > k_star).sum()) if k_star is not None else 0
    rc_mask = deg > k_star if k_star is not None else np.zeros(c.n, bool)
    b_rc = int((rc_mask & (kout >= RATIO * np.maximum(kin, 1e-9))).sum())
    i_rc = int((rc_mask & (kin >= RATIO * np.maximum(kout, 1e-9))).sum())

    res = dict(
        n_neurons=int(c.n), n_edges_directed=int(c.n_edges),
        syn_threshold=c.syn_threshold,
        asymmetry_census=rows,
        rich_club=dict(k=ks.tolist(), phi=phi.tolist(),
                       phi_null_mean=null.mean(0).tolist(),
                       phi_null_std=null.std(0).tolist(),
                       phi_normalised=phin.tolist(), z=z.tolist(),
                       k_star=k_star, n_rich_club=rc,
                       broadcasters_in_rich_club=b_rc,
                       integrators_in_rich_club=i_rc,
                       rich_club_fraction=float(rc / c.n)),
        n_null=n_null,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "rich_club"}, indent=2))
    print(json.dumps({k: v for k, v in res["rich_club"].items()
                      if k in ("k_star", "n_rich_club", "rich_club_fraction",
                               "broadcasters_in_rich_club",
                               "integrators_in_rich_club")}, indent=2))


def sp_triu(a):
    coo = a.tocoo()
    m = coo.row < coo.col
    return coo.row[m], coo.col[m]


def undirected_from_directed(p, q, n):
    import scipy.sparse as sp
    a = sp.csr_matrix((np.ones(len(p), np.int8), (p, q)), shape=(n, n))
    a = a + a.T
    a.data[:] = 1
    a = a.tocsr(); a.setdiag(0); a.eliminate_zeros()
    deg = np.asarray(a.sum(1)).ravel().astype(np.int64)
    u, v = sp_triu(a)
    return u, v, deg


if __name__ == "__main__":
    main(n_null=int(sys.argv[1]) if len(sys.argv) > 1 else 5)
