"""delta-hyperbolicity of a real hierarchy, as a reference point.

A delta_rel value means nothing in isolation. WordNet's noun hypernym graph is
the canonical tree that hyperbolic embeddings were built for (Nickel & Kiela),
so it fixes the low end of the scale; a matched Gaussian fixes the high end.
Both are measured with the same sampled estimator used on the activations.
"""
import os, sys, json
from collections import deque, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geometry as g

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_hypernyms(path):
    """WordNet database format: '@' pointers are hypernyms (is-a parents)."""
    edges = defaultdict(list)
    for line in open(path, encoding="latin-1"):
        if line.startswith("  "):
            continue                       # licence header
        head = line.split("|")[0].split()
        if len(head) < 6:
            continue
        off = head[0]
        n_words = int(head[3], 16)
        i = 4 + 2 * n_words                # skip the word/lex_id pairs
        if i >= len(head):
            continue
        n_ptr = int(head[i]); i += 1
        for _ in range(n_ptr):
            if i + 3 >= len(head):
                break
            sym, tgt = head[i], head[i + 1]
            i += 4
            if sym in ("@", "@i"):
                edges[off].append(tgt)
                edges[tgt].append(off)
    return edges


def sampled_delta(edges, n_pts=700, seed=0):
    rng = np.random.default_rng(seed)
    nodes = list(edges)
    # largest connected component
    seen, best = set(), []
    for s in nodes:
        if s in seen:
            continue
        comp, q = [], deque([s]); seen.add(s)
        while q:
            u = q.popleft(); comp.append(u)
            for v in edges[u]:
                if v not in seen:
                    seen.add(v); q.append(v)
        if len(comp) > len(best):
            best = comp
    sel = list(rng.choice(best, min(n_pts, len(best)), replace=False))
    index = {s: i for i, s in enumerate(sel)}
    D = np.full((len(sel), len(sel)), np.nan)
    for si, s in enumerate(sel):
        dist = {s: 0}; q = deque([s])
        while q:
            u = q.popleft()
            for v in edges[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1; q.append(v)
        for t, ti in index.items():
            D[si, ti] = dist.get(t, np.nan)
    ok = ~np.isnan(D).any(1)
    D = D[np.ix_(ok, ok)]
    return len(best), D


if __name__ == "__main__":
    edges = parse_hypernyms(os.path.join(HERE, "data/wordnet/data.noun"))
    ncomp, D = sampled_delta(edges)
    res = g.delta_hyperbolicity(D, 200000)
    resm = g.delta_max(D)
    print(f"WordNet nouns: {len(edges)} synsets, largest component {ncomp}, "
          f"{D.shape[0]} sampled")
    print("  tree   delta_rel mean %.4f  p95 %.4f  MAX %.4f  diam %.0f"
          % (res["delta_rel_mean"], res["delta_rel_p95"], resm["delta_rel"], res["diam"]))
    # Gaussian at the dimensionality of the model, for the other end of the scale
    for d in (192,):
        G = np.random.default_rng(0).normal(size=(700, d))
        DG = np.linalg.norm(G[:, None] - G[None], axis=-1)
        rg = g.delta_hyperbolicity(DG, 200000)
        rgm = g.delta_max(DG)
        print("  gauss R^%d delta_rel mean %.4f  p95 %.4f  MAX %.4f"
              % (d, rg["delta_rel_mean"], rg["delta_rel_p95"], rgm["delta_rel"]))
    json.dump(dict(wordnet=res, wordnet_max=resm, n_synsets=len(edges)),
              open(os.path.join(HERE, "results", "wordnet.json"), "w"), indent=1)
