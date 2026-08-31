"""Balanced Forman curvature (Topping et al., 2022, "Understanding over-squashing
and bottlenecks on graphs via curvature") and the SDRF rewiring built on it.

For an edge (i, j) of an unweighted graph with degrees d_i, d_j >= 1:

    Ric(i,j) = 2/d_i + 2/d_j - 2
             + 2 * |T(i,j)| / max(d_i,d_j) + |T(i,j)| / min(d_i,d_j)
             + (1/gamma_max) * (|S_i| + |S_j|)

T(i,j) are the triangles at the edge; S_i are the neighbours of i (excluding j
and the triangle nodes) that close a 4-cycle through j without a diagonal;
gamma_max is the largest number of such 4-cycles any single node participates in.
Negative curvature marks the edges that bottleneck message passing.

The exact per-edge cost is O(d_i * d_j), so `bfc_edges` computes curvature for a
supplied subset of edges; a random subset of tens of thousands is more than
enough to compare two edge-curvature distributions.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def bfc_edges(a: sp.csr_matrix, edges: np.ndarray) -> np.ndarray:
    """Balanced Forman curvature for `edges` (an (m,2) array of node pairs)."""
    a = a.tocsr()
    indptr, indices = a.indptr, a.indices
    n = a.shape[0]
    deg = np.diff(indptr).astype(np.float64)
    mark_i = np.zeros(n, np.int8)
    mark_j = np.zeros(n, np.int8)
    out = np.empty(len(edges))

    for e in range(len(edges)):
        i, j = int(edges[e, 0]), int(edges[e, 1])
        ni = indices[indptr[i]:indptr[i + 1]]
        nj = indices[indptr[j]:indptr[j + 1]]
        di, dj = deg[i], deg[j]
        if di <= 1 or dj <= 1:
            out[e] = 0.0
            continue
        mark_i[ni] = 1
        mark_j[nj] = 1
        tri = ni[(mark_j[ni] == 1)]
        tri = tri[tri != j]
        n_tri = len(tri)
        # S_i: neighbours of i that are not j, not neighbours of j
        si = ni[(mark_j[ni] == 0) & (ni != j)]
        sj = nj[(mark_i[nj] == 0) & (nj != i)]
        gamma = 0
        cnt_i = 0
        for k in si:
            nk = indices[indptr[k]:indptr[k + 1]]
            # 4-cycles i-k-w-j with w in S_j (no diagonal k~j, w~i by construction)
            m = np.count_nonzero((mark_j[nk] == 1) & (mark_i[nk] == 0) & (nk != i))
            if m:
                cnt_i += 1
                if m > gamma:
                    gamma = m
        cnt_j = 0
        for w in sj:
            nw = indices[indptr[w]:indptr[w + 1]]
            m = np.count_nonzero((mark_i[nw] == 1) & (mark_j[nw] == 0) & (nw != j))
            if m:
                cnt_j += 1
                if m > gamma:
                    gamma = m
        val = (2.0 / di + 2.0 / dj - 2.0
               + 2.0 * n_tri / max(di, dj) + n_tri / min(di, dj))
        if gamma > 0:
            val += (cnt_i + cnt_j) / gamma
        out[e] = val
        mark_i[ni] = 0
        mark_j[nj] = 0
    return out


def sample_edges(a: sp.csr_matrix, m: int, rng) -> np.ndarray:
    coo = sp.triu(a, 1).tocoo()
    idx = rng.choice(coo.nnz, size=min(m, coo.nnz), replace=False)
    return np.stack([coo.row[idx], coo.col[idx]], 1)


def sdrf(a: sp.csr_matrix, n_iter: int, rng, tau: float = 20.0,
         max_removal_curv: float = 0.0, cand: int = 200):
    """Stochastic discrete Ricci flow (Topping et al.): repeatedly add an edge
    that most improves the curvature of the currently most negative edge, and
    optionally remove the most positively curved edge.

    Returns the rewired adjacency and a log of the minimum curvature.
    """
    a = a.tolil(copy=True)
    log = []
    for _ in range(n_iter):
        acsr = a.tocsr()
        edges = sample_edges(acsr, 4000, rng)
        curv = bfc_edges(acsr, edges)
        worst = edges[int(np.argmin(curv))]
        i, j = int(worst[0]), int(worst[1])
        log.append(float(curv.min()))
        ni = acsr.indices[acsr.indptr[i]:acsr.indptr[i + 1]]
        nj = acsr.indices[acsr.indptr[j]:acsr.indptr[j + 1]]
        # candidate improvements: connect a neighbour of i to a neighbour of j
        if len(ni) == 0 or len(nj) == 0:
            continue
        ks = rng.choice(ni, size=min(len(ni), int(np.sqrt(cand)) + 1), replace=False)
        ws = rng.choice(nj, size=min(len(nj), int(np.sqrt(cand)) + 1), replace=False)
        best, best_gain = None, -np.inf
        base = curv.min()
        for k in ks:
            for w in ws:
                if k == w or acsr[int(k), int(w)] != 0:
                    continue
                a[int(k), int(w)] = 1
                a[int(w), int(k)] = 1
                g = bfc_edges(a.tocsr(), np.array([[i, j]]))[0] - base
                a[int(k), int(w)] = 0
                a[int(w), int(k)] = 0
                if g > best_gain:
                    best_gain, best = g, (int(k), int(w))
        if best is not None and best_gain > 0:
            a[best[0], best[1]] = 1
            a[best[1], best[0]] = 1
            if max_removal_curv is not None:
                curv2 = bfc_edges(a.tocsr(), edges)
                hi = edges[int(np.argmax(curv2))]
                if curv2.max() > max_removal_curv:
                    a[int(hi[0]), int(hi[1])] = 0
                    a[int(hi[1]), int(hi[0])] = 0
    return a.tocsr(), log
