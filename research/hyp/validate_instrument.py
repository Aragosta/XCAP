"""Calibrate the structural test before trusting it on activations.

Naive versions fail: delta_rel does not separate WordNet from a Gaussian, and
raw 'hyperbolic embeds better' favours a Gaussian *more* than a tree, because
negative curvature helps any cloud whose distances are too spread for a flat
low-dimensional fit. The fix is to score each cloud against a null with its own
mean and covariance, so only structure beyond second order can register.
"""
import sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geometry as g
from distortion import hyp_advantage

torch.set_num_threads(2)


def tree_cloud(depth=8, branch=2, d=192, seed=0, noise=0.02):
    """Points in R^d whose Euclidean distances approximate a tree metric."""
    rng = np.random.default_rng(seed)
    parent, nodes = {0: -1}, [0]
    frontier, nxt = [0], 1
    for _ in range(depth):
        new = []
        for p in frontier:
            for _b in range(branch):
                parent[nxt] = p; nodes.append(nxt); new.append(nxt); nxt += 1
        frontier = new
        if len(nodes) > 400:
            break
    n = len(nodes)
    adj = {i: [] for i in nodes}
    for c, p in parent.items():
        if p >= 0:
            adj[c].append(p); adj[p].append(c)
    from collections import deque
    D = np.zeros((n, n))
    for s in range(n):
        dist = {s: 0}; q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1; q.append(v)
        D[s] = [dist[t] for t in range(n)]
    # classical MDS: the closest Euclidean point cloud to that tree metric
    J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    o = np.argsort(w)[::-1][:d]
    X = V[:, o] * np.sqrt(np.clip(w[o], 0, None))
    if X.shape[1] < d:
        X = np.hstack([X, np.zeros((n, d - X.shape[1]))])
    Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
    X = X @ Q
    return X + rng.normal(0, noise * np.abs(X).mean(), X.shape)


def score(X, k=8, seed=0):
    """Hyperbolic advantage of X, relative to its covariance-matched null."""
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    a = hyp_advantage(D, k=k, seed=seed)["advantage"]
    N = g.matched_gaussian(X, seed)
    DN = np.linalg.norm(N[:, None] - N[None], axis=-1)
    b = hyp_advantage(DN, k=k, seed=seed)["advantage"]
    return a, b, a / b


if __name__ == "__main__":
    print("cloud             adv(data)  adv(null)   RATIO   (>1 = tree-like beyond covariance)")
    T = tree_cloud()
    for name, X in [("tree metric MDS", T),
                    ("isotropic gauss", np.random.default_rng(1).normal(size=(T.shape[0], 192))),
                    ("gauss w/ spectrum", g.matched_gaussian(T, 3))]:
        a, b, r = score(X)
        print("%-18s %8.2f %10.2f %8.3f" % (name, a, b, r))
