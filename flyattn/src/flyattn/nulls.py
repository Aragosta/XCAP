"""Null models. The house rule for this project: the baseline is a
degree-preserving rewiring or a hot-limit S1/H2 sample, never Erdos-Renyi.

`rewire_undirected` / `rewire_directed` are vectorised double-edge swaps
(Maslov-Sneppen). Swaps are proposed in batches; a batch is filtered for
self-loops, for duplicates inside the batch, and against the current edge set,
so the result is always a simple graph with the exact degree sequence of the
input.
"""
from __future__ import annotations

import numpy as np


def _keys(u: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
    return u.astype(np.int64) * n + v.astype(np.int64)


def rewire_undirected(u: np.ndarray, v: np.ndarray, n: int, *,
                      swaps_per_edge: float = 10.0, batch: int = 200_000,
                      rng: np.random.Generator | None = None,
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Degree-preserving rewiring of a simple undirected edge list (u<v)."""
    rng = rng or np.random.default_rng(0)
    u = u.astype(np.int64).copy()
    v = v.astype(np.int64).copy()
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    u, v = lo, hi
    m = len(u)
    key_set = np.sort(_keys(u, v, n))
    target = int(swaps_per_edge * m)
    done = 0
    while done < target:
        b = min(batch, target - done)
        e1 = rng.integers(0, m, b)
        e2 = rng.integers(0, m, b)
        flip = rng.random(b) < 0.5
        a1, b1 = u[e1], v[e1]
        a2, b2 = np.where(flip, u[e2], v[e2]), np.where(flip, v[e2], u[e2])
        # proposed: (a1,b2) and (a2,b1)
        n1u, n1v = np.minimum(a1, b2), np.maximum(a1, b2)
        n2u, n2v = np.minimum(a2, b1), np.maximum(a2, b1)
        ok = (e1 != e2) & (a1 != b2) & (a2 != b1)
        k1, k2 = _keys(n1u, n1v, n), _keys(n2u, n2v, n)
        ok &= k1 != k2
        # not already present
        for k in (k1, k2):
            idx = np.searchsorted(key_set, k)
            idx_c = np.clip(idx, 0, len(key_set) - 1)
            ok &= key_set[idx_c] != k
        # unique edge slots and unique new keys within the batch
        ok = _dedup(ok, e1, e2, k1, k2)
        sel = np.flatnonzero(ok)
        if len(sel):
            old = np.concatenate([_keys(u[e1[sel]], v[e1[sel]], n),
                                  _keys(u[e2[sel]], v[e2[sel]], n)])
            u[e1[sel]], v[e1[sel]] = n1u[sel], n1v[sel]
            u[e2[sel]], v[e2[sel]] = n2u[sel], n2v[sel]
            new = np.concatenate([k1[sel], k2[sel]])
            key_set = _replace(key_set, old, new)
        done += b
    return u.astype(np.int32), v.astype(np.int32)


def rewire_directed(pre: np.ndarray, post: np.ndarray, n: int, *,
                    swaps_per_edge: float = 10.0, batch: int = 200_000,
                    rng: np.random.Generator | None = None,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Degree-preserving rewiring keeping in- and out-degree of every node."""
    rng = rng or np.random.default_rng(0)
    p = pre.astype(np.int64).copy()
    q = post.astype(np.int64).copy()
    m = len(p)
    key_set = np.sort(_keys(p, q, n))
    target = int(swaps_per_edge * m)
    done = 0
    while done < target:
        b = min(batch, target - done)
        e1 = rng.integers(0, m, b)
        e2 = rng.integers(0, m, b)
        # swap targets: (p1,q2), (p2,q1)
        k1 = _keys(p[e1], q[e2], n)
        k2 = _keys(p[e2], q[e1], n)
        ok = (e1 != e2) & (p[e1] != q[e2]) & (p[e2] != q[e1]) & (k1 != k2)
        for k in (k1, k2):
            idx = np.clip(np.searchsorted(key_set, k), 0, len(key_set) - 1)
            ok &= key_set[idx] != k
        ok = _dedup(ok, e1, e2, k1, k2)
        sel = np.flatnonzero(ok)
        if len(sel):
            old = np.concatenate([_keys(p[e1[sel]], q[e1[sel]], n),
                                  _keys(p[e2[sel]], q[e2[sel]], n)])
            nq1, nq2 = q[e2[sel]].copy(), q[e1[sel]].copy()
            q[e1[sel]], q[e2[sel]] = nq1, nq2
            key_set = _replace(key_set, old, np.concatenate([k1[sel], k2[sel]]))
        done += b
    return p.astype(np.int32), q.astype(np.int32)


def _dedup(ok, e1, e2, k1, k2):
    """Keep only proposals that touch each edge slot once and add unique keys."""
    idx = np.flatnonzero(ok)
    if not len(idx):
        return ok
    seen_slots: set[int] = set()
    seen_keys: set[int] = set()
    keep = np.zeros(len(idx), bool)
    e1i, e2i, k1i, k2i = e1[idx], e2[idx], k1[idx], k2[idx]
    for j in range(len(idx)):
        a, b_, x, y = int(e1i[j]), int(e2i[j]), int(k1i[j]), int(k2i[j])
        if a in seen_slots or b_ in seen_slots or x in seen_keys or y in seen_keys:
            continue
        seen_slots.add(a); seen_slots.add(b_)
        seen_keys.add(x); seen_keys.add(y)
        keep[j] = True
    out = np.zeros_like(ok)
    out[idx[keep]] = True
    return out


def _replace(key_set: np.ndarray, old: np.ndarray, new: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(key_set, np.sort(old))
    mask = np.ones(len(key_set), bool)
    mask[pos] = False
    return np.sort(np.concatenate([key_set[mask], new]))
