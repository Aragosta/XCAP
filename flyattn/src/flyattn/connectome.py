"""FlyWire v783 connectome loading and graph construction.

Data source: Codex public snapshot 783 (Dorkenwald et al. 2024; Schlegel et al. 2024),
    https://storage.googleapis.com/flywire-data/codex/data/fafb/783/

Conventions used throughout (fixed here so every experiment shares them):
  * Edges are aggregated over neuropils: a (pre, post) pair gets one edge whose
    weight is the total synapse count.
  * Default synapse threshold is >= 5, the threshold used in the FlyWire
    connectome papers for a "reliable" connection.
  * `undirected_simple()` collapses direction and removes self-loops. Clustering
    and curvature are defined on that object.
"""
from __future__ import annotations

import gzip
import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

DATA_DIR = os.environ.get(
    "FLYWIRE_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "flywire"),
)
DEFAULT_SYN_THRESHOLD = 5


@dataclass
class Connectome:
    """Directed, synapse-weighted connectome on a compact node index."""

    root_ids: np.ndarray           # (N,) int64, node index -> FlyWire root id
    pre: np.ndarray                # (E,) int32 node indices
    post: np.ndarray               # (E,) int32
    weight: np.ndarray             # (E,) int32 synapse counts
    syn_threshold: int

    @property
    def n(self) -> int:
        return len(self.root_ids)

    @property
    def n_edges(self) -> int:
        return len(self.pre)

    def index_of(self) -> dict[int, int]:
        return {int(r): i for i, r in enumerate(self.root_ids)}

    def directed_csr(self) -> sp.csr_matrix:
        return sp.csr_matrix(
            (self.weight.astype(np.float64), (self.pre, self.post)),
            shape=(self.n, self.n),
        )

    def undirected_simple(self) -> sp.csr_matrix:
        """Binary, symmetric, zero-diagonal adjacency."""
        a = sp.csr_matrix(
            (np.ones(len(self.pre), np.int8), (self.pre, self.post)),
            shape=(self.n, self.n),
        )
        a = a + a.T
        a.data[:] = 1
        a = a.tocsr()
        a.setdiag(0)
        a.eliminate_zeros()
        return a.astype(np.int8)

    def in_out_degree(self) -> tuple[np.ndarray, np.ndarray]:
        """Unweighted in/out degree over thresholded edges."""
        out = np.bincount(self.pre, minlength=self.n)
        inn = np.bincount(self.post, minlength=self.n)
        return inn, out

    def in_out_strength(self) -> tuple[np.ndarray, np.ndarray]:
        out = np.bincount(self.pre, weights=self.weight, minlength=self.n)
        inn = np.bincount(self.post, weights=self.weight, minlength=self.n)
        return inn, out


def load_connectome(syn_threshold: int = DEFAULT_SYN_THRESHOLD,
                    data_dir: str = DATA_DIR) -> Connectome:
    """Load connections.csv.gz, aggregate over neuropils, threshold, compact."""
    path = os.path.join(data_dir, "connections.csv.gz")
    pre_l, post_l, w_l = [], [], []
    with gzip.open(path, "rt") as fh:
        header = fh.readline().rstrip("\n").split(",")
        assert header[:4] == ["pre_root_id", "post_root_id", "neuropil", "syn_count"], header
        for line in fh:
            f = line.split(",", 4)
            pre_l.append(f[0])
            post_l.append(f[1])
            w_l.append(f[3])
    pre_raw = np.array(pre_l, dtype=np.int64)
    post_raw = np.array(post_l, dtype=np.int64)
    w_raw = np.array(w_l, dtype=np.int64)

    # aggregate duplicate (pre, post) rows coming from different neuropils
    ids = np.unique(np.concatenate([pre_raw, post_raw]))
    pi = np.searchsorted(ids, pre_raw).astype(np.int64)
    qi = np.searchsorted(ids, post_raw).astype(np.int64)
    key = pi * len(ids) + qi
    order = np.argsort(key, kind="stable")
    key = key[order]
    w_sorted = w_raw[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    agg_key = key[starts]
    agg_w = np.add.reduceat(w_sorted, starts)

    keep = agg_w >= syn_threshold
    agg_key, agg_w = agg_key[keep], agg_w[keep]
    pre_i = (agg_key // len(ids)).astype(np.int64)
    post_i = (agg_key % len(ids)).astype(np.int64)

    # drop self-loops, then compact to nodes that still carry an edge
    ok = pre_i != post_i
    pre_i, post_i, agg_w = pre_i[ok], post_i[ok], agg_w[ok]
    used = np.unique(np.concatenate([pre_i, post_i]))
    remap = np.full(len(ids), -1, np.int64)
    remap[used] = np.arange(len(used))
    return Connectome(
        root_ids=ids[used],
        pre=remap[pre_i].astype(np.int32),
        post=remap[post_i].astype(np.int32),
        weight=agg_w.astype(np.int32),
        syn_threshold=syn_threshold,
    )


def load_classification(data_dir: str = DATA_DIR) -> dict[int, dict[str, str]]:
    path = os.path.join(data_dir, "classification.csv.gz")
    out: dict[int, dict[str, str]] = {}
    with gzip.open(path, "rt") as fh:
        cols = fh.readline().rstrip("\n").split(",")
        for line in fh:
            f = line.rstrip("\n").split(",")
            out[int(f[0])] = dict(zip(cols[1:], f[1:]))
    return out


def load_coordinates(data_dir: str = DATA_DIR) -> dict[int, np.ndarray]:
    """root_id -> mean nm-scale position (nanometres; FAFB voxels are 4x4x40 nm)."""
    path = os.path.join(data_dir, "coordinates.csv.gz")
    acc: dict[int, list[np.ndarray]] = {}
    with gzip.open(path, "rt") as fh:
        fh.readline()
        for line in fh:
            rid, rest = line.split(",", 1)
            lo, hi = rest.index("["), rest.index("]")
            xyz = np.fromstring(rest[lo + 1:hi], sep=" ", dtype=np.float64)
            if xyz.size == 3:
                acc.setdefault(int(rid), []).append(xyz)
    return {k: np.mean(v, axis=0) * np.array([4.0, 4.0, 40.0]) for k, v in acc.items()}


def giant_component(a: sp.csr_matrix) -> np.ndarray:
    """Node indices of the largest connected component of a symmetric adjacency."""
    from scipy.sparse.csgraph import connected_components

    n_comp, labels = connected_components(a, directed=False)
    sizes = np.bincount(labels)
    return np.flatnonzero(labels == sizes.argmax())
