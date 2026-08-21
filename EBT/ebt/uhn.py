"""A coherent framework for energy attention: similarity -> separation -> projection.

Every single-shot associative memory -- classical Hopfield, sparse distributed
memory, dense associative memory, the modern continuous Hopfield network, and
transformer attention -- is the same three operations (Millidge et al., ICML
2022, "Universal Hopfield Networks"):

    out = sep( sim(q, M) ) @ P

    sim   how close is the query to each memory       [B,K]
    sep   sharpen that into a retrieval weighting     [B,K]
    proj  read out the associated value               [B,d_out]

The models differ only in which sim and sep they pick:

    classical Hopfield      dot        + identity
    sparse distributed mem  hamming    + threshold
    dense assoc. memory     dot        + polynomial
    modern Hopfield / attn  dot        + softmax
    this repo's sparsemax   dot        + sparsemax

which makes the design space explicit and the empty cells obvious.  Millidge et
al. report that swapping the dot product for a Euclidean or Manhattan distance
gives substantially more robust retrieval and higher capacity; the sparse
Hopfield line (Santos et al.) reports the same for swapping softmax for
sparsemax.  Nobody appears to have combined them, and that combination is the
interesting cell here.

The three properties this module is built to test
-------------------------------------------------
1. *dissimilarity is structurally encoded* -- the sim stage is a real metric,
   so "no match" has a representation (large distance), which a dot product
   cannot express independently of magnitude.
2. *memory can be saved* -- the memory M is a free argument, so it can be the
   raw patterns or a much smaller set of prototypes.
3. *generalisation is inherent* -- a metric memory answers queries it has never
   stored by their distance to what it has, which is nearest-prototype
   classification, not lookup.

A note on fairness
------------------
The similarity functions have wildly different scales: q.k grows like sqrt(d),
||q-k||^2 like d, ||q-k||_1 like d*sqrt(d).  Feeding them to the same softmax
temperature would compare temperatures, not similarity functions.  Scores are
therefore standardised per query row before separation, unless disabled.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .sparsemax import sparsemax

SIMILARITIES = ("dot", "cosine", "euclidean", "manhattan")
SEPARATIONS = ("identity", "softmax", "sparsemax", "poly", "max")


def similarity(q: Tensor, M: Tensor, kind: str = "dot") -> Tensor:
    """[B,d] x [K,d] -> [B,K], larger meaning more similar in every case."""
    if kind == "dot":
        return q @ M.T
    if kind == "cosine":
        unit = torch.nn.functional.normalize
        return unit(q, dim=-1) @ unit(M, dim=-1).T
    if kind == "euclidean":
        return -torch.cdist(q, M).pow(2)
    if kind == "manhattan":
        return -torch.cdist(q, M, p=1.0)
    raise ValueError(f"unknown similarity {kind!r}; expected one of {SIMILARITIES}")


def standardise(s: Tensor) -> Tensor:
    """Zero-mean, unit-std per query row, so one temperature suits every sim."""
    return (s - s.mean(-1, keepdim=True)) / s.std(-1, keepdim=True).clamp(min=1e-9)


def separate(s: Tensor, kind: str = "softmax", beta: float = 1.0, degree: int = 3) -> Tensor:
    """Sharpen similarity scores into retrieval weights."""
    if kind == "identity":
        return s
    if kind == "softmax":
        return torch.softmax(beta * s, dim=-1)
    if kind == "sparsemax":
        return sparsemax(beta * s, dim=-1)
    if kind == "poly":
        # dense associative memory: rectify then raise to a power
        p = torch.relu(s).pow(degree)
        return p / p.sum(-1, keepdim=True).clamp(min=1e-9)
    if kind == "max":
        return torch.zeros_like(s).scatter_(-1, s.argmax(-1, keepdim=True), 1.0)
    raise ValueError(f"unknown separation {kind!r}; expected one of {SEPARATIONS}")


def uhn(q: Tensor, M: Tensor, P: Tensor | None = None, sim: str = "dot",
        sep: str = "softmax", beta: float = 1.0, degree: int = 3,
        standardised: bool = True) -> Tensor:
    """One associative-memory read: sep(sim(q, M)) @ P.

    ``P`` defaults to ``M`` (autoassociative); pass a different matrix for a
    heteroassociative memory, which is what attention's values are.
    """
    s = similarity(q, M, sim)
    if standardised:
        s = standardise(s)
    return separate(s, sep, beta, degree) @ (M if P is None else P)


def kmeans(X: Tensor, k: int, iters: int = 25, generator: torch.Generator | None = None
           ) -> tuple[Tensor, Tensor]:
    """Lloyd's algorithm.  Returns (centroids [k,d], assignment [N])."""
    n = X.size(0)
    perm = torch.randperm(n, generator=generator)[:k]
    centroids = X[perm].clone()
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        assign = torch.cdist(X, centroids).argmin(-1)
        for c in range(k):
            hit = assign == c
            if hit.any():
                centroids[c] = X[hit].mean(0)
    return centroids, assign


def compress(X: Tensor, k: int, generator: torch.Generator | None = None
             ) -> tuple[Tensor, Tensor]:
    """Replace N memories by k prototypes: the 'memory can be saved' operation."""
    if k >= X.size(0):
        return X.clone(), torch.arange(X.size(0))
    return kmeans(X, k, generator=generator)
