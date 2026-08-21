"""Attention as associative memory: the energy framework in its own terms.

This module tests the energy view where it makes *sharp* predictions, with no
training involved.  The claims come from three lines of work:

Ramsauer et al. (2020), "Hopfield Networks is All You Need"
    The continuous modern Hopfield energy is

        E(xi) = -lse(beta, X xi) + 1/2 ||xi||^2 + beta^-1 log M + 1/2 max_i ||x_i||^2

    and one CCCP step of minimising it is exactly

        xi' = X^T softmax(beta X xi)

    i.e. one row of transformer attention.  The theory then predicts:
      (a) capacity exponential in the dimension,
      (b) convergence in a *single* step for well-separated patterns, with
          retrieval error exponentially small in the separation,
      (c) when patterns are *not* well separated, the fixed point is close to
          the *mean* of the similar patterns -- a metastable state.  Retrieval
          fails by averaging.

Santos et al., "Sparse and Structured Hopfield Networks" (2024) + the
Hopfield-Fenchel-Young framework
    Replacing the entropic regulariser with a quadratic one turns lse into
    sparsemax and gives a family of energies

        E_Omega(xi) = 1/2 ||xi||^2 - Omega*(beta X xi) / beta

    where Omega* is the convex conjugate.  Softmax <-> log-sum-exp, sparsemax
    <-> a quadratic regulariser.  The prediction that matters: sparsity margins
    permit *exact* one-step retrieval, which the softmax energy provably cannot
    achieve because its output always has full support.

Hoover et al. (2023), "Energy Transformer"
    Runs the update recurrently to a fixed point rather than once.  So the
    energy view's real structural claim is not the score function -- it is that
    a layer is one step of an optimisation you are allowed to iterate.

Prediction (c) and the sparsemax margin result together give the cleanest test
of the user's combined framework: build memories that are deliberately *not*
well separated, and see which normaliser still retrieves the right one.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .sparsemax import sparsemax


def _probs(z: Tensor, kind: str) -> Tensor:
    if kind == "softmax":
        return torch.softmax(z, dim=-1)
    if kind == "sparsemax":
        return sparsemax(z, dim=-1)
    if kind == "sigmoid":
        # not a member of the Fenchel-Young family: the weights do not
        # normalise, so there is no matching energy.  Included because it is
        # the non-competitive gate under test elsewhere in this repo.
        return torch.sigmoid(z)
    raise ValueError(f"unknown kind {kind!r}")


def scores(xi: Tensor, X: Tensor, beta: float = 1.0, score: str = "dot") -> Tensor:
    """Similarity scores for the update.

    "dot"    beta * xi.x_j          -- what the modern Hopfield update uses
    "energy" -beta/2 * ||xi-x_j||^2 -- the pairwise energy itself

    They differ by -beta/2 ||x_j||^2 (the ||xi||^2 term is constant within a
    row), so they agree only when all memories have the same norm.  When they
    do not, argmax of the dot product is not argmin of the energy, and the two
    retrieve *different* memories.
    """
    if score == "dot":
        return beta * (xi @ X.T)
    if score == "energy":
        return -0.5 * beta * torch.cdist(xi, X).pow(2)
    raise ValueError(f"unknown score {score!r}")


def update(xi: Tensor, X: Tensor, beta: float = 1.0, kind: str = "softmax",
           score: str = "dot") -> Tensor:
    """One associative-memory step.  xi: [B,d], X: [M,d] -> [B,d]."""
    p = _probs(scores(xi, X, beta, score), kind)
    if kind == "sigmoid":
        return (p @ X) / (1.0 + p.sum(-1, keepdim=True))
    return p @ X


def energy(xi: Tensor, X: Tensor, beta: float = 1.0, kind: str = "softmax") -> Tensor:
    """Hopfield-Fenchel-Young energy.  Returns [B].

    E(xi) = 1/2 ||xi||^2 - Omega*(beta X xi)/beta  (+ constants in xi)

    with Omega* the conjugate of the regulariser: log-sum-exp for softmax, and
    for sparsemax  Omega*(z) = p.z - 1/2||p||^2  at p = sparsemax(z).
    """
    z = beta * (xi @ X.T)
    quad = 0.5 * xi.pow(2).sum(-1)
    if kind == "softmax":
        conj = torch.logsumexp(z, dim=-1)
    elif kind == "sparsemax":
        p = sparsemax(z, dim=-1)
        conj = (p * z).sum(-1) - 0.5 * p.pow(2).sum(-1)
    else:
        raise ValueError(f"{kind!r} has no Fenchel-Young energy")
    const = math.log(X.size(0)) / beta + 0.5 * X.pow(2).sum(-1).max()
    return quad - conj / beta + const


def retrieve(xi: Tensor, X: Tensor, beta: float = 1.0, kind: str = "softmax",
             steps: int = 1, score: str = "dot") -> tuple[Tensor, list[Tensor]]:
    """Iterate the update.  Returns the final state and the energy trajectory.

    The energy trajectory is only reported for the dot-product score, whose
    Fenchel-Young energy is the one implemented in :func:`energy`.
    """
    traj = []
    track = kind != "sigmoid" and score == "dot"
    if track:
        traj.append(energy(xi, X, beta, kind))
    for _ in range(steps):
        xi = update(xi, X, beta, kind, score)
        if track:
            traj.append(energy(xi, X, beta, kind))
    return xi, traj


def separation(X: Tensor) -> Tensor:
    """Ramsauer's pattern separation: Delta_i = min_j (x_i.x_i - x_i.x_j), j != i."""
    gram = X @ X.T
    diag = gram.diagonal()
    gap = diag[:, None] - gram
    gap.fill_diagonal_(float("inf"))
    return gap.min(dim=1).values


def retrieval_accuracy(xi: Tensor, X: Tensor, target: Tensor, beta: float = 1.0,
                       kind: str = "softmax", steps: int = 1,
                       score: str = "dot") -> dict[str, float]:
    """Fraction of queries whose retrieved state lands nearest the right pattern."""
    out, _ = retrieve(xi, X, beta, kind, steps, score)
    exact_err = (out - X[target]).norm(dim=-1) / X[target].norm(dim=-1).clamp(min=1e-9)
    # Cosine-nearest is the primary measure and Euclidean-nearest is reported
    # beside it, because gates that do not normalise (sigmoid) return the right
    # direction at the wrong scale: judging them by Euclidean distance measures
    # the scale, not the retrieval.
    unit = torch.nn.functional.normalize
    cos_nearest = (unit(out, dim=-1) @ unit(X, dim=-1).T).argmax(-1)
    return {
        "correct": float((cos_nearest == target).float().mean()),
        "euclid_correct": float((torch.cdist(out, X).argmin(-1) == target).float().mean()),
        "relative_error": float(exact_err.mean()),
        "exact_frac": float((exact_err < 0.01).float().mean()),
        "norm_ratio": float((out.norm(dim=-1) / X[target].norm(dim=-1)).mean()),
    }


def make_patterns(n: int, d: int, generator: torch.Generator,
                  clusters: int | None = None, spread: float = 0.1) -> Tensor:
    """Random patterns, optionally arranged in tight clusters.

    Clustered patterns are the interesting case: they are *not* well separated,
    which is exactly the regime where the theory predicts softmax retrieval
    degenerates into averaging.
    """
    if clusters is None:
        return torch.randn(n, d, generator=generator)
    centres = torch.randn(clusters, d, generator=generator)
    idx = torch.arange(n) % clusters
    return centres[idx] + spread * torch.randn(n, d, generator=generator)


def argmin_energy(xi: Tensor, X: Tensor) -> Tensor:
    """Hard retrieval: the memory of lowest *pairwise* energy E(q,m)=||q-m||^2.

    This is the readout the energy definition implies directly -- zero energy
    means identity, so the lowest-energy memory is the best match.  It is NOT
    what attention does: attention returns a convex combination sum_j p_j m_j.
    Comparing the two separates a failure of the energy landscape from a
    failure of the averaging readout.
    """
    return torch.cdist(xi, X).argmin(-1)


def energy_gap(xi: Tensor, X: Tensor, target: Tensor) -> Tensor:
    """E(q, m_second_best) - E(q, m_target): how much the right memory wins by."""
    e = torch.cdist(xi, X).pow(2)
    correct = e.gather(1, target[:, None]).squeeze(1)
    e = e.scatter(1, target[:, None], float("inf"))
    return e.min(-1).values - correct
