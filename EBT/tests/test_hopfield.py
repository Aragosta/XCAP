"""The energy framework's own predictions, as tests.

Each test names the claim it checks and the source of the claim.  These need no
training: the theory is about the update rule itself.
"""
import math

import pytest
import torch

from ebt.hopfield import (argmin_energy, energy, energy_gap, make_patterns,
                          retrieval_accuracy, retrieve, scores, separation, update)

D = 64


def g(seed=0):
    return torch.Generator().manual_seed(seed)


# ------------------------------------------------- the update IS attention
def test_the_update_rule_is_one_row_of_softmax_attention():
    """xi' = X^T softmax(beta X xi) -- Ramsauer et al. (2020), eq. for the CCCP step."""
    X = make_patterns(16, D, g())
    xi = torch.randn(4, D, generator=g(1))
    beta = 1.7
    ref = torch.softmax(beta * (xi @ X.T), dim=-1) @ X
    assert torch.allclose(update(xi, X, beta, "softmax"), ref, atol=1e-5)


def test_energy_decreases_monotonically_under_the_update():
    """CCCP guarantees descent; this is the sanity check that the energy is the
    right one for the update rule (a mismatch shows up immediately here)."""
    for kind in ("softmax", "sparsemax"):
        X = make_patterns(32, D, g(), clusters=4, spread=0.3)
        xi = X + 0.4 * torch.randn(X.shape, generator=g(2))
        _, traj = retrieve(xi, X, 1.0, kind, steps=6)
        e = torch.stack(traj)
        tol = 1e-5 * e.abs().max()        # float32 round-off at energies of O(10)
        assert (e[1:] <= e[:-1] + tol).all(), kind


def test_a_stored_pattern_is_a_fixed_point_when_patterns_are_separated():
    X = make_patterns(16, D, g())
    out, _ = retrieve(X, X, beta=1.0, kind="softmax", steps=1)
    assert (out - X).norm(dim=-1).max() < 1e-2 * X.norm(dim=-1).mean()


def test_one_update_suffices_for_well_separated_patterns():
    """Ramsauer et al.: the update converges after one step for separated patterns."""
    X = make_patterns(64, D, g())
    xi = X + 0.5 * torch.randn(X.shape, generator=g(3))
    one, _ = retrieve(xi, X, 1.0, "softmax", steps=1)
    five, _ = retrieve(xi, X, 1.0, "softmax", steps=5)
    assert (one - five).norm(dim=-1).max() < 1e-3 * X.norm(dim=-1).mean()


# --------------------------------------------------------------- capacity
@pytest.mark.parametrize("m", [64, 512, 2048])
def test_capacity_does_not_saturate_at_many_patterns(m):
    """Capacity is exponential in d, so d=64 must handle thousands of patterns."""
    X = make_patterns(m, D, g())
    xi = X + 0.5 * torch.randn(X.shape, generator=g(4))
    r = retrieval_accuracy(xi, X, torch.arange(m), 1.0, "softmax", steps=1)
    assert r["correct"] > 0.99


def test_retrieval_error_shrinks_as_separation_grows():
    """The error is exponentially small in the separation -- so it must at least
    be monotone in it, once the patterns are separated at all.

    Below separation ~0 the network is in the metastable regime and the error
    is dominated by which cluster mean it lands on, not by the separation, so
    the sweep starts above that.
    """
    errs, seps = [], []
    for spread in (0.3, 1.0, 3.0):
        X = make_patterns(32, D, g(), clusters=4, spread=spread)
        xi = X + 0.05 * torch.randn(X.shape, generator=g(5))
        errs.append(retrieval_accuracy(xi, X, torch.arange(32), 1.0, "softmax")["relative_error"])
        seps.append(float(separation(X).mean()))
    assert seps == sorted(seps), "spread must increase separation"
    assert errs == sorted(errs, reverse=True), f"error must fall as separation grows: {errs}"


# ------------------------------------------------- the metastable failure
def test_unseparated_patterns_collapse_to_their_cluster_mean():
    """Ramsauer et al.: with similar patterns the fixed point is a metastable
    state near their mean.  This is retrieval failing by *averaging*.

    Note this is a property of the **dot-product** update, which is what the
    modern Hopfield network uses.  Scoring with the pairwise energy instead
    removes the failure entirely -- see the tests below.
    """
    clusters = 4
    X = make_patterns(32, D, g(), clusters=clusters, spread=0.05)
    idx = torch.arange(32) % clusters
    means = torch.stack([X[idx == c].mean(0) for c in range(clusters)])
    out, _ = retrieve(X, X, 1.0, "softmax", steps=1)
    to_pattern = (out - X).norm(dim=-1)
    to_mean = (out - means[idx]).norm(dim=-1)
    assert (to_mean < to_pattern).float().mean() > 0.9


def test_well_separated_patterns_do_not_collapse():
    clusters = 4
    X = make_patterns(32, D, g(), clusters=clusters, spread=2.0)
    idx = torch.arange(32) % clusters
    means = torch.stack([X[idx == c].mean(0) for c in range(clusters)])
    out, _ = retrieve(X, X, 1.0, "softmax", steps=1)
    assert ((out - means[idx]).norm(dim=-1) < (out - X).norm(dim=-1)).float().mean() < 0.1


def test_sparsemax_resists_the_averaging_more_than_softmax():
    """Sparse Hopfield: a sparsity margin permits exact retrieval where the
    entropic (softmax) energy cannot, because softmax output has full support."""
    clusters = 4
    X = make_patterns(32, D, g(), clusters=clusters, spread=0.15)
    idx = torch.arange(32) % clusters
    means = torch.stack([X[idx == c].mean(0) for c in range(clusters)])
    frac = {}
    for kind in ("softmax", "sparsemax"):
        out, _ = retrieve(X, X, 1.0, kind, steps=1)
        frac[kind] = float(((out - means[idx]).norm(dim=-1)
                            < (out - X).norm(dim=-1)).float().mean())
    assert frac["sparsemax"] < frac["softmax"]


def test_sparsemax_can_retrieve_a_pattern_exactly_and_softmax_cannot():
    """The sharpest difference the framework predicts: exact vs approximate.

    Softmax has full support, so its fixed point is a convex combination of
    *all* patterns and can never be one of them exactly.  Sparsemax can hit a
    vertex of the simplex and return the pattern itself.

    The distinction is only *observable* in a window of separation, which is
    worth knowing before anyone leans on it: below it both fail, and above it
    softmax's residual weight on the other patterns underflows float32 and its
    error rounds to exactly zero too.  This test sits inside that window.
    """
    X = make_patterns(16, D, g(), clusters=4, spread=1.0)
    assert 10.0 < float(separation(X).mean()) < 200.0, "inside the observable window"
    r_sparse = retrieval_accuracy(X, X, torch.arange(16), 1.0, "sparsemax")
    r_soft = retrieval_accuracy(X, X, torch.arange(16), 1.0, "softmax")
    assert r_sparse["relative_error"] == 0.0
    assert r_soft["relative_error"] > 0.0


def test_the_exactness_gap_vanishes_into_float32_at_high_separation():
    """The companion fact: at large separation softmax is exact *numerically*."""
    X = make_patterns(16, D, g())
    assert float(separation(X).mean()) > 20.0
    r_soft = retrieval_accuracy(X, X, torch.arange(16), 1.0, "softmax")
    assert r_soft["relative_error"] == 0.0


# --------------------------------------------------------------- plumbing
def test_separation_matches_its_definition():
    X = make_patterns(8, D, g())
    gram = X @ X.T
    ref = torch.stack([min(gram[i, i] - gram[i, j] for j in range(8) if j != i)
                       for i in range(8)])
    assert torch.allclose(separation(X), ref, atol=1e-4)


def test_sigmoid_has_no_fenchel_young_energy():
    """It is not a normalised distribution, so it is outside the family."""
    with pytest.raises(ValueError):
        energy(torch.randn(2, D), make_patterns(8, D, g()), 1.0, "sigmoid")


def test_sigmoid_retrieves_the_direction_but_not_the_scale():
    """Why the retrieval metric is cosine-based: the non-competitive gate does
    not normalise, so Euclidean nearest-neighbour would measure its scale."""
    X = make_patterns(128, D, g())
    xi = X + 0.5 * torch.randn(X.shape, generator=g(6))
    r = retrieval_accuracy(xi, X, torch.arange(128), 1.0, "sigmoid")
    assert r["correct"] > 0.9
    assert r["euclid_correct"] < 0.2
    assert r["norm_ratio"] < 0.5


# ================= dot product vs the pairwise energy as the score ============
# The modern Hopfield update scores memories with xi.x_j, not with -||xi-x_j||^2.
# The two differ by -||x_j||^2/2, so they rank memories differently whenever the
# memories have different norms -- and then the metastable "averaging" failure
# above is really a norm-bias failure of the dot product, not a limit of
# associative memory.

def test_dot_and_energy_scores_differ_exactly_by_the_memory_norm():
    X = make_patterns(8, D, g())
    xi = torch.randn(4, D, generator=g(1))
    dot = scores(xi, X, 1.0, "dot")
    eng = scores(xi, X, 1.0, "energy")
    # -1/2||xi-x||^2 = xi.x - 1/2||x||^2 - 1/2||xi||^2
    ref = dot - 0.5 * X.pow(2).sum(-1)[None, :] - 0.5 * xi.pow(2).sum(-1)[:, None]
    assert torch.allclose(eng, ref, atol=1e-3)


def test_the_lowest_energy_memory_is_always_the_right_one():
    """E(q,m) = 0 iff q = m, so the pairwise energy cannot be wrong about which
    memory a noisy query came from -- whatever the separation."""
    for spread in (0.05, 0.15, 0.35):
        X = make_patterns(32, D, g(), clusters=4, spread=spread)
        t = torch.arange(32)
        q = X + 0.5 * spread * torch.randn(X.shape, generator=g(7))
        assert (argmin_energy(q, X) == t).float().mean() == 1.0
        assert (energy_gap(q, X, t) > 0).all()


def test_energy_scoring_retrieves_where_dot_scoring_fails():
    """The decisive comparison: same memories, same gate, same beta."""
    X = make_patterns(32, D, g(), clusters=4, spread=0.05)
    t = torch.arange(32)
    q = X + 0.025 * torch.randn(X.shape, generator=g(8))
    dot = retrieval_accuracy(q, X, t, 16.0, "softmax", 1, "dot")["correct"]
    eng = retrieval_accuracy(q, X, t, 16.0, "softmax", 1, "energy")["correct"]
    assert eng == 1.0
    assert dot < 0.5


def test_sharpening_rescues_energy_scoring_but_not_dot_scoring():
    """If raising beta does not fix retrieval, the *score* is wrong, not the
    temperature: an infinitely sharp gate returns argmax of whatever it ranks."""
    X = make_patterns(32, D, g(), clusters=4, spread=0.05)
    t = torch.arange(32)
    q = X + 0.025 * torch.randn(X.shape, generator=g(8))
    for score, expect in (("energy", True), ("dot", False)):
        cold = retrieval_accuracy(q, X, t, 1.0, "softmax", 1, score)["correct"]
        hot = retrieval_accuracy(q, X, t, 256.0, "softmax", 1, score)["correct"]
        assert (hot > cold + 0.5) == expect, (score, cold, hot)


def test_with_equal_norm_memories_the_two_scores_are_the_same_function():
    """||q-k||^2 = 2 - 2 q.k on the unit sphere, so QK-normalisation collapses
    the distinction -- which is why the difference never shows up in models
    that already normalise queries and keys."""
    X = torch.nn.functional.normalize(make_patterns(32, D, g(), clusters=4, spread=0.05), dim=-1)
    t = torch.arange(32)
    q = X + 0.025 * torch.randn(X.shape, generator=g(8))
    for beta in (1.0, 16.0, 256.0):
        dot = retrieval_accuracy(q, X, t, beta, "softmax", 1, "dot")
        eng = retrieval_accuracy(q, X, t, beta, "softmax", 1, "energy")
        assert dot["correct"] == eng["correct"]


def test_energy_scoring_is_stable_under_iteration():
    """With the dot product, iterating drifts into the metastable average; with
    the energy it is a fixed point."""
    X = make_patterns(32, D, g(), clusters=4, spread=0.35)
    t = torch.arange(32)
    q = X + 0.7 * torch.randn(X.shape, generator=g(9))
    one = retrieval_accuracy(q, X, t, 16.0, "softmax", 1, "energy")["correct"]
    five = retrieval_accuracy(q, X, t, 16.0, "softmax", 5, "energy")["correct"]
    assert five >= one - 1e-6

    dot_one = retrieval_accuracy(q, X, t, 1.0, "softmax", 1, "dot")["correct"]
    dot_five = retrieval_accuracy(q, X, t, 1.0, "softmax", 5, "dot")["correct"]
    assert dot_five < dot_one - 0.1


def test_a_minimal_example_where_the_dot_product_picks_the_wrong_memory():
    """Two memories, one near and one merely large.  The dot product prefers
    the large one; the energy prefers the near one, which is the answer."""
    near = torch.tensor([[1.0, 0.0]])
    far_but_large = torch.tensor([[3.0, 0.0]])
    X = torch.cat([near, far_but_large])
    q = torch.tensor([[1.05, 0.0]])
    assert scores(q, X, 1.0, "dot").argmax(-1).item() == 1, "dot picks the large memory"
    assert scores(q, X, 1.0, "energy").argmax(-1).item() == 0, "energy picks the near one"
    assert argmin_energy(q, X).item() == 0
