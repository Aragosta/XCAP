"""The similarity -> separation -> projection framework."""
import math

import pytest
import torch

from ebt.hopfield import make_patterns, update
from ebt.uhn import (SEPARATIONS, SIMILARITIES, compress, kmeans, separate,
                     similarity, standardise, uhn)

D, K, B = 32, 16, 8


def g(seed=0):
    return torch.Generator().manual_seed(seed)


# ------------------------------------------- known models are cells of the grid
def test_dot_plus_softmax_is_exactly_transformer_attention():
    q, M = torch.randn(B, D, generator=g()), torch.randn(K, D, generator=g(1))
    V = torch.randn(K, D, generator=g(2))
    beta = 1 / math.sqrt(D)
    ref = torch.softmax(beta * (q @ M.T), dim=-1) @ V
    assert torch.allclose(uhn(q, M, V, "dot", "softmax", beta, standardised=False), ref, atol=1e-5)


def test_dot_plus_softmax_is_exactly_the_modern_hopfield_update():
    M = make_patterns(K, D, g())
    q = M + 0.3 * torch.randn(M.shape, generator=g(3))
    ref = update(q, M, beta=2.0, kind="softmax")
    assert torch.allclose(uhn(q, M, None, "dot", "softmax", 2.0, standardised=False), ref, atol=1e-5)


def test_max_separation_is_hard_nearest_neighbour_lookup():
    M = make_patterns(K, D, g())
    q = M + 0.05 * torch.randn(M.shape, generator=g(4))
    out = uhn(q, M, None, "euclidean", "max")
    assert torch.allclose(out, M, atol=1e-5), "max + a metric returns the memory itself"


# -------------------------------------------------------------- the sim stage
@pytest.mark.parametrize("kind", SIMILARITIES)
def test_every_similarity_ranks_an_identical_memory_first(kind):
    M = make_patterns(K, D, g())
    s = similarity(M, M, kind)
    assert (s.argmax(-1) == torch.arange(K)).all()


def test_metric_similarities_are_symmetric_and_zero_on_the_diagonal():
    M = make_patterns(K, D, g())
    for kind in ("euclidean", "manhattan"):
        s = similarity(M, M, kind)
        assert torch.allclose(s, s.T, atol=1e-4)
        assert torch.allclose(s.diagonal(), torch.zeros(K), atol=1e-3)


def test_the_dot_product_is_not_maximised_by_the_closest_memory():
    """The structural point: a dot product cannot separate 'close' from 'large'."""
    M = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    q = torch.tensor([[1.05, 0.0]])
    assert similarity(q, M, "dot").argmax(-1).item() == 1
    assert similarity(q, M, "euclidean").argmax(-1).item() == 0
    assert similarity(q, M, "manhattan").argmax(-1).item() == 0


def test_cosine_and_euclidean_agree_on_unit_norm_memories():
    M = torch.nn.functional.normalize(make_patterns(K, D, g()), dim=-1)
    q = torch.nn.functional.normalize(torch.randn(B, D, generator=g(5)), dim=-1)
    assert torch.equal(similarity(q, M, "cosine").argmax(-1),
                       similarity(q, M, "euclidean").argmax(-1))


# -------------------------------------------------------------- the sep stage
@pytest.mark.parametrize("kind", ["softmax", "sparsemax", "poly", "max"])
def test_separations_produce_distributions(kind):
    s = torch.randn(B, K, generator=g())
    p = separate(s, kind, beta=2.0)
    assert torch.allclose(p.sum(-1), torch.ones(B), atol=1e-5)
    assert (p >= 0).all()


def test_separations_differ_in_how_much_they_zero_out():
    s = standardise(torch.randn(B, K, generator=g()))
    zeros = {k: float((separate(s, k, beta=2.0) == 0).float().mean())
             for k in ("softmax", "sparsemax", "max")}
    assert zeros["softmax"] == 0.0
    assert 0.0 < zeros["sparsemax"] < zeros["max"]
    assert zeros["max"] == (K - 1) / K


def test_identity_separation_does_not_normalise():
    s = torch.randn(B, K, generator=g())
    assert torch.equal(separate(s, "identity"), s)


def test_standardisation_puts_every_similarity_on_one_scale():
    """Without it, one temperature means something different per similarity."""
    M, q = make_patterns(K, D, g()), torch.randn(B, D, generator=g(6))
    for kind in SIMILARITIES:
        s = standardise(similarity(q, M, kind))
        assert abs(float(s.mean())) < 1e-5
        assert abs(float(s.std(dim=-1).mean()) - 1.0) < 1e-3


# ------------------------------------------------------------- memory budget
def test_compression_returns_the_requested_number_of_slots():
    X = make_patterns(64, D, g())
    mem, assign = compress(X, 8, generator=g())
    assert mem.shape == (8, D)
    assert assign.shape == (64,) and int(assign.max()) < 8


def test_compression_is_a_no_op_when_the_budget_covers_everything():
    X = make_patterns(16, D, g())
    mem, assign = compress(X, 16, generator=g())
    assert torch.equal(mem, X) and torch.equal(assign, torch.arange(16))


def test_kmeans_prototypes_sit_closer_to_their_cluster_than_random_points():
    centres = torch.randn(4, D, generator=g())
    X = centres.repeat_interleave(16, 0) + 0.2 * torch.randn(64, D, generator=g(7))
    proto, assign = kmeans(X, 4, generator=g())
    d_proto = torch.cdist(X, proto).min(-1).values.mean()
    d_rand = torch.cdist(X, torch.randn(4, D, generator=g(8))).min(-1).values.mean()
    assert d_proto < d_rand


def test_a_compressed_memory_still_answers_by_class():
    """Memory can be saved: 8 prototypes for 128 patterns still classifies."""
    centres = torch.randn(8, D, generator=g())
    label = torch.arange(8).repeat_interleave(16)
    X = centres[label] + 0.6 * torch.randn(128, D, generator=g(9))
    proto, assign = compress(X, 8, generator=g())
    slot_label = torch.stack([torch.bincount(label[assign == c], minlength=8).argmax()
                              for c in range(8)])
    P = torch.nn.functional.one_hot(slot_label, 8).float()
    q = X + 0.5 * torch.randn(X.shape, generator=g(10))
    pred = uhn(q, proto, P, "euclidean", "softmax", beta=8.0).argmax(-1)
    assert float((pred == label).float().mean()) > 0.8


def test_a_metric_memory_answers_queries_it_never_stored():
    """Generalisation is inherent: unseen points are answered by distance."""
    centres = torch.randn(8, D, generator=g())
    label = torch.arange(8).repeat_interleave(16)
    X = centres[label] + 0.6 * torch.randn(128, D, generator=g(11))
    P = torch.nn.functional.one_hot(label, 8).float()
    novel = centres[label] + 0.6 * torch.randn(128, D, generator=g(12))
    pred = uhn(novel, X, P, "euclidean", "softmax", beta=8.0).argmax(-1)
    assert float((pred == label).float().mean()) > 0.85


def test_unknown_names_raise():
    with pytest.raises(ValueError):
        similarity(torch.randn(2, D), torch.randn(K, D), "nope")
    with pytest.raises(ValueError):
        separate(torch.randn(2, K), "nope")
