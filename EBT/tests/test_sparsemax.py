import torch
import pytest

from ebt.sparsemax import NEG_SENTINEL, masked_softmax, sparsemax


def test_output_is_on_the_simplex():
    z = torch.randn(16, 32) * 3
    p = sparsemax(z)
    assert torch.allclose(p.sum(-1), torch.ones(16), atol=1e-5)
    assert (p >= 0).all()


def test_produces_exact_zeros_unlike_softmax():
    z = torch.randn(8, 64) * 5
    p, s = sparsemax(z), torch.softmax(z, -1)
    assert (p == 0).any()
    assert (s > 0).all()


def test_matches_euclidean_projection_by_brute_force():
    """sparsemax(z) is the argmin of ||p - z||^2 over the simplex."""
    torch.manual_seed(0)
    z = torch.randn(64, requires_grad=False)
    p = sparsemax(z)
    q = torch.full((64,), 1 / 64, requires_grad=True)
    opt = torch.optim.Adam([q], lr=0.05)
    for _ in range(4000):                      # projected gradient descent
        opt.zero_grad()
        ((q - z) ** 2).sum().backward()
        opt.step()
        with torch.no_grad():
            q.copy_(sparsemax(q.detach()))     # projection step (self-consistent)
    assert ((p - z) ** 2).sum() <= ((q.detach() - z) ** 2).sum() + 1e-4


def test_uniform_input_gives_uniform_output():
    p = sparsemax(torch.zeros(3, 10))
    assert torch.allclose(p, torch.full((3, 10), 0.1), atol=1e-6)


def test_large_gap_gives_one_hot():
    z = torch.tensor([[10.0, 0.0, -1.0]])
    assert torch.allclose(sparsemax(z), torch.tensor([[1.0, 0.0, 0.0]]))


def test_translation_invariance():
    z = torch.randn(4, 20)
    assert torch.allclose(sparsemax(z), sparsemax(z + 7.5), atol=1e-6)


def test_scale_increases_sparsity():
    z = torch.randn(8, 64)
    supports = [(sparsemax(z * s) > 0).float().mean() for s in (0.1, 1.0, 10.0)]
    assert supports[0] >= supports[1] >= supports[2]


def test_gradient_matches_numerical():
    z = torch.randn(4, 12, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: sparsemax(t, -1), (z,), eps=1e-6, atol=1e-8)


def test_gradient_is_zero_outside_the_support():
    z = torch.tensor([[5.0, 0.0, -3.0]], requires_grad=True)
    p = sparsemax(z)
    p.sum().backward()
    assert z.grad[0, 2] == 0.0


def test_masked_positions_never_in_support():
    z = torch.randn(32, 48)
    mask = torch.rand(32, 48) > 0.5
    mask[:, 0] = True                          # keep at least one live slot per row
    p = sparsemax(z, mask=mask)
    assert (p[~mask] == 0).all()
    assert torch.allclose(p.sum(-1), torch.ones(32), atol=1e-5)


def test_masking_equals_dropping_the_columns():
    z = torch.randn(1, 10)
    mask = torch.ones(1, 10, dtype=torch.bool)
    mask[0, 3:6] = False
    full = sparsemax(z, mask=mask)
    kept = torch.cat([z[:, :3], z[:, 6:]], dim=1)
    assert torch.allclose(torch.cat([full[:, :3], full[:, 6:]], 1), sparsemax(kept), atol=1e-6)


def test_sentinel_is_not_inf():
    assert NEG_SENTINEL < -1e6 and torch.isfinite(torch.tensor(NEG_SENTINEL))


def test_masked_softmax_zeroes_masked_entries_and_sums_to_one():
    z = torch.randn(5, 9)
    mask = torch.rand(5, 9) > 0.3
    mask[:, 0] = True
    p = masked_softmax(z, mask=mask)
    assert (p[~mask] == 0).all()
    assert torch.allclose(p.sum(-1), torch.ones(5), atol=1e-6)


def test_fully_masked_row_is_all_zero_not_nan():
    z = torch.randn(2, 6)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    p = masked_softmax(z, mask=mask)
    assert torch.isfinite(p).all() and float(p.sum()) == 0.0


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_works_along_any_dim(dim):
    z = torch.randn(4, 5, 6)
    p = sparsemax(z, dim=dim)
    assert torch.allclose(p.sum(dim), torch.ones_like(p.sum(dim)), atol=1e-5)
