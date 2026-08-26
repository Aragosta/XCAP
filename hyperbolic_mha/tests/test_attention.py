"""Attention-module correctness, including the parity that makes the ablation controlled."""

import pytest
import torch

from hmha.attention import EuclideanMHA, HyperbolicMHA, build_attention
from hmha.lorentz import lorentz_inner

D_MODEL, N_HEADS, B, T = 32, 4, 2, 12
torch.manual_seed(0)


def _x():
    return torch.randn(B, T, D_MODEL)


def _both():
    return EuclideanMHA(D_MODEL, N_HEADS), HyperbolicMHA(D_MODEL, N_HEADS)


@pytest.mark.parametrize("kind", ["euclidean", "hyperbolic"])
def test_output_shape(kind):
    attn = build_attention(kind, D_MODEL, N_HEADS)
    assert attn(_x()).shape == (B, T, D_MODEL)


def test_parameter_count_parity():
    """The whole comparison rests on this: same params, only geometry differs."""
    euc, hyp = _both()
    n_euc = sum(p.numel() for p in euc.parameters())
    n_hyp = sum(p.numel() for p in hyp.parameters())
    # hyperbolic adds only the (frozen by default) per-head curvature scalars
    assert n_hyp - n_euc == N_HEADS
    n_hyp_trainable = sum(p.numel() for p in hyp.parameters() if p.requires_grad)
    assert n_hyp_trainable == n_euc
    assert abs(n_hyp - n_euc) / n_euc < 0.005


def test_learnable_curvature_is_trainable_and_positive():
    hyp = HyperbolicMHA(D_MODEL, N_HEADS, learnable_curvature=True, curvature=2.0)
    assert hyp.raw_curvature.requires_grad
    assert torch.allclose(hyp.curvature, torch.full_like(hyp.curvature, 2.0), atol=1e-5)
    # softplus keeps c > 0 even if the optimiser drives the raw value very negative
    with torch.no_grad():
        hyp.raw_curvature.fill_(-50.0)
    assert (hyp.curvature > 0).all()


@pytest.mark.parametrize("kind", ["euclidean", "hyperbolic"])
def test_causality_via_gradients(kind):
    """Output at position i must not depend on any input at position j > i."""
    attn = build_attention(kind, D_MODEL, N_HEADS)
    x = _x().requires_grad_(True)
    i = 4
    attn(x)[:, i].sum().backward()
    future_grad = x.grad[:, i + 1 :].abs().max()
    assert future_grad == 0, f"leak from the future: {future_grad}"
    assert x.grad[:, : i + 1].abs().max() > 0, "should depend on the past"


def test_euclidean_stats_path_matches_fused_path():
    """need_stats swaps the fused kernel for an explicit one; they must agree."""
    attn = EuclideanMHA(D_MODEL, N_HEADS).eval()
    x = _x()
    with torch.no_grad():
        assert torch.allclose(attn(x, need_stats=False), attn(x, need_stats=True), atol=1e-5)


def test_signature_trick_matches_explicit_lorentz_inner():
    """The fast score path must equal the pairwise Lorentz product it stands in for."""
    from hmha.lorentz import expmap0

    hyp = HyperbolicMHA(D_MODEL, N_HEADS)
    c = hyp.curvature
    q = torch.randn(1, N_HEADS, 5, hyp.head_dim, dtype=torch.float64)
    k = torch.randn(1, N_HEADS, 5, hyp.head_dim, dtype=torch.float64)
    q_h, k_h = expmap0(q, c.double()), expmap0(k, c.double())

    fast = (q_h * hyp.signature.double()) @ k_h.transpose(-1, -2)
    slow = torch.stack(
        [
            torch.stack([lorentz_inner(q_h[0, h, i], k_h[0, h, j]) for j in range(5)])
            for h in range(N_HEADS)
            for i in range(5)
        ]
    ).view(N_HEADS, 5, 5)
    assert torch.allclose(fast[0], slow, atol=1e-10)


@pytest.mark.parametrize("aggregation", ["lorentz_centroid", "klein", "tangent_mean"])
def test_aggregation_variants_run_and_are_finite(aggregation):
    attn = HyperbolicMHA(D_MODEL, N_HEADS, aggregation=aggregation)
    x = _x().requires_grad_(True)
    out = attn(x)
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in attn.parameters() if p.grad is not None)


@pytest.mark.parametrize("score_sign", ["corrected", "spec"])
@pytest.mark.parametrize("score_scale", ["sqrt_d", "learned"])
def test_score_options_run_and_are_finite(score_sign, score_scale):
    attn = HyperbolicMHA(D_MODEL, N_HEADS, score_sign=score_sign, score_scale=score_scale)
    out = attn(_x())
    assert torch.isfinite(out).all()


def test_score_sign_actually_changes_attention():
    """If the two sign conventions gave the same weights, the flag would be a no-op."""
    torch.manual_seed(1)
    corrected = HyperbolicMHA(D_MODEL, N_HEADS, score_sign="corrected")
    spec = HyperbolicMHA(D_MODEL, N_HEADS, score_sign="spec")
    spec.load_state_dict(corrected.state_dict())
    x = _x()
    with torch.no_grad():
        assert not torch.allclose(corrected(x), spec(x), atol=1e-4)


def test_spec_sign_attends_to_the_farthest_token():
    """Concrete consequence of the reference's sign: mass lands on the far token."""
    from hmha.lorentz import expmap0, lorentz_distance

    c = 1.0
    q = torch.zeros(1, 4)  # at the origin
    k = torch.tensor([[0.05, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])  # near, far
    q_h, k_h = expmap0(q, c), expmap0(k, c)
    dist = lorentz_distance(q_h.expand_as(k_h), k_h, c)
    assert dist[1] > dist[0]

    inner = lorentz_inner(q_h, k_h)
    spec_pick = torch.softmax(-inner, dim=-1).argmax()
    corrected_pick = torch.softmax(inner, dim=-1).argmax()
    assert spec_pick.item() == 1, "spec sign concentrates on the far key"
    assert corrected_pick.item() == 0, "corrected sign concentrates on the near key"


@pytest.mark.parametrize("kind", ["euclidean", "hyperbolic"])
def test_deterministic_under_fixed_seed(kind):
    x = _x()
    outs = []
    for _ in range(2):
        torch.manual_seed(7)
        outs.append(build_attention(kind, D_MODEL, N_HEADS).eval()(x))
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("curvature", [0.25, 1.0, 4.0])
def test_finite_across_curvatures_and_input_scales(curvature):
    attn = HyperbolicMHA(D_MODEL, N_HEADS, curvature=curvature)
    for scale in [1e-4, 1.0, 20.0]:
        x = (torch.randn(B, T, D_MODEL) * scale).requires_grad_(True)
        out = attn(x)
        out.sum().backward()
        assert torch.isfinite(out).all(), f"non-finite out, c={curvature} scale={scale}"
        assert torch.isfinite(x.grad).all(), f"non-finite grad, c={curvature} scale={scale}"
        x.grad = None


def test_stats_are_recorded():
    attn = HyperbolicMHA(D_MODEL, N_HEADS)
    attn(_x(), need_stats=True)
    assert attn.last_attn_entropy is not None and attn.last_attn_entropy >= 0
    assert attn.last_radius is not None and attn.last_radius >= 0


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        HyperbolicMHA(D_MODEL, 5)  # not divisible
    with pytest.raises(ValueError):
        HyperbolicMHA(D_MODEL, N_HEADS, score_sign="nonsense")
    with pytest.raises(ValueError):
        HyperbolicMHA(D_MODEL, N_HEADS, aggregation="nonsense")
    with pytest.raises(ValueError):
        build_attention("quaternionic", D_MODEL, N_HEADS)
