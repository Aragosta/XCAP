"""Manifold correctness. These are a gate: nothing trains until they pass."""

import math

import pytest
import torch

from hmha.lorentz import (
    _MAX_TANGENT_ARG,
    expmap0,
    klein_gyromidpoint,
    logmap0,
    lorentz_centroid,
    lorentz_distance,
    lorentz_inner,
    lorentz_linear,
    project_to_manifold,
    signature_matrix,
)

CURVATURES = [0.25, 1.0, 4.0]
torch.manual_seed(0)


def _tangent(*shape, scale=1.0):
    return torch.randn(*shape, dtype=torch.float64) * scale


@pytest.mark.parametrize("c", CURVATURES)
def test_expmap0_lands_on_manifold(c):
    """<x,x>_L == -1/c. This is the test the spec's exp map fails."""
    x = expmap0(_tangent(64, 16), c)
    inner = lorentz_inner(x, x)
    assert torch.allclose(inner, torch.full_like(inner, -1.0 / c), atol=1e-8)
    assert (x[..., 0] > 0).all(), "must stay on the upper sheet"


@pytest.mark.parametrize("c", CURVATURES)
def test_spec_expmap_without_sqrt_c_is_off_manifold(c):
    """Documents correction #1: the spec's spatial part violates the constraint."""
    u = _tangent(32, 8)
    norm = u.norm(dim=-1, keepdim=True)
    sqrt_c = math.sqrt(c)
    spec = torch.cat(
        [torch.cosh(sqrt_c * norm) / sqrt_c, (u / norm) * torch.sinh(sqrt_c * norm)], dim=-1
    )
    inner = lorentz_inner(spec, spec)
    target = torch.full_like(inner, -1.0 / c)
    if abs(c - 1.0) < 1e-12:
        # at c == 1 the missing 1/sqrt(c) factor is 1, so the spec happens to be right
        assert torch.allclose(inner, target, atol=1e-8)
    else:
        assert not torch.allclose(inner, target, atol=1e-3)


@pytest.mark.parametrize("c", CURVATURES)
def test_exp_log_roundtrip(c):
    u = _tangent(64, 16, scale=0.7)
    assert torch.allclose(logmap0(expmap0(u, c), c), u, atol=1e-9)


@pytest.mark.parametrize("c", CURVATURES)
def test_log_exp_roundtrip(c):
    x = expmap0(_tangent(64, 16, scale=0.7), c)
    assert torch.allclose(expmap0(logmap0(x, c), c), x, atol=1e-9)


@pytest.mark.parametrize("c", CURVATURES)
def test_project_to_manifold(c):
    x = project_to_manifold(_tangent(32, 8, scale=3.0), c)
    inner = lorentz_inner(x, x)
    assert torch.allclose(inner, torch.full_like(inner, -1.0 / c), atol=1e-8)


@pytest.mark.parametrize("c", CURVATURES)
def test_distance_is_a_metric(c):
    x = expmap0(_tangent(16, 8, scale=0.5), c)
    y = expmap0(_tangent(16, 8, scale=0.5), c)
    z = expmap0(_tangent(16, 8, scale=0.5), c)

    assert torch.allclose(lorentz_distance(x, y, c), lorentz_distance(y, x, c), atol=1e-9)
    assert lorentz_distance(x, x, c).abs().max() < 1e-3
    assert (lorentz_distance(x, y, c) >= 0).all()
    d_xz = lorentz_distance(x, z, c)
    assert (d_xz <= lorentz_distance(x, y, c) + lorentz_distance(y, z, c) + 1e-9).all()


@pytest.mark.parametrize("c", CURVATURES)
def test_distance_from_origin_matches_tangent_norm(c):
    """exp_o is a radial isometry: d(o, exp_o(u)) == ||u||."""
    u = _tangent(32, 8, scale=0.6)
    x = expmap0(u, c)
    origin = expmap0(torch.zeros_like(u), c)
    assert torch.allclose(lorentz_distance(origin, x, c), u.norm(dim=-1), atol=1e-7)


@pytest.mark.parametrize("c", CURVATURES)
def test_signature_matmul_equals_pairwise_inner(c):
    """The Q M K^T shortcut must equal the explicit pairwise Lorentz product."""
    q = expmap0(_tangent(6, 5), c)
    k = expmap0(_tangent(7, 5), c)
    m = signature_matrix(q.shape[-1], dtype=q.dtype)

    fast = q @ m @ k.transpose(-1, -2)
    slow = torch.stack([torch.stack([lorentz_inner(qi, kj) for kj in k]) for qi in q])
    assert torch.allclose(fast, slow, atol=1e-10)


@pytest.mark.parametrize("c", CURVATURES)
def test_negative_inner_is_increasing_in_distance(c):
    """Documents correction #2: -<q,k>_L grows with distance.

    So softmax(-<q,k>_L), as the spec writes it, puts its mass on the *farthest*
    tokens. The corrected score +<q,k>_L is decreasing in distance.
    """
    q = expmap0(_tangent(64, 8, scale=0.8), c)
    k = expmap0(_tangent(64, 8, scale=0.8), c)
    dist = lorentz_distance(q, k, c)
    neg_inner = -lorentz_inner(q, k)

    order_d = torch.argsort(dist)
    assert torch.all(torch.diff(neg_inner[order_d]) >= -1e-9), "-<q,k>_L must increase with d"
    assert torch.allclose(neg_inner, torch.cosh(math.sqrt(c) * dist) / c, atol=1e-7)


@pytest.mark.parametrize("c", CURVATURES)
def test_centroid_lands_on_manifold(c):
    pts = expmap0(_tangent(4, 10, 8, scale=0.5), c)
    w = torch.softmax(torch.randn(4, 10, dtype=torch.float64), dim=-1)
    out = lorentz_centroid(w, pts, c)
    inner = lorentz_inner(out, out)
    assert torch.allclose(inner, torch.full_like(inner, -1.0 / c), atol=1e-8)
    assert (out[..., 0] > 0).all()


@pytest.mark.parametrize("c", CURVATURES)
def test_centroid_of_one_point_is_that_point(c):
    pts = expmap0(_tangent(4, 3, 8, scale=0.5), c)
    w = torch.zeros(4, 3, dtype=torch.float64)
    w[:, 1] = 1.0
    assert torch.allclose(lorentz_centroid(w, pts, c), pts[:, 1], atol=1e-9)


@pytest.mark.parametrize("c", CURVATURES)
def test_klein_gyromidpoint_lands_on_manifold(c):
    pts = expmap0(_tangent(4, 10, 8, scale=0.5), c)
    w = torch.softmax(torch.randn(4, 10, dtype=torch.float64), dim=-1)
    out = klein_gyromidpoint(w, pts, c)
    inner = lorentz_inner(out, out)
    assert torch.allclose(inner, torch.full_like(inner, -1.0 / c), atol=1e-7)


@pytest.mark.parametrize("c", CURVATURES)
def test_klein_gyromidpoint_of_one_point_is_that_point(c):
    pts = expmap0(_tangent(4, 3, 8, scale=0.5), c)
    w = torch.zeros(4, 3, dtype=torch.float64)
    w[:, 2] = 1.0
    assert torch.allclose(klein_gyromidpoint(w, pts, c), pts[:, 2], atol=1e-7)


@pytest.mark.parametrize("c", CURVATURES)
def test_lorentz_linear_stays_on_manifold(c):
    x = expmap0(_tangent(16, 8, scale=0.5), c)
    w = torch.randn(12, 8, dtype=torch.float64) * 0.1
    y = lorentz_linear(x, w, c)
    inner = lorentz_inner(y, y)
    assert y.shape == (16, 13)
    assert torch.allclose(inner, torch.full_like(inner, -1.0 / c), atol=1e-8)


def test_small_curvature_approaches_euclidean():
    """As c -> 0 the manifold flattens: geodesic distance -> Euclidean distance."""
    u, v = _tangent(64, 8, scale=0.5), _tangent(64, 8, scale=0.5)
    euclid = (u - v).norm(dim=-1)
    errors = []
    for c in [1e-1, 1e-3, 1e-5]:
        d = lorentz_distance(expmap0(u, c), expmap0(v, c), c)
        errors.append((d - euclid).abs().max().item())
    assert errors[0] > errors[1] > errors[2], f"should converge, got {errors}"
    assert errors[-1] < 1e-3


@pytest.mark.parametrize("c", CURVATURES)
def test_gradients_are_finite_at_the_origin(c):
    """The origin is where arcosh-based log maps blow up. asinh must not."""
    u = torch.zeros(4, 8, dtype=torch.float64, requires_grad=True)
    logmap0(expmap0(u, c), c).sum().backward()
    assert torch.isfinite(u.grad).all()


@pytest.mark.parametrize("c", CURVATURES)
def test_gradients_are_finite_for_extreme_norms(c):
    for scale in [1e-6, 1.0, 50.0]:
        u = (_tangent(8, 16) * scale).requires_grad_(True)
        x = expmap0(u, c)
        w = torch.softmax(torch.randn(8, dtype=torch.float64), dim=-1)
        out = lorentz_centroid(w.unsqueeze(0), x.unsqueeze(0), c)
        logmap0(out, c).sum().backward()
        assert torch.isfinite(u.grad).all(), f"non-finite grad at scale {scale}"


@pytest.mark.parametrize("c", CURVATURES)
def test_large_inputs_do_not_overflow_in_float32(c):
    """Tangent norms are clamped, so float32 sinh/cosh must not reach inf."""
    u = torch.randn(16, 8) * 1e4
    x = expmap0(u, c)
    assert torch.isfinite(x).all()
    assert (x[..., 0] > 0).all()
    # the clamp caps the geodesic radius rather than producing inf
    assert x.abs().max() <= math.cosh(_MAX_TANGENT_ARG) / math.sqrt(c) * 1.01


def test_distance_grows_exponentially_with_volume():
    """The property motivating the whole exercise: hyperbolic balls hold more.

    The number of points that fit at pairwise distance >= 1 grows faster than
    the Euclidean count at matched radius. Checked here as: the manifold's
    geodesic distance exceeds the tangent-space distance between the same
    points (negative curvature spreads points apart).
    """
    c = 1.0
    u, v = _tangent(256, 4, scale=1.5), _tangent(256, 4, scale=1.5)
    geodesic = lorentz_distance(expmap0(u, c), expmap0(v, c), c)
    assert (geodesic >= (u - v).norm(dim=-1) - 1e-6).all()
