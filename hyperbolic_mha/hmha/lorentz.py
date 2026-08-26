"""Lorentz (hyperboloid) model of hyperbolic space.

Convention
----------
A point ``x`` in R^{D+1} is split as ``x = [x_0, x_s]`` with ``x_s`` in R^D.
The Lorentz (Minkowski) inner product is

    <x, y>_L = -x_0 y_0 + x_s . y_s

and the manifold of curvature ``-c`` (``c > 0``) is

    H^D_c = { x : <x, x>_L = -1/c,  x_0 > 0 }

so ``x_0 = sqrt(1/c + ||x_s||^2)``. The origin is ``o = [1/sqrt(c), 0, ..., 0]``.

Notes on the reference formulas
-------------------------------
Three corrections relative to the spec this module was built from; each is
covered by a test in ``tests/test_lorentz.py``.

1. ``expmap0`` needs a ``1/sqrt(c)`` factor on the *spatial* part. Without it
   the result does not satisfy ``<x, x>_L = -1/c``.
2. ``logmap0`` is written with ``asinh`` rather than ``arcosh``. The two are
   equivalent on the manifold, but ``arcosh`` has an infinite derivative at 1,
   which is exactly where points near the origin live, and it produces NaN
   gradients there.
3. The Lorentz factor ``gamma = 1/sqrt(1 - c||v||^2)`` belongs to the *Klein*
   model, not the Lorentz one. ``klein_gyromidpoint`` converts to Klein
   coordinates first; ``lorentz_centroid`` is the direct Lorentz analogue and
   needs no ``gamma``.
"""

from __future__ import annotations

import torch

# sinh/cosh overflow in float32 around |arg| ~ 89; clamp well below that so the
# products downstream stay finite too.
_MAX_TANGENT_ARG = 15.0
_EPS = 1e-7


def lorentz_inner(x: torch.Tensor, y: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    """<x, y>_L = -x_0 y_0 + x_s . y_s, contracting the last dimension."""
    prod = x * y
    time_part = prod[..., :1]
    space_part = prod[..., 1:].sum(dim=-1, keepdim=True)
    out = space_part - time_part
    return out if keepdim else out.squeeze(-1)


def project_to_manifold(x_space: torch.Tensor, c: torch.Tensor | float) -> torch.Tensor:
    """Build a valid manifold point from a free spatial part.

    Sets the time coordinate to ``sqrt(1/c + ||x_s||^2)``, which is the only
    value making the point satisfy the manifold constraint. The time coordinate
    is therefore *derived*, never a free parameter -- this is what keeps the
    hyperbolic parameter count equal to the Euclidean one.
    """
    c = _as_tensor(c, x_space)
    x_time = torch.sqrt(1.0 / c + x_space.pow(2).sum(dim=-1, keepdim=True))
    return torch.cat([x_time, x_space], dim=-1)


def expmap0(u: torch.Tensor, c: torch.Tensor | float) -> torch.Tensor:
    """Exponential map at the origin: tangent vector ``u`` in R^D -> manifold point.

        x_0   = cosh(sqrt(c)||u||) / sqrt(c)
        x_s   = sinh(sqrt(c)||u||) / (sqrt(c)||u||) * u

    The ``sinh(z)/z`` form is used so the ``u -> 0`` limit is well behaved
    (it tends to 1, giving ``x_s -> u``) instead of dividing by ``||u||``.
    """
    c = _as_tensor(c, u)
    sqrt_c = c.sqrt()
    norm = u.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    arg = (sqrt_c * norm).clamp(max=_MAX_TANGENT_ARG)

    x_time = torch.cosh(arg) / sqrt_c
    # Divide by sqrt_c*norm, NOT by arg: when arg is unclamped the two are equal
    # and the factor reduces to the safe sinh(z)/z form, but once arg saturates
    # this keeps ||x_s|| bounded by sinh(MAX)/sqrt_c. Dividing by arg would still
    # scale with ||u|| and overflow for large inputs.
    x_space = (torch.sinh(arg) / (sqrt_c * norm)) * u
    return torch.cat([x_time, x_space], dim=-1)


def logmap0(x: torch.Tensor, c: torch.Tensor | float) -> torch.Tensor:
    """Logarithmic map at the origin: manifold point -> tangent vector in R^D.

    Exact inverse of :func:`expmap0`. On the manifold ``||x_s|| = sinh(sqrt(c) d)/sqrt(c)``
    where ``d`` is the geodesic distance to the origin, so ``d = asinh(sqrt(c)||x_s||)/sqrt(c)``
    and the tangent vector is ``d * x_s/||x_s||``.
    """
    c = _as_tensor(c, x)
    sqrt_c = c.sqrt()
    x_space = x[..., 1:]
    norm = x_space.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    # asinh(z)/z -> 1 as z -> 0, so this stays smooth at the origin.
    scale = torch.asinh(sqrt_c * norm) / (sqrt_c * norm)
    return scale * x_space


def lorentz_distance(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor | float) -> torch.Tensor:
    """Geodesic distance ``arcosh(-c <x,y>_L) / sqrt(c)``.

    The argument is clamped to ``>= 1 + eps``: it is analytically ``>= 1`` but
    rounding can push it just below, and ``arcosh`` has an infinite derivative
    at 1 regardless, so the clamp also keeps ``d(x, x) = 0`` differentiable.
    The epsilon is dtype-aware -- a fixed 1e-6 would put a floor of ~1.4e-3 on
    every distance, which is coarse enough to show up as ``d(x, x) != 0``.
    """
    c = _as_tensor(c, x)
    eps = torch.finfo(x.dtype).eps
    arg = (-c * lorentz_inner(x, y)).clamp_min(1.0 + eps)
    return torch.acosh(arg) / c.sqrt()


def lorentz_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    c: torch.Tensor | float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hyperbolic linear map ``exp_o(W log_o(x))``.

    Drops to the tangent space at the origin, applies an ordinary linear map
    there, and lifts the result back onto the manifold. ``x`` is a manifold
    point of ambient dim ``D_in + 1``; the output has ambient dim ``D_out + 1``.
    """
    u = logmap0(x, c)
    v = torch.nn.functional.linear(u, weight, bias)
    return expmap0(v, c)


def _weighted_sum(weights: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """``sum_j w_j x_j``, with or without a leading query axis on the weights.

    Accepts ``weights`` of shape ``(..., N)`` -> output ``(..., D+1)``, or
    ``(..., Q, N)`` -> output ``(..., Q, D+1)``. Attention needs the second form
    (one weighted mean per query); the plain form is the natural API elsewhere.
    """
    if weights.dim() == points.dim() - 1:
        return torch.matmul(weights.unsqueeze(-2), points).squeeze(-2)
    return torch.matmul(weights, points)


def lorentz_centroid(
    weights: torch.Tensor,
    points: torch.Tensor,
    c: torch.Tensor | float,
) -> torch.Tensor:
    """Weighted Lorentz centroid (the manifold analogue of a weighted mean).

        m = sum_j w_j x_j ,   out = m / (sqrt(c) * |<m, m>_L|^(1/2))

    A convex combination of upper-sheet points is timelike, so ``<m,m>_L < 0``
    and the renormalisation lands exactly back on the manifold.

    Args:
        weights: ``(..., N)`` or ``(..., Q, N)`` non-negative weights.
        points:  ``(..., N, D+1)`` manifold points.
    """
    c = _as_tensor(c, points)
    m = _weighted_sum(weights, points)
    # <m,m>_L is negative for timelike m; take |.| before the sqrt and clamp so a
    # near-lightlike sum (possible under extreme attention) cannot divide by zero.
    denom = lorentz_inner(m, m, keepdim=True).abs().clamp_min(_EPS).sqrt()
    return m / (c.sqrt() * denom)


def klein_gyromidpoint(
    weights: torch.Tensor,
    points: torch.Tensor,
    c: torch.Tensor | float,
) -> torch.Tensor:
    """Einstein gyromidpoint, computed in Klein coordinates.

    This is the aggregation the reference spec describes via the Lorentz factor
    ``gamma = 1/sqrt(1 - c||v||^2)``. That factor is only meaningful in the
    Klein model, so convert first: ``k = x_s / (sqrt(c) x_0)`` puts a Lorentz
    point into the Klein ball of radius ``1/sqrt(c)``. Average with weights
    ``w_j gamma_j``, then map back.
    """
    c = _as_tensor(c, points)
    sqrt_c = c.sqrt()
    k = points[..., 1:] / (sqrt_c * points[..., :1])
    gamma = 1.0 / (1.0 - c * k.pow(2).sum(dim=-1, keepdim=True)).clamp_min(_EPS).sqrt()

    gamma_flat = gamma.squeeze(-1)
    if weights.dim() == points.dim():
        # weights carry a query axis; give gamma a matching one so it broadcasts
        # across queries rather than against a batch dimension.
        gamma_flat = gamma_flat.unsqueeze(-2)
    w = weights * gamma_flat
    m = _weighted_sum(w, k) / w.sum(dim=-1, keepdim=True).clamp_min(_EPS)

    m_time = 1.0 / (sqrt_c * (1.0 - c * m.pow(2).sum(dim=-1, keepdim=True)).clamp_min(_EPS).sqrt())
    return torch.cat([m_time, sqrt_c * m_time * m], dim=-1)


def signature_matrix(dim: int, device=None, dtype=None) -> torch.Tensor:
    """``M = diag(-1, 1, ..., 1)`` of size ``dim`` (the ambient dim ``D+1``).

    Lets the whole pairwise Lorentz inner product be a single matmul:
    ``Q M K^T``, so the attention scores hit the same fast matmul kernels as
    ordinary dot-product attention.
    """
    m = torch.ones(dim, device=device, dtype=dtype)
    m[0] = -1.0
    return torch.diag(m)


def _as_tensor(c: torch.Tensor | float, ref: torch.Tensor) -> torch.Tensor:
    """Curvature as a tensor on ``ref``'s device/dtype, shaped to broadcast."""
    if isinstance(c, torch.Tensor):
        return c.to(device=ref.device, dtype=ref.dtype)
    return torch.tensor(float(c), device=ref.device, dtype=ref.dtype)
