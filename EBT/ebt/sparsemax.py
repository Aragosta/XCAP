"""Sparsemax: Euclidean projection onto the probability simplex.

Reference: Martins & Astudillo, "From Softmax to Sparsemax" (ICML 2016).

    sparsemax(z) = argmin_{p in simplex} ||p - z||^2

Unlike softmax, the solution has *exact* zeros: only the coordinates above a
threshold tau(z) survive.  The Jacobian is supported only on that set, which is
what makes it usable as a drop-in normaliser inside attention.

Masking convention
------------------
Positions where ``mask`` is False are replaced by a large finite negative
sentinel rather than -inf.  A -inf would poison the cumulative sums used by the
threshold search; the sentinel is provably never selected into the support (see
tests/test_sparsemax.py::test_masked_positions_never_in_support).
"""

from __future__ import annotations

import torch
from torch import Tensor

NEG_SENTINEL = -1e9


def _threshold_and_support(z: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    """Return (tau, support_size) for the sparsemax projection along ``dim``."""
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    cumsum = z_sorted.cumsum(dim) - 1.0
    rhos = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim()
    shape[dim] = -1
    rhos = rhos.view(shape)

    support = (rhos * z_sorted) > cumsum          # k such that 1 + k*z_(k) > sum_{j<=k} z_(j)
    support_size = support.sum(dim=dim, keepdim=True)
    tau = cumsum.gather(dim, support_size - 1) / support_size.to(z.dtype)
    return tau, support_size


class _SparsemaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: Tensor, dim: int) -> Tensor:
        # sparsemax is translation invariant; shift for numerical stability.
        z = z - z.max(dim=dim, keepdim=True).values
        tau, _ = _threshold_and_support(z, dim)
        out = torch.clamp(z - tau, min=0.0)
        ctx.save_for_backward(out)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (out,) = ctx.saved_tensors
        dim = ctx.dim
        nonzeros = (out > 0).to(grad_out.dtype)
        # J = diag(s) - s s^T / |S|  restricted to the support S
        masked_grad = grad_out * nonzeros
        v = masked_grad.sum(dim=dim, keepdim=True) / nonzeros.sum(dim=dim, keepdim=True).clamp(min=1)
        return nonzeros * (grad_out - v), None


def sparsemax(z: Tensor, dim: int = -1, mask: Tensor | None = None) -> Tensor:
    """Sparsemax along ``dim``.  ``mask`` is True for positions that may be kept."""
    if mask is not None:
        z = z.masked_fill(~mask, NEG_SENTINEL)
    return _SparsemaxFunction.apply(z, dim)


def masked_softmax(z: Tensor, dim: int = -1, mask: Tensor | None = None) -> Tensor:
    """Softmax with the same masking signature as :func:`sparsemax`."""
    if mask is not None:
        z = z.masked_fill(~mask, float("-inf"))
    out = torch.softmax(z, dim=dim)
    if mask is not None:
        out = torch.nan_to_num(out, nan=0.0)  # rows that are fully masked
    return out


NORMALISERS = {"softmax": masked_softmax, "sparsemax": sparsemax}


def get_normaliser(name: str):
    if name not in NORMALISERS:
        raise ValueError(f"unknown normaliser {name!r}; expected one of {sorted(NORMALISERS)}")
    return NORMALISERS[name]
