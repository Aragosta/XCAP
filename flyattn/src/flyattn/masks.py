"""Attention masks generated from the S1/H2 model, and the connectome-derived
and rewired masks used as comparators.

The sequence position *is* the angular coordinate: node i sits at angle
2*pi*i/T on the circle. Then the inverse temperature beta interpolates between
two familiar architectures rather than between "structured" and "random":

    beta -> 1+      the angular term stops constraining anything and the mask is
                    a configuration-model random sparse mask (the hot limit, which
                    is the null model - not ER, not "random shuffle")
    beta large      the mask collapses onto a local band, i.e. sliding-window
                    attention with a heavy-tailed window-size distribution

and beta = 2D = 2 is the boundary above which system-spanning long-range links
disappear. Degree sequence and density are held fixed across the sweep, so the
only thing changing is where the same number of edges is placed.
"""
from __future__ import annotations

import numpy as np
import torch

from . import s1h2


def powerlaw_degrees(n, mean_degree, gamma, rng, k_min=1.0):
    """Continuous power-law hidden degrees with a prescribed mean."""
    u = rng.random(n)
    k = k_min * (1.0 - u) ** (-1.0 / (gamma - 1.0))
    k = np.clip(k, k_min, n - 1.0)
    return k * (mean_degree / k.mean())


def s1_mask(seq_len, beta, mean_degree, gamma, rng, calib_iters=6,
            positions_as_angles=True):
    """Sample an S1 graph on `seq_len` nodes and return a boolean mask.

    Degrees are calibrated so the realised mean degree matches `mean_degree` at
    every beta, which is what makes the sweep a pure structure sweep.
    """
    n = seq_len
    kt = powerlaw_degrees(n, mean_degree, gamma, rng)
    theta = (np.arange(n) / n * 2 * np.pi if positions_as_angles
             else np.sort(rng.random(n) * 2 * np.pi))
    R = n / (2 * np.pi)
    kappa, mu = s1h2.calibrate_kappa(kt, theta, beta, R, iters=calib_iters)
    u, v = s1h2.sample_edges(kappa, theta, beta, mu, R, rng)
    m = np.zeros((n, n), bool)
    m[u, v] = True
    m[v, u] = True
    np.fill_diagonal(m, True)
    return torch.from_numpy(m)


def configuration_mask(seq_len, mean_degree, gamma, rng):
    """Hot-limit reference: same degree sequence, no geometry at all."""
    n = seq_len
    k = np.round(powerlaw_degrees(n, mean_degree, gamma, rng)).astype(int)
    if k.sum() % 2:
        k[np.argmax(k)] += 1
    stubs = np.repeat(np.arange(n), k)
    rng.shuffle(stubs)
    m = np.zeros((n, n), bool)
    a, b = stubs[0::2], stubs[1::2]
    ok = a != b
    m[a[ok], b[ok]] = True
    m[b[ok], a[ok]] = True
    np.fill_diagonal(m, True)
    return torch.from_numpy(m)


def window_mask(seq_len, half_width):
    i = np.arange(seq_len)
    m = np.abs(i[:, None] - i[None, :]) <= half_width
    return torch.from_numpy(m)


def global_token_mask(seq_len, n_global, kind="symmetric", base=None, rng=None):
    """BigBird-style global tokens (T8).

    kind="symmetric"   global tokens read everything and are read by everything
    kind="asymmetric"  half are integrators (full read, restricted write) and half
                       broadcasters (restricted read, full write), the split the
                       M3 census says the fly actually uses
    """
    m = (torch.zeros(seq_len, seq_len, dtype=torch.bool) if base is None
         else base.clone())
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(seq_len, size=n_global, replace=False)
    if kind == "symmetric":
        m[idx, :] = True
        m[:, idx] = True
    else:
        half = len(idx) // 2
        integ, broad = idx[:half], idx[half:]
        m[integ, :] = True          # integrators read everything
        m[:, broad] = True          # broadcasters are read by everything
    m.fill_diagonal_(True)
    return m


def density(mask: torch.Tensor, causal=True) -> float:
    if causal:
        c = torch.tril(torch.ones_like(mask))
        return float((mask & c.bool()).sum()) / float(c.sum())
    return float(mask.float().mean())


def ensure_reachable(mask: torch.Tensor) -> torch.Tensor:
    """Guarantee every causal row has at least one allowed key (itself)."""
    mask = mask.clone()
    mask.fill_diagonal_(True)
    return mask
