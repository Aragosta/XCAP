"""The S1/H2 model: sampling, hidden-degree calibration, and inverse temperature fitting.

S1 (isomorphic to H2 after the standard change of variables): N nodes are placed
uniformly on a circle of radius R = N / 2*pi, node i carries a hidden degree
kappa_i, and

    p_ij = 1 / (1 + (d_ij / (mu * kappa_i * kappa_j))**beta),   d_ij = R * dtheta_ij
    mu   = beta * sin(pi/beta) / (2*pi*<k>)                      (valid for beta > 1)

beta is the inverse temperature. beta -> 1+ is the hot limit, where the angular
term stops mattering and the ensemble degenerates to a soft configuration model;
beta > 1 is the geometric (clustered) phase; beta > 2D = 2 is the regime where
system-spanning long-range links vanish. Clustering increases monotonically with
beta, which is what makes the temperature identifiable from clustering alone.

Everything is O(N^2) but chunked and vectorised; N up to ~4e4 is a couple of
minutes per pass on CPU.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

CHUNK_PAIRS = 20_000_000


def mu_of(beta: float, kbar: float) -> float:
    if beta <= 1.0:
        # hot side: mu has a different closed form; the calibration loop fixes
        # the scale anyway, so use the beta -> 1+ limit as a starting point.
        beta = 1.0 + 1e-6
    return beta * np.sin(np.pi / beta) / (2.0 * np.pi * kbar)


def _row_chunk(n: int) -> int:
    return max(1, min(n, CHUNK_PAIRS // max(n, 1)))


def _angular_distance(t_i: np.ndarray, t_j: np.ndarray) -> np.ndarray:
    """Chord-wise angular separation in [0, pi]; float32 to halve the pass cost."""
    d = np.abs(t_i[:, None] - t_j[None, :]).astype(np.float32)
    return np.minimum(d, np.float32(2.0 * np.pi) - d)


def expected_degrees(kappa, theta, beta, mu, R):
    """Analytic E[k_i] under the S1 ensemble (chunked over rows)."""
    n = len(kappa)
    out = np.zeros(n)
    k32 = kappa.astype(np.float32)
    b32, one = np.float32(beta), np.float32(1.0)
    step = _row_chunk(n)
    for s in range(0, n, step):
        e = min(n, s + step)
        d = _angular_distance(theta[s:e], theta) * np.float32(R)
        d /= np.float32(mu) * k32[s:e, None]
        d /= k32[None, :]
        p = one / (one + d ** b32)
        np.fill_diagonal(p[:, s:e], np.float32(0.0))
        out[s:e] = p.sum(1, dtype=np.float64)
    return out


def calibrate_kappa(k_target, theta, beta, R, iters=10, rho=0.4, verbose=False):
    """Adjust hidden degrees so the ensemble reproduces the target degree sequence.

    Damped multiplicative updates with a per-step ratio clip: the undamped
    Mercator-style update (rho = 1) oscillates and diverges on heavy-tailed
    degree sequences at low mean degree, which is exactly our regime.
    """
    kappa = np.maximum(k_target.astype(np.float64), 0.5).copy()
    kbar = k_target.mean()
    mu = mu_of(beta, kbar)
    best, best_err = kappa.copy(), np.inf
    for it in range(iters):
        ek = expected_degrees(kappa, theta, beta, mu, R)
        err = float(np.abs(ek - k_target).mean())
        if err < best_err:
            best_err, best = err, kappa.copy()
        if verbose:
            print(f"    calib it{it} mean|dk|={err:.4f} <E k>={ek.mean():.3f}", flush=True)
        ratio = np.clip((k_target / np.maximum(ek, 1e-9)) ** rho, 0.5, 2.0)
        kappa = np.maximum(kappa * ratio, 1e-3)
    return best, mu


def sample_edges(kappa, theta, beta, mu, R, rng):
    """Draw one S1 graph; returns an upper-triangular edge list."""
    n = len(kappa)
    us, vs = [], []
    k32 = kappa.astype(np.float32)
    b32, one = np.float32(beta), np.float32(1.0)
    step = _row_chunk(n)
    cols = np.arange(n)
    for s in range(0, n, step):
        e = min(n, s + step)
        d = _angular_distance(theta[s:e], theta) * np.float32(R)
        d /= np.float32(mu) * k32[s:e, None]
        d /= k32[None, :]
        p = one / (one + d ** b32)
        # keep only j > i so each unordered pair is considered once
        p[np.arange(s, e)[:, None] >= cols[None, :]] = np.float32(0.0)
        hit = rng.random(p.shape, dtype=np.float32) < p
        ri, ci = np.nonzero(hit)
        us.append(ri + s)
        vs.append(ci)
    return np.concatenate(us).astype(np.int32), np.concatenate(vs).astype(np.int32)


def adjacency(u, v, n) -> sp.csr_matrix:
    a = sp.csr_matrix((np.ones(len(u), np.int8), (u, v)), shape=(n, n))
    a = a + a.T
    a.data[:] = 1
    a = a.tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    return a


def mean_clustering(a: sp.csr_matrix, chunk: int = 4000) -> float:
    """Mean local clustering coefficient over nodes of degree >= 2."""
    deg = np.asarray(a.sum(1)).ravel().astype(np.float64)
    af = a.astype(np.float32)
    tri = np.zeros(a.shape[0])
    for s in range(0, a.shape[0], chunk):
        e = min(a.shape[0], s + chunk)
        blk = (af[s:e] @ af).multiply(af[s:e])
        tri[s:e] = np.asarray(blk.sum(1)).ravel()
    ok = deg >= 2
    c = np.zeros_like(tri)
    c[ok] = tri[ok] / (deg[ok] * (deg[ok] - 1.0))
    return float(c[ok].mean())


def clustering_at_beta(k_target, beta, rng, calib_iters=4, reps=1):
    """Mean clustering of S1 samples with a calibrated degree sequence."""
    n = len(k_target)
    R = n / (2.0 * np.pi)
    theta = np.sort(rng.random(n) * 2.0 * np.pi)
    kappa, mu = calibrate_kappa(k_target, theta, beta, R, iters=calib_iters)
    cs, degs = [], []
    for _ in range(reps):
        u, v = sample_edges(kappa, theta, beta, mu, R, rng)
        a = adjacency(u, v, n)
        cs.append(mean_clustering(a))
        degs.append(float(np.asarray(a.sum(1)).mean()))
    return float(np.mean(cs)), float(np.mean(degs)), kappa, theta


def fit_beta(k_target, c_target, rng, lo=1.05, hi=6.0, tol=0.002, max_iter=12,
             verbose=True):
    """Bisection on beta to match a target mean clustering coefficient."""
    trace = []
    c_lo, _, _, _ = clustering_at_beta(k_target, lo, rng)
    c_hi, _, _, _ = clustering_at_beta(k_target, hi, rng)
    trace += [(lo, c_lo), (hi, c_hi)]
    if verbose:
        print(f"    beta={lo:.3f} c={c_lo:.4f} | beta={hi:.3f} c={c_hi:.4f} "
              f"| target c={c_target:.4f}", flush=True)
    if c_target < c_lo:
        return lo, trace, "below_range"
    if c_target > c_hi:
        return hi, trace, "above_range"
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        c_mid, _, _, _ = clustering_at_beta(k_target, mid, rng)
        trace.append((mid, c_mid))
        if verbose:
            print(f"    beta={mid:.3f} c={c_mid:.4f}", flush=True)
        if abs(c_mid - c_target) < tol:
            return mid, trace, "converged"
        if c_mid < c_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), trace, "max_iter"


def fit_gamma(degrees: np.ndarray, kmin_grid=None):
    """Discrete power-law MLE with a KS-minimising lower cutoff (Clauset et al.)."""
    d = np.asarray(degrees)
    d = d[d > 0]
    if kmin_grid is None:
        kmin_grid = np.unique(np.percentile(d, np.arange(50, 99, 1)).astype(int))
        kmin_grid = kmin_grid[kmin_grid >= 2]
    best = None
    for kmin in kmin_grid:
        x = d[d >= kmin]
        if len(x) < 200:
            continue
        # continuous approximation with the standard 1/2 correction
        gamma = 1.0 + len(x) / np.sum(np.log(x / (kmin - 0.5)))
        xs = np.sort(x)
        emp = np.arange(1, len(xs) + 1) / len(xs)
        theo = 1.0 - (xs / (kmin - 0.5)) ** (1.0 - gamma)
        ks = np.max(np.abs(emp - theo))
        if best is None or ks < best[2]:
            best = (float(gamma), int(kmin), float(ks), int(len(x)))
    return dict(gamma=best[0], k_min=best[1], ks=best[2], n_tail=best[3])
