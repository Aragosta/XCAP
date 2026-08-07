"""
FFD.py — fractionally differenced features, and the search for the d that buys
stationarity at the smallest possible cost in memory.

WHAT THIS IS
------------
A price series is I(1): you cannot regress on it, and every stationarity fix in common
use — returns, log returns, z-scores of returns — is a first difference, which is
d = 1, which throws away the ENTIRE level. López de Prado's point is that d = 1 is
overkill. Stationarity is usually reached somewhere around d ≈ 0.3–0.5, and everything
between that d and 1 is memory you deleted for nothing.

The transform is the binomial expansion of (1 − B)^d with real d:

    x_t = Σ_{k≥0} w_k · y_{t−k},     w_0 = 1,   w_k = −w_{k−1} · (d − k + 1) / k

At integer d the series terminates (d = 1 → [1, −1], the plain difference). At
fractional d it does not: w_k ~ k^(−d−1)/Γ(−d), a power-law tail. FFD (fixed-width
window) truncates it at |w_k| < THRES and applies the SAME finite window at every t, so
every observation is driven by the same weight vector. The alternative — an expanding
window, w renormalised each t — is not translation invariant: early observations get a
different filter than late ones, which is precisely the drift you were trying to remove.
This file only implements FFD.

WHAT "OPTIMAL d" MEANS HERE
---------------------------
    d* = min { d : ADF(x(d)) rejects a unit root at ALPHA }

Minimum, not argmax of anything. Both criteria point the same way and only one of them
is a real constraint:

  * stationarity is a THRESHOLD — the ADF statistic falls monotonically in d, so there
    is a first d that passes, and every larger d also passes.
  * memory is MONOTONE DECREASING in d — corr(x(d), y) falls from 1.0 at d = 0 to
    ~0.0 at d = 1 without a local optimum to find.

So there is nothing to optimise jointly. The binding constraint picks d, memory is the
thing being conserved, and any "score" combining them would only be a disguised choice
of ALPHA. The one real judgement call is ALPHA: 5% is the default; 1% buys a bigger d.

Because the ADF stat is monotone in d, the grid scan is followed by a BISECTION between
the last failing rung and the first passing one — REFINE halvings take d to grid/2^R
resolution for R extra ADF fits, not a finer grid over the whole range.

ZERO SUM — why the standard truncation is origin-dependent
----------------------------------------------------------
The exact weights sum to (1 − 1)^d = 0 for every d > 0. Truncated at |w_k| < THRES
they do not, and the surviving mass S = Σ_{k<K} w_k sits on the level y_{t−K+1}. Two
consequences, one cosmetic and one not:

  * NOT COSMETIC. x_t is then not invariant to the origin of y. Add c to y and every
    x_t moves by S·c. The origin of log ADJUSTED price is arbitrary — it is set by the
    back-adjustment base, not by anything about the asset — so a feature that depends
    on it is partly measuring the vendor's bookkeeping. Cross-sectionally the leaked
    term is a stale log price LEVEL, which is share-count arbitrary. At d = 0.31,
    THRES 1e-5, it is 35% of the feature's variance (284 securities, ≥3000 bars).
  * The fix is exact and free. The dropped tail is Σ_{k≥K} w_k·y_{t−k}, whose weights
    sum to −S; approximating y_{t−k} ≈ y_{t−K+1} across it gives x_t − S·y_{t−K+1},
    which is what ZERO_SUM subtracts. Plain truncation approximates that same tail by
    ZERO — i.e. by the arbitrary origin. So zero-sum is not a correction bolted on top
    of FFD, it is a strictly better approximation of the filter FFD is truncating, and
    it makes Σw exactly 0 and x exactly origin-invariant. It is ON by default;
    ZERO_SUM = False reproduces the published FFD.

`leak` is still reported everywhere: with ZERO_SUM it is no longer a defect, it is the
size of the tail being approximated, and so a direct read on how much of the filter the
window is actually carrying.

WHAT THE ADF VERDICT IS AND IS NOT — measured, on random walks
--------------------------------------------------------------
A fixed-K zero-sum filter of a random walk is a finite MA of iid increments, so it is
STATIONARY for every d > 0, exactly. The window, not d, is doing the work. What the
ADF is really reporting at d < 0.5 is whether the remaining persistence is detectable
over the sample, not a population property. 200 random walks, n = 6000, share declared
stationary at 5%:

    d          0.20    0.30    0.40    0.50    0.60
    truncated  52.0%   86.5%  100.0%  100.0%  100.0%
    zero-sum   38.5%   70.5%   99.5%  100.0%  100.0%

The exact filter would give I(1−d), stationary only for d > 0.5, so everything left of
that column is the finite window. Note what this does and does not blame: dropping the
leak entirely moves d = 0.3 from 86.5% to 70.5%, so the leak is a MINORITY of it. This
is a property of the published method, not of this implementation.

So do not read d* = 0.3 as "prices are I(0.3)" — `gph()` puts this panel at d ≈ 0.98,
i.e. I(1), and prints alongside d* in `scan` for exactly that reason. Read it as the
smallest d at which a K-wide window leaves no ADF-detectable unit root over this
sample. For a FEATURE that is a perfectly usable calibration. For an inference about
the price process it is not.

BURN-IN vs MEMORY — the trade that is actually binding
------------------------------------------------------
The first K−1 bars of every asset are NaN. THRES is the dial, and it does not trade
burn-in against the leak (ZERO_SUM already handled that) — it trades burn-in against
MEMORY, which is the whole point of the method. Measured on this panel, d* refit per
rung (pooled over 1200 sampled names):

    THRES    d*       K     rows kept
    1e-3    0.050     41       98.8%
    1e-4    0.167    521       84.8%
    1e-5    0.322   2068       49.0%

There is no free lunch in that table: FFD's memory IS the tail, so buying back rows
sells the memory that motivated the transform. But the memory is worth less than it
looks — see the next section — and the burn-in costs more than it looks, because the
rows it deletes are not a random half. Scoring the CONTROLS on all rows vs on the rows
the filter survives (section 5 of `diag` recomputes this every run):

                          K = 2068 (THRES 1e-5)      K = 518 (THRES 1e-4, default)
    madist_p  h63          0.0230 → 0.0050            0.0199 → 0.0131
    mom12_1   h63          0.0211 → 0.0027            0.0217 → 0.0132

Same features, same screen, only the row set differs. The surviving subsample is
HARDER, not merely smaller — it is long-history liquid names, the least inefficient
slice of the universe — and at K = 2068 it costs both controls ~4x at h63. That is the
whole case for THRES = 1e-4: per row the two widths are a tie, so the narrow window was
paying a 4x selection tax for memory that buys nothing.

DOES IT BEAT THE MATCHED CONTROL? — no, and that is the headline
----------------------------------------------------------------
The comparison this file used to make was against `madist_*`, a 252-bar rolling
z-score, while the filter under test was K ≈ 2068 bars wide. That hands FFD 8x the
lookback and credits the win to fractional differencing. `maK_*` fixes it: same K,
minp = K so it burns in identically, only the WEIGHT PROFILE differs — FFD's power-law
tail against a flat mean. That is the entire claim of the method. Paired NW t of the
IC difference, positive = FFD wins, each on its own common support:

    K        ffd − maK  h1        h63          ffdz − maK  h1        h63
      41      +0.0037 (t 1.82)  −0.0000 (0.00)   +0.0038 (2.05)  +0.0035 (1.18)
     521      −0.0012 (−0.59)   −0.0070 (−1.15)  +0.0013 (0.72)  +0.0054 (1.15)
    2068      −0.0071 (−3.22)   −0.0123 (−2.35)  −0.0055 (−2.38) −0.0058 (−1.35)

At the width where FFD is supposed to be interesting it LOSES, significantly. At
K = 521 it is a coin flip. Only at K = 41 — where there is essentially no fractional
tail left to speak of — does it edge ahead. Cross-sectional rank corr of ffdz with its
own maK is +0.81 at both 521 and 2068: it is the same signal, differently weighted.
At the shipped default `diag` reproduces the coin flip on the full panel: ffd_p − maK_p
runs −0.0018/−0.0002/−0.0019/−0.0068 (|t| ≤ 1.12), ffdz_p − maK_p −0.0006/+0.0027/
+0.0039/+0.0048 (|t| ≤ 1.32). Nothing significant in either direction.

And the control tells you where the signal actually was. maK_p − madist_p — two flat
rolling z-scores differing only in window, 518 bars against 252 — is +0.0042 (t 2.51)
at h1 and +0.0047 (t 2.38) at h5. LOOKBACK is what moved the IC. Fractional weighting,
tested against its own window, moved nothing. Every "FFD wins" this file used to print
was that lookback difference wearing the transform's name.

The volume leg looked like the exception and is not. Against the mismatched control it
wins big (ffdz_v5 − ma252_v = +0.0109, t 2.95 at h63); against maK_v5 it is
−0.0044 (t −1.15), and maK_v5 outscores it outright (0.0217 vs 0.0173). The entire
apparent edge was the 252-vs-2804 lookback gap.

What survives is the ORIGINAL López de Prado claim, which is about d = 1, not about the
fractional tail: plain returns carry nothing cross-sectionally (ret1 IC ≈ 0 to −0.005)
while a memory-carrying level of log price carries ~0.010–0.015. FFD gets you such a
level. So does subtracting a moving average, for one rolling window and no ADF search.

ONE d, OR ONE PER ASSET?
------------------------
Both are emitted, and they are not interchangeable:

    ffd     the POOLED d (median of the per-asset d*). This is the cross-sectional
            feature. Two assets filtered at different d are on different scales — the
            variance and the autocorrelation of x both depend on d — so ranking them
            against each other is ranking their filters as much as their prices.
    ffd_own the per-asset d*. Use for per-asset time-series work (a regressor in an
            asset's own model), never for a cross-sectional rank.

`d_own` is emitted alongside so the choice stays visible and reversible.

SPEED
-----
The cost is the convolution: O(T · K) per (asset, d), and K is large exactly where the
transform is interesting — at THRES 1e-5, d = 0.2 needs ~3.4k weights. Three things
pay for it:

  * EARLY EXIT. The grid is scanned ascending and stops at the first d that passes.
    d* is typically the 3rd–6th rung, not the 20th.
  * njit + prange over assets for the direct convolution and for every ADF fit; the
    ADF's lag selection reuses ONE cross-product matrix over all candidate lags, so
    autolag costs O(T·p²) once instead of once per lag.
  * an FFT path (numpy, O(T log T)) that takes over when K is large, fanned out over
    assets with joblib threads — numpy's FFT drops the GIL, so threads are real
    parallelism with no panel copy per worker.

    python research/FFD.py scan      # d-curve + d* distribution on a sample
    python research/FFD.py build     # features → data/_staging/ffd.parquet
    python research/FFD.py validate  # ADF vs statsmodels, FFT vs direct, zero-sum
    python research/FFD.py bench
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit, prange

PANEL = "data/_staging/alpha_panel.parquet"
OUT = "data/_staging/ffd.parquet"

DGRID = np.round(np.arange(0.0, 1.0001, 0.05), 4)   # ascending; early exit makes the
                                                    # tail rungs free when d* is small
THRES = 1e-4            # weight truncation. Smaller → longer memory, wider window.
                        # 1e-4 (d* ≈ 0.17, K ≈ 521) rather than 1e-5 (K ≈ 2068): on
                        # COMMON rows the two are a statistical tie (ffdz Δ +0.0033,
                        # t 1.04 at h1), but 1e-4 scores 14.5M rows over 5,170 names
                        # against 8.5M over 3,343. The gain is COVERAGE, not a better
                        # filter, and it is the difference between a feature that runs
                        # on the universe and one that runs on long-history survivors.
MAX_WIDTH = 4096        # hard cap on K, so a tiny d cannot allocate an unbounded filter
ALPHA = 0.05            # ADF size. The one judgement call in the whole file.
ZERO_SUM = True         # anchor the truncated tail at y_{t−K+1} so Σw = 0 exactly and
                        # x is origin-invariant. See ZERO SUM. False = published FFD.
REFINE = 6              # bisections after the grid → resolution 0.05/64 ≈ 0.0008
MIN_OBS = 252           # a d* from less than a year of bars is noise
SAMPLE = 1500           # assets used by `scan` to fix the pooled d
FFT_MIN_K = 256         # above this width the FFT path wins

HORIZONS = (1, 5, 21, 63)   # forward-return horizons written to the parquet
ZWIN = 252              # trailing window for the vol-normalised reductions

# ── weights ─────────────────────────────────────────────────────────────────
@njit(cache=True)
def _weights(d, thres, cap, zero_sum):
    """(w, tail). FFD weights truncated at |w_k| < thres, NEWEST-first — w[0] = w_0
    multiplies y_t, which is the order the convolution wants.

    `tail` is Σw BEFORE any correction: the mass the exact filter puts beyond the
    window. With zero_sum it is subtracted from the OLDEST weight, which is the same
    thing as approximating the dropped tail by its nearest available observation
    (see ZERO SUM in the header) and leaves Σw = 0 to machine precision. It is returned
    either way because it measures how much of the filter the window is carrying."""
    w = np.empty(cap, dtype=np.float64)
    w[0] = 1.0
    k = 1
    while k < cap:
        v = -w[k - 1] * (d - k + 1.0) / k
        if abs(v) < thres:
            break
        w[k] = v
        k += 1
    w = w[:k]
    tail = 0.0
    for i in range(k):
        tail += w[i]
    if zero_sum:
        w[k - 1] -= tail
    return w, tail


def ffd_weights(d: float, thres: float = THRES, cap: int = MAX_WIDTH,
                zero_sum: bool = ZERO_SUM) -> np.ndarray:
    """Public wrapper. w[k] multiplies y_{t−k}."""
    return _weights(float(d), float(thres), int(cap), zero_sum)[0]


# ── transform ───────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=True, inline="always")
def _ffd_into(y, w, out):
    """out[t] = Σ_k w[k]·y[t−k]; NaN for the first K−1 bars, which have no full window."""
    n = y.shape[0]
    K = w.shape[0]
    for t in range(min(K - 1, n)):
        out[t] = np.nan
    for t in range(K - 1, n):
        acc = 0.0
        for k in range(K):
            acc += w[k] * y[t - k]
        out[t] = acc


@njit(cache=True, parallel=True, fastmath=True)
def _ffd_panel_direct(y, starts, ends, w, out):
    for a in prange(starts.shape[0]):
        _ffd_into(y[starts[a]:ends[a]], w, out[starts[a]:ends[a]])


def _next_fast(n: int) -> int:
    """Smallest 5-smooth integer ≥ n. numpy's FFT is fastest on these."""
    m = 1
    while m < n:
        m *= 2
    best = m
    f2 = 1
    while f2 < 2 * n:
        f3 = f2
        while f3 < 2 * n:
            f5 = f3
            while f5 < 2 * n:
                if n <= f5 < best:
                    best = f5
                f5 *= 5
            f3 *= 3
        f2 *= 2
    return best


def _ffd_fft_into(y, w, out):
    """Same result as `_ffd_into` in O(T log T). Used when K is large; numpy's FFT
    releases the GIL, which is what makes the joblib thread fan-out below worth having."""
    n, K = y.shape[0], w.shape[0]
    if n < K:
        out[:] = np.nan
        return
    m = _next_fast(n + K - 1)
    z = np.fft.irfft(np.fft.rfft(y, m) * np.fft.rfft(w, m), m)[:n]
    out[:K - 1] = np.nan
    out[K - 1:] = z[K - 1:]


def ffd(y: np.ndarray, d: float, thres: float = THRES, cap: int = MAX_WIDTH,
        w: np.ndarray | None = None, zero_sum: bool = ZERO_SUM) -> np.ndarray:
    """Fractionally difference one series. Picks the direct or FFT path by width."""
    y = np.ascontiguousarray(y, dtype=np.float64)
    w = _weights(float(d), float(thres), int(cap), zero_sum)[0] if w is None else w
    out = np.empty(y.shape[0], dtype=np.float64)
    if w.shape[0] >= FFT_MIN_K and y.shape[0] > 512:
        _ffd_fft_into(y, w, out)
    else:
        _ffd_into(y, w, out)
    return out


def ffd_panel(y, starts, ends, d, thres=THRES, cap=MAX_WIDTH, n_jobs=-1,
              zero_sum=ZERO_SUM):
    """Transform a stacked panel at ONE d. Direct+prange for narrow filters, FFT over
    joblib threads for wide ones."""
    w = _weights(float(d), float(thres), int(cap), zero_sum)[0]
    out = np.empty(y.shape[0], dtype=np.float64)
    if w.shape[0] < FFT_MIN_K:
        _ffd_panel_direct(y, starts, ends, w, out)
        return out, w
    Parallel(n_jobs=n_jobs, backend="threading", batch_size=64)(
        delayed(_ffd_fft_into)(y[s:e], w, out[s:e]) for s, e in zip(starts, ends))
    return out, w


# ── ADF ─────────────────────────────────────────────────────────────────────
# Δy_t = α + γ·y_{t−1} + Σ_{j=1..p} δ_j·Δy_{t−j} + ε_t ,  H0: γ = 0, statistic t(γ).
# Design column order is [y_{t−1}, Δy_{t−1} … Δy_{t−p}, 1] so that a p' < p model is
# the leading block plus the last column — one cross-product matrix serves every
# candidate lag, which is what makes AIC selection cheap enough to run per (asset, d).

# MacKinnon (2010) response-surface constants for τ_c (constant, no trend), N = 1:
# crit = b∞ + b1/T + b2/T² + b3/T³, at 1% / 5% / 10%.
_MK_C = np.array([
    [-3.43035, -6.5393, -16.786, -79.433],
    [-2.86154, -2.8903,  -4.234, -40.040],
    [-2.56677, -1.5384,  -2.809,   0.000],
], dtype=np.float64)


@njit(cache=True, inline="always")
def _adf_crit(nobs, level):
    """level: 0 = 1%, 1 = 5%, 2 = 10%."""
    t = 1.0 / nobs
    b = _MK_C[level]
    return b[0] + b[1] * t + b[2] * t * t + b[3] * t * t * t


@njit(cache=True, inline="always")
def _chol(A, p, L):
    """A → lower Cholesky in L. False if not positive definite."""
    for j in range(p):
        s = A[j, j]
        for k in range(j):
            s -= L[j, k] * L[j, k]
        if s <= 1e-300:
            return False
        L[j, j] = np.sqrt(s)
        for i in range(j + 1, p):
            t = A[i, j]
            for k in range(j):
                t -= L[i, k] * L[j, k]
            L[i, j] = t / L[j, j]
    return True


@njit(cache=True, inline="always")
def _chol_solve(L, b, p, x):
    for i in range(p):                                   # forward
        s = b[i]
        for k in range(i):
            s -= L[i, k] * x[k]
        x[i] = s / L[i, i]
    for i in range(p - 1, -1, -1):                       # back
        s = x[i]
        for k in range(i + 1, p):
            s -= L[k, i] * x[k]
        x[i] = s / L[i, i]


@njit(cache=True)
def _adf_normal(x, lag, start, XtX, Xty):
    """Cross-products for the ADF design on rows [start, n−1) of Δx. Returns (yty, nrow).

    `start` fixes the sample: lag selection must compare models on IDENTICAL rows or the
    information criterion is comparing sample sizes, not fits."""
    n = x.shape[0]
    p = lag + 2
    for i in range(p):
        Xty[i] = 0.0
        for j in range(p):
            XtX[i, j] = 0.0
    yty = 0.0
    nrow = 0
    for i in range(start, n - 1):
        dyi = x[i + 1] - x[i]
        if not np.isfinite(dyi) or not np.isfinite(x[i]):
            continue
        ok = True
        for j in range(1, lag + 1):
            if not np.isfinite(x[i - j + 1]) or not np.isfinite(x[i - j]):
                ok = False
                break
        if not ok:
            continue
        for a in range(p):
            va = x[i] if a == 0 else (1.0 if a == p - 1
                                      else x[i - a + 1] - x[i - a])
            Xty[a] += va * dyi
            for b in range(a, p):
                vb = x[i] if b == 0 else (1.0 if b == p - 1
                                          else x[i - b + 1] - x[i - b])
                XtX[a, b] += va * vb
        yty += dyi * dyi
        nrow += 1
    for a in range(p):
        for b in range(a):
            XtX[a, b] = XtX[b, a]
    return yty, nrow


@njit(cache=True, inline="always")
def _adf_solve(XtX, Xty, yty, nrow, p, L, beta, rhs):
    """t(γ) and ssr from the normal equations. γ is coefficient 0 by construction."""
    if nrow <= p + 1 or not _chol(XtX, p, L):
        return np.nan, np.nan
    for i in range(p):
        rhs[i] = Xty[i]
    _chol_solve(L, rhs, p, beta)
    ssr = yty
    for i in range(p):
        ssr -= beta[i] * Xty[i]
    if ssr <= 0.0:
        return np.nan, np.nan
    for i in range(p):
        rhs[i] = 1.0 if i == 0 else 0.0
    _chol_solve(L, rhs, p, rhs)                          # rhs ← (XtX)⁻¹ e₀
    v = ssr / (nrow - p) * rhs[0]
    return (beta[0] / np.sqrt(v) if v > 0.0 else np.nan), ssr


@njit(cache=True)
def _adf(x, maxlag, autolag):
    """ADF t-statistic with a constant. autolag: 1 = AIC over 0…maxlag, 0 = fixed maxlag.

    AIC selects on the maxlag-truncated sample (all candidates see the same rows), then
    the winner is REFIT on its own larger sample — the statsmodels convention."""
    n = x.shape[0]
    if n < maxlag + 12:
        return np.nan, -1, 0
    P = maxlag + 2
    XtX = np.empty((P, P), dtype=np.float64)
    Xty = np.empty(P, dtype=np.float64)
    L = np.zeros((P, P), dtype=np.float64)
    beta = np.empty(P, dtype=np.float64)
    rhs = np.empty(P, dtype=np.float64)
    yty, nrow = _adf_normal(x, maxlag, maxlag, XtX, Xty)
    best = maxlag
    if autolag == 1:
        A = np.empty((P, P), dtype=np.float64)
        c = np.empty(P, dtype=np.float64)
        bic = np.inf
        for k in range(maxlag + 1):
            p = k + 2
            for i in range(p):                            # leading block + const column
                si = i if i < p - 1 else P - 1
                c[i] = Xty[si]
                for j in range(p):
                    sj = j if j < p - 1 else P - 1
                    A[i, j] = XtX[si, sj]
            _, ssr = _adf_solve(A, c, yty, nrow, p, L, beta, rhs)
            if np.isfinite(ssr):
                aic = nrow * np.log(ssr / nrow) + 2.0 * p
                if aic < bic:
                    bic = aic
                    best = k
    if best != maxlag:
        yty, nrow = _adf_normal(x, best, best, XtX, Xty)
    t, _ = _adf_solve(XtX, Xty, yty, nrow, best + 2, L, beta, rhs)
    return t, best, nrow


def adf(x: np.ndarray, maxlag: int | None = None, autolag: bool = True):
    """(stat, usedlag, nobs, crit5). Constant, no trend. NaNs are dropped by the kernel."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    ml = _schwert(x.shape[0]) if maxlag is None else int(maxlag)
    t, lag, nobs = _adf(x, ml, 1 if autolag else 0)
    return t, lag, nobs, (_adf_crit(nobs, 1) if nobs else np.nan)


def _schwert(n: int, cap: int = 20) -> int:
    """Schwert's rule, capped. The cap is a speed knob: the ADF cross-product is O(T·p²)
    and beyond ~20 lags it dominates the convolution without moving the statistic."""
    return max(1, min(cap, int(np.ceil(12.0 * (max(n, 1) / 100.0) ** 0.25))))


# ── the search ──────────────────────────────────────────────────────────────
@njit(cache=True, inline="always")
def _corr_valid(a, b):
    """Pearson correlation over rows where BOTH are finite — the memory measure."""
    n = a.shape[0]
    sa = sb = saa = sbb = sab = 0.0
    m = 0
    for i in range(n):
        if np.isfinite(a[i]) and np.isfinite(b[i]):
            sa += a[i]; sb += b[i]
            saa += a[i] * a[i]; sbb += b[i] * b[i]; sab += a[i] * b[i]
            m += 1
    if m < 3:
        return np.nan
    va = saa - sa * sa / m
    vb = sbb - sb * sb / m
    if va <= 0.0 or vb <= 0.0:
        return np.nan
    return (sab - sa * sb / m) / np.sqrt(va * vb)


@njit(cache=True, inline="always")
def _eval_d(y, d, thres, cap, maxlag, autolag, zero_sum, buf):
    """(adf stat, corr with y, width, nobs, leak) at one d."""
    w, leak = _weights(d, thres, cap, zero_sum)
    K = w.shape[0]
    if K > y.shape[0]:
        return np.nan, np.nan, K, 0, leak
    _ffd_into(y, w, buf)
    t, _, nobs = _adf(buf[K - 1:], maxlag, autolag)
    return t, _corr_valid(buf, y), K, nobs, leak


@njit(cache=True)
def _optimal_d_one(y, dgrid, thres, cap, level, maxlag, autolag, refine, min_obs,
                   zero_sum):
    """Smallest d on the grid whose FFD series rejects a unit root, then bisected.

    Returns (d, stat, crit, corr, width, nobs, leak). d = NaN when no rung qualifies —
    a real outcome, not an error to paper over with a fallback.

    There is no admissibility gate on `leak` here, and an earlier version of this file
    was wrong to have one. The reasoning was that a rejection under a leaky window is
    the window talking rather than the series; the reasoning is right but the leak is
    not the mechanism — removing it entirely still calls 70.5% of random walks
    stationary at d = 0.3, against 86.5% with it (see WHAT THE ADF VERDICT IS AND IS
    NOT). The finite window is doing that, so gating on leak bought accuracy that was
    not there and cost real resolution in d. ZERO_SUM removes the leak for the reason
    that IS defensible — origin invariance — and the ADF's limits are documented
    instead of being papered over with a floor.

    The bisection is valid because t(γ) is monotone decreasing in d; the grid brackets
    the crossing and REFINE halvings sharpen it. Both endpoints stay ones the ADF
    actually accepted, so the returned stat always belongs to the returned d."""
    n = y.shape[0]
    buf = np.empty(n, dtype=np.float64)
    if n < min_obs:
        return np.nan, np.nan, np.nan, np.nan, 0.0, 0.0, np.nan
    ml = maxlag
    if ml <= 0:
        ml = max(1, min(20, int(np.ceil(12.0 * (n / 100.0) ** 0.25))))
    lo = -1.0
    for g in range(dgrid.shape[0]):
        d = dgrid[g]
        t, c, K, nobs, lk = _eval_d(y, d, thres, cap, ml, autolag, zero_sum, buf)
        if not np.isfinite(t):
            continue
        if t <= _adf_crit(nobs, level):
            hi_d, hi_t, hi_c, hi_K, hi_n, hi_l = d, t, c, K, nobs, lk
            for _ in range(refine):                       # bisect (lo, hi_d]
                if lo < 0.0:
                    break
                mid = 0.5 * (lo + hi_d)
                mt, mc, mK, mn, mlk = _eval_d(y, mid, thres, cap, ml, autolag,
                                              zero_sum, buf)
                if np.isfinite(mt) and mt <= _adf_crit(mn, level):
                    hi_d, hi_t, hi_c, hi_K, hi_n, hi_l = mid, mt, mc, mK, mn, mlk
                else:
                    lo = mid
            return (hi_d, hi_t, _adf_crit(hi_n, level), hi_c,
                    float(hi_K), float(hi_n), hi_l)
        lo = d
    return np.nan, np.nan, np.nan, np.nan, 0.0, 0.0, np.nan


@njit(cache=True, parallel=True)
def _optimal_d_panel(y, starts, ends, dgrid, thres, cap, level, maxlag, autolag,
                     refine, min_obs, zero_sum, out):
    for a in prange(starts.shape[0]):
        r = _optimal_d_one(y[starts[a]:ends[a]], dgrid, thres, cap, level, maxlag,
                           autolag, refine, min_obs, zero_sum)
        for j in range(7):
            out[a, j] = r[j]


@njit(cache=True, parallel=True)
def _dcurve_panel(y, starts, ends, dgrid, thres, cap, maxlag, autolag, zero_sum,
                  stat, corr):
    """The FULL curve — no early exit. `scan` only; the build never needs it."""
    for a in prange(starts.shape[0]):
        s, e = starts[a], ends[a]
        buf = np.empty(e - s, dtype=np.float64)
        ml = maxlag
        if ml <= 0:
            ml = max(1, min(20, int(np.ceil(12.0 * ((e - s) / 100.0) ** 0.25))))
        for g in range(dgrid.shape[0]):
            t, c, K, nobs, lk = _eval_d(y[s:e], dgrid[g], thres, cap, ml, autolag,
                                        zero_sum, buf)
            stat[a, g] = t
            corr[a, g] = c


_COLS = ("d", "adf", "crit", "corr", "width", "nobs", "leak")
_LEVEL = {0.01: 0, 0.05: 1, 0.10: 2}


def optimal_d(y: np.ndarray, dgrid=DGRID, thres=THRES, cap=MAX_WIDTH, alpha=ALPHA,
              maxlag: int | None = None, autolag: bool = True, refine: int = REFINE,
              min_obs: int = MIN_OBS, zero_sum: bool = ZERO_SUM) -> dict:
    """d* for one series. See `_optimal_d_one` for the criterion."""
    r = _optimal_d_one(np.ascontiguousarray(y, dtype=np.float64),
                       np.asarray(dgrid, dtype=np.float64), thres, cap,
                       _LEVEL[alpha], 0 if maxlag is None else maxlag,
                       1 if autolag else 0, refine, min_obs, zero_sum)
    return dict(zip(_COLS, r))


def optimal_d_panel(y, starts, ends, dgrid=DGRID, thres=THRES, cap=MAX_WIDTH,
                    alpha=ALPHA, maxlag=None, autolag=True, refine=REFINE,
                    min_obs=MIN_OBS, zero_sum=ZERO_SUM) -> pd.DataFrame:
    out = np.empty((starts.shape[0], 7), dtype=np.float64)
    _optimal_d_panel(y, starts, ends, np.asarray(dgrid, dtype=np.float64), thres, cap,
                     _LEVEL[alpha], 0 if maxlag is None else maxlag,
                     1 if autolag else 0, refine, min_obs, zero_sum, out)
    return pd.DataFrame(out, columns=list(_COLS))


# ── an independent read on memory: GPH ──────────────────────────────────────
def gph(y: np.ndarray, power: float = 0.5) -> tuple[float, float]:
    """Geweke–Porter-Hudak log-periodogram estimate of the integration order d, and its
    standard error. Regress log I(λ_j) on −log(4 sin²(λ_j/2)) over the lowest
    m = ⌊n^power⌋ Fourier frequencies; the slope IS d.

    This exists as a CROSS-CHECK on d*, because the two disagree in a specific and
    informative way. GPH reads the memory of the series itself and never touches a
    truncated filter, so it is immune to the leakage above; it is also badly biased in
    small samples and by short-run dynamics, which is why it does not replace the ADF
    search. Agreement (d* ≈ d_gph − 0.5, the distance to the stationarity boundary) is
    evidence the answer is about the series. Disagreement points at the window.

    se = π/√(24m) is the standard GPH asymptotic, exact for the known-variance
    log-periodogram regression."""
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = y.shape[0]
    m = int(n ** power)
    if n < 64 or m < 8:
        return np.nan, np.nan
    I = np.abs(np.fft.rfft(y - y.mean(), n)[1:m + 1]) ** 2 / (2.0 * np.pi * n)
    lam = 2.0 * np.pi * np.arange(1, m + 1) / n
    x = -np.log(4.0 * np.sin(lam / 2.0) ** 2)
    good = I > 0
    if good.sum() < 8:
        return np.nan, np.nan
    x, lg = x[good], np.log(I[good])
    xc = x - x.mean()
    return float(xc @ (lg - lg.mean()) / (xc @ xc)), float(np.pi / np.sqrt(24.0 * m))


def gph_panel(y, starts, ends, power: float = 0.5, n_jobs: int = -1) -> np.ndarray:
    """GPH per asset. numpy's FFT drops the GIL, so threads are real parallelism."""
    r = Parallel(n_jobs=n_jobs, backend="threading", batch_size=64)(
        delayed(gph)(y[s:e], power) for s, e in zip(starts, ends))
    return np.array([v[0] for v in r], dtype=np.float64)


# ── panel plumbing ──────────────────────────────────────────────────────────
def _load(panel: str = PANEL, sample: int | None = None, seed: int = 0):
    """(df, y, starts, ends). y is log adjusted price, the series FFD is meant for."""
    import duckdb

    where = "adj > 0 and close > 0"
    if sample:
        ids = duckdb.sql(f"""select distinct security_id from '{panel}' where {where}
                             using sample {int(sample)} rows (reservoir, {seed})""").df()
        where += f" and security_id in ({','.join(map(str, ids.security_id))})"
    df = duckdb.sql(f"""select security_id, date, close, adj, volume from '{panel}'
                        where {where} order by security_id, date""").df()
    sid = df.security_id.to_numpy()
    b = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1], True])
    y = np.ascontiguousarray(np.log(df.adj.to_numpy(np.float64)))
    return df, y, b[:-1].copy(), b[1:].copy()


def scan(panel: str = PANEL, sample: int = SAMPLE, dgrid=DGRID, alpha: float = ALPHA,
         thres: float = THRES) -> pd.DataFrame:
    """The d-curve and the d* distribution. This is what fixes the pooled d."""
    t0 = time.time()
    df, y, starts, ends = _load(panel, sample)
    print(f"[scan] {len(df):,} rows, {len(starts):,} securities ({time.time() - t0:.1f}s)",
          flush=True)

    g = np.asarray(dgrid, dtype=np.float64)
    t1 = time.time()
    stat = np.empty((len(starts), len(g)), dtype=np.float64)
    corr = np.empty_like(stat)
    _dcurve_panel(y, starts, ends, g, thres, MAX_WIDTH, 0, 1, ZERO_SUM, stat, corr)
    print(f"[scan] curve: {len(starts):,} × {len(g)} rungs in {time.time() - t1:.1f}s")

    n = np.array([e - s for s, e in zip(starts, ends)], dtype=np.float64)
    crit = np.array([_adf_crit(max(m, 30.0), _LEVEL[alpha]) for m in n])[:, None]
    keep = n >= MIN_OBS
    print("\n" + "=" * 78)
    print(f"1. d-CURVE   (mean over {keep.sum():,} securities with ≥ {MIN_OBS} bars, "
          f"ADF at {alpha:.0%})")
    print("=" * 78)
    print(f"  {'d':>6}{'ADF':>9}{'crit':>8}{'%stat':>8}{'corr(x,y)':>11}"
          f"{'width':>8}{'tail':>8}{'rows':>8}")
    for j, d in enumerate(g):
        s, c = stat[keep, j], corr[keep, j]
        w, tail = _weights(float(d), thres, MAX_WIDTH, ZERO_SUM)
        ok = np.isfinite(s)
        print(f"  {d:6.2f}{np.nanmean(s):9.2f}{crit[keep].mean():8.2f}"
              f"{np.mean(s[ok] <= crit[keep][ok, 0]) * 100:7.1f}%"
              f"{np.nanmean(c):11.3f}{len(w):8d}{tail:8.3f}"
              f"{np.maximum(n - len(w) + 1, 0).sum() / n.sum() * 100:7.1f}%")
    print("  %stat = share of securities stationary at that d — NOT a population")
    print("  property below d = 0.5; see the random-walk table in the header.")
    print("  corr is with log price: the memory that survives. tail = Σw before the")
    print("  zero-sum anchor, i.e. how much of the filter lies outside the window.")
    print("  rows = share of panel rows surviving the K−1 burn-in. That column and the")
    print("  corr column are the trade; d = 1 (returns) is the corr ≈ 0 end of it.")

    t2 = time.time()
    D = optimal_d_panel(y, starts, ends, g, thres, MAX_WIDTH, alpha)
    D["security_id"] = df.security_id.to_numpy()[starts]
    print(f"\n[scan] d* with early exit + {REFINE} bisections: {time.time() - t2:.1f}s "
          f"(vs {time.time() - t1 - (time.time() - t2):.1f}s for the full curve)")

    d = D.d.to_numpy()
    got = np.isfinite(d)
    print("\n" + "=" * 78)
    print("2. d* DISTRIBUTION")
    print("=" * 78)
    print(f"  resolved     {got.sum():,} / {len(D):,}  ({got.mean():.1%}); the rest have "
          f"< {MIN_OBS} bars or reject at no d ≤ {g[-1]:.2f}")
    print(f"  zero-sum     {ZERO_SUM}   (x is origin-invariant; `tail` is the mass "
          f"being approximated, not a defect)")
    if got.any():
        q = np.percentile(d[got], [5, 25, 50, 75, 95])
        print("  percentiles  " + "  ".join(f"p{p}={v:.3f}" for p, v in
                                            zip((5, 25, 50, 75, 95), q)))
        print(f"  mean {d[got].mean():.3f}   median {np.median(d[got]):.3f}   "
              f"corr at d* {D.loc[got, 'corr'].mean():.3f}   "
              f"window {D.loc[got, 'width'].median():.0f} bars   "
              f"leak {D.loc[got, 'leak'].mean():.3f}")
        print(f"\n  POOLED d = {np.median(d[got]):.2f}  (median; the cross-sectional "
              f"feature uses this)")

    t3 = time.time()
    D["gph"] = gph_panel(y, starts, ends)
    ok = got & np.isfinite(D.gph.to_numpy()) & (n >= MIN_OBS)
    print("\n" + "=" * 78)
    print("3. GPH CROSS-CHECK   (log-periodogram d, no filter involved)")
    print("=" * 78)
    print(f"  d_gph  mean {D.loc[ok, 'gph'].mean():.3f}  median "
          f"{D.loc[ok, 'gph'].median():.3f}   (log price is I(1), so ≈ 1.0 is the "
          f"sanity check)")
    print(f"  d_gph − d*   mean {(D.gph - D.d)[ok].mean():+.3f}   corr(d*, d_gph) "
          f"{np.corrcoef(D.loc[ok, 'd'], D.loc[ok, 'gph'])[0, 1]:+.3f}")
    print("  d_gph is what the SERIES is; d* is what the WINDOW plus this sample size")
    print("  cannot distinguish from stationary. The gap has a floor near +0.5 — the")
    print("  distance to the stationarity boundary — and runs wider because a K-wide")
    print(f"  filter is stationary at any d > 0. ({time.time() - t3:.1f}s)")
    return D


@njit(cache=True, parallel=True)
def _fwd_returns(y, starts, ends, horizons, out):
    """Forward log return measured FROM t. The implementation lag is applied to the
    SIGNAL in the harness, so these stay convention-free — same as trendscan.py."""
    for a in prange(starts.shape[0]):
        s, e = starts[a], ends[a]
        n = e - s
        for h in range(horizons.shape[0]):
            H = horizons[h]
            for t in range(n):
                out[s + t, h] = y[s + t + H] - y[s + t] if t + H < n else np.nan


@njit(cache=True, parallel=True)
def _roll_mean(x, starts, ends, win, out):
    for a in prange(starts.shape[0]):
        s, e = starts[a], ends[a]
        acc = 0.0
        for t in range(s, e):
            acc += x[t]
            if t - s >= win:
                acc -= x[t - win]
            out[t] = acc / min(t - s + 1, win)


def _one_series(name, y, starts, ends, dgrid, alpha, thres, d_pooled, n_jobs):
    """d* per asset, the pooled-d transform, and the own-d transform for one input."""
    t1 = time.time()
    D = optimal_d_panel(y, starts, ends, np.asarray(dgrid, dtype=np.float64),
                        thres, MAX_WIDTH, alpha)
    d = D.d.to_numpy()
    got = np.isfinite(d)
    dp = float(np.median(d[got])) if d_pooled is None else float(d_pooled)
    x, w = ffd_panel(y, starts, ends, dp, thres, MAX_WIDTH, n_jobs)
    print(f"[build] {name:5s} d*: {got.sum():,}/{len(D):,} resolved, median {dp:.3f}, "
          f"K = {len(w):,} → {np.isfinite(x).mean():.1%} of rows "
          f"({time.time() - t1:.1f}s)", flush=True)

    # Own-d pass. Grouping assets by their (rounded) d amortises the weight vector and
    # keeps this one convolution per asset rather than one per (asset, grid rung).
    xo = np.full(y.shape[0], np.nan, dtype=np.float64)
    key = np.where(got, np.round(d, 3), np.nan)
    for dv in np.unique(key[got]):
        idx = np.flatnonzero(key == dv)
        wv = _weights(float(dv), thres, MAX_WIDTH, ZERO_SUM)[0]
        Parallel(n_jobs=n_jobs, backend="threading", batch_size=32)(
            delayed(_ffd_fft_into if len(wv) >= FFT_MIN_K else _ffd_into)(
                y[starts[a]:ends[a]], wv, xo[starts[a]:ends[a]]) for a in idx)
    return D, d, dp, x, xo, w


def build(panel: str = PANEL, out_path: str = OUT, d_pooled: float | None = None,
          dgrid=DGRID, alpha: float = ALPHA, thres: float = THRES,
          n_jobs: int = -1) -> pd.DataFrame:
    """FFD of log price AND log volume, each at its own pooled d and its own per-asset
    d, plus the forward returns and screens the diagnostics need.

    Volume gets the same treatment because the question "how much of this series is a
    unit root" is not specific to prices, and log volume answers it differently — it is
    far closer to stationary already, so d* lands lower, which means a WIDER window and
    a bigger burn-in. That asymmetry is the interesting part and it is printed."""
    t0 = time.time()
    df, y, starts, ends = _load(panel)
    n = len(df)
    print(f"[build] {n:,} rows, {len(starts):,} securities ({time.time() - t0:.1f}s)",
          flush=True)

    # log1p, not log: zero-volume bars are real (halts, thin names) and dropping them
    # would break the bar contiguity every rolling object here depends on.
    v = np.ascontiguousarray(np.log1p(df.volume.to_numpy(np.float64)))

    P = _one_series("price", y, starts, ends, dgrid, alpha, thres, d_pooled, n_jobs)
    V = _one_series("vol", v, starts, ends, dgrid, alpha, thres, d_pooled, n_jobs)

    fwd = np.empty((n, len(HORIZONS)), dtype=np.float64)
    _fwd_returns(y, starts, ends, np.asarray(HORIZONS, dtype=np.int64), fwd)
    adv = np.empty(n, dtype=np.float64)
    _roll_mean(df.close.to_numpy(np.float64) * df.volume.to_numpy(np.float64),
               starts, ends, 21, adv)

    rep = np.repeat(np.arange(len(starts)), ends - starts)
    o = pd.DataFrame({
        "security_id": df.security_id.to_numpy(),
        "date": df.date.to_numpy(),
        "close": df.close.to_numpy(np.float32),
        "adv21": adv.astype(np.float32),
        "logp": y.astype(np.float32),
        "logv": v.astype(np.float32),
    })
    for tag, (D, d, dp, x, xo, w) in (("p", P), ("v", V)):
        o[f"ffd_{tag}"] = x.astype(np.float32)        # pooled d — cross-sectional
        o[f"ffdown_{tag}"] = xo.astype(np.float32)    # per-asset d — time-series only
        o[f"d_{tag}"] = d[rep].astype(np.float32)
        o[f"adf_{tag}"] = D.adf.to_numpy()[rep].astype(np.float32)
        o[f"leak_{tag}"] = D.leak.to_numpy()[rep].astype(np.float32)
        o.attrs[f"d_pooled_{tag}"] = dp
    for i, h in enumerate(HORIZONS):
        o[f"fwd{h}"] = fwd[:, i].astype(np.float32)

    o.to_parquet(out_path, index=False)
    print(f"[build] wrote {out_path}: {len(o):,} rows × {len(o.columns)} cols, "
          f"{time.time() - t0:.1f}s")
    for tag, src, (D, d, dp, x, xo, w) in (("price", y, P), ("vol", v, V)):
        print(f"[build] {tag:5s} pooled d {dp:.3f}  K {len(w):,}  burn-in "
              f"{1 - np.isfinite(x).mean():.1%} of rows  corr(ffd, log{tag[0]}) "
              f"{_corr_valid(x, src):.3f}")
    return o


# ── checks ──────────────────────────────────────────────────────────────────
def validate() -> None:
    """Every kernel against an independent implementation. Nothing here is a smoke test:
    the ADF is compared to statsmodels bit-for-bit-ish, the weights to Γ, the FFT path
    to the direct one."""
    rng = np.random.default_rng(0)
    fails = 0
    ok = True

    print("1. WEIGHTS   (raw, zero_sum off — the binomial expansion itself)")
    for d in (0.0, 0.25, 0.5, 0.75, 1.0):
        w = ffd_weights(d, 1e-6, zero_sum=False)
        k = np.arange(len(w))
        closed = np.array([(-1.0) ** i * float(np.prod([(d - j) / (j + 1)
                                                        for j in range(i)])) for i in k])
        e = np.max(np.abs(w - closed))
        print(f"   d={d:.2f}  K={len(w):5d}  w0={w[0]:.3f} w1={w[1] if len(w) > 1 else 0:+.3f}"
              f"   max|w − binomial| = {e:.2e}")
        fails += e > 1e-12
    w1 = ffd_weights(1.0, 1e-6, zero_sum=False)
    assert len(w1) == 2 and abs(w1[1] + 1.0) < 1e-15, "d=1 must be the plain difference"
    print("   d=1 terminates at [1, −1] ✓")

    print("\n2. TRANSFORM   (FFT path vs direct, and d=1 vs np.diff)")
    y = np.cumsum(rng.standard_normal(4000))
    for d in (0.3, 0.6):
        w = ffd_weights(d, 1e-5)
        a = np.empty(len(y)); _ffd_into(y, w, a)
        b = np.empty(len(y)); _ffd_fft_into(y, w, b)
        e = np.nanmax(np.abs(a - b))
        print(f"   d={d}  K={len(w):5d}  max|direct − fft| = {e:.2e}")
        fails += e > 1e-6
    e = np.nanmax(np.abs(ffd(y, 1.0)[1:] - np.diff(y)))
    print(f"   d=1  max|ffd − np.diff| = {e:.2e}")
    fails += e > 1e-12

    print("\n3. ADF   (vs statsmodels.tsa.stattools.adfuller, regression='c')")
    try:
        from statsmodels.tsa.stattools import adfuller
        for name, x in (("white noise", rng.standard_normal(1000)),
                        ("random walk", np.cumsum(rng.standard_normal(1000))),
                        ("AR(1) .95", None),
                        ("ffd d=0.4", ffd(y, 0.4))):
            if x is None:
                x = np.zeros(1000)
                for i in range(1, 1000):
                    x[i] = 0.95 * x[i - 1] + rng.standard_normal()
            x = x[np.isfinite(x)]
            ml = _schwert(len(x))
            for auto in (True, False):
                mine = adf(x, maxlag=ml, autolag=auto)
                ref = adfuller(x, maxlag=ml, regression="c",
                               autolag="aic" if auto else None)
                de, dl = abs(mine[0] - ref[0]), mine[1] - ref[2]
                print(f"   {name:12s} autolag={str(auto):5s} "
                      f"t={mine[0]:8.3f} (ref {ref[0]:8.3f}, Δ {de:.2e})  "
                      f"lag={mine[1]:2d} (ref {ref[2]:2d})  crit5={mine[3]:.3f} "
                      f"(ref {ref[4]['5%']:.3f})")
                fails += de > 1e-6 or dl != 0 or abs(mine[3] - ref[4]["5%"]) > 5e-3
    except ImportError:
        print("   statsmodels not installed in this interpreter — SKIPPED. Run under an"
              "\n   interpreter that has it; this is the check that matters most.")
        ok = False

    print("\n4. ZERO SUM   (Σw = 0, and the origin-invariance it buys)")
    c = 7.3                                        # an arbitrary change of origin
    for d in (0.2, 0.31, 0.5):
        w0, tail = _weights(d, THRES, MAX_WIDTH, False)
        w1_, _ = _weights(d, THRES, MAX_WIDTH, True)
        a0, a1 = ffd(y, d, w=w0), ffd(y, d, w=w1_)
        b0, b1 = ffd(y + c, d, w=w0), ffd(y + c, d, w=w1_)
        # the anchored form must equal truncated − tail·y_{t−K+1}, exactly
        K = len(w0)
        ref = a0.copy()
        ref[K - 1:] -= tail * y[:len(y) - K + 1]
        print(f"   d={d:.2f} K={K:5d}  Σw {w0.sum():+.3f} → {w1_.sum():+.1e}   "
              f"shift under y+{c}: {np.nanmax(np.abs(b0 - a0)):.3f} → "
              f"{np.nanmax(np.abs(b1 - a1)):.1e}   "
              f"max|anchored − (x − tail·y_lag)| = {np.nanmax(np.abs(a1 - ref)):.1e}")
        fails += (abs(w1_.sum()) > 1e-12 or np.nanmax(np.abs(b1 - a1)) > 1e-9
                  or np.nanmax(np.abs(a1 - ref)) > 1e-9)
    print("   Plain truncation moves with the origin of y in exact proportion to Σw;")
    print("   log adjusted price HAS no meaningful origin, which is the whole argument.")

    print("\n5. SEARCH   (ARFIMA(0,d₀,0) built by UNtruncated (1−B)^(−d₀) on white noise)")
    print("   GPH, which never sees a filter, should recover d₀. d* will sit WELL below")
    print("   d₀ − 0.5: a K-wide filter is stationary at any d > 0, so the ADF is")
    print("   reporting detectable persistence, not the population order. See header.")
    for d0 in (0.6, 0.8, 1.0):
        e = rng.standard_normal(20000)
        wl = _weights(-d0, 1e-12, 20000, False)[0]              # long-memory generator, no cut
        z = np.convolve(e, wl)[:20000][6000:]         # burn in the filter's own history
        r = optimal_d(z)
        g, se = gph(z)
        print(f"   d₀ = {d0:.2f}   d* = {r['d']:.3f}  adf {r['adf']:7.2f} vs crit "
              f"{r['crit']:.2f}  corr {r['corr']:.3f}  K {int(r['width']):5d}  "
              f"leak {r['leak']:+.3f}   d_gph = {g:.3f} ± {se:.3f}")
        fails += abs(g - d0) > 3.0 * se + 0.15

    print(f"\n{'PASS' if fails == 0 else str(fails) + ' FAILURE(S)'}"
          f"{'' if ok else '  (ADF check skipped)'}")


def bench(n_assets: int = 500, n_obs: int = 4000) -> None:
    rng = np.random.default_rng(0)
    y = np.ascontiguousarray(np.concatenate(
        [np.cumsum(rng.standard_normal(n_obs)) for _ in range(n_assets)]))
    starts = np.arange(n_assets, dtype=np.int64) * n_obs
    ends = starts + n_obs
    print(f"panel {n_assets:,} × {n_obs:,} = {len(y):,} rows")

    optimal_d(y[:n_obs])                                       # warm the JIT
    for tag, kw in (("full curve", None), ("d* early-exit", {})):
        t = time.time()
        if kw is None:
            s = np.empty((n_assets, len(DGRID))); c = np.empty_like(s)
            _dcurve_panel(y, starts, ends, np.asarray(DGRID), THRES, MAX_WIDTH, 0, 1, s, c)
        else:
            optimal_d_panel(y, starts, ends)
        print(f"  {tag:16s} {time.time() - t:6.2f}s")
    for d in (0.4, 0.05):
        w = ffd_weights(d, THRES)
        for path in ("direct", "fft"):
            t = time.time()
            if path == "direct":
                out = np.empty(len(y)); _ffd_panel_direct(y, starts, ends, w, out)
            else:
                out = np.empty(len(y))
                Parallel(n_jobs=-1, backend="threading", batch_size=64)(
                    delayed(_ffd_fft_into)(y[s:e], w, out[s:e])
                    for s, e in zip(starts, ends))
            print(f"  transform d={d:.2f} K={len(w):5d} {path:7s} {time.time() - t:6.2f}s")


if __name__ == "__main__":
    sys.path.insert(0, "research")
    args = [a for a in sys.argv[1:] if not a.startswith("-")] or ["scan"]
    if "validate" in args:
        validate()
    if "bench" in args:
        bench()
    if "scan" in args:
        scan()
    if "build" in args:
        build()
