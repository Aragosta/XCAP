"""
trendscan.py — a rolling OLS trend scan. Emits a SURFACE, not a signal.

WHAT THIS IS
------------
At every (asset, date) it fits log price on time over a ladder of trailing window
lengths, and keeps the whole term structure. Three quantities per rung, from one set of
rolling sums:

    tb_L   slope t          direction, in units of the fit's own noise
    tc_L   curvature t      the exact orthogonal complement of the slope
    sd_L   residual sd      the noise scale itself

Everything uses data ≤ t. López de Prado's trend scanning is a forward-looking
*labelling* method; this is the same machinery run backward so it can be traded.

`sd` is the denominator of both t-stats, and it is also a signal in its own right — the
low-vol effect, arriving free from sums already computed (IC_h1 0.0134, t 4.29, at a
turnover of 0.0145). It is NOT a trend-scan object and should not be described as one.

A fourth quantity, `rz` — the endpoint residual, "overextension" — was emitted and has
been REMOVED. It measured 0.851 correlated with curvature, because it was taken from the
LINEAR fit while the surface also fits a quadratic: the endpoint is exactly where the
quadratic term is largest, so `rz_L ≈ c·L²/6/sd` and the linear fit dumps the whole
curvature signal into its last residual. Adding it to a composite already holding
curvature was significantly harmful (ΔIC −0.0020, t −4.05 at h1) at up to 4× the
turnover. Taking it from the QUADRATIC fit instead would make it orthogonal to both by
construction; that is the only version worth re-introducing. See TRENDSCAN.md §3.7.

WHAT THIS FILE DOES NOT DECIDE
------------------------------
No skip. No argmax. No fast/slow split. No composite. Those are REDUCTIONS of the
surface, they are one array read each, and they belong to whatever consumes this —
a signal, a feature block, a label. Baking any of them into the build costs a
dimension the scan already spans, and every measured mistake in this project's
history has been exactly that:

  * the 12-rung ladder MEAN was the stored trend leg; a single 252 rung beats it at
    every horizon (paired t 2.60 at h1) at 60% of the turnover. The surface was
    right and the reduction was wrong.
  * `slow`(argmax) vs `fast`(ladder) were stored as two separate legs with two
    different mechanisms. Held to one grid and one skip, the argmax LOSES for slow —
    the reverse of what was claimed — and for fast the two are indistinguishable.
  * a fast/slow split at all: dropping the fast leg from the composite is free at h1/h5
    and better at h21/h63, while it carried 23× the trend leg's turnover.

Only the trend leg is measurably load-bearing (dropping it costs ΔIC −0.0064, t −6.04
at h1). TRENDSCAN.md records all of it; every number came from reducing this stored
surface, with no rescan — which is the argument for storing it.

THE GRID
--------
One ladder, spanning short to long. There is no "fast ladder": a 5-day rung is just a
rung. Roughly geometric, because adjacent rungs are near-redundant — 117 daily-spaced
windows measured N_eff 2.14, so fine spacing buys columns, not information.

Windows longer than an asset's history are skipped rather than voiding the row, so a
252 rung does not require 252 bars of history. NOTE the consequence: any reduction that
averages across rungs then averages over a VARYING number of them, and since |t| ∝ L^1.5
its scale tracks listing age. Reduce with a single rung, or mask on rung count.

SPEED
-----
The naive scan is O(T · ΣL). Each window is a rolling OLS updated in O(1) per bar via
centred sums, so the surface costs O(T · |GRID|), njit-compiled and parallel over
assets, with an exact recompute every RESYNC bars to stop drift. 19M rows × 10 rungs
scan in ~3s; the whole build including the parquet write is ~40s.

    python research/trendscan.py build     # surface → data/_staging/trendscan.parquet
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from numba import njit, prange

PANEL = "data/_staging/alpha_panel.parquet"
OUT = "data/_staging/trendscan.parquet"

# One ladder, short to long, roughly geometric. A rung is a rung; there is no fast/slow
# distinction here because the data does not support hardcoding one.
GRID = np.array([5, 10, 21, 42, 63, 84, 126, 168, 210, 252], dtype=np.int64)
EMIT = ("tb", "tc", "sd")

HORIZONS = (1, 5, 21, 63)   # forward-return horizons written to the parquet
RESYNC = 2048                                       # exact-recompute cadence, anti-drift

# Bar integrity. The vendor EOD contains isolated bars belonging to a DIFFERENT
# instrument — a whole OHLC row at ~27× the surrounding level, with its own volume, that
# unwinds exactly on the next bar (security 270686, 2015-09-18: 519.17 → 14199.60 →
# 519.17). The bar is internally consistent, so low ≤ {open,close} ≤ high passes it, and
# the ADV screen actively *selects* for it: the spike inflates adv21 for 21 bars and
# pulls the name into the liquid universe. 3,388 such bars across 299 of 5,604
# securities; in the loss panel 665 rows of 6.66M carried 78% of Σy².
#
# Detected on ADJUSTED price and by REVERSAL, which is what makes it split-safe: a real
# split is already in `adj` so it never shows here, and a *missed* split does not unwind.
# Flagged bars are dropped, not interpolated — the price on those bars is unknown. Since
# the move reverts, dropping leaves the level series continuous.
BAD_SPIKE = 0.80       # |log return| a single bar must not exceed unexplained (≈ ×2.2)
BAD_REVERT = 0.15      # ...and that unwinds to within this fraction of itself
BAD_WINDOW = 3         # ...within this many bars (a spike can plateau before reverting)
# The second defect is a LEVEL BREAK, and it is not repairable bar by bar either. Two
# causes, one treatment:
#
#   missed corporate action — security 283067, 2004-07-30: 8.99 → 74.30 overnight
#     (×8.26) with split_factor 1.0 and adv21 running 70M → 95M → 101M straight through.
#     Dollar volume is continuous across it, which is the signature of a reverse split
#     the adjustment factors never saw; a genuine +726% session spikes volume, not
#     share count down.
#   spliced / recycled ticker — security 258690 alternates a dead $0.005 stub
#     (volume 0) with a real $4,700 name (volume 250k) on consecutive bars, and
#     security 258234 resumes 2 years later at 4× the price. `phase1_checks` already
#     names this: "the price series and the action series describe different companies
#     that shared a symbol".
#
# Either way the price base changes, so bars before and after are not one series and no
# window may span the break. They are CUT into separate series rather than dropped: the
# data on both sides is fine, only its continuity is not. Dropping the securities whole
# cost 3.89% of the panel for nothing.
BAD_BREAK = 1.60       # |log return| that ends a series (≈ ×5 / −80%)
BAD_GAP = 365          # ...as does a calendar gap of this many days (suspension, relist)
MIN_SEG = 5            # segments shorter than the shortest rung carry no surface at all


# ── kernels ─────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=True, inline="always")
def _moments(Lf):
    """Centred design moments for x = 0…L−1 with u = x − (L−1)/2.
    Σu = Σu³ = 0, so the linear and quadratic basis vectors are orthogonal: the slope is
    IDENTICAL in both fits and the curvature is its exact orthogonal complement."""
    Su2 = Lf * (Lf * Lf - 1.0) / 12.0
    Su4 = Lf * (Lf * Lf - 1.0) * (3.0 * Lf * Lf - 7.0) / 240.0
    return Su2, Su4 - Su2 * Su2 / Lf


@njit(cache=True, fastmath=True)
def _ladder_one(y, ladder, tout, cout, sout):
    """All three quantities at each rung — levels, nothing collapsed.

    Slope and curvature come out of the SAME rolling sums, so `tc` alongside `tb` costs
    three extra flops and no extra pass. That is why emitting the full surface is not a
    luxury: reducing it later is strictly cheaper than scanning twice."""
    n = y.shape[0]
    for k in range(ladder.shape[0]):
        L = ladder[k]
        for t in range(n):
            tout[t, k] = np.nan
            cout[t, k] = np.nan
            sout[t, k] = np.nan
        if n < L or L < 5:
            continue
        Lf = float(L)
        m = (Lf - 1.0) / 2.0
        Su2, Su4_c = _moments(Lf)
        Sy = 0.0
        Syy = 0.0
        W1 = 0.0
        W2 = 0.0
        for i in range(L):
            v = y[i]
            Sy += v
            Syy += v * v
            W1 += i * v
            W2 += i * i * v
        for t in range(L - 1, n):
            if t > L - 1:
                if (t - (L - 1)) % RESYNC == 0:
                    Sy = 0.0
                    Syy = 0.0
                    W1 = 0.0
                    W2 = 0.0
                    for i in range(L):
                        v = y[t - L + 1 + i]
                        Sy += v
                        Syy += v * v
                        W1 += i * v
                        W2 += i * i * v
                else:
                    yo = y[t - L]
                    yn = y[t]
                    W2 = W2 - 2.0 * W1 + Sy - yo + (Lf - 1.0) * (Lf - 1.0) * yn
                    W1 = W1 - Sy + yo + (Lf - 1.0) * yn
                    Sy = Sy - yo + yn
                    Syy = Syy - yo * yo + yn * yn
            Suy = W1 - m * Sy
            Su2y_c = (W2 - 2.0 * m * W1 + m * m * Sy) - Su2 * Sy / Lf
            b = Suy / Su2
            sse1 = (Syy - Sy * Sy / Lf) - b * Suy
            sd = np.sqrt(sse1 / (Lf - 2.0)) if sse1 > 0.0 else 0.0
            sout[t, k] = sd
            tout[t, k] = 0.0 if sd <= 0.0 else b * np.sqrt(Su2) / sd
            c = Su2y_c / Su4_c
            sse2 = sse1 - c * Su2y_c
            cout[t, k] = 0.0 if sse2 <= 0.0 else c / np.sqrt(sse2 / ((Lf - 3.0) * Su4_c))


@njit(cache=True, parallel=True)
def _ladder_panel(y, starts, ends, ladder, tout, cout, sout):
    for a in prange(starts.shape[0]):
        s, e = starts[a], ends[a]
        if e - s >= 2:
            _ladder_one(y[s:e], ladder, tout[s:e], cout[s:e], sout[s:e])


@njit(cache=True, parallel=True)
def _fwd_returns(y, starts, ends, horizons, out):
    for a in prange(starts.shape[0]):
        s, e = starts[a], ends[a]
        n = e - s
        for h in range(horizons.shape[0]):
            H = horizons[h]
            for t in range(n):
                out[s + t, h] = y[s + t + H] - y[s + t] if t + H < n else np.nan


@njit(cache=True, parallel=True)
def _bad_bars(y, starts, ends, spike, revert, window, out):
    """Flag isolated price prints: a jump > `spike` in log that unwinds within `window`.

    `y` is log adjusted price. For a bar t with |r_t| > spike, walk forward up to
    `window` bars looking for the cumulative move to return to where it started. If it
    does, every bar from t to the bar before the recovery is a spurious print.
    """
    for a in prange(starts.shape[0]):
        s, e = starts[a], ends[a]
        for t in range(s + 1, e):
            r = y[t] - y[t - 1]
            if r < spike and r > -spike:
                continue
            cum = r
            for k in range(1, window + 1):
                if t + k >= e:
                    break
                cum += y[t + k] - y[t + k - 1]
                if abs(cum) < revert * abs(r):
                    for j in range(t, t + k):
                        out[j] = True
                    break


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


# ── build ───────────────────────────────────────────────────────────────────
def build(panel: str = PANEL, out_path: str = OUT, grid=GRID, emit=EMIT) -> pd.DataFrame:
    import duckdb

    t0 = time.time()
    df = duckdb.sql(f"""select security_id, date, close, adj, volume from '{panel}'
                        where adj > 0 and close > 0 order by security_id, date""").df()
    print(f"[build] {len(df):,} rows, {df.security_id.nunique():,} securities "
          f"({time.time() - t0:.1f}s)", flush=True)

    sid = df.security_id.to_numpy()
    b = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1], True])
    starts, ends = b[:-1].copy(), b[1:].copy()

    # Bar integrity, BEFORE anything is fitted or any target is measured. A spurious
    # print corrupts every window that contains it (up to 252 bars of tb/tc/sd) and both
    # legs of fwd{h}; there is no downstream place to repair it. See BAD_SPIKE.
    t2 = time.time()
    ly = np.log(df.adj.to_numpy(np.float64))
    bad = np.zeros(len(df), dtype=np.bool_)
    _bad_bars(ly, starts, ends, BAD_SPIKE, BAD_REVERT, BAD_WINDOW, bad)
    n_bad, sec_bad = int(bad.sum()), df.security_id.to_numpy()[bad]
    print(f"[build] bar integrity: dropped {n_bad:,} spurious prints "
          f"({n_bad / len(df):.4%}) across {len(np.unique(sec_bad)):,} securities "
          f"({time.time() - t2:.1f}s)", flush=True)
    if n_bad:
        df = df.loc[~bad].reset_index(drop=True)
        sid = df.security_id.to_numpy()
        b = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1], True])
        starts, ends = b[:-1].copy(), b[1:].copy()
        ly = np.log(df.adj.to_numpy(np.float64))

    # Whatever survives the bar rule and is still impossible is a level break. Cut the
    # series there, so no rolling window and no fwd{h} ever spans it. Downstream code
    # is unchanged: it already works on contiguous per-security blocks, and a cut is
    # simply one more block boundary.
    r = np.diff(ly, prepend=ly[0])
    gap = np.diff(df.date.to_numpy("datetime64[D]").astype(np.int64), prepend=0)
    price_cut, gap_cut = np.abs(r) > BAD_BREAK, gap > BAD_GAP
    price_cut[starts] = gap_cut[starts] = False       # row 0 of a security differences
    cut = price_cut | gap_cut                         # against the previous security
    seg = np.cumsum(np.r_[True, sid[1:] != sid[:-1]] | cut)
    b = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1], True])
    starts, ends = b[:-1].copy(), b[1:].copy()
    n_cut = int(cut.sum())
    short = (ends - starts) < MIN_SEG
    print(f"[build] level breaks: {n_cut:,} cuts ({int(price_cut.sum()):,} "
          f"price, {int(gap_cut.sum()):,} calendar) split {len(np.unique(sid)):,} "
          f"securities into {len(starts):,} series; dropping {int(short.sum()):,} "
          f"segments shorter than {MIN_SEG} bars", flush=True)
    if short.any():
        keep = np.ones(len(df), dtype=np.bool_)
        for s, e in zip(starts[short], ends[short]):
            keep[s:e] = False
        df = df.loc[keep].reset_index(drop=True)
        sid = df.security_id.to_numpy()
        seg = seg[keep]
        b = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1], True])
        starts, ends = b[:-1].copy(), b[1:].copy()

    # Log adjusted price, centred per asset. Slope and t are invariant to a level shift;
    # centring only keeps the rolling sums well conditioned.
    y = np.log(df.adj.to_numpy(np.float64))
    for s, e in zip(starts, ends):
        y[s:e] -= y[s:e].mean()

    t1 = time.time()
    tb = np.empty((len(df), len(grid)), dtype=np.float64)
    tc, sd = np.empty_like(tb), np.empty_like(tb)
    _ladder_panel(y, starts, ends, np.asarray(grid, dtype=np.int64), tb, tc, sd)
    print(f"[build] surface: {len(grid)} rungs × 3 in {time.time() - t1:.1f}s", flush=True)

    fwd = np.empty((len(df), len(HORIZONS)), dtype=np.float64)
    _fwd_returns(y, starts, ends, np.asarray(HORIZONS, dtype=np.int64), fwd)
    dv = df.close.to_numpy(np.float64) * df.volume.to_numpy(np.float64)
    adv = np.empty(len(df), dtype=np.float64)
    _roll_mean(dv, starts, ends, 21, adv)

    o = pd.DataFrame({
        "security_id": df.security_id.to_numpy(),
        "date": df.date.to_numpy(),
        "close": df.close.to_numpy(np.float32),
        "adv21": adv.astype(np.float32),
    })
    # RAW. No skip baked in — the window length and any end-lag are two axes of one
    # 2-D surface; hardcoding the second collapses it.
    for name, A in zip(("tb", "tc", "sd"), (tb, tc, sd)):
        if name in emit:
            for j, L in enumerate(grid):
                o[f"{name}_{L}"] = A[:, j].astype(np.float32)
    # Targets, for downstream diagnostics. Measured FROM t; any implementation lag is
    # applied to the SIGNAL by the consumer, so these stay convention-free.
    for i, h in enumerate(HORIZONS):
        o[f"fwd{h}"] = fwd[:, i].astype(np.float32)

    o.to_parquet(out_path, index=False)
    print(f"[build] wrote {out_path}: {len(o):,} rows × {len(o.columns)} cols "
          f"({len(grid)} rungs × {len(emit)}), {time.time() - t0:.1f}s")
    return o


if __name__ == "__main__":
    build()
