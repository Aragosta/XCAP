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
    python research/trendscan.py diag      # score the reductions in reductions()
    python research/trendscan.py           # both
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

HORIZONS = (1, 5, 21, 63)
LAG = 1                                             # implementation lag, in bars.
# Decide at close t, trade at close t+1, earn t+1 → t+1+h. Pairing a signal with the
# return measured from its OWN close grants free instantaneous fills, and that bar is a
# large slice of the half-life of anything fast. Matches BACKTEST.py's lag=1.
SKIP = 21                                           # the "−1" of 12−1, for REDUCTIONS
MIN_ADV = 1e6
MIN_PRICE = 5.0
RESYNC = 2048                                       # exact-recompute cadence, anti-drift


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
    # RAW. No skip baked in: skipped[t] == raw[t−SKIP], so any end-lag is one shift away
    # (see `skip()`). The window length and the end-lag are two axes of one 2-D surface;
    # hardcoding the second collapses it.
    for name, A in zip(("tb", "tc", "sd"), (tb, tc, sd)):
        if name in emit:
            for j, L in enumerate(grid):
                o[f"{name}_{L}"] = A[:, j].astype(np.float32)
    # Targets, for the diagnostics only. Measured FROM t; the implementation lag is
    # applied to the SIGNAL in the harness, so these stay convention-free.
    for i, h in enumerate(HORIZONS):
        o[f"fwd{h}"] = fwd[:, i].astype(np.float32)

    o.to_parquet(out_path, index=False)
    print(f"[build] wrote {out_path}: {len(o):,} rows × {len(o.columns)} cols "
          f"({len(grid)} rungs × {len(emit)}), {time.time() - t0:.1f}s")
    return o


# ── reductions ──────────────────────────────────────────────────────────────
# A reduction turns the surface into something rankable. They live here, downstream of
# the build, so that changing one costs a re-read and not a rescan.
def surface(df: pd.DataFrame, kind: str, grid=GRID) -> np.ndarray:
    """The (rows × rungs) matrix for one quantity."""
    return df[[f"{kind}_{L}" for L in grid]].to_numpy(np.float64)


def skip(df: pd.DataFrame, cols, k: int = SKIP) -> np.ndarray:
    """End the window k bars early — the "−1" of 12−1. Reading the surface k rows back
    is what turns a 12-month scan into a 12−1 scan: no rescan, one shift."""
    return df.groupby("security_id", observed=True, sort=False)[list(cols)].shift(k) \
             .to_numpy(np.float64)


def rowmean(A: np.ndarray) -> np.ndarray:
    """Ladder mean. Averages over however many rungs the asset's history supports —
    see the listing-age note in the header."""
    ok = np.isfinite(A)
    cnt = ok.sum(axis=1)
    return np.where(cnt > 0, np.where(ok, A, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)


def argmax_abs(A: np.ndarray) -> np.ndarray:
    """Value at argmax|·| across rungs — LdP's endogenous horizon. NOT a neutral
    selector: |t| ∝ L^1.5 draws it toward the longest rung regardless of information."""
    M = np.where(np.isfinite(A), np.abs(A), -1.0)
    j = M.argmax(axis=1)
    v = np.take_along_axis(A, j[:, None], axis=1)[:, 0]
    return np.where(M.max(axis=1) >= 0.0, v, np.nan)


def rung(df: pd.DataFrame, kind: str, L: int) -> np.ndarray:
    return df[f"{kind}_{L}"].to_numpy(np.float64)


# The reductions `diag` scores. This dict is a CHOICE, deliberately visible and
# deliberately not in the build. Edit it freely; nothing downstream depends on it.
#
# THE SKIP IS NOT SYMMETRIC, and this is measured, not aesthetic:
#   trend       WANTS the skip. Ending the window a month early removes short-horizon
#               reversal from the slope estimate.
#   curvature   is HURT by it. No-skip beats skip by +0.0072 IC at h21 (t 2.20) and
#               +0.0087 at h63 (t 1.97) for the argmax, +0.0101 at h63 (t 2.20) for the
#               ladder. Curvature is measuring the bend that is happening NOW.
def reductions(df: pd.DataFrame, grid=GRID) -> dict:
    tb_s = skip(df, [f"tb_{L}" for L in grid])
    tc_r = surface(df, "tc", grid)
    return {
        "trend_top": tb_s[:, -1],                     # single longest rung, 12−1
        "trend_lad": rowmean(tb_s),                   # ladder mean
        "trend_am": argmax_abs(tb_s),                 # endogenous horizon
        "ncurv_lad": -rowmean(tc_r),                  # ladder mean, no skip
        "ncurv_am": -argmax_abs(tc_r),                # endogenous horizon
    }


# NOT the answer — the thing to beat, and it currently LOSES to its own trend leg.
# Equal-weighting trend with curvature is significantly WORSE than trend alone at h1
# (ΔIC −0.0038, t −3.57) and a wash at h21/h63, because equal weight over-allocates to
# a leg that is ~9× weaker at h1 (0.0019 vs 0.0182). Curvature is nonetheless a real,
# near-orthogonal bet (N_eff 1.91/2.00) that covers the trend leg's 2009-16 hole
# (h21 IC 0.0153 vs the trend's 0.0121 there). So the open question is WEIGHTING and
# horizon, not whether curvature works — and "equal weight beats every fitted
# allocator" was never tested against the one-leg baseline until now.
COMPOSITES = {"combo_ew": ["trend_top", "ncurv_lad"]}


# ── evaluation harness ──────────────────────────────────────────────────────
# Shared by diag() and the decisions scripts. Every IC in this project comes through
# here, so a convention fixed here is fixed everywhere.
def _nw_lag(h: int) -> int:
    """Bartlett bandwidth for an h-overlapping series.

    Overlapping forward returns induce an MA(h−1) in the IC series, so h−1 is the bare
    MINIMUM that covers it and 1.5h is the usual conservative choice. This file used
    `lag = h` — barely above the minimum, which understates the long-run variance and
    inflates t most at h63, the horizon carrying the largest claims."""
    return max(int(np.ceil(1.5 * h)), 1)


def _nw_t(x: np.ndarray, lag: int) -> float:
    """Newey-West t of the mean."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return np.nan
    d = x - x.mean()
    v = (d @ d) / n
    for l in range(1, lag + 1):
        v += 2.0 * (1.0 - l / (lag + 1.0)) * ((d[l:] @ d[:-l]) / n)
    return np.nan if v <= 0 else x.mean() / np.sqrt(v / n)


def apply_lag(df: pd.DataFrame, cols, lag: int = LAG) -> pd.DataFrame:
    """Shift each signal forward `lag` bars WITHIN a security. In place.

    Pairing signal(t) with the return measured FROM t grants execution at the very close
    that produced the signal — free, instantaneous fills. BACKTEST.py's convention is
    that a signal at t earns y[t+1+h] − y[t+1]; shifting the signal one bar and leaving
    fwd where it is reproduces that exactly, without rebuilding the targets.

    Must run BEFORE the liquidity screen — the screen drops rows and so breaks the bar
    contiguity this shift depends on."""
    if lag <= 0:
        return df
    df.sort_values(["security_id", "date"], inplace=True, kind="stable")
    g = df.groupby("security_id", observed=True, sort=False)
    df[list(cols)] = g[list(cols)].shift(lag)
    return df


def ic_series(df: pd.DataFrame, names, horizons=HORIZONS, composites=None):
    """Cross-sectional Spearman IC per date → {(name, h): Series indexed by date}.

    `composites` maps a name to a list of columns that are equal-weight averaged AFTER
    ranking (ranks, not levels — the legs are on wildly different scales)."""
    composites = composites or {}
    base = list(dict.fromkeys(list(names) + [c for v in composites.values() for c in v]))
    g = df.groupby("date", observed=True)
    R = pd.DataFrame({c: g[c].rank(pct=True)
                      for c in base + [f"fwd{h}" for h in horizons]})
    for name, parts in composites.items():
        R[name] = R[list(parts)].mean(axis=1)
    k = df.date.to_numpy()
    D = {c: R[c] - R[c].groupby(k, observed=True).transform("mean") for c in R.columns}

    def ser(a, h):
        x, yv = D[a], D[f"fwd{h}"]
        num = (x * yv).groupby(k, observed=True).sum()
        den = np.sqrt((x * x).groupby(k, observed=True).sum()
                      * (yv * yv).groupby(k, observed=True).sum())
        return (num / den).replace([np.inf, -np.inf], np.nan)

    out = list(names) + list(composites)
    return {(c, h): ser(c, h) for c in out for h in horizons}, R


def ic_table(S, names, horizons=HORIZONS, indent="  ") -> None:
    print(f"{indent}{'signal':16s}"
          + "".join(f"{'IC_h' + str(h):>10s}{'t':>7s}" for h in horizons))
    for c in names:
        print(f"{indent}{c:16s}" + "".join(
            f"{S[(c, h)].mean():10.4f}{_nw_t(S[(c, h)].to_numpy(), _nw_lag(h)):7.2f}"
            for h in horizons), flush=True)


def paired_table(S, pairs, horizons=HORIZONS, indent="  ") -> None:
    """NW t of the DIFFERENCE series. Separate t-stats do not test whether a gap is
    real: the IC series share dates and are ~0.8 correlated, so the paired difference is
    far better determined than either level."""
    for a, b in pairs:
        d = {h: (S[(a, h)] - S[(b, h)]).dropna() for h in horizons}
        print(f"{indent}{a:16s} − {b:16s}" + "".join(
            f"{d[h].mean():10.4f}{_nw_t(d[h].to_numpy(), _nw_lag(h)):7.2f}"
            for h in horizons))


def load(out_path: str = OUT, grid=GRID, kinds=("tb", "tc"), lag: int = LAG) -> pd.DataFrame:
    """Read the surface, apply the reductions, lag, then screen. Order matters — the
    reductions need raw bar contiguity, and so does the lag, and the screen destroys it."""
    t0 = time.time()
    cols = [f"{k}_{L}" for k in kinds for L in grid]
    df = pd.read_parquet(out_path, columns=["security_id", "date", "close", "adv21"]
                         + cols + [f"fwd{h}" for h in HORIZONS])
    n0 = len(df)
    df.sort_values(["security_id", "date"], inplace=True, kind="stable")
    red = reductions(df, grid)
    df = pd.concat([df[["security_id", "date", "close", "adv21"]
                       + [f"fwd{h}" for h in HORIZONS]],
                    pd.DataFrame(red, index=df.index)], axis=1)
    apply_lag(df, list(red), lag)
    df = df[(df.adv21 >= MIN_ADV) & (df.close >= MIN_PRICE)].dropna(subset=list(red))
    print(f"[load] {n0:,} → {len(df):,} rows  (lag {lag}b, ADV ≥ ${MIN_ADV:,.0f}, "
          f"px ≥ ${MIN_PRICE:.0f}), {time.time() - t0:.1f}s")
    print(f"[load] {df.date.min()} → {df.date.max()}, "
          f"{df.security_id.nunique():,} securities, {df.date.nunique():,} dates\n",
          flush=True)
    return df


def diag(out_path: str = OUT, grid=GRID) -> None:
    t0 = time.time()
    df = load(out_path, grid)
    NAMES = [c for c in df.columns
             if c not in ("security_id", "date", "close", "adv21")
             and not c.startswith("fwd")]
    S, R = ic_series(df, NAMES, composites=COMPOSITES)
    ALL = NAMES + list(COMPOSITES)

    print("=" * 84)
    print(f"1. INFORMATION COEFFICIENT   (cross-sectional Spearman, {LAG}-bar lag, "
          f"NW at 1.5h)")
    print("=" * 84)
    ic_table(S, ALL)
    base = max(NAMES, key=lambda c: S[(c, 21)].mean())
    print(f"\n  PAIRED vs the best single reduction ({base}) — NW t of the DIFFERENCE.")
    paired_table(S, [(c, base) for c in ALL if c != base])

    print("\n" + "=" * 84)
    print("2. MONOTONICITY   (mean fwd21 log return in bp, by signal quintile)")
    print("=" * 84)
    for c in ALL:
        q = pd.qcut(R[c], 5, labels=False, duplicates="drop")
        m = df.groupby(q, observed=True).fwd21.mean() * 1e4
        mono = bool((np.diff(m.to_numpy()) > 0).all())
        print(f"  {c:16s}" + " ".join(f"Q{i + 1}:{v:7.1f}" for i, v in enumerate(m))
              + f"   Q5−Q1 = {m.iloc[-1] - m.iloc[0]:6.1f} bp  monotone={mono}")

    print("\n" + "=" * 84)
    print("3. TURNOVER   (mean |Δsignal| per name-day, normalised by σ(signal))")
    print("=" * 84)
    d2 = df.sort_values(["security_id", "date"])
    for c in NAMES:
        s = d2.groupby("security_id", observed=True)[c].diff().abs().mean()
        print(f"  {c:16s}{s / d2[c].std():8.4f}")
    print("  Cost is proportional to this; nothing here is net of spreads.")

    print("\n" + "=" * 84)
    print("4. BREADTH   (N_eff = tr(Σ)²/tr(Σ²) on the daily IC series)")
    print("=" * 84)
    legs = next(iter(COMPOSITES.values()))
    P = pd.DataFrame({c: S[(c, 1)] for c in legs}).dropna()
    C = np.cov(P.to_numpy().T)
    print("  " + pd.DataFrame(np.corrcoef(P.to_numpy().T), index=P.columns,
                              columns=P.columns)
          .to_string(float_format=lambda x: f"{x:+.3f}").replace("\n", "\n  "))
    print(f"\n  N_eff = {np.trace(C) ** 2 / np.trace(C @ C):.2f}   (max {len(legs)}.00)")
    print("  Only meaningful because each leg carries alpha on its own. N_eff alone is")
    print("  gameable — orthogonal noise raises it without limit.")

    print("\n" + "=" * 84)
    print("5. STABILITY   (IC by sub-period — a full-sample mean can hide a dead decade)")
    print("=" * 84)
    eras = [("2000-2008", 2000, 2008), ("2009-2016", 2009, 2016), ("2017-2026", 2017, 2026)]
    print(f"  {'signal':16s}" + "".join(f"{e + ' h1/h21':>24s}" for e, _, _ in eras))
    for c in ALL:
        row = ""
        for _, lo, hi in eras:
            for h in (1, 21):
                s = S[(c, h)]
                m = ((s.index >= pd.Timestamp(f"{lo}-01-01"))
                     & (s.index <= pd.Timestamp(f"{hi}-12-31")))
                row += f"{s[m].mean():12.4f}"
        print(f"  {c:16s}{row}")

    print(f"\n[diag] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")] or ["build", "diag"]
    if "build" in args:
        build()
    if "diag" in args:
        diag()
