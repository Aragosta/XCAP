"""
trendscan_eval.py — everything that is done WITH the scan. No signal is defined here.

trendscan.py emits one thing: `tb_L`, the raw slope t at each rung. Every choice made
after that lives here — targets, liquidity, the skip, cross-rung rescaling, ranking,
neutralisation, deciles, costs. The two files ship together on purpose: the harness was
deleted from this project once (commit c6c8e41, along with `loss_test.py`) and every
number in TRENDSCAN.md §3 became unreproducible.

    python research/trendscan_eval.py check   # neutralisation + decile-bucketing asserts
    python research/trendscan_eval.py eval    # IC tables + paired t — which column ranks
    python research/trendscan_eval.py signal  # IS it a signal: neutralised IC, deciles
    python research/trendscan_eval.py cost    # BACKTEST.py, with spreads and commissions

`signal` and `cost` answer different questions and the second does not gate the first: a
long-only top-decile book conflates signal quality with portfolio construction and beta.
Read `signal` while the deliverable is a signal — see TRENDSCAN.md §3.13.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from numba import njit, prange

from trendscan import GRID, build

# IC is still rising at h252 (§3.13), so 126 and 252 are here to find the peak, not to be
# traded: at h252 the overlapping windows leave ~27 independent observations.
HORIZONS = (1, 5, 21, 63, 126, 252)
LAG = 1              # decide at close t, trade at t+1. Pairing a signal with the return
#                      from its own close grants free fills. Matches BACKTEST.py.
SKIP = 21            # the "−1" of 12−1, applied here, never in the build
MIN_ADV = 1e6
MIN_PRICE = 5.0

# Null sd of tb_L under a driftless random walk is κ·√L, not 1: the t divides by an se
# assuming information accrues at L^1.5, under a unit root it accrues at √L. Rescaling is
# monotone within a rung so it cannot improve IC; it buys comparability ACROSS rungs,
# which is what makes an absolute |z| gate possible at all. Measured on 40k paths. §3.9.
KAPPA_B = {5: 1.83, 10: 1.39, 21: 1.37, 42: 1.38, 63: 1.40,
           84: 1.39, 126: 1.39, 168: 1.40, 210: 1.41, 252: 1.41}


def zscale(t, L, kappa=KAPPA_B):
    return t / (kappa[L] * np.sqrt(L))


# ── per-segment kernels ─────────────────────────────────────────────────────
# Everything below runs on the contiguous per-segment blocks the build hands over. They
# are numba because pandas' groupby().rolling() on 19M rows is both slow and returns a
# MultiIndex whose order is only incidentally the frame's — a silent misalignment on
# columns every baseline depends on.
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
def _roll_std(x, starts, ends, win, minp, out):
    """Trailing sd within a segment."""
    for a in prange(starts.shape[0]):
        s, e = starts[a], ends[a]
        acc = 0.0
        acc2 = 0.0
        for t in range(s, e):
            v = x[t]
            acc += v
            acc2 += v * v
            if t - s >= win:
                o = x[t - win]
                acc -= o
                acc2 -= o * o
            n = min(t - s + 1, win)
            if n < minp:
                out[t] = np.nan
            else:
                var = (acc2 - acc * acc / n) / (n - 1.0)
                out[t] = np.sqrt(var) if var > 0.0 else np.nan


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


def _segbounds(df: pd.DataFrame):
    """Contiguous [start, end) of each segment. `df` must be sorted by (seg, date)."""
    seg = df.seg.to_numpy()
    b = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1], True])
    return b[:-1].copy(), b[1:].copy()


def targets(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Forward log returns and 21-bar ADV. Measured FROM t; any implementation lag is
    applied to the SIGNAL by `apply_lag`, so these stay convention-free. Not in the build
    — the answer key is not part of the signal."""
    starts, ends = _segbounds(df)
    y = df.y.to_numpy(np.float64)
    fwd = np.empty((len(df), len(horizons)))
    _fwd_returns(y, starts, ends, np.asarray(horizons, dtype=np.int64), fwd)
    adv = np.empty(len(df))
    _roll_mean(df.close.to_numpy(np.float64) * df.volume.to_numpy(np.float64),
               starts, ends, 21, adv)
    out = pd.DataFrame({f"fwd{h}": fwd[:, i] for i, h in enumerate(horizons)})
    out["adv21"] = adv
    return out


def skip(df: pd.DataFrame, cols, k: int = SKIP) -> np.ndarray:
    """End the window k bars early — the "−1" of 12−1. Reading the surface k rows back is
    what turns a 12-month scan into a 12−1 scan: no rescan, one shift. Grouped on `seg`,
    so the shift never reaches back across a level break."""
    return df.groupby("seg", observed=True, sort=False)[list(cols)].shift(k) \
             .to_numpy(np.float64)


def _trailing_vol(df: pd.DataFrame) -> np.ndarray:
    """Trailing 252-bar sd of the 1-bar return, min 126. The denominator of the flat
    baseline and, negated, the low-vol control."""
    starts, ends = _segbounds(df)
    r1 = np.diff(df.y.to_numpy(np.float64), prepend=0.0)
    r1[starts] = 0.0
    out = np.empty(len(df))
    _roll_std(r1, starts, ends, 252, 126, out)
    return out


def reductions(df: pd.DataFrame, grid=GRID) -> dict:
    """The signals scored by `evaluate`. This dict is a CHOICE, deliberately visible and
    deliberately not in the build.

    NOT here: `argmax_abs` and the ladder mean. §3.1/§3.2 killed both twice (the single
    top rung beats the ladder at every horizon at 27% of the turnover; a dense argmax grid
    is worth exactly 0.0000). They are settled, and re-adding them would re-litigate."""
    cols = [f"tb_{L}" for L in grid]
    S = skip(df, cols)
    out = {"tb_top": S[:, -1]}                              # the 252 rung, 12−1
    for L in (21, 63, 126):                                 # §3.1's rung question,
        out[f"tb_{L}"] = S[:, list(grid).index(L)]          # re-asked on clean data

    # The unfitted baselines every trend column must beat, over the IDENTICAL window
    # [t−273, t−21] the reductions above use. §4's recorded failure mode was killing a
    # baseline without ever measuring it, so these are not optional.
    g = df.groupby("seg", observed=True, sort=False)["y"]
    ret = (g.shift(SKIP) - g.shift(SKIP + 252)).to_numpy(np.float64)
    vol = _trailing_vol(df)
    out["flat_ret"] = ret
    out["flat_ret_vol"] = ret / vol
    out["nvol"] = -vol                                      # low-vol control, no skip
    return {k: v for k, v in out.items() if np.isfinite(v).any()}


# ── evaluation harness ──────────────────────────────────────────────────────
# Every IC in this project comes through here, so a convention fixed here is fixed
# everywhere: 1-bar lag, per-date Spearman, Newey-West at ceil(1.5h).
def _nw_lag(h: int) -> int:
    """Bartlett bandwidth for an h-overlapping series. Overlapping forward returns induce
    an MA(h−1) in the IC series, so h−1 is the bare MINIMUM that covers it and 1.5h is the
    usual conservative choice. An earlier revision used `lag = h`, which understates the
    long-run variance and inflates t most at h63 — the horizon carrying the largest
    claims."""
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
    """Shift each signal forward `lag` bars WITHIN a segment. In place.

    Pairing signal(t) with the return measured FROM t grants execution at the very close
    that produced the signal — free, instantaneous fills. BACKTEST.py's convention is that
    a signal at t earns y[t+1+h] − y[t+1]; shifting the signal one bar and leaving fwd
    where it is reproduces that exactly, without rebuilding the targets.

    Must run BEFORE the liquidity screen — the screen drops rows and so breaks the bar
    contiguity this shift depends on."""
    if lag <= 0:
        return df
    df[list(cols)] = df.groupby("seg", observed=True, sort=False)[list(cols)].shift(lag)
    return df


def ic_series(df: pd.DataFrame, names, horizons=HORIZONS):
    """Cross-sectional Spearman IC per date → {(name, h): Series indexed by date}."""
    g = df.groupby("date", observed=True)
    R = pd.DataFrame({c: g[c].rank(pct=True)
                      for c in list(names) + [f"fwd{h}" for h in horizons]})
    k = df.date.to_numpy()
    D = {c: R[c] - R[c].groupby(k, observed=True).transform("mean") for c in R.columns}

    return ({(c, h): _ic_from(D[c], D[f"fwd{h}"], k) for c in names for h in horizons},
            R, D, k)


def _ic_from(x, yv, k):
    """Per-date correlation of two ALREADY per-date-demeaned rank vectors. Factored out so
    that a neutralised signal is scored by the exact same statistic as a raw one — the
    residual of a rank is no longer a rank, and re-ranking it would silently change the
    estimator mid-comparison."""
    num = (x * yv).groupby(k, observed=True).sum()
    den = np.sqrt((x * x).groupby(k, observed=True).sum()
                  * (yv * yv).groupby(k, observed=True).sum())
    return (num / den).replace([np.inf, -np.inf], np.nan)


def _proj_out(x, z, k):
    """Strip the per-date OLS projection of x onto z. Both demeaned per date."""
    num = (x * z).groupby(k, observed=True).transform("sum")
    den = (z * z).groupby(k, observed=True).transform("sum")
    return x - z * (num / den.replace(0.0, np.nan))


def _resid(x, controls, k):
    """Exact per-date JOINT OLS residual of x on `controls`, by Gram-Schmidt: orthogonalise
    the controls against each other first and repeated single projections are identical to
    a joint solve — no per-date matrix, no lstsq over 6,900 dates. Projecting out
    correlated controls one at a time WITHOUT that step would not be."""
    basis = []
    for z in controls:
        for b in basis:
            z = _proj_out(z, b, k)
        basis.append(z)
    for b in basis:
        x = _proj_out(x, b, k)
    return x


def ic_table(S, names, horizons=HORIZONS, indent="  ") -> None:
    print(f"{indent}{'signal':16s}"
          + "".join(f"{'IC_h' + str(h):>10s}{'t':>7s}" for h in horizons))
    for c in names:
        print(f"{indent}{c:16s}" + "".join(
            f"{S[(c, h)].mean():10.4f}{_nw_t(S[(c, h)].to_numpy(), _nw_lag(h)):7.2f}"
            for h in horizons), flush=True)


def paired_table(S, pairs, horizons=HORIZONS, indent="  ") -> None:
    """NW t of the DIFFERENCE series. Separate t-stats do not test whether a gap is real:
    the IC series share dates and are ~0.8 correlated, so the paired difference is far
    better determined than either level."""
    for a, b in pairs:
        d = {h: (S[(a, h)] - S[(b, h)]).dropna() for h in horizons}
        print(f"{indent}{a:14s} − {b:14s}" + "".join(
            f"{d[h].mean():10.4f}{_nw_t(d[h].to_numpy(), _nw_lag(h)):7.2f}"
            for h in horizons))


def load(df: pd.DataFrame | None = None, grid=GRID, lag: int = LAG) -> pd.DataFrame:
    """Scan, reduce, lag, then screen. Order matters — the reductions need raw bar
    contiguity, and so does the lag, and the screen destroys it."""
    t0 = time.time()
    if df is None:
        df = build(grid=grid)
    df = df.sort_values(["seg", "date"], kind="stable").reset_index(drop=True)
    n0 = len(df)
    red = reductions(df, grid)
    out = pd.concat([df[["security_id", "seg", "date", "close"]],
                     targets(df), pd.DataFrame(red)], axis=1)
    apply_lag(out, list(red), lag)
    out = out[(out.adv21 >= MIN_ADV) & (out.close >= MIN_PRICE)]
    print(f"\n[load] {n0:,} → {len(out):,} rows  (lag {lag}b, ADV ≥ ${MIN_ADV:,.0f}, "
          f"px ≥ ${MIN_PRICE:.0f}), {time.time() - t0:.1f}s")
    print(f"[load] {out.date.min():%Y-%m-%d} → {out.date.max():%Y-%m-%d}, "
          f"{out.security_id.nunique():,} securities, {out.date.nunique():,} dates\n",
          flush=True)
    return out


ERAS = [("2000-2008", 2000, 2008), ("2009-2016", 2009, 2016), ("2017-2026", 2017, 2026)]


def evaluate(df: pd.DataFrame | None = None, grid=GRID) -> None:
    t0 = time.time()
    d = load(df, grid)
    NAMES = [c for c in d.columns
             if c not in ("security_id", "seg", "date", "close", "adv21")
             and not c.startswith("fwd")]
    d = d.dropna(subset=NAMES + [f"fwd{h}" for h in HORIZONS], how="all")
    S, R, D, kd = ic_series(d, NAMES)

    print("=" * 84)
    print(f"1. INFORMATION COEFFICIENT  (cross-sectional Spearman, {LAG}-bar lag, "
          f"NW at 1.5h)")
    print("=" * 84)
    ic_table(S, NAMES)

    print("\n  PAIRED vs tb_top — NW t of the DIFFERENCE. This is the only test that")
    print("  says whether a gap is real; the levels above are ~0.8 correlated.")
    paired_table(S, [(c, "tb_top") for c in NAMES if c != "tb_top"])

    print("\n" + "=" * 84)
    print("2. TURNOVER  (mean |Δ per-date rank| per name-day)")
    print("=" * 84)
    # On RANKS, not levels. Normalising |Δsignal| by σ(signal) is what an earlier
    # revision did, and it is unusable on a ratio signal: `flat_ret_vol` divides by a
    # trailing vol that can be near zero, so a few enormous values inflate σ and the
    # ratio collapses to 0.0001 — a signal that looks free to trade because its own
    # outliers set the scale. The book trades per-date ranks, so measure those.
    for c in NAMES:
        s = d.assign(_r=R[c]).groupby("seg", observed=True)["_r"].diff().abs().mean()
        print(f"  {c:16s}{s:8.4f}")
    print("  Cost is proportional to this; nothing here is net of spreads. `cost` is.")

    print("\n" + "=" * 84)
    print("3. STABILITY  (IC by era — a full-sample mean can hide a dead decade)")
    print("=" * 84)
    print(f"  {'signal':16s}" + "".join(f"{e + ' h1/h21':>24s}" for e, _, _ in ERAS))
    for c in NAMES:
        row = ""
        for _, lo, hi in ERAS:
            for h in (1, 21):
                s = S[(c, h)]
                m = ((s.index >= pd.Timestamp(f"{lo}-01-01"))
                     & (s.index <= pd.Timestamp(f"{hi}-12-31")))
                row += f"{s[m].mean():12.4f}"
        print(f"  {c:16s}{row}")

    print("\n" + "=" * 84)
    print("4. DELISTING TRUNCATION  (fwd{h} is NaN in a segment's final h bars, so a")
    print("   name's terminal decline is excluded from every h21/h63 IC)")
    print("=" * 84)
    last = d.groupby("seg", observed=True).date.transform("max")
    dead_seg = d.loc[last < d.date.max() - pd.Timedelta(days=90), "seg"].unique()
    dead = d.seg.isin(dead_seg)
    print(f"  segments ending >90d before panel end: {len(dead_seg):,} of "
          f"{d.seg.nunique():,}  ({dead.mean():.2%} of rows)")
    for h in (21, 63):
        print(f"  fwd{h}: {d.loc[dead, f'fwd{h}'].isna().mean():.2%} of those rows have "
              f"no target at all — the directly truncated share")
    print("\n  That share is the whole of the IC-side exposure, and it is small. Do NOT")
    print("  read a survivors-only IC as the bias: dropping every dying name removes")
    print("  ~41% of rows concentrated in the highest-IC era, so the gap it shows is")
    print("  era mix, not truncation. The consequence that matters is P&L, where a")
    print("  departed name is held FLAT rather than written down — measured in `cost`,")
    print("  which runs with and without `delist_return`.")

    print(f"\n[eval] done in {time.time() - t0:.1f}s")


# ── signal diagnostics ─────────────────────────────────────
# `eval` asks "which column has the highest IC". That is a ranking question. These are the
# questions you ask of a SIGNAL, and none of them involve a portfolio:
#
#   1. does it still carry information once the cheap baseline is projected out?
#   2. where does the IC peak in horizon?
#   3. is the relationship monotone across the cross-section, or only in the tail?
#   4. is it a trend signal, or a low-vol / small-name bet in a trend costume?
#
# Q1 is the one that matters. A signal that dies when you project out 12-1/vol is not a
# trend signal, it is an expensive way to compute 12-1/vol.
CONTROLS = {
    "flat_ret_vol": ["flat_ret_vol"],           # the cheap baseline
    "vol+size":     ["nvol", "ladv"],           # is it really low-vol / small-name?
    "all":          ["flat_ret_vol", "nvol", "ladv"],
}


def signal_diag(df: pd.DataFrame | None = None, grid=GRID,
                names=("tb_top",)) -> None:
    t0 = time.time()
    d = load(df, grid)
    d = d.assign(ladv=np.log(d.adv21.to_numpy(np.float64)))
    names = [c for c in names if c in d.columns]
    need = list(names) + ["flat_ret_vol", "nvol", "ladv"]
    S, R, D, k = ic_series(d, need)

    print("=" * 84)
    print("1. RAW IC BY HORIZON  (where does it peak? h126/h252 are new)")
    print("=" * 84)
    ic_table(S, need)
    print("\n  h252's NW t rests on ~27 independent observations. Read the IC, not the t.")

    print("\n" + "=" * 84)
    print("2. NEUTRALISED IC  - per-date JOINT OLS residual on the controls, scored by")
    print("   the identical statistic. THE test: what survives the cheap alternative?")
    print("=" * 84)
    for label, ctrl in CONTROLS.items():
        ctrl = [c for c in ctrl if c in D]
        print(f"\n  after projecting out: {', '.join(ctrl)}")
        Sn = {}
        for c in names:
            r = _resid(D[c], [D[z] for z in ctrl], k)
            for h in HORIZONS:
                Sn[(c, h)] = _ic_from(r, D[f"fwd{h}"], k)
        ic_table(Sn, names, indent="    ")
        for c in names:
            row = ""
            for h in HORIZONS:
                raw = S[(c, h)].mean()
                row += f"{Sn[(c, h)].mean() / raw:17.0%}" if abs(raw) > 1e-9 else f"{'-':>17s}"
            print(f"    {c:14s} retained" + row)

    print("\n" + "=" * 84)
    print("3. DECILE MONOTONICITY  (mean forward return in bp, per-date deciles then")
    print("   averaged over dates - a tail-only signal is a fragile one)")
    print("=" * 84)
    for c in names:
        # A row with no signal yet (the top rung needs 273 bars) has a NaN pct-rank, and
        # NaN.astype(int) is 0 on this platform — silently dumping every history-less row
        # into D1 and poisoning the spread. Mask, never cast blind.
        rr = R[c].to_numpy()
        ok = np.isfinite(rr)
        dec = np.minimum((rr[ok] * 10).astype(int), 9)
        sub = d.loc[ok]
        print(f"\n  {c}  ({ok.mean():.1%} of rows have a signal)")
        print("    h   " + "".join(f"{'D' + str(i + 1):>8s}" for i in range(10))
              + f"{'D10-D1':>10s}{'mono':>8s}")
        for h in (21, 63, 252):
            m = (sub.assign(_d=dec).groupby(["date", "_d"], observed=True)[f"fwd{h}"]
                 .mean().groupby("_d").mean() * 1e4)
            v = np.array([m.get(i, np.nan) for i in range(10)], dtype=float)
            up = int(np.nansum(np.diff(v) > 0))
            print(f"    {h:<4d}" + "".join(f"{x:8.0f}" for x in v)
                  + f"{v[9] - v[0]:10.0f}{str(up) + '/9':>8s}")

    print("\n  `mono` counts rising steps out of 9. 9/9 is monotone; ~5/9 is noise around")
    print("  a tail effect. Deciles are equal-COUNT, not equal-risk - a monotone table")
    print("  with a vol gradient across it is still partly a vol bet, which is what")
    print("  section 2's vol+size row is there to catch.")
    print(f"\n[signal] done in {time.time() - t0:.1f}s")


# ── the arbiter ─────────────────────────────────────────────────────────────
def cost(df: pd.DataFrame | None = None, capital: float = 5e7,
         top: float = 0.10, rebal: int = 21) -> None:
    """BACKTEST.py with spreads and commissions. Nothing above is net of costs; this is.

    Two books, both long-only equal-weight on the top decile, monthly:
      tb_top   the candidate            (LOSS.md §3's configuration)
      flat_ret the unfitted baseline    (plain 12−1 return, no scan at all)

    If the candidate does not beat the baseline HERE, the IC tables are decoration."""
    sys.path.insert(0, "research")
    from BACKTEST import backtest, spread_costs

    d = load(df)
    for c in ("tb_top", "flat_ret"):
        if c not in d.columns:
            sys.exit(f"{c} missing from reductions")

    d = d.assign(tk=d.security_id.astype(str))
    px = d.pivot_table(index="date", columns="tk", values="close", aggfunc="last")
    dv = d.pivot_table(index="date", columns="tk", values="adv21", aggfunc="last")
    tc = spread_costs(dv)
    exec_dates = px.index[::rebal]
    # Names that stop printing well before the panel ends have delisted. Writing them
    # off is not optional on a long-only book: LOSS.md §5.4 flags hold-flat as an
    # upward bias that has never been fixed, and BACKTEST.py takes the fix directly.
    last = d.groupby("tk").date.max()
    dead = last[last < d.date.max() - pd.Timedelta(days=90)]

    wcache: dict[str, pd.DataFrame] = {}

    def weights(name):
        if name not in wcache:       # a 17M-row pivot; the delist comparison reuses it
            sig = d.pivot_table(index="date", columns="tk", values=name, aggfunc="last") \
                   .reindex(index=px.index, columns=px.columns)
            w = (sig.rank(axis=1, pct=True) >= 1.0 - top).astype(float)
            wcache[name] = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        return wcache[name]

    def run(name, delist):
        w = weights(name)
        return backtest(
            w, px, freq=252, lag=LAG, signal_dates=list(exec_dates),
            transaction_cost=tc, capital=capital, dollar_volume=dv,
            impact_coef=0.02,            # ≈ 1.0 × daily σ of liquid US equities. NOT a
            borrow_fee=0.0,              # round number — results are quadratic in it.
            delist_return=-0.30 if delist else None,
            delist_dates=last.loc[dead.index] if delist else None,
            track_trades=False)

    books = ("tb_top", "flat_ret")
    # `turnover` and `cost_frac` are per-date Series in BACKTEST's dict; the annualised
    # scalars are what belongs in a summary row.
    keys = ["ann_return", "ann_vol", "sharpe", "max_drawdown",
            "ann_turnover", "ann_cost_drag"]

    def f(v):                        # a Series here once cost a 10-minute run
        return f"{float(v):13.4f}" if np.isscalar(v) or np.ndim(v) == 0 else f"{'—':>13s}"
    print("\n" + "=" * 84)
    print(f"NET OF COSTS  (long-only top {top:.0%}, rebal {rebal}b, ${capital:,.0f}, "
          f"IBKR + spreads,")
    print(f"               impact 0.02, borrow 0.0)")
    print("=" * 84)
    print(f"  {'book':10s}" + "".join(f"{k:>13s}" for k in keys))
    res = {}
    for name in books:
        res[name] = run(name, True)
        print(f"  {name:10s}" + "".join(f(res[name].get(k, np.nan)) for k in keys))

    print(f"\n  Delisting: −30% terminal write-off on {len(dead):,} names that stop")
    print("  printing >90d before the panel ends. LOSS.md §5.4 flags hold-flat as a")
    print("  never-fixed upward bias on a long-only book; this is the size of it.")
    print(f"  {'book':10s}{'Sharpe w/ delist':>18s}{'hold-flat':>12s}{'Δ':>9s}")
    for name in books:
        a = res[name].get("sharpe", np.nan)
        b = run(name, False).get("sharpe", np.nan)
        print(f"  {name:10s}{a:18.4f}{b:12.4f}{b - a:9.4f}")

    print("\n  The unfitted baseline `flat_ret` is in the table on purpose. This")
    print("  configuration is the survivor of ~40 comparisons on ONE panel: quote the")
    print("  trial count with any Sharpe taken from it, and deflate (LdP ch14) before")
    print("  believing it. A candidate that does not beat `flat_ret` HERE has nothing.")


def selfcheck() -> None:
    """Two silent-failure modes, both of which produce plausible numbers."""
    rng = np.random.default_rng(0)

    # Neutralisation. Two properties define it, and a silent break in either would show up
    # as a plausible-looking IC rather than an error, which is exactly the failure this
    # project keeps having.
    nd, ns = 120, 40
    dt = pd.date_range("2010-01-01", periods=nd, freq="B")
    zz = [rng.normal(0, 1, nd) for _ in range(ns)]
    dd = pd.concat([pd.DataFrame({"date": dt, "x": zz[s], "c": 0.3 * zz[s] +
                                  rng.normal(0, 1, nd), "fwd1": rng.normal(0, 1, nd)})
                    for s in range(ns)], ignore_index=True)
    _, _, Dv, kv = ic_series(dd, ["x", "c"], horizons=(1,))
    r = _resid(Dv["x"], [Dv["c"]], kv)
    dot = float((r * Dv["c"]).groupby(kv, observed=True).sum().abs().max())
    assert dot < 1e-10, f"residual not orthogonal to control per date: {dot:.2e}"
    r0 = _resid(Dv["x"], [Dv["x"]], kv)          # neutralise a signal by itself
    assert float((r0 * r0).sum()) == 0.0, "self-neutralisation left something behind"

    # NaN.astype(int) is 0 on arm64 and INT_MIN on x86 — either way it is not a decile.
    # This bit the decile table once by quietly filing every history-less row under D1.
    rr = np.array([0.05, 0.95, np.nan])
    ok = np.isfinite(rr)
    assert ok.sum() == 2 and set(np.minimum((rr[ok] * 10).astype(int), 9)) == {0, 9}, \
        "decile bucketing must mask NaN ranks, not cast them"

    print("selfcheck ok")


if __name__ == "__main__":
    cmd = sys.argv[1:2] or ["eval"]
    if cmd == ["check"]:
        selfcheck()
    elif cmd == ["eval"]:
        evaluate()
    elif cmd == ["signal"]:
        signal_diag()
    elif cmd == ["cost"]:
        cost()
    else:
        sys.exit(f"unknown command {cmd[0]!r}; try: check | eval | signal | cost")
