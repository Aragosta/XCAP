"""
trendscan.py — a rolling OLS trend scan. The signal, and nothing else.

At every (asset, date) it fits log adjusted price on time over a ladder of trailing
window lengths and emits ONE number per rung:

    tb_L   slope t-stat

That is the whole surface. No second path, no scale column, no target, no reduction, no
scoring. Everything you might do WITH a t — absolute magnitude, cross-rung rescaling,
ranking, skip, liquidity screening, forward horizons, deciles, costs — happens after the
fact and is not in this tree right now. It was cut deliberately (see TRENDSCAN.md's
status note); recover it from git history rather than rewriting it from memory:

    git show 9810fa7:research/trendscan_eval.py

Everything uses data ≤ t: López de Prado's trend scanning is a forward-looking
*labelling* method, this is the same machinery run backward.

What the build still does, and why it is not "extra": it rejects bars before it fits
anything (§2.1). A spurious print corrupts every window containing it and there is no
downstream place to repair it.

The surface is not persisted: the scan is ~6s and a stale multi-GB parquet outlives the
harness that validated it.

Findings live in TRENDSCAN.md. Nothing here scores anything.

    python research/trendscan.py check    # kernel + lstsq agreement + null scale
    python research/trendscan.py build    # scan, print, discard (add a path to write)
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from numba import njit, prange

PANEL = "data/_staging/alpha_panel.parquet"

# Roughly geometric. Adjacent rungs are near-redundant (117 daily-spaced windows measured
# N_eff 2.14), so finer spacing buys columns, not information. TRENDSCAN.md §3.2.
GRID = np.array([5, 10, 21, 42, 63, 84, 126, 168, 210, 252], dtype=np.int64)

# Winsorising the returns fed to the filter is a real thing and it measured as the signal
# (§3.5, §3.13). We are not doing it here: this file emits the raw fit, nothing else.

RESYNC = 2048                                       # exact-recompute cadence, anti-drift

# Bar integrity — the vendor ships isolated bars belonging to a DIFFERENT instrument, and
# the ADV screen actively selects for them. Detected on ADJUSTED price and by REVERSAL,
# which is what makes it split-safe. Worked examples and counts: TRENDSCAN.md §2.1.
BAD_SPIKE = 0.80       # |log return| a single bar must not exceed unexplained (≈ ×2.2)
BAD_REVERT = 0.15      # ...and that unwinds to within this fraction of itself
BAD_WINDOW = 3         # ...within this many bars (a spike can plateau before reverting)
# A LEVEL BREAK (missed corporate action, or a recycled ticker) is not repairable bar by
# bar: the price base changes, so no window may span it. Cut, not dropped — the data on
# both sides is fine, only its continuity is not.
BAD_BREAK = 1.60       # |log return| that ends a series (≈ ×5 / −80%)
BAD_GAP = 365          # ...as does a calendar gap of this many days (suspension, relist)
MIN_SEG = 5            # segments shorter than the shortest rung carry no surface at all


# ── kernels ─────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def _ladder_one(y, ladder, tout, sout):
    """Slope t at each rung — levels, nothing collapsed.

    The design is centred on u = x − (L−1)/2, so Σu = 0 and Su2 = L(L²−1)/12 is a closed
    form, not a sum. Three rolling sums (Σy, Σy², Σi·y) carry the whole fit.

    `sout` is the residual sd. It is the t's denominator and is written out for the
    selfcheck only — it is a scale reading, not a trend-scan object, and `build` throws
    it away. Low-vol as an anomaly is §3.6's business, not this file's.

    Curvature (`tc`, the exact orthogonal complement of the slope) used to come out of a
    fourth sum Σi²·y here. It measured zero on returns on all ten rungs, so it and its
    rank-2 recursion are gone; TRENDSCAN.md §3.7 keeps the algebra."""
    n = y.shape[0]
    for k in range(ladder.shape[0]):
        L = ladder[k]
        for t in range(n):
            tout[t, k] = np.nan
            sout[t, k] = np.nan
        if n < L or L < 5:
            continue
        Lf = float(L)
        m = (Lf - 1.0) / 2.0
        Su2 = Lf * (Lf * Lf - 1.0) / 12.0
        Sy = 0.0
        Syy = 0.0
        W1 = 0.0
        for i in range(L):
            v = y[i]
            Sy += v
            Syy += v * v
            W1 += i * v
        for t in range(L - 1, n):
            if t > L - 1:
                if (t - (L - 1)) % RESYNC == 0:
                    Sy = 0.0
                    Syy = 0.0
                    W1 = 0.0
                    for i in range(L):
                        v = y[t - L + 1 + i]
                        Sy += v
                        Syy += v * v
                        W1 += i * v
                else:
                    yo = y[t - L]
                    yn = y[t]
                    W1 = W1 - Sy + yo + (Lf - 1.0) * yn
                    Sy = Sy - yo + yn
                    Syy = Syy - yo * yo + yn * yn
            Suy = W1 - m * Sy
            b = Suy / Su2
            sse1 = (Syy - Sy * Sy / Lf) - b * Suy
            sd = np.sqrt(sse1 / (Lf - 2.0)) if sse1 > 0.0 else 0.0
            sout[t, k] = sd
            # A degenerate denominator emits NaN, not 0.0: a flat window is an ABSENT
            # reading, and 0.0 lands it at mid-rank instead. 0.08% of screened rows have
            # sd_252 < 1e-4 and the bottom-1%-sd rows carry mean |tb_252| 33.7 vs 17.9.
            tout[t, k] = np.nan if sd <= 0.0 else b * np.sqrt(Su2) / sd


@njit(cache=True, parallel=True)
def _ladder_panel(y, starts, ends, ladder, tout, sout):
    for a in prange(starts.shape[0]):
        s, e = starts[a], ends[a]
        if e - s >= 2:
            _ladder_one(y[s:e], ladder, tout[s:e], sout[s:e])


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


# ── build ───────────────────────────────────────────────────────────────────
def build(panel: str = PANEL, out_path: str | None = None, grid=GRID) -> pd.DataFrame:
    """Scan the panel and return the surface. `out_path` writes a parquet if you really
    want one; nothing in this file does, because recomputing is cheaper than storing."""
    import duckdb

    t0 = time.time()
    df = duckdb.sql(f"""select security_id, date, close, adj, volume from '{panel}'
                        where adj > 0 and close > 0 order by security_id, date""").df()
    print(f"[build] {len(df):,} rows, {df.security_id.nunique():,} securities "
          f"({time.time() - t0:.1f}s)", flush=True)

    sid = df.security_id.to_numpy()
    b = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1], True])
    starts, ends = b[:-1].copy(), b[1:].copy()

    # Bar integrity, BEFORE anything is fitted. A spurious print corrupts every window
    # that contains it (up to 252 bars of tb); there is no downstream place to repair
    # it. See BAD_SPIKE.
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
    # series there, so no rolling window ever spans it. Downstream code is unchanged: it
    # already works on contiguous blocks, and a cut is simply one more block boundary.
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

    # Log adjusted price, centred per segment. Slope and t are invariant to a level
    # shift; centring only keeps the rolling sums well conditioned.
    y = np.log(df.adj.to_numpy(np.float64))
    for s, e in zip(starts, ends):
        y[s:e] -= y[s:e].mean()

    t1 = time.time()
    gl = np.asarray(grid, dtype=np.int64)
    shape = (len(df), len(grid))
    # float32 — this is the stored precision anyway, and a 19M×10 surface in float64 is
    # 1.5 GB of nothing.
    tb = np.full(shape, np.nan, dtype=np.float32)
    scratch = np.full(shape, np.nan, dtype=np.float32)   # residual sd: written, discarded
    _ladder_panel(y, starts, ends, gl, tb, scratch)
    del scratch
    print(f"[build] surface: {len(grid)} rungs in {time.time() - t1:.1f}s", flush=True)

    # `seg`, not `security_id`, is the unit of contiguity: a level break CUTS a security
    # into two series, and any shift or diff that spans the cut is reading across a
    # discontinuity the build exists to remove. Everything downstream groups on this.
    segid = np.empty(len(df), dtype=np.int32)
    for i, (s, e) in enumerate(zip(starts, ends)):
        segid[s:e] = i
    o = pd.DataFrame({
        "security_id": df.security_id.to_numpy(),
        "seg": segid,
        "date": df.date.to_numpy(),
        "close": df.close.to_numpy(np.float32),
        "volume": df.volume.to_numpy(np.float64),
        "y": y.astype(np.float32),          # centred log adj price. Targets, liquidity
    })                                      # and every flat baseline derive from these.
    # RAW. No skip baked in — the window length and any end-lag are two axes of one
    # 2-D surface; hardcoding the second collapses it.
    for j, L in enumerate(grid):
        o[f"tb_{L}"] = tb[:, j]

    if out_path:
        o.to_parquet(out_path, index=False)
        print(f"[build] wrote {out_path}", flush=True)
    print(f"[build] {len(o):,} rows × {len(o.columns)} cols ({len(grid)} rungs), "
          f"{time.time() - t0:.1f}s", flush=True)
    return o


def selfcheck() -> None:
    """The kernel is the only non-trivial branch left in this file."""
    rng = np.random.default_rng(0)

    # A degenerate window emits NaN, not 0.0.
    y = np.zeros(300)
    g = np.array([252], dtype=np.int64)
    t_, s_ = (np.full((300, 1), 7.0) for _ in range(2))
    _ladder_one(y, g, t_, s_)
    assert np.isnan(t_[299, 0]), "flat window did not emit NaN"
    assert np.isnan(t_[0, 0]), "pre-window rows not NaN-filled"

    # The rolling sums must reproduce a plain least-squares fit. This is what the removed
    # quadratic-orthogonality assertions were implicitly checking; with the quadratic term
    # gone, this is the only thing between the recursion and a silent wrong slope.
    for L in (5, 21, 252):
        u = np.arange(L) - (L - 1) / 2.0
        yv = rng.normal(0, 1, L).cumsum()
        t_, s_ = (np.empty((L, 1)) for _ in range(2))
        _ladder_one(yv, np.array([L], dtype=np.int64), t_, s_)
        X = np.column_stack([np.ones(L), u])
        resid = yv - X @ np.linalg.lstsq(X, yv, rcond=None)[0]
        sd_ref = np.sqrt(resid @ resid / (L - 2))
        b_ref = np.linalg.lstsq(X, yv, rcond=None)[0][1]
        t_ref = b_ref * np.sqrt(L * (L * L - 1) / 12.0) / sd_ref
        assert abs(s_[L - 1, 0] - sd_ref) < 1e-9 * max(sd_ref, 1.0), f"sd wrong at L={L}"
        assert abs(t_[L - 1, 0] - t_ref) < 1e-7 * max(abs(t_ref), 1.0), f"tb wrong at L={L}"

    # The O(1) update must not drift. Check well past a RESYNC boundary, not just the
    # first window — the recursion is the whole reason this file is fast.
    L, n = 21, RESYNC + 500
    yv = rng.normal(0, 0.01, n).cumsum()
    t_, s_ = (np.empty((n, 1)) for _ in range(2))
    _ladder_one(yv, np.array([L], dtype=np.int64), t_, s_)
    X = np.column_stack([np.ones(L), np.arange(L) - (L - 1) / 2.0])
    for t in (L - 1, RESYNC, RESYNC + 1, n - 1):
        w = yv[t - L + 1:t + 1]
        rr = w - X @ np.linalg.lstsq(X, w, rcond=None)[0]
        assert abs(s_[t, 0] - np.sqrt(rr @ rr / (L - 2))) < 1e-9, f"sums drifted at t={t}"

    # Null sd of tb_L under a driftless random walk is κ·√L, not 1 — the t divides by an
    # se assuming information accrues at L^1.5, under a unit root it accrues at √L. The
    # measured κ (§3.9) is not applied anywhere in this file, because rescaling is a
    # reduction; this asserts the kernel still produces the null it was measured on.
    L = 63
    yv = rng.normal(0, 0.02, (2000, L)).cumsum(1)
    tt = np.empty(2000)
    for i in range(2000):
        t_, s_ = (np.empty((L, 1)) for _ in range(2))
        _ladder_one(yv[i], np.array([L], dtype=np.int64), t_, s_)
        tt[i] = t_[L - 1, 0]
    k = tt.std() / np.sqrt(L)
    assert 1.26 < k < 1.54, f"null of tb_{L} is {k:.3f}·sqrt(L), not ~1.40·sqrt(L)"

    print("selfcheck ok")


if __name__ == "__main__":
    cmd = sys.argv[1:2] or ["check"]
    if cmd == ["check"]:
        selfcheck()
    elif cmd == ["build"]:
        build(out_path=sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        sys.exit(f"unknown command {cmd[0]!r}; try: check | build")
