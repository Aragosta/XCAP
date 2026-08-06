"""
BACKTEST.py — lean, generic portfolio backtester.

Public API
----------
    backtest(weights, prices, ...)         weights → equity, metrics, trades, yearly
    walk_forward(signal_fn, prices, ...)   rolling train→test out-of-sample harness
    spread_costs(dollar_volume)            liquidity-tiered one-way spread
    yearly_summary(result)                 per-calendar-year breakdown
    results_backtest(strategies, ...)      summary table + equity/drawdown chart

Conventions
-----------
- Resolution is DAILY. Rebalance less often via `signal_dates`; a monthly price panel
  cannot see an intra-month delisting, stop or gap.
- Drift mode: weights are set on execution dates and drift with returns in between.
- Timing: a signal at index t executes at t+lag+1, so weights decided with information
  up to t earn the return from t+lag → t+lag+1. On a daily panel `lag=1` is honest.
- Costs are ONE-WAY per unit traded notional: period cost is Σⱼ tcⱼ·|Δwⱼ|, charged on
  every share traded, both sides — not 0.5·Σ|Δw|.
- Cost reporting is an annualized drag in pp/yr (`ann_*_drag`), comparable to
  `ann_return`. `ann_cost_drag` is ALL-IN; commission and impact are components of it.
- Delisting returns are OFF by default: a name that leaves `prices` is held flat, not
  written down. On a survivorship-free panel that is the single largest inflator in
  this file — pass `delist_return` (-1.0, or a per-name Series) to book the write-off.
- NOT modelled: EUR/USD exposure for a EUR-based account holding USD stocks.
"""
import warnings

import numpy as np
import pandas as pd

try:                                              # optional JIT for the hot loop
    from numba import jit
except Exception:                                 # pragma: no cover - fallback
    def jit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco(args[0]) if (len(args) == 1 and callable(args[0]) and not kwargs) else deco


# ── cost model ──────────────────────────────────────────────────────────────
# One-way spread by trailing MONTHLY dollar volume. Passing a DAILY figure (~21×
# smaller) demotes every name a tier or two and roughly triples the modelled cost.
SPREAD_TIERS = (
    (1e9, 0.0005),   # ≥ $1B/mo  →  5 bps
    (1e8, 0.0010),   # ≥ $100M   → 10 bps
    (1e7, 0.0025),   # ≥ $10M    → 25 bps
    (1e6, 0.0060),   # ≥ $1M     → 60 bps
    (0.0, 0.0150),   # < $1M     → 150 bps
)

# IBKR Ireland (EU entity), tiered, US stocks — verified against interactivebrokers.ie,
# July 2026. USD 0.0035/sh + ~0.0002/sh exchange & clearing pass-through, USD 0.35 per
# order minimum, capped at 0.5% of trade value. The EU entity caps tiered at 0.5%, not
# the 1% the US entity publishes; near the $5 price floor that cap is often binding.
# Sell-side regulatory fees (SEC Section 31 ~0.2 bp of sale value, FINRA TAF
# ~$0.000195/sh) are inside the pass-through rounding.
IBKR = dict(commission_per_share=0.0037, commission_min=0.35, commission_max_pct=0.005)


def spread_costs(dollar_volume: pd.DataFrame, *, tiers: tuple = SPREAD_TIERS,
                 lookback: int = 3) -> pd.DataFrame:
    """Per-name one-way spread (fraction of traded notional) from trailing MONTHLY
    dollar volume. Each name gets the cost of the highest threshold it clears;
    NaN volume → worst tier, so a data gap can never flatter the run."""
    dv = dollar_volume.rolling(lookback, min_periods=1).mean()
    out = pd.DataFrame(max(c for _, c in tiers), index=dv.index, columns=dv.columns)
    for thresh, c in sorted(tiers, key=lambda x: x[0]):       # ascending → highest wins
        out = out.mask(dv >= thresh, c)
    return out


# ── input coercion ──────────────────────────────────────────────────────────
def _as_prices_df(prices) -> pd.DataFrame:
    if isinstance(prices, pd.Series):
        return prices.to_frame()
    if isinstance(prices, pd.DataFrame):
        return prices
    raise TypeError("`prices` must be a pandas Series or DataFrame.")


def _as_weights_df(weights, *, index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    """Coerce weights (DataFrame/Series/array) to a dates×tickers DataFrame on `index`."""
    if len(index) == 0:
        raise ValueError("`prices` has no rows.")

    if isinstance(weights, pd.DataFrame):
        w = weights.reindex(columns=columns)
    elif isinstance(weights, pd.Series):
        if weights.index.difference(columns).empty:          # static weights by ticker
            w = weights.reindex(columns).astype(float).to_frame().T.set_axis([index[0]])
        elif len(columns) == 1 and weights.index.difference(index).empty:  # single asset, by date
            w = weights.to_frame(columns[0])
        else:
            raise ValueError("`weights` Series must be indexed by tickers, or by dates when single-asset.")
    else:
        arr = np.asarray(weights, dtype=float)
        if arr.ndim == 1:
            if arr.shape[0] != len(columns):
                raise ValueError("1D `weights` length must match number of `prices` columns.")
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != len(columns) or arr.shape[0] not in (1, len(index)):
            raise ValueError("`weights` array must be 1D (N,) or 2D (T, N) aligned to prices.")
        rows = index if arr.shape[0] == len(index) else [index[0]]
        w = pd.DataFrame(arr, index=rows, columns=columns)

    # Every branch above produces rows stamped on some subset of dates; they all end the
    # same way — hold each row until the next one, treat anything unset as flat.
    if w.index.equals(index):
        return w.fillna(0.0)
    return w.sort_index().reindex(index=index, method="ffill").fillna(0.0)


def _at_exec(val, exec_dates: pd.Index, cols: pd.Index, *, fill=None) -> np.ndarray:
    """Align a scalar, a by-ticker Series or a (dates×tickers) frame to the (n_exec, N)
    matrix of values in force at each execution date.

    Rows are selected BEFORE columns: reindexing to the full daily index first would
    materialize the whole (T, N) matrix, which is the allocation this design exists to
    avoid. `method="ffill"` on a sorted source gives the last value at or before each
    execution date — identical to ffill-then-slice, without the big intermediate.

    `fill` decides what an unknown entry becomes, and every use of it fails CLOSED —
    "max" for a rate (worst spread seen) and "min" for a capacity (thinnest volume
    seen), so a hole in the input can never flatter the run. `fill=None` leaves NaN
    for the caller to handle.
    """
    n = len(exec_dates)
    if not isinstance(val, (pd.DataFrame, pd.Series)):
        if val < 0:
            raise ValueError("rate must be >= 0.")
        return np.full((n, len(cols)), float(val))

    src = val if isinstance(val, pd.DataFrame) else val.to_frame().T
    if not src.index.is_monotonic_increasing:
        src = src.sort_index()
    aligned = (src.reindex(index=exec_dates, method="ffill")
               if isinstance(src.index, pd.DatetimeIndex) and not src.index.equals(exec_dates)
               else src)
    out = aligned.reindex(columns=cols).to_numpy(float)
    if out.shape[0] == 1 != n:               # by-ticker Series: one row, held everywhere
        out = np.repeat(out, n, axis=0)
    if out.shape[0] != n:
        raise ValueError("execution input must be date-indexed or a single row.")

    if fill is not None:
        # Worst case over the WHOLE source, not just the execution rows — an unknown
        # entry is unknown against everything that was ever observed.
        known = src.to_numpy(float)
        known = known[np.isfinite(known)]
        if known.size:
            out = np.where(np.isnan(out), known.min() if fill == "min" else known.max(), out)
    return out


def _check_freq(idx: pd.Index, freq: float) -> None:
    """Warn when `freq` contradicts the actual spacing of the price index; every
    annualized metric scales by √freq, so a mismatch silently rescales Sharpe."""
    if len(idx) < 3 or not isinstance(idx, pd.DatetimeIndex):
        return
    span_years = (idx[-1] - idx[0]).days / 365.25
    if span_years <= 0:
        return
    implied = (len(idx) - 1) / span_years
    if not (0.5 * freq <= implied <= 2.0 * freq):
        warnings.warn(
            f"`prices` has ~{implied:.0f} rows/year but freq={freq:g}. "
            f"Annualized metrics will be wrong by √({implied / freq:.1f}).",
            stacklevel=3,
        )


# ── core engine (JIT) ───────────────────────────────────────────────────────
@jit(nopython=True, cache=True)
def _drift_core(rets, ex_px, ex_adv, ex_w, ex_tc, short_mult,
                bf_arr, bf_scalar, bf_is_arr, ppy, exec_slot,
                capital, cps, cmin, cmaxpct, impact_coef,
                track, w_pre, w_post, tr_cost, n_dates, n_assets):
    """
    Drift backtest: one-way trading cost (spread + commission + impact) on every
    rebalance, plus a short-borrow fee every period on short notional.

    Inputs prefixed `ex_` are indexed by EXECUTION SLOT, not by date — they are read
    only on rebalance rows, so on a daily panel with monthly rebalancing they are ~20×
    smaller than a full (T, N) matrix. `exec_slot[i]` maps date i to its slot, else -1.
    `rets` is the only full-size input; `bf_arr` is too, hence the scalar fast path.

    Returns equity, turnover_gross, cost_frac, borrow_frac, comm_frac, impact_frac.
    cost_frac is ALL-IN; comm_frac and impact_frac are components of it.
    """
    equity = np.ones(n_dates)
    turnover_gross = np.zeros(n_dates)
    cost_frac = np.zeros(n_dates)
    borrow_frac = np.zeros(n_dates)
    comm_frac = np.zeros(n_dates)
    impact_frac = np.zeros(n_dates)
    w = np.zeros(n_assets)

    for i in range(n_dates):
        s = exec_slot[i]
        if s >= 0:                                 # rebalance at start of period i (i ≥ 1)
            t_over = 0.0
            cost = 0.0
            comm = 0.0
            imp = 0.0
            # Orders are sized on the book at the close of the PRIOR period, since the
            # trade happens at the start of period i. Mirrored in _build_blotter.
            book = equity[i - 1] * capital
            for j in range(n_assets):
                tgt = ex_w[s, j]
                dwj = tgt - w[j]
                adw = dwj if dwj >= 0.0 else -dwj
                if adw == 0.0:
                    continue
                t_over += adw
                if track:
                    w_pre[s, j] = w[j]
                    w_post[s, j] = tgt
                cj = ex_tc[s, j] * short_mult if tgt < 0.0 else ex_tc[s, j]

                # Per-share commission and square-root impact are non-linear in order
                # size, so they only exist once `capital` is known: the same weight
                # change is a different cost at $1M and at $1B.
                if book > 0.0:
                    notional = book * adw
                    p = ex_px[s, j]
                    if cps > 0.0 and p == p and p > 0.0:
                        fee = cps * (notional / p)
                        if fee < cmin:
                            fee = cmin
                        if cmaxpct > 0.0 and fee > cmaxpct * notional:
                            fee = cmaxpct * notional
                        cj += fee / notional
                        comm += fee / book
                    if impact_coef > 0.0:
                        a = ex_adv[s, j]
                        if a == a and a > 0.0:
                            ik = impact_coef * np.sqrt(notional / a)
                            cj += ik
                            imp += ik * adw

                cost += cj * adw
                if track:
                    tr_cost[s, j] = cj
                w[j] = tgt
            if cost > 0.999:
                cost = 0.999
            turnover_gross[i] = t_over
            cost_frac[i] = cost
            comm_frac[i] = comm
            impact_frac[i] = imp

        if i > 0:
            rp = 0.0
            borrow = 0.0
            for j in range(n_assets):
                r = rets[i, j]
                if r == r:                          # not NaN
                    rp += w[j] * r
                # Weights drift by (1+r)/(1+rp) with both factors ≥ 0, so a weight is
                # negative only if it was set negative — no separate "any short" flag.
                if w[j] < 0.0:
                    borrow += (-w[j]) * (bf_arr[i, j] if bf_is_arr else bf_scalar)
            borrow /= ppy
            if borrow > 0.999:
                borrow = 0.999
            borrow_frac[i] = borrow
            equity[i] = equity[i - 1] * (1.0 + rp)
            if s >= 0:
                equity[i] *= (1.0 - cost_frac[i])
            equity[i] *= (1.0 - borrow)
            denom = 1.0 + rp
            if denom > 0.0 and np.isfinite(denom):  # drift weights to next period
                for j in range(n_assets):
                    r = rets[i, j]
                    if r != r:
                        r = 0.0
                    w[j] = w[j] * (1.0 + r) / denom

    return equity, turnover_gross, cost_frac, borrow_frac, comm_frac, impact_frac


# ── metrics ─────────────────────────────────────────────────────────────────
def _tail_mean(sorted_vals: np.ndarray, pct: float) -> float:
    """Mean of the worst `1-pct` fraction of an ascending-sorted array (CVaR / CDaR)."""
    if len(sorted_vals) == 0:
        return np.nan
    k = int(np.ceil((1.0 - pct) * len(sorted_vals)))
    tail = sorted_vals[:k] if 0 < k < len(sorted_vals) else sorted_vals
    return float(np.mean(tail))


def _ann_drag(frac: pd.Series, freq: float, n: int) -> float:
    """Annualized compounded drag (pp/yr) of a per-period cost-fraction series."""
    if n <= 0:
        return np.nan
    keep = float(np.prod(1.0 - np.clip(frac.to_numpy()[1:], 0.0, 0.999)))
    return -1.0 if keep <= 0 else 1.0 - keep ** (freq / n)


def _metrics(equity, turnover, cost_frac, borrow_frac, comm_frac, impact_frac, freq, rf) -> dict:
    """Performance/risk metrics from an equity curve and its cost series."""
    net_ret = equity.pct_change(fill_method=None).fillna(0.0).rename("returns")
    net_ret.iloc[0] = 0.0
    drawdown = equity / equity.cummax() - 1.0

    # Exclude the structural period-0 zero (no position yet) but keep genuine interior
    # zero-return periods — dropping those would inflate Sharpe.
    r = net_ret.to_numpy()[1:]
    n = len(r)
    sd = float(np.std(r, ddof=1)) if n > 1 else np.nan
    total_factor = float(equity.iloc[-1] / equity.iloc[0])
    # A book losing >100% drives equity <= 0, and a negative base to a fractional power
    # is complex -> every downstream metric raises TypeError. Report a total loss.
    if not np.isfinite(total_factor) or total_factor <= 0:
        ann_return = -1.0
    else:
        ann_return = total_factor ** (freq / n) - 1.0 if n > 0 else np.nan

    rf_per = (1.0 + rf) ** (1.0 / freq) - 1.0
    ann_vol = sd * np.sqrt(freq) if n > 1 else np.nan
    # Shared numerator for Sharpe and Sortino. Using geometric ann_return for one and
    # arithmetic for the other makes them incomparable: at 30% vol they differ ~4.5pp.
    ann_excess = (float(r.mean()) - rf_per) * freq if n else np.nan
    sharpe = float(ann_excess / ann_vol) if (n > 1 and ann_vol > 0) else np.nan

    neg = r[r < 0]
    downside = float(np.std(neg, ddof=1) * np.sqrt(freq)) if len(neg) > 1 else np.nan
    sortino = float(ann_excess / downside) if (downside > 0) else np.nan

    cvar = _tail_mean(np.sort(r), 0.95)
    avg_turnover = float(turnover.mean())

    return {
        "returns": net_ret,
        "equity": equity,
        "drawdown": drawdown,
        "total_return": total_factor - 1.0,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": float(drawdown.min()),
        "avg_drawdown": float(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0,
        "cdar": _tail_mean(np.sort(drawdown.to_numpy()), 0.95),
        "cvar": cvar,
        "cvar_ann": cvar * np.sqrt(freq) if np.isfinite(cvar) else np.nan,
        "downside_deviation": downside,
        "expectancy": float(r.mean()) if n else np.nan,
        "win_rate": float((r > 0).mean()) if n else 0.0,
        "turnover": turnover,
        "cost_frac": cost_frac,
        "borrow_frac": borrow_frac,
        "commission_frac": comm_frac,
        "impact_frac": impact_frac,
        "avg_turnover": avg_turnover,
        "ann_turnover": avg_turnover * freq,
        "ann_cost_drag": _ann_drag(cost_frac, freq, n),
        "ann_commission_drag": _ann_drag(comm_frac, freq, n),
        "ann_impact_drag": _ann_drag(impact_frac, freq, n),
        "ann_borrow_drag": _ann_drag(borrow_frac, freq, n),
    }


# ── main backtest ───────────────────────────────────────────────────────────
def backtest(
    weights,
    prices,
    *,
    freq: int = 252,
    lag: int = 1,
    signal_dates: list | None = None,
    transaction_cost=0.0,
    short_cost_mult: float = 1.5,
    borrow_fee: float = 0.0,
    risk_free_rate: float = 0.0,
    capital: float = 0.0,
    commission_per_share: float = 0.0,
    commission_min: float = 0.0,
    commission_max_pct: float = 0.0,
    impact_coef: float = 0.0,
    dollar_volume=None,
    raw_prices=None,
    delist_return=None,
    track_trades: bool = True,
) -> dict:
    """
    Drift backtest of a portfolio defined by `weights` over `prices`.

    weights          : DataFrame (dates×tickers), Series (by ticker, or by date if
                       single asset), or array (N,) / (T, N).
    prices           : date-indexed DataFrame/Series. Run DAILY and rebalance via
                       `signal_dates` — a monthly panel cannot see an intra-month
                       delisting, stop or gap.
    freq             : annualization factor (252 daily, 12 monthly). Must match the
                       spacing of `prices`; a mismatch warns rather than corrects.
    lag              : signal→execution lag; weights at t earn t+lag → t+lag+1. On a
                       daily panel `lag=1` is honest — signal on today's close, trade
                       tomorrow's. `lag=0` assumes you trade the close you signalled on.
    signal_dates     : rebalance dates (default: every date).
    transaction_cost : one-way SPREAD — scalar or (dates×tickers) from `spread_costs`.
    short_cost_mult  : extra execution penalty on short *trades*.
    borrow_fee       : ANNUAL short-borrow rate, charged each period on short notional.

    Size-dependent costs, inert unless `capital` > 0 (see the `IBKR` preset):
    capital          : account size in the price currency. Per-share commission and
                       market impact are non-linear in order size, so they cannot be
                       expressed in bps and need an AUM to mean anything.
    commission_per_share / commission_min / commission_max_pct : broker schedule.
    impact_coef      : square-root impact, cost += coef·√(notional / ADV). Needs
                       `dollar_volume`; this is what bounds strategy capacity, and on
                       a concentrated book it usually dominates every other cost.
                       CALIBRATE IT — the standard form is c·σ_daily·√(Q/ADV) with
                       c ≈ 0.5-1.0, so `impact_coef` ≈ 0.5-1.0 × the daily volatility
                       of the names traded (≈0.01-0.03 for liquid US equities), NOT a
                       round number. Results are quadratically sensitive to it.
    dollar_volume    : (dates×tickers) average DAILY dollar volume, for impact.
    raw_prices       : (dates×tickers) as-traded prices for the per-share commission
                       and the blotter. Defaults to `prices` — pass the unadjusted
                       series if `prices` is back-adjusted, or historical share counts
                       and therefore commissions will be badly wrong.
    delist_return    : terminal return booked the day after a name's last print, for
                       names whose price never returns before the end of `prices`.
                       Scalar (-1.0 writes the position off in full) or a Series by
                       ticker for CRSP-style per-name delisting returns; names left out
                       of the Series keep the default hold-flat behaviour. None (the
                       default) models no delisting at all, which flatters any run on a
                       survivorship-free panel.
    track_trades     : build the per-trade blotter.

    Returns the `_metrics` dict plus 'trades', 'yearly' and 'capital'.
    """
    if lag < 0:
        raise ValueError("`lag` must be >= 0.")
    if capital < 0:
        raise ValueError("`capital` must be >= 0.")

    px = _as_prices_df(prices).sort_index()
    if px.shape[0] == 0 or px.shape[1] == 0:
        raise ValueError("`prices` is empty.")
    _check_freq(px.index, float(freq))

    idx, cols = px.index, px.columns
    n_dates, n_assets = px.shape
    w_vals = _as_weights_df(weights, index=idx, columns=cols).to_numpy()
    pv = px.to_numpy(float)
    # Last KNOWN price per name. `pv` stays raw — the tradability check below needs to
    # see the holes — while everything priced off a prior close reads this instead.
    pv_ffill = pd.DataFrame(pv).ffill().to_numpy(float)

    # execution timing: signal at t → execute at t+lag+1
    if signal_dates is None:
        signal_dates = idx.tolist()
    exec_w = {}
    for ts in (pd.Timestamp(x) for x in signal_dates):
        if ts not in idx:
            continue
        sp = idx.get_loc(ts)
        ep = sp + lag + 1
        if ep >= n_dates:
            continue
        row = w_vals[sp].copy()
        row[np.isnan(pv[ep - 1])] = 0.0              # can't trade without a fill price
        exec_w[ep] = row

    exec_rows = np.array(sorted(exec_w), dtype=np.int64)
    exec_slot = np.full(n_dates, -1, dtype=np.int64)
    exec_slot[exec_rows] = np.arange(len(exec_rows))
    n_exec = len(exec_rows)
    ex_w = np.array([exec_w[i] for i in exec_rows]) if n_exec else np.zeros((0, n_assets))

    # Spread, execution price and ADV are read only on rebalance rows, so they are
    # aligned straight to those rows — never to the full daily index.
    #
    # The FILL is at the close of row ep-1: _drift_core sets weights at the top of row
    # ep and then applies rets[ep], which is the ep-1 → ep move. So every execution
    # input is read one row before the slot it belongs to; reading row ep would price
    # the trade off data that did not exist when the order was sent. exec_rows ≥ 1
    # always (ep = sp+lag+1), so -1 never wraps.
    fill_rows = exec_rows - 1
    exec_dates = idx[fill_rows]
    ex_tc = _at_exec(transaction_cost, exec_dates, cols, fill="max")
    ex_px = (_at_exec(_as_prices_df(raw_prices), exec_dates, cols)
             if raw_prices is not None else pv_ffill[fill_rows])
    # Unknown fill price → last known close, matching the returns denominator. Skipping
    # the charge instead (the old NaN behaviour) makes a data hole free to trade.
    ex_px = np.where(np.isnan(ex_px), pv_ffill[fill_rows], ex_px)
    ex_adv = (_at_exec(dollar_volume, exec_dates, cols, fill="min")
              if dollar_volume is not None else np.full((n_exec, n_assets), np.nan))

    # Borrow accrues every period, so it is the one rate kept at full (T, N) — and only
    # when it actually varies by name.
    bf_is_arr = isinstance(borrow_fee, (pd.DataFrame, pd.Series))
    bf_arr = (_as_weights_df(borrow_fee, index=idx, columns=cols).to_numpy(float)
              if bf_is_arr else np.zeros((1, 1)))
    bf_scalar = 0.0 if bf_is_arr else float(borrow_fee)
    if not bf_is_arr and bf_scalar < 0:
        raise ValueError("`borrow_fee` must be >= 0.")

    scratch = (n_exec, n_assets) if track_trades else (0, n_assets)
    w_pre, w_post, tr_cost = (np.zeros(scratch) for _ in range(3))

    # Equivalent to px.pct_change(fill_method=None) with ±inf mapped to NaN, computed
    # in place: the pandas form allocates several copies of a (T, N) frame, which on a
    # full daily panel is gigabytes.
    # Denominator is the ffilled price, so a return is priced off the last KNOWN close,
    # not the immediately-prior row: with a raw denominator, one NaN day (data hole,
    # halted print) makes the day right after it NaN too (x / NaN), silently erasing
    # that day's real price move instead of just deferring it. A true delisting is
    # unaffected — price never returns, so the ffilled value keeps producing 0.0 return,
    # same as the NaN it replaces.
    rets_np = np.empty_like(pv)
    rets_np[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(pv[1:], pv_ffill[:-1], out=rets_np[1:])
        rets_np[1:] -= 1.0
    rets_np[~np.isfinite(rets_np)] = np.nan

    if delist_return is not None:
        # A name whose price stops for good is currently carried at its last print and
        # later "sold" there — a bankruptcy and a takeunder both come back as 0%. Book
        # the terminal return on the row after the last print instead: the weight then
        # drifts by (1 + delist_return) and the position is written down for real.
        dr = (pd.Series(delist_return, index=cols) if np.isscalar(delist_return)
              else pd.Series(delist_return).reindex(cols)).to_numpy(float)
        if np.any(dr < -1.0):
            raise ValueError("`delist_return` must be >= -1.0 (-1.0 is a total loss).")
        traded = np.isfinite(pv)
        ever = traded.any(axis=0)
        # Last row that printed, per name; `ever` guards the all-NaN column.
        last = n_dates - 1 - traded[::-1].argmax(axis=0)
        gone = ever & (last < n_dates - 1) & np.isfinite(dr)
        rets_np[last[gone] + 1, np.flatnonzero(gone)] = dr[gone]

    equity_np, turn_np, cost_np, borrow_np, comm_np, imp_np = _drift_core(
        rets_np, ex_px, ex_adv, ex_w, ex_tc, float(short_cost_mult),
        bf_arr, bf_scalar, bf_is_arr, float(freq), exec_slot,
        float(capital), float(commission_per_share), float(commission_min),
        float(commission_max_pct), float(impact_coef),
        bool(track_trades), w_pre, w_post, tr_cost, n_dates, n_assets,
    )

    equity = pd.Series(equity_np, index=idx)
    series = lambda a, name: pd.Series(a, index=idx, name=name)
    out = _metrics(
        equity,
        series(0.5 * turn_np, "turnover"),          # one-way turnover (reporting)
        series(cost_np, "cost_frac"),
        series(borrow_np, "borrow_frac"),
        series(comm_np, "commission_frac"),
        series(imp_np, "impact_frac"),
        float(freq), float(risk_free_rate),
    )
    out["capital"] = capital
    out["trades"] = (_build_blotter(exec_dates, cols, w_pre, w_post, tr_cost,
                                    equity_np[exec_rows - 1], ex_px, capital)
                     if track_trades and n_exec else pd.DataFrame(columns=_TRADE_COLS))
    out["yearly"] = yearly_summary(out, freq=float(freq), rf=float(risk_free_rate))
    return out


# ── trade blotter ───────────────────────────────────────────────────────────
_TRADE_COLS = ["date", "ticker", "action", "side", "w_before", "w_after", "dw",
               "weight_traded", "price", "cost_frac", "notional", "shares", "cost"]


def _build_blotter(exec_dates, cols, w_pre, w_post, tr_cost, book_equity, ex_px, capital) -> pd.DataFrame:
    """One row per name actually traded at each rebalance, dated at the close it fills
    on (one row before the slot the new weights first earn). `notional`/`shares`/`cost`
    are currency amounts and are NaN unless `capital` is set; `weight_traded` and
    `cost_frac` are always meaningful. `book_equity` is equity at the close BEFORE each
    execution — the base orders are sized against, matching `_drift_core`."""
    dw = w_post - w_pre
    rows, cs = np.nonzero(dw != 0.0)
    if len(rows) == 0:
        return pd.DataFrame(columns=_TRADE_COLS)

    d = dw[rows, cs]
    price = ex_px[rows, cs]
    cost_frac = tr_cost[rows, cs]
    notional = (book_equity[rows] * capital) * np.abs(d) if capital > 0 else np.full(len(d), np.nan)
    df = pd.DataFrame({
        "date": pd.DatetimeIndex(exec_dates)[rows],
        "ticker": np.asarray(cols)[cs],
        "action": np.where(w_pre[rows, cs] == 0.0, "OPEN",
                    np.where(w_post[rows, cs] == 0.0, "CLOSE", "REBALANCE")),
        "side": np.where(d > 0, "BUY", "SELL"),
        "w_before": w_pre[rows, cs],
        "w_after": w_post[rows, cs],
        "dw": d,
        "weight_traded": np.abs(d),
        "price": price,
        "cost_frac": cost_frac,
        "notional": notional,
        "shares": np.where(price > 0, notional / price, np.nan),
        "cost": notional * cost_frac,
    })
    return df.sort_values(["date", "ticker"], ignore_index=True)[_TRADE_COLS]


# ── generic out-of-sample walk-forward ──────────────────────────────────────
def walk_forward(signal_fn, prices, *, train: int, test: int, lag: int = 1,
                 signal_kwargs: dict | None = None, **backtest_kwargs) -> dict:
    """
    Rolling train→test out-of-sample backtest. Each block is fit on a `train`-row
    window and traded over the next `test` rows, so every traded return is strictly
    out-of-sample:

        [── train ──][── test ──]
                     [── train ──][── test ──]   (step = test)

    signal_fn(train_prices, **signal_kwargs) -> weights
        A static Series (by ticker) held over the block, or a DataFrame covering the
        test dates. Called on the TRAIN slice only — it never sees test data.

    Further keywords (freq, transaction_cost, capital, impact_coef, ...) pass through
    to `backtest`, which returns the same dict over the stitched OOS span.
    """
    px = _as_prices_df(prices).sort_index()
    n = px.shape[0]
    if train <= 0 or test <= 0:
        raise ValueError("`train` and `test` must be positive.")
    if train + test > n:
        raise ValueError(f"Need ≥ train+test ({train + test}) rows; got {n}.")
    kw = signal_kwargs or {}

    weight_rows, block_starts = {}, []
    for start in range(train, n, test):
        test_px = px.iloc[start:start + test]
        if test_px.shape[0] == 0:
            break
        w = signal_fn(px.iloc[start - train:start], **kw)
        block_starts.append(test_px.index[0])
        if isinstance(w, pd.DataFrame):
            # Test rows only. A signal_fn that returns a frame over the slice it was
            # handed — the ordinary shape — otherwise writes its TRAIN rows here, and
            # since train windows overlap the previous block's test window, a later
            # block overwrites already-decided OOS weights with in-sample ones.
            w = w.reindex(columns=px.columns)
            weight_rows.update(w.loc[w.index.intersection(test_px.index)].iterrows())
        else:                                          # static weights → hold the block
            # Stamped once at the block start; `backtest` ffills it and the book drifts
            # from there. Re-stamping every date would re-target daily, churning a
            # "hold" into `test` rebalances a block.
            weight_rows[test_px.index[0]] = pd.Series(w).reindex(px.columns).fillna(0.0)

    if not weight_rows:
        raise ValueError("walk_forward produced no weights.")

    return backtest(
        pd.DataFrame(weight_rows).T.sort_index(), px.loc[block_starts[0]:],
        lag=lag, signal_dates=sorted(weight_rows), **backtest_kwargs,
    )


# ── reporting ───────────────────────────────────────────────────────────────
_SUMMARY_METRICS = [
    ("Annual Return", "ann_return", ".2%"),
    ("Annual Volatility", "ann_vol", ".2%"),
    ("Sharpe Ratio", "sharpe", ".3f"),
    ("Sortino Ratio", "sortino_ratio", ".3f"),
    ("Expectancy", "expectancy", ".4f"),
    ("Total Return", "total_return", ".2%"),
    ("Max Drawdown", "max_drawdown", ".2%"),
    ("Avg Drawdown", "avg_drawdown", ".2%"),
    ("CDaR (95%)", "cdar", ".2%"),
    ("CVaR (95%)", "cvar_ann", ".2%"),
    ("Downside Deviation", "downside_deviation", ".2%"),
    ("Ann. Turnover", "ann_turnover", ".2%"),
    # pp/yr, comparable to Annual Return. Trading Cost is all-in; the two "of which"
    # lines are components of it, not extras to add on.
    ("Ann. Trading Cost", "ann_cost_drag", ".2%"),
    ("  of which commission", "ann_commission_drag", ".2%"),
    ("  of which impact", "ann_impact_drag", ".2%"),
    ("Ann. Borrow Cost", "ann_borrow_drag", ".2%"),
]

_YEARLY_SUMS = (("turnover", "Turnover"), ("cost_frac", "Cost"), ("borrow_frac", "Borrow"),
                ("commission_frac", "Commission"), ("impact_frac", "Impact"))


def yearly_summary(res: dict, *, freq: float = 252, rf: float = 0.0) -> pd.DataFrame:
    """
    Per-calendar-year breakdown of a backtest result.

    Returns are CHAINED from the period returns within the year, never taken as
    equity[last]/equity[first] — the latter drops the first period of every year and
    can report a year that doubled as +0.00%.
    """
    ret = res.get("returns")
    if ret is None or len(ret) == 0:
        return pd.DataFrame()
    ret = ret.iloc[1:]                      # drop the structural period-0 zero
    trades = res.get("trades")
    rf_per = (1.0 + rf) ** (1.0 / freq) - 1.0

    rows = []
    for yr, r in ret.groupby(ret.index.year):
        r = r.dropna()
        n = len(r)
        if n == 0:
            continue
        eq = (1.0 + r).cumprod()
        vol = float(r.std(ddof=1) * np.sqrt(freq)) if n > 1 else np.nan
        neg = r[r < 0]
        dsd = float(neg.std(ddof=1) * np.sqrt(freq)) if len(neg) > 1 else np.nan
        ann_excess = (float(r.mean()) - rf_per) * freq
        row = {
            "Year": int(yr),
            "Periods": n,
            "Return": float(eq.iloc[-1] - 1.0),
            "Vol": vol,
            "Sharpe": float(ann_excess / vol) if vol > 0 else np.nan,
            "Sortino": float(ann_excess / dsd) if dsd > 0 else np.nan,
            # Rebased to the start of the year: how far we fell during this year, not
            # how far below the all-time peak we sat.
            "MaxDD": float((eq / eq.cummax() - 1.0).min()),
            "Best": float(r.max()),
            "Worst": float(r.min()),
            "WinRate": float((r > 0).mean()),
        }
        for key, name in _YEARLY_SUMS:
            s = res.get(key)
            if s is not None:
                row[name] = float(s[s.index.year == yr].sum())
        if trades is not None and len(trades):
            t = trades[trades["date"].dt.year == yr]
            row["Trades"] = int(len(t))
            row["Rebalances"] = int(t["date"].nunique())
        rows.append(row)
    return pd.DataFrame(rows).set_index("Year") if rows else pd.DataFrame()


def results_backtest(strategies: dict, *, title: str | None = None,
                     figsize: tuple = (14, 8)) -> dict:
    """
    Summary table + equity/drawdown chart for one or many backtest result dicts.
    `strategies`: {name: result_dict}. Returns {'fig','axes','summary_df','yearly_df'};
    `yearly_df` has a (strategy, metric) column MultiIndex.
    """
    import matplotlib.pyplot as plt

    if "equity" in strategies:                          # single result → wrap
        strategies = {strategies.get("name", "Strategy"): strategies}
    if not strategies:
        raise ValueError("At least one strategy must be provided.")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=figsize, gridspec_kw={"height_ratios": [2, 1]}
    )
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    for name, res in strategies.items():
        if "equity" in res:
            eq = res["equity"].dropna()
            ax1.plot(eq.index, eq / eq.iloc[0], label=name, linewidth=1.5)
        if "drawdown" in res:
            dd = res["drawdown"]
            ax2.fill_between(dd.index, dd.values, 0, alpha=0.3, label=name)
    ax1.set_ylabel("Equity (normalized)"); ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3); ax1.legend(loc="best", fontsize=9)
    ax1.set_title("Cumulative Returns")
    ax2.set_ylabel("Drawdown"); ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3); ax2.legend(loc="best", fontsize=9)
    plt.tight_layout()

    summary_df = pd.DataFrame({
        name: {disp: (f"{res.get(key, np.nan):{fmt}}" if pd.notna(res.get(key, np.nan)) else "N/A")
               for disp, key, fmt in _SUMMARY_METRICS}
        for name, res in strategies.items()
    }).T

    yearly = {name: res["yearly"] for name, res in strategies.items()
              if isinstance(res.get("yearly"), pd.DataFrame) and len(res["yearly"])}
    yearly_df = pd.concat(yearly, axis=1) if yearly else pd.DataFrame()

    return {"fig": fig, "axes": (ax1, ax2), "summary_df": summary_df, "yearly_df": yearly_df}


# ── smoke test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=2000)
    px = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, size=(2000, 6)), axis=0)),
        index=dates, columns=[f"A{i}" for i in range(6)],
    )
    dv = pd.DataFrame(rng.uniform(1e6, 2e8, size=px.shape), index=dates, columns=px.columns)
    month_ends = px.resample("ME").last().index.intersection(px.index)

    # daily panel, monthly rebalance, equal weight
    w = pd.Series(1 / 6, index=px.columns)
    r = backtest(w, px, signal_dates=list(month_ends), transaction_cost=0.001)
    print(f"[daily]    ann={r['ann_return']:.2%}  sharpe={r['sharpe']:.2f}  maxDD={r['max_drawdown']:.2%}")

    # full cost stack: tiered spread + IBKR Ireland commissions + market impact
    r2 = backtest(w, px, signal_dates=list(month_ends), capital=50e6,
                  transaction_cost=spread_costs(dv * 21), dollar_volume=dv,
                  impact_coef=0.1, **IBKR)
    print(f"[costed]   ann={r2['ann_return']:.2%}  cost={r2['ann_cost_drag']:.2%}/yr "
          f"(comm {r2['ann_commission_drag']:.2%}, impact {r2['ann_impact_drag']:.2%})")
    print(f"           {len(r2['trades'])} trades over {r2['yearly'].shape[0]} years")

    # out-of-sample walk-forward
    def mom_signal(train_px):
        m = train_px.iloc[-1] / train_px.iloc[0] - 1.0
        return (m == m.max()).astype(float)           # hold last winner
    rw = walk_forward(mom_signal, px, train=252, test=21, transaction_cost=0.001)
    print(f"[walkfwd]  ann={rw['ann_return']:.2%}  sharpe={rw['sharpe']:.2f}")

    # gap check: a one-day price gap must not erase the move once price reappears
    gap_dates = pd.bdate_range("2020-01-01", periods=5)
    gap_px = pd.DataFrame({"A": [100.0, 100.0, np.nan, 90.0, 90.0]}, index=gap_dates)
    gap_r = backtest(pd.Series({"A": 1.0}), gap_px, signal_dates=[gap_dates[0]], lag=0)
    assert abs(gap_r["returns"].iloc[3] - (90.0 / 100.0 - 1.0)) < 1e-9, \
        "price gap silently erased the recovery-day return"

    # tradability is judged on the fill close, which is known when the order is sent —
    # not on the row after it, which is not
    gap2 = backtest(pd.Series({"A": 1.0}), gap_px, signal_dates=[gap_dates[1]], lag=0)
    assert len(gap2["trades"]) == 1, "trade suppressed using the next day's price"

    # delistings: a name that stops printing is written off, not held flat
    dl_dates = pd.bdate_range("2020-01-01", periods=10)
    dl_px = pd.DataFrame({"A": [100.0] * 4 + [np.nan] * 6,
                          "B": [100.0] * 10}, index=dl_dates)
    dl_w = pd.Series({"A": 0.5, "B": 0.5})
    assert abs(backtest(dl_w, dl_px, signal_dates=[dl_dates[0]])["total_return"]) < 1e-12, \
        "default should still hold a vanished name flat"
    dl = backtest(dl_w, dl_px, signal_dates=[dl_dates[0]], delist_return=-1.0)
    assert abs(dl["total_return"] + 0.5) < 1e-12, \
        f"delisted half the book should cost 50%, got {dl['total_return']:.4%}"
    assert abs(dl["equity"].iloc[-1] - dl["equity"].iloc[4]) < 1e-12, \
        "write-off leaked past the day after the last print"
    # a name still trading on the last row has not delisted
    assert abs(backtest(pd.Series({"B": 1.0}), dl_px[["B"]], signal_dates=[dl_dates[0]],
                        delist_return=-1.0)["total_return"]) < 1e-12, \
        "a live name was written off at the end of the sample"
    # per-name returns, and names absent from the Series keep the old behaviour
    dl_s = backtest(dl_w, dl_px, signal_dates=[dl_dates[0]],
                    delist_return=pd.Series({"A": -0.7}))
    assert abs(dl_s["total_return"] + 0.35) < 1e-12, "per-ticker delist_return misapplied"
    dl_px2 = dl_px.assign(C=[100.0] * 6 + [np.nan] * 4)
    dl_s2 = backtest(pd.Series({"A": 0.5, "B": 0.25, "C": 0.25}), dl_px2,
                     signal_dates=[dl_dates[0]], delist_return=pd.Series({"A": -1.0}))
    assert abs(dl_s2["total_return"] + 0.5) < 1e-12, \
        "a delisted name left out of the Series should still be held flat"
    # the last print's own move is real and must survive the write-off booked after it
    dl_px3 = pd.DataFrame({"A": [100.0, 100.0, 100.0, 120.0] + [np.nan] * 6,
                           "B": [100.0] * 10}, index=dl_dates)
    dl3 = backtest(dl_w, dl_px3, signal_dates=[dl_dates[0]], delist_return=-1.0)
    assert abs(dl3["equity"].iloc[3] - 1.1) < 1e-12, \
        "write-off landed on the last print instead of the day after it"
    try:
        backtest(dl_w, dl_px, signal_dates=[dl_dates[0]], delist_return=-1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("delist_return below -100% was accepted")

    # exec timing: the fill is the close the position was opened at, not the next one
    t_dates = pd.bdate_range("2020-01-01", periods=3)
    t_px = pd.DataFrame({"A": [100.0, 150.0, 150.0]}, index=t_dates)
    t_r = backtest(pd.Series({"A": 1.0}), t_px, signal_dates=[t_dates[0]], lag=0,
                   capital=1e6)
    t_trade = t_r["trades"].iloc[0]
    assert abs(t_r["equity"].iloc[1] - 1.5) < 1e-9, "position missed the move it was on for"
    assert t_trade["price"] == 100.0 and t_trade["date"] == t_dates[0], \
        f"blotter priced the fill off the wrong row: {t_trade['date']} @ {t_trade['price']}"

    # exec inputs must not be read from the future: a spread that is free on the fill
    # close and ruinous the day after must charge the free one
    t_tc = pd.DataFrame({"A": [0.0, 0.10, 0.10]}, index=t_dates)
    t_c = backtest(pd.Series({"A": 1.0}), t_px, signal_dates=[t_dates[0]], lag=0,
                   transaction_cost=t_tc)
    assert t_c["cost_frac"].iloc[1] == 0.0, "spread was read from the row after the fill"

    # a by-ticker spread is one row held across EVERY execution date (needs more than
    # one of them to bite) and must match spelling the same rates out per date
    tc_s = pd.Series([0.002, 0.004] * 3, index=px.columns)
    r_ts = backtest(w, px, signal_dates=list(month_ends), transaction_cost=tc_s)
    r_td = backtest(w, px, signal_dates=list(month_ends),
                    transaction_cost=pd.DataFrame([tc_s] * len(px), index=px.index))
    assert abs(r_ts["ann_cost_drag"] - r_td["ann_cost_drag"]) < 1e-12, \
        "per-ticker spread was not held across execution dates"

    # an unknown spread costs the worst spread observed, never nothing
    t_h = pd.DataFrame({"A": [np.nan, np.nan, 0.10]}, index=t_dates)
    t_w = backtest(pd.Series({"A": 1.0}), t_px, signal_dates=[t_dates[0]], lag=0,
                   transaction_cost=t_h)
    assert abs(t_w["cost_frac"].iloc[1] - 0.10) < 1e-12, "unknown spread traded for free"

    # walk_forward: a static signal is HELD over its block, not re-targeted daily.
    # Weights must be multi-name for this to bite — a 100%-single-name book cannot
    # drift away from its target, so it churns nothing either way.
    wf = walk_forward(lambda t: pd.Series(1 / 6, index=t.columns), px.iloc[:600],
                      train=252, test=21)
    assert wf["trades"]["date"].nunique() <= len(range(252, 600, 21)), \
        "static walk-forward weights rebalanced more than once per block"

    # walk_forward: only test-window rows are traded. This signal is flat in-sample and
    # invested out-of-sample, so it should stay invested throughout. Train windows
    # overlap the previous block's test window, and unclipped they overwrite those
    # already-decided OOS weights with the in-sample flat ones — leaving the book out
    # of the market on almost every day it was supposed to be holding.
    def leaky_signal(train_px):
        fwd = pd.bdate_range(train_px.index[-1], periods=22)[1:]
        return pd.concat([
            pd.DataFrame(0.0, index=train_px.index, columns=train_px.columns),
            pd.DataFrame(1 / 6, index=fwd, columns=train_px.columns),
        ])
    wl = walk_forward(leaky_signal, px.iloc[:600], train=252, test=21)
    assert (wl["returns"].abs() > 1e-12).mean() > 0.9, \
        "in-sample weight rows overwrote already-decided OOS weights"

    # missing cost data must fail closed, not free
    imp_dv = dv.copy()
    imp_dv.iloc[:, :] = np.nan
    imp_dv.iloc[0, 0] = 1e6                           # one known print → the floor
    r_nan = backtest(w, px, signal_dates=list(month_ends), capital=50e6,
                     dollar_volume=imp_dv, impact_coef=0.1)
    assert r_nan["ann_impact_drag"] > 0, "NaN ADV bought an impact-free fill"
    # and it is the THINNEST volume observed, not the friendliest one
    imp_dv2 = dv.copy()
    imp_dv2.iloc[:, :] = np.nan
    imp_dv2.iloc[0, 0], imp_dv2.iloc[0, 1] = 1e6, 1e12
    ikw = dict(signal_dates=list(month_ends), capital=50e6, impact_coef=0.1)
    r_min = backtest(w, px, dollar_volume=imp_dv2, **ikw)
    r_ref = backtest(w, px, dollar_volume=pd.DataFrame(1e6, *dv.axes), **ikw)
    assert abs(r_min["ann_impact_drag"] - r_ref["ann_impact_drag"]) < 1e-12, \
        "unknown ADV fell back to the deepest volume instead of the thinnest"

    # ...same for an unknown raw fill price: commission is per share, so no price means
    # no share count, and the charge used to be skipped outright. One name with no raw
    # price at all must fall back to its adjusted close and be charged identically.
    raw_holes = px.copy()
    raw_holes.iloc[:, 0] = np.nan
    ckw = dict(signal_dates=list(month_ends), capital=50e6, commission_per_share=0.01)
    r_holes = backtest(w, px, raw_prices=raw_holes, **ckw)
    r_full = backtest(w, px, raw_prices=px, **ckw)
    assert abs(r_holes["ann_commission_drag"] - r_full["ann_commission_drag"]) < 1e-12, \
        "unknown fill price skipped commission"

    # per-ticker borrow rates: a Series by ticker used to raise TypeError
    r_bf = backtest(pd.Series({c: -1 / 6 for c in px.columns}), px,
                    signal_dates=list(month_ends),
                    borrow_fee=pd.Series(0.05, index=px.columns))
    assert r_bf["ann_borrow_drag"] > 0.04, "per-ticker borrow_fee was not charged"
    print("OK")
