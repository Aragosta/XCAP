"""
BACKTEST.py — lean, generic portfolio backtester.

Public API
----------
    backtest(weights, prices, ...)        weights → PnL, equity, and metrics
    walk_forward(signal_fn, prices, ...)   generic out-of-sample (rolling train→test) harness
    tiered_transaction_costs(dollar_vol)   size/liquidity-tiered per-name cost model
    results_backtest(strategies, ...)      summary table + equity/drawdown chart

Conventions
-----------
- Drift mode: weights are set on execution dates and drift with returns in between.
- Timing (no look-ahead): a signal at index t executes at t+lag+1, so weights decided
  with information up to t earn the return from t+lag → t+lag+1.
- Costs: `transaction_cost` is a ONE-WAY cost per unit traded notional.  Period cost is
  Σⱼ tcⱼ·|Δwⱼ| (charged on every share traded, both buys and sells) — not 0.5·Σ|Δw|.
  It may be a scalar (flat) or a (dates × tickers) DataFrame (e.g. liquidity-tiered).
"""
import numpy as np
import pandas as pd

try:                                              # optional JIT for the hot loop
    from numba import jit
except Exception:                                 # pragma: no cover - fallback
    def jit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco(args[0]) if (len(args) == 1 and callable(args[0]) and not kwargs) else deco


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
        if w.index.equals(index):
            return w.fillna(0.0)
        return w.sort_index().reindex(index=index, method="ffill").fillna(0.0)

    if isinstance(weights, pd.Series):
        if weights.index.difference(columns).empty:          # static weights by ticker
            row = weights.reindex(columns).to_numpy(float)
            return pd.DataFrame([row], index=[index[0]], columns=columns) \
                     .reindex(index=index, method="ffill").fillna(0.0)
        if len(columns) == 1 and weights.index.difference(index).empty:   # single asset, by date
            return weights.sort_index().to_frame(columns[0]) \
                          .reindex(index=index, method="ffill").fillna(0.0)
        raise ValueError("`weights` Series must be indexed by tickers, or by dates when single-asset.")

    arr = np.asarray(weights, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != len(columns):
            raise ValueError("1D `weights` length must match number of `prices` columns.")
        return pd.DataFrame([arr], index=[index[0]], columns=columns) \
                 .reindex(index=index, method="ffill").fillna(0.0)
    if arr.ndim == 2 and arr.shape[1] == len(columns):
        if arr.shape[0] == len(index):
            return pd.DataFrame(arr, index=index, columns=columns).fillna(0.0)
        if arr.shape[0] == 1:
            return pd.DataFrame(arr, index=[index[0]], columns=columns) \
                     .reindex(index=index, method="ffill").fillna(0.0)
    raise ValueError("`weights` array must be 1D (N,) or 2D (T, N) aligned to prices.")


# ── cost model ──────────────────────────────────────────────────────────────
# (threshold dollar-volume, one-way cost) — a name trading ≥ threshold/month pays
# that cost; microcaps fall to the wide-spread tiers.  Loosely DeMiguel et al. (2020).
DEFAULT_COST_TIERS = (
    (1e9, 0.0005),   # ≥ $1B/mo  →  5 bps
    (1e8, 0.0010),   # ≥ $100M   → 10 bps
    (1e7, 0.0025),   # ≥ $10M    → 25 bps
    (1e6, 0.0060),   # ≥ $1M     → 60 bps
    (0.0, 0.0150),   # < $1M     → 150 bps
)


def tiered_transaction_costs(
    dollar_volume: pd.DataFrame,
    *,
    tiers: tuple = DEFAULT_COST_TIERS,
    lookback: int = 3,
) -> pd.DataFrame:
    """
    Per-name ONE-WAY transaction cost (fraction of traded notional) from trailing
    dollar volume.  Each name gets the cost of the highest dollar-volume threshold
    it clears; illiquid names pay the wide-spread tiers.  Never NaN (NaN volume →
    worst tier), so it is safe to feed straight into `backtest(transaction_cost=...)`.
    """
    return _tiered(dollar_volume, tiers, lookback)


# Annual short-BORROW fee by liquidity (general collateral → hard-to-borrow).
# IBKR: GC ~0.25-0.5%/yr; HTB 5-50%+/yr, microcaps 100%+ (often unborrowable — see
# the SSR locate filter in compute_eligibility, which removes those entirely).
DEFAULT_BORROW_TIERS = (
    (1e9, 0.0025),   # ≥ $1B/mo  →  0.25%/yr  (general collateral)
    (1e8, 0.0100),   # ≥ $100M   →  1%/yr
    (1e7, 0.0500),   # ≥ $10M    →  5%/yr
    (1e6, 0.2500),   # ≥ $1M     → 25%/yr     (hard to borrow)
    (0.0, 0.5000),   # < $1M     → 50%/yr
)


def tiered_borrow_fees(
    dollar_volume: pd.DataFrame,
    *,
    tiers: tuple = DEFAULT_BORROW_TIERS,
    lookback: int = 3,
) -> pd.DataFrame:
    """Per-name ANNUAL short-borrow fee from trailing dollar volume (see tiers)."""
    return _tiered(dollar_volume, tiers, lookback)


def _tiered(dollar_volume: pd.DataFrame, tiers: tuple, lookback: int) -> pd.DataFrame:
    dv  = dollar_volume.rolling(lookback, min_periods=1).mean()
    out = pd.DataFrame(max(c for _, c in tiers), index=dv.index, columns=dv.columns)
    for thresh, c in sorted(tiers, key=lambda x: x[0]):       # ascending → highest wins
        out = out.mask(dv >= thresh, c)
    return out


# ── core engine (JIT) ───────────────────────────────────────────────────────
@jit(nopython=True, cache=True)
def _drift_core(rets, w_target, tc, short_mult, bf, ppy, exec_idx, n_dates, n_assets):
    """
    Drift backtest with per-name one-way trading costs + short-borrow holding fees.

    rets      : (T, N) period returns (NaN treated as 0).
    w_target  : (T, N) target weights, populated at execution rows.
    tc        : (T, N) one-way trading-cost fractions.
    short_mult: trading-cost multiplier for short trades (execution penalty).
    bf        : (T, N) ANNUAL borrow-fee rates; charged each period on short notional.
    ppy       : periods per year (annual borrow fee → per-period = bf / ppy).
    exec_idx  : (T,) 1 where a rebalance executes.
    Returns   : equity, turnover_gross, cost_frac (trading), borrow_frac (holding).
    """
    equity = np.ones(n_dates)
    turnover_gross = np.zeros(n_dates)
    cost_frac = np.zeros(n_dates)
    borrow_frac = np.zeros(n_dates)
    w = np.zeros(n_assets)

    for i in range(n_dates):
        if exec_idx[i] == 1:                       # rebalance at start of period i (i ≥ 1 always)
            t_over = 0.0
            cost = 0.0
            for j in range(n_assets):
                dwj = w_target[i, j] - w[j]
                adw = dwj if dwj >= 0.0 else -dwj
                t_over += adw
                ## short trades pay short_mult× the transaction cost (harder execution);
                ## the ongoing borrow fee is charged separately below.
                cj = tc[i, j] * short_mult if w_target[i, j] < 0.0 else tc[i, j]
                cost += cj * adw                   # one-way cost on full traded notional
                w[j] = w_target[i, j]
            if cost > 0.999:
                cost = 0.999
            turnover_gross[i] = t_over
            cost_frac[i] = cost

        if i > 0:
            rp = 0.0
            borrow = 0.0
            for j in range(n_assets):
                r = rets[i, j]
                if r == r:                          # not NaN
                    rp += w[j] * r
                if w[j] < 0.0:                      # borrow fee on short notional held this period
                    borrow += (-w[j]) * bf[i, j]
            borrow /= ppy
            if borrow > 0.999:
                borrow = 0.999
            borrow_frac[i] = borrow
            equity[i] = equity[i - 1] * (1.0 + rp)
            if exec_idx[i] == 1:
                equity[i] *= (1.0 - cost_frac[i])
            equity[i] *= (1.0 - borrow)             # short-borrow holding fee
            denom = 1.0 + rp
            if denom > 0.0 and np.isfinite(denom):  # drift weights to next period
                for j in range(n_assets):
                    r = rets[i, j]
                    if r != r:
                        r = 0.0
                    w[j] = w[j] * (1.0 + r) / denom

    return equity, turnover_gross, cost_frac, borrow_frac


# ── metrics ─────────────────────────────────────────────────────────────────
def _tail_mean(sorted_vals: np.ndarray, pct: float) -> float:
    """Mean of the worst `1-pct` fraction of an ascending-sorted array (CVaR / CDaR)."""
    if len(sorted_vals) == 0:
        return np.nan
    k = int(np.ceil((1.0 - pct) * len(sorted_vals)))
    tail = sorted_vals[:k] if 0 < k < len(sorted_vals) else sorted_vals
    return float(np.mean(tail))


def _metrics(equity, turnover, cost_frac, borrow_frac, freq, rf) -> dict:
    """Performance/risk metrics from an equity curve (and turnover/cost series)."""
    net_ret = equity.pct_change(fill_method=None).fillna(0.0).rename("returns")
    net_ret.iloc[0] = 0.0
    drawdown = equity / equity.cummax() - 1.0

    # Stats exclude the structural period-0 zero (no position yet) but keep genuine
    # interior zero-return periods — dropping those would inflate Sharpe.
    r = net_ret.to_numpy()[1:]
    n = len(r)
    total_factor = float(equity.iloc[-1] / equity.iloc[0])
    # A book that loses more than 100% drives equity <= 0, and a NEGATIVE base to a fractional power is
    # COMPLEX -> every downstream metric raises TypeError. Report a total loss instead of crashing the run.
    if not np.isfinite(total_factor) or total_factor <= 0:
        ann_return = -1.0
    else:
        ann_return = total_factor ** (freq / n) - 1.0 if n > 0 else np.nan
    ann_vol = float(np.std(r, ddof=1) * np.sqrt(freq)) if n > 1 else np.nan
    rf_per = (1.0 + rf) ** (1.0 / freq) - 1.0
    sharpe = float((r.mean() - rf_per) / np.std(r, ddof=1) * np.sqrt(freq)) if (n > 1 and ann_vol > 0) else np.nan

    neg = r[r < 0]
    downside = float(np.std(neg, ddof=1) * np.sqrt(freq)) if len(neg) > 1 else np.nan
    sortino = float((ann_return - rf) / downside) if (downside and np.isfinite(downside) and downside > 0) else np.nan

    dd_sorted = np.sort(drawdown.to_numpy())
    ret_sorted = np.sort(r)
    cvar = _tail_mean(ret_sorted, 0.95)
    pos, negr = r[r > 0], r[r < 0]
    win_rate = len(pos) / n if n else 0.0
    expectancy = win_rate * (pos.mean() if len(pos) else 0.0) + (len(negr) / n if n else 0.0) * (negr.mean() if len(negr) else 0.0)

    cost_amount = (equity * cost_frac / (1.0 - cost_frac).clip(lower=1e-12)).where(cost_frac > 0, 0.0)
    borrow_amount = (equity * borrow_frac / (1.0 - borrow_frac).clip(lower=1e-12)).where(borrow_frac > 0, 0.0)
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
        "cdar": _tail_mean(dd_sorted, 0.95),
        "cvar": cvar,
        "cvar_ann": cvar * np.sqrt(freq) if np.isfinite(cvar) else np.nan,
        "downside_deviation": downside,
        "expectancy": float(expectancy),
        "win_rate": win_rate,
        "turnover": turnover,
        "cost_frac": cost_frac,
        "avg_turnover": avg_turnover,
        "ann_turnover": avg_turnover * freq,
        "total_turnover": float(turnover.sum()),
        "total_cost": float(cost_amount.sum()),
        "ann_borrow": float(borrow_frac.to_numpy()[1:].mean() * freq) if n else np.nan,
        "total_borrow": float(borrow_amount.sum()),
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
    borrow_fee=0.0,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Vectorized drift backtest of a portfolio defined by `weights` over `prices`.

    weights          : DataFrame (dates×tickers), Series (by ticker, or by date if single
                       asset), or array (N,) / (T, N).
    prices           : DataFrame/Series of prices, date-indexed.
    freq             : annualization factor (252 daily, 12 monthly).
    lag              : signal→execution lag; weights at t earn the t+lag → t+lag+1 return.
    signal_dates     : rebalance dates (default: every date).
    transaction_cost : one-way trade cost — scalar or (dates×tickers) DataFrame
                       (`tiered_transaction_costs`).
    short_cost_mult  : extra execution penalty on short *trades* (default 1.5×).
    borrow_fee       : ANNUAL short-borrow holding fee charged each period on short
                       notional — scalar or (dates×tickers) DataFrame (`tiered_borrow_fees`).
    Returns a metrics dict (see `_metrics`).
    """
    if lag < 0:
        raise ValueError("`lag` must be >= 0.")

    px = _as_prices_df(prices).sort_index()
    if px.shape[0] == 0 or px.shape[1] == 0:
        raise ValueError("`prices` is empty.")

    idx, cols = px.index, px.columns
    n_dates, n_assets = px.shape
    w_target = _as_weights_df(weights, index=idx, columns=cols)
    rets = px.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    # rate matrices (scalar → broadcast; DataFrame → align, worst-tier fallback)
    tc_np = _rate_matrix(transaction_cost, idx, cols, n_dates, n_assets)
    bf_np = _rate_matrix(borrow_fee, idx, cols, n_dates, n_assets)

    # execution timing: signal at t → execute at t+lag+1
    if signal_dates is None:
        signal_dates = idx.tolist()
    exec_idx = np.zeros(n_dates, dtype=np.int64)
    w_target_np = np.zeros((n_dates, n_assets))
    w_vals = w_target.to_numpy()
    for ts in (pd.Timestamp(x) for x in signal_dates):
        if ts not in idx:
            continue
        sp = idx.get_loc(ts)
        ep = sp + lag + 1
        if ep >= n_dates:
            continue
        exec_idx[ep] = 1
        row = w_vals[sp].copy()
        row[px.iloc[ep].isna().to_numpy()] = 0.0     # can't trade a name with no execution price
        w_target_np[ep] = row

    equity_np, turn_np, cost_np, borrow_np = _drift_core(
        rets.to_numpy(float), w_target_np, tc_np, float(short_cost_mult),
        bf_np, float(freq), exec_idx, n_dates, n_assets
    )

    equity = pd.Series(equity_np, index=idx)
    turnover = pd.Series(0.5 * turn_np, index=idx, name="turnover")    # one-way turnover (reporting)
    cost_frac = pd.Series(cost_np, index=idx, name="cost_frac")
    borrow_frac = pd.Series(borrow_np, index=idx, name="borrow_frac")
    out = _metrics(equity, turnover, cost_frac, borrow_frac, float(freq), float(risk_free_rate))
    out["transaction_cost"] = transaction_cost if np.isscalar(transaction_cost) else "tiered"
    return out


def _rate_matrix(val, idx, cols, n_dates, n_assets) -> np.ndarray:
    """Coerce a scalar or (dates×tickers) DataFrame rate into a dense (T, N) array."""
    if isinstance(val, (pd.DataFrame, pd.Series)):
        df = val if isinstance(val, pd.DataFrame) else val.to_frame().T
        fallback = float(np.nanmax(df.to_numpy())) if df.size else 0.0
        return df.reindex(index=idx, columns=cols).ffill().fillna(fallback).to_numpy(float)
    if val < 0:
        raise ValueError("rate must be >= 0.")
    return np.full((n_dates, n_assets), float(val))


# ── generic out-of-sample walk-forward ──────────────────────────────────────
def walk_forward(
    signal_fn,
    prices,
    *,
    train: int,
    test: int,
    freq: int = 252,
    lag: int = 1,
    transaction_cost=0.0,
    borrow_fee=0.0,
    risk_free_rate: float = 0.0,
    signal_kwargs: dict | None = None,
) -> dict:
    """
    Generic rolling train→test out-of-sample backtest.

    For each block the model is fit on a `train`-row window and traded over the next
    `test`-row window — so every traded return is strictly out-of-sample:

        [── train ──][── test ──]
                     [── train ──][── test ──]   (step = test)

    signal_fn(train_prices, **signal_kwargs) -> weights
        Returns either a static weight Series (by ticker) held over the test block, or a
        DataFrame (dates×tickers) covering the test dates.  Called once per block on the
        TRAIN slice only — it never sees test data.

    Returns the same metrics dict as `backtest()`, computed over the stitched OOS span.
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
        train_px = px.iloc[start - train:start]
        test_px = px.iloc[start:start + test]
        if test_px.shape[0] == 0:
            break
        w = signal_fn(train_px, **kw)
        block_starts.append(test_px.index[0])
        if isinstance(w, pd.DataFrame):
            for d, row in w.reindex(columns=px.columns).iterrows():
                weight_rows[d] = row
        else:                                          # static weights → hold across the block
            row = pd.Series(w).reindex(px.columns).fillna(0.0)
            for d in test_px.index:
                weight_rows[d] = row

    if not weight_rows:
        raise ValueError("walk_forward produced no weights.")

    weights = pd.DataFrame(weight_rows).T.sort_index()
    oos_px = px.loc[block_starts[0]:]
    return backtest(
        weights, oos_px,
        freq=freq, lag=lag, signal_dates=sorted(weight_rows),
        transaction_cost=transaction_cost, borrow_fee=borrow_fee, risk_free_rate=risk_free_rate,
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
    ("Ann. Borrow", "ann_borrow", ".2%"),
    ("Total Cost", "total_cost", ".2%"),
]


def _yearly_returns(returns: pd.Series, equity: pd.Series, drawdown: pd.Series) -> pd.DataFrame:
    rows = []
    for yr in sorted(set(returns.index.year)):
        eq = equity[equity.index.year == yr].dropna()
        dd = drawdown[drawdown.index.year == yr].dropna()
        if len(eq) == 0:
            continue
        rows.append({
            "Year": yr,
            "Return": eq.iloc[-1] / eq.iloc[0] - 1.0 if eq.iloc[0] > 0 else np.nan,
            "Max DD": float(dd.min()) if len(dd) else np.nan,
        })
    return pd.DataFrame(rows).set_index("Year") if rows else pd.DataFrame()


def results_backtest(strategies: dict, *, title: str | None = None,
                     figsize: tuple = (14, 8), **_ignored) -> dict:
    """
    Summary table + equity/drawdown chart for one or many backtest result dicts.
    `strategies`: {name: result_dict}.  Returns {'fig','axes','summary_df','yearly_df'}.
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

    summary = {}
    for name, res in strategies.items():
        summary[name] = {
            disp: (f"{res.get(key, np.nan):{fmt}}" if pd.notna(res.get(key, np.nan)) else "N/A")
            for disp, key, fmt in _SUMMARY_METRICS
        }
    summary_df = pd.DataFrame(summary).T

    yearly = {}
    for name, res in strategies.items():
        if {"returns", "equity", "drawdown"} <= res.keys():
            y = _yearly_returns(res["returns"], res["equity"], res["drawdown"])
            if len(y):
                yearly[name] = y
    yearly_df = pd.DataFrame()
    if yearly:
        years = sorted(set().union(*[df.index for df in yearly.values()]))
        yearly_df = pd.DataFrame(index=years)
        for name, y in yearly.items():
            yearly_df[f"{name} Return"] = y["Return"]
            yearly_df[f"{name} Max DD"] = y["Max DD"]

    return {"fig": fig, "axes": (ax1, ax2), "summary_df": summary_df, "yearly_df": yearly_df}


# ── smoke test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-01-31", periods=180, freq="ME")
    px = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.01, 0.05, size=(180, 6)), axis=0)),
        index=dates, columns=[f"A{i}" for i in range(6)],
    )

    # 1) static equal-weight long-only
    w = pd.Series(1 / 6, index=px.columns)
    r = backtest(w, px, freq=12, lag=0, transaction_cost=0.001)
    print(f"[static]  ann={r['ann_return']:.2%}  sharpe={r['sharpe']:.2f}  maxDD={r['max_drawdown']:.2%}")

    # 2) tiered costs from a synthetic dollar-volume matrix
    dv = pd.DataFrame(rng.uniform(1e5, 5e9, size=px.shape), index=dates, columns=px.columns)
    tc = tiered_transaction_costs(dv)
    r2 = backtest(w, px, freq=12, lag=0, transaction_cost=tc)
    print(f"[tiered]  ann={r2['ann_return']:.2%}  ann_cost={r2['total_cost']:.2%}")

    # 3) generic out-of-sample walk-forward (momentum signal)
    def mom_signal(train_px):
        m = train_px.iloc[-1] / train_px.iloc[0] - 1.0
        return (m == m.max()).astype(float)           # hold last winner
    rw = walk_forward(mom_signal, px, train=24, test=3, freq=12, lag=0, transaction_cost=0.001)
    print(f"[walkfwd] ann={rw['ann_return']:.2%}  sharpe={rw['sharpe']:.2f}")
    print("OK")