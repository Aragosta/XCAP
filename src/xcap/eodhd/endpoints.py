"""Typed request builders, one per EODHD endpoint we use.

Keeping URL/param construction here means the job modules never hand-assemble
URLs, and the ledger keys stay consistent across runs.
"""

from __future__ import annotations

from .client import RequestSpec


def exchanges_list() -> RequestSpec:
    """All supported exchanges. 1 call."""
    return RequestSpec(endpoint="exchanges-list", path="/exchanges-list/", key="ALL")


def eod(ticker: str) -> RequestSpec:
    """Full daily OHLCV history for one security. 1 call.

    Deliberately unbounded: a date-limited request costs the same single call,
    so archiving everything keeps the start-year decision reversible. The date
    floor is applied when building parquet, not when fetching.

    OHLC values are raw — adjusted for neither splits nor dividends. The vendor's
    adjusted_close is retained only so xcap.corpactions can reconcile against it.
    """
    return RequestSpec(endpoint="eod", path=f"/eod/{ticker}", key=ticker,
                       params={"period": "d", "order": "a"})


def splits(ticker: str) -> RequestSpec:
    """Split history. 1 call. Required to rebuild adjustment factors locally."""
    return RequestSpec(endpoint="splits", path=f"/splits/{ticker}", key=ticker,
                       params={"from": "1900-01-01"})


def dividends(ticker: str) -> RequestSpec:
    """Dividend history. 1 call. Ex-dates plus declaration/record/payment dates."""
    return RequestSpec(endpoint="dividends", path=f"/div/{ticker}", key=ticker,
                       params={"from": "1900-01-01"})


def exchange_symbol_list(exchange_code: str, *, delisted: bool = False) -> RequestSpec:
    """Tickers for one exchange. 1 call.

    Without `delisted`, EODHD returns only tickers active in the past month.
    With `delisted=1` it returns *only* inactive tickers — the two lists are
    disjoint by design, and both are required to avoid survivorship bias.
    """
    params: dict[str, object] = {}
    if delisted:
        params["delisted"] = 1
    return RequestSpec(
        endpoint="exchange-symbol-list",
        path=f"/exchange-symbol-list/{exchange_code}",
        key=f"{exchange_code}{'.delisted' if delisted else '.active'}",
        params=params,
    )
