"""Typed request builders, one per EODHD endpoint we use.

Keeping URL/param construction here means the job modules never hand-assemble
URLs, and the ledger keys stay consistent across runs.
"""

from __future__ import annotations

from .client import RequestSpec


def exchanges_list() -> RequestSpec:
    """All supported exchanges. 1 call."""
    return RequestSpec(endpoint="exchanges-list", path="/exchanges-list/", key="ALL")


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
